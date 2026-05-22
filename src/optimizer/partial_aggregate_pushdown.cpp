#include "duckdb/optimizer/partial_aggregate_pushdown.hpp"

#include "duckdb/function/aggregate/distributive_functions.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/generic_common.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

PartialAggregatePushdown::PartialAggregatePushdown(Optimizer &optimizer_p) : optimizer(optimizer_p) {
}

struct PartialAggregatePushdownHeuristics {
	static constexpr idx_t MIN_DIMENSION_GROUPS = 1;
	static constexpr idx_t MIN_AGGREGATE_TO_DIMENSION_RATIO = 4;
	static constexpr idx_t MAX_JOIN_SELECTIVITY_INV = 8;
	static constexpr idx_t MAX_EXTRA_LOWER_GROUPS = 1;
};

struct PartialAggregatePushdownInfo {
	idx_t aggregate_side;
	idx_t dimension_side;
	TableIndex lower_group_index;
	TableIndex lower_aggregate_index;
	idx_t join_key_count;
	unordered_set<TableIndex> side_bindings[2];
	vector<ColumnBinding> lower_group_bindings;
	column_binding_map_t<ColumnBinding> lower_group_map;
	column_binding_map_t<LogicalType> lower_group_types;
	// Per-aggregate classification: true = fact-side (exported to lower aggregate + combined
	// in upper), false = dimension-side (kept in upper aggregate unchanged).
	// Dimension-side aggregates are only allowed for MIN and MAX — the only aggregates that
	// are correct after pushdown regardless of how many fact rows share a join key (the
	// lower aggregate reduces the fact side to one row per join key, so COUNT/SUM over
	// dimension columns would observe fewer rows than the original join).
	vector<bool> is_fact_side_aggregate;
};

static bool IsSubset(const unordered_set<TableIndex> &bindings, const unordered_set<TableIndex> &side_bindings) {
	for (auto &binding : bindings) {
		if (side_bindings.find(binding) == side_bindings.end()) {
			return false;
		}
	}
	return true;
}

static unordered_set<TableIndex> GetExpressionBindings(const Expression &expr) {
	unordered_set<TableIndex> bindings;
	LogicalJoin::GetExpressionBindings(expr, bindings);
	return bindings;
}

static bool GetColumnBinding(const Expression &expr, ColumnBinding &binding) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return false;
	}
	binding = expr.Cast<BoundColumnRefExpression>().binding;
	return true;
}

// Returns true for aggregates that are correct when left in the upper aggregate after
// pushdown — i.e., they operate over the post-join result where the fact side has been
// reduced to one row per join key. MIN and MAX are correct because:
//   - Under unique dimension join keys: each dim attribute appears once per group → same result.
//   - Under duplicate dimension join keys: the join still returns all dim rows (the dim side
//     is unmodified); MIN/MAX over those rows = MIN/MAX over the original N-row join result.
// COUNT and SUM are explicitly excluded: they produce wrong results when fewer fact rows are
// visible in the upper aggregate (post-pushdown count per group = 1, not the original N).
static bool IsSupportedDimSideAggregate(const BoundAggregateExpression &expr) {
	if (expr.IsDistinct() || expr.filter || expr.order_bys || expr.children.size() != 1) {
		return false;
	}
	const auto &name = expr.function.GetName();
	return name == "min" || name == "max";
}

static bool IsSupportedAggregate(const BoundAggregateExpression &expr) {
	if (expr.IsDistinct() || expr.filter || expr.order_bys) {
		return false;
	}
	if (expr.children.size() > 1) {
		return false;
	}
	// Gate on HasGetStateTypeCallback (not HasStateCombineCallback) intentionally:
	// ExportAggregateFunction::Bind produces AGGREGATE_STATE (struct-typed) only when
	// get_state_type is set; otherwise it falls back to LEGACY_AGGREGATE_STATE (opaque blob).
	// The upper combine aggregate (CombineAggrFun) only accepts AGGREGATE_STATE — there is
	// no AggregateFunction overload for LEGACY_AGGREGATE_STATE. Aggregates lacking
	// get_state_type (e.g. skew, approx_count_distinct) are therefore correctly excluded:
	// they have a combine callback but cannot participate in the EXPORT_STATE →
	// COMBINE_AGGR → FINALIZE pipeline this pass builds.
	if (!expr.function.HasGetStateTypeCallback()) {
		return false;
	}
	return true;
}

static bool ContainsAggregateInput(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return true;
	}
	for (auto &child : op.children) {
		if (ContainsAggregateInput(*child)) {
			return true;
		}
	}
	return false;
}

static bool GetPushdownOperators(LogicalOperator &op, LogicalAggregate *&aggr, LogicalComparisonJoin *&join) {
	if (op.type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY || op.children.size() != 1) {
		return false;
	}
	aggr = &op.Cast<LogicalAggregate>();
	if (!aggr->grouping_functions.empty() || aggr->groups.empty() || aggr->expressions.empty()) {
		return false;
	}
	auto &child = *op.children[0];
	if (child.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return false;
	}
	join = &child.Cast<LogicalComparisonJoin>();
	return join->join_type == JoinType::INNER && !join->HasProjectionMap() && join->children.size() == 2 &&
	       !join->conditions.empty();
}

static bool GetExpressionSide(const Expression &expr, const PartialAggregatePushdownInfo &info, idx_t &side) {
	auto bindings = GetExpressionBindings(expr);
	if (bindings.empty()) {
		return false;
	}
	if (IsSubset(bindings, info.side_bindings[0])) {
		side = 0;
		return true;
	}
	if (IsSubset(bindings, info.side_bindings[1])) {
		side = 1;
		return true;
	}
	return false;
}

static bool FindAggregateSide(const LogicalAggregate &aggr, PartialAggregatePushdownInfo &info) {
	const idx_t n = aggr.expressions.size();
	info.is_fact_side_aggregate.assign(n, false);

	optional_idx aggregate_side; // fact side: determined by first non-COUNT(*) aggregate
	bool has_count_star = false;

	// First pass: classify each aggregate as fact-side or potential dim-side.
	for (idx_t i = 0; i < n; i++) {
		auto &expr = aggr.expressions[i];
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
		auto &aggregate = expr->Cast<BoundAggregateExpression>();
		if (aggregate.children.empty()) {
			// COUNT(*) — no input column, does not constrain the aggregate side.
			// Will be marked fact-side after aggregate_side is determined.
			has_count_star = true;
			continue;
		}
		idx_t side;
		if (!GetExpressionSide(*aggregate.children[0], info, side)) {
			return false;
		}
		if (!aggregate_side.IsValid()) {
			// First non-COUNT(*) aggregate determines the fact side.
			aggregate_side = side;
			info.is_fact_side_aggregate[i] = true;
		} else if (aggregate_side.GetIndex() == side) {
			// Same side as already determined fact side — pushable.
			info.is_fact_side_aggregate[i] = true;
		} else {
			// Opposite side: allowed only for MIN / MAX.
			// These are safe because the lower aggregate does not change how many
			// dim rows appear per group in the upper join result — MIN/MAX over
			// one or N identical dim values yields the same result either way.
			if (!IsSupportedAggregate(aggregate) || !IsSupportedDimSideAggregate(aggregate)) {
				return false;
			}
			info.is_fact_side_aggregate[i] = false; // dim-side: stays in upper unchanged
		}
	}

	if (!aggregate_side.IsValid()) {
		// Pure COUNT(*) — infer fact side from GROUP BY columns.
		if (!has_count_star) {
			return false;
		}
		optional_idx inferred_dimension_side;
		for (auto &group : aggr.groups) {
			idx_t side;
			if (!GetExpressionSide(*group, info, side)) {
				return false;
			}
			if (!inferred_dimension_side.IsValid()) {
				inferred_dimension_side = side;
			} else if (inferred_dimension_side.GetIndex() != side) {
				return false;
			}
		}
		if (!inferred_dimension_side.IsValid()) {
			return false;
		}
		aggregate_side = 1 - inferred_dimension_side.GetIndex();
	}

	// Mark all COUNT(*)s as fact-side (pushed to lower aggregate).
	for (idx_t i = 0; i < n; i++) {
		if (aggr.expressions[i]->Cast<BoundAggregateExpression>().children.empty()) {
			info.is_fact_side_aggregate[i] = true;
		}
	}

	info.aggregate_side = aggregate_side.GetIndex();
	info.dimension_side = 1 - info.aggregate_side;
	return true;
}

static bool PassesCardinalityHeuristic(LogicalComparisonJoin &join, const PartialAggregatePushdownInfo &info,
                                       ClientContext &context) {
	auto &aggregate_child = *join.children[info.aggregate_side];
	auto &dimension_child = *join.children[info.dimension_side];
	// Fall back to EstimateCardinality when the join-order optimizer has not yet
	// populated the estimated_cardinality fields. EstimateCardinality walks the
	// subtree recursively (LogicalGet reads from table statistics / function
	// cardinality callbacks) and caches the result, so subsequent calls are cheap.
	const idx_t agg_card = aggregate_child.has_estimated_cardinality
	                           ? aggregate_child.estimated_cardinality
	                           : aggregate_child.EstimateCardinality(context);
	const idx_t dim_card = dimension_child.has_estimated_cardinality
	                           ? dimension_child.estimated_cardinality
	                           : dimension_child.EstimateCardinality(context);
	if (agg_card < PartialAggregatePushdownHeuristics::MIN_AGGREGATE_TO_DIMENSION_RATIO * dim_card) {
		return false;
	}
	const idx_t join_card = join.has_estimated_cardinality ? join.estimated_cardinality
	                                                       : join.EstimateCardinality(context);
	if (join_card * PartialAggregatePushdownHeuristics::MAX_JOIN_SELECTIVITY_INV < agg_card) {
		return false;
	}
	return true;
}

static bool GetJoinSideExpressions(JoinCondition &condition, const PartialAggregatePushdownInfo &info,
                                   unique_ptr<Expression> *&aggregate_expr, unique_ptr<Expression> *&dimension_expr) {
	if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
		return false;
	}
	auto left_bindings = GetExpressionBindings(condition.GetLHS());
	auto right_bindings = GetExpressionBindings(condition.GetRHS());
	if (left_bindings.empty() || right_bindings.empty()) {
		return false;
	}
	if (IsSubset(left_bindings, info.side_bindings[info.aggregate_side]) &&
	    IsSubset(right_bindings, info.side_bindings[info.dimension_side])) {
		aggregate_expr = &condition.LeftReference();
		dimension_expr = &condition.RightReference();
		return true;
	}
	if (IsSubset(right_bindings, info.side_bindings[info.aggregate_side]) &&
	    IsSubset(left_bindings, info.side_bindings[info.dimension_side])) {
		aggregate_expr = &condition.RightReference();
		dimension_expr = &condition.LeftReference();
		return true;
	}
	return false;
}

static bool ValidateJoinConditions(LogicalComparisonJoin &join, const PartialAggregatePushdownInfo &info) {
	for (auto &condition : join.conditions) {
		unique_ptr<Expression> *aggregate_expr;
		unique_ptr<Expression> *dimension_expr;
		if (!GetJoinSideExpressions(condition, info, aggregate_expr, dimension_expr)) {
			return false;
		}
		ColumnBinding join_key;
		if (!GetColumnBinding(**aggregate_expr, join_key)) {
			return false;
		}
	}
	return true;
}

static bool HasWideDimensionGroups(const LogicalAggregate &aggr, const PartialAggregatePushdownInfo &info) {
	idx_t dimension_group_count = 0;
	for (auto &group : aggr.groups) {
		ColumnBinding group_binding;
		if (!GetColumnBinding(*group, group_binding)) {
			return false;
		}
		idx_t side;
		if (!GetExpressionSide(*group, info, side)) {
			return false;
		}
		if (side == info.dimension_side) {
			dimension_group_count++;
		}
	}
	return dimension_group_count >= PartialAggregatePushdownHeuristics::MIN_DIMENSION_GROUPS;
}

static bool AnalyzePushdown(LogicalAggregate &aggr, LogicalComparisonJoin &join, PartialAggregatePushdownInfo &info,
                            ClientContext &context) {
	LogicalJoin::GetTableReferences(*join.children[0], info.side_bindings[0]);
	LogicalJoin::GetTableReferences(*join.children[1], info.side_bindings[1]);
	if (!FindAggregateSide(aggr, info)) {
		return false;
	}
	if (ContainsAggregateInput(*join.children[info.aggregate_side])) {
		return false;
	}
	if (info.side_bindings[info.dimension_side].size() != 1) {
		return false;
	}
	if (!PassesCardinalityHeuristic(join, info, context)) {
		return false;
	}
	if (!ValidateJoinConditions(join, info)) {
		return false;
	}
	return HasWideDimensionGroups(aggr, info);
}

static void AddLowerGroup(PartialAggregatePushdownInfo &info, ColumnBinding binding, const LogicalType &type) {
	if (info.lower_group_map.find(binding) != info.lower_group_map.end()) {
		return;
	}
	auto lower_binding = ColumnBinding(info.lower_group_index, ProjectionIndex(info.lower_group_bindings.size()));
	info.lower_group_bindings.push_back(binding);
	info.lower_group_map[binding] = lower_binding;
	info.lower_group_types[binding] = type;
}

static void BuildLowerGroupMap(LogicalAggregate &aggr, LogicalComparisonJoin &join,
                               PartialAggregatePushdownInfo &info) {
	for (auto &condition : join.conditions) {
		unique_ptr<Expression> *aggregate_expr;
		unique_ptr<Expression> *dimension_expr;
		if (!GetJoinSideExpressions(condition, info, aggregate_expr, dimension_expr)) {
			continue;
		}
		ColumnBinding binding;
		if (!GetColumnBinding(**aggregate_expr, binding)) {
			continue;
		}
		AddLowerGroup(info, binding, (*aggregate_expr)->GetReturnType());
	}
	info.join_key_count = info.lower_group_bindings.size();
	for (auto &group : aggr.groups) {
		auto &group_ref = group->Cast<BoundColumnRefExpression>();
		if (info.lower_group_map.find(group_ref.binding) != info.lower_group_map.end()) {
			continue;
		}
		idx_t side;
		GetExpressionSide(*group, info, side);
		if (side == info.aggregate_side) {
			AddLowerGroup(info, group_ref.binding, group->GetReturnType());
		}
	}
}

static bool PassesLowerGroupHeuristic(const PartialAggregatePushdownInfo &info) {
	return info.lower_group_bindings.size() <=
	       info.join_key_count + PartialAggregatePushdownHeuristics::MAX_EXTRA_LOWER_GROUPS;
}

static bool BindPushdownAggregates(ClientContext &context, LogicalAggregate &aggr,
                                   const PartialAggregatePushdownInfo &info,
                                   vector<unique_ptr<Expression>> &lower_aggregates,
                                   vector<unique_ptr<Expression>> &upper_aggregates) {
	auto combine_function = CombineAggrFun::GetFunction();
	FunctionBinder function_binder(context);
	idx_t lower_idx = 0; // index into lower_aggregates (fact-side only)

	for (idx_t i = 0; i < aggr.expressions.size(); i++) {
		auto aggregate_copy = unique_ptr_cast<Expression, BoundAggregateExpression>(aggr.expressions[i]->Copy());

		if (!info.is_fact_side_aggregate[i]) {
			// Dimension-side aggregate (MIN/MAX on dim column): keep in upper aggregate
			// unchanged. No lower aggregate is created for this slot.
			upper_aggregates.push_back(std::move(aggregate_copy));
			continue;
		}

		// Fact-side aggregate: export state to lower, combine in upper.
		auto lower_aggregate = ExportAggregateFunction::Bind(std::move(aggregate_copy));
		auto lower_type = lower_aggregate->GetReturnType();
		if (lower_type.id() != LogicalTypeId::AGGREGATE_STATE) {
			return false;
		}

		vector<unique_ptr<Expression>> arguments;
		auto lower_binding = ColumnBinding(info.lower_aggregate_index, ProjectionIndex(lower_idx++));
		arguments.push_back(make_uniq<BoundColumnRefExpression>(lower_type, lower_binding));
		auto upper_aggregate = function_binder.BindAggregateFunction(combine_function, std::move(arguments));
		if (upper_aggregate->GetReturnType().id() != LogicalTypeId::AGGREGATE_STATE) {
			return false;
		}
		lower_aggregates.push_back(std::move(lower_aggregate));
		upper_aggregates.push_back(std::move(upper_aggregate));
	}
	return true;
}

static vector<unique_ptr<Expression>> CreateLowerGroups(const PartialAggregatePushdownInfo &info) {
	vector<unique_ptr<Expression>> lower_groups;
	for (auto &binding : info.lower_group_bindings) {
		auto type = info.lower_group_types.at(binding);
		lower_groups.push_back(make_uniq<BoundColumnRefExpression>(type, binding));
	}
	return lower_groups;
}

static unique_ptr<LogicalAggregate> CreateLowerAggregate(LogicalAggregate &aggr, LogicalComparisonJoin &join,
                                                         PartialAggregatePushdownInfo &info,
                                                         vector<unique_ptr<Expression>> lower_aggregates) {
	auto lower_aggr =
	    make_uniq<LogicalAggregate>(info.lower_group_index, info.lower_aggregate_index, std::move(lower_aggregates));
	lower_aggr->groups = CreateLowerGroups(info);
	lower_aggr->children.push_back(std::move(join.children[info.aggregate_side]));
	lower_aggr->ResolveOperatorTypes();
	lower_aggr->estimated_cardinality = aggr.estimated_cardinality;
	lower_aggr->has_estimated_cardinality = aggr.has_estimated_cardinality;
	return lower_aggr;
}

static unique_ptr<LogicalComparisonJoin> CreateJoin(LogicalComparisonJoin &join, PartialAggregatePushdownInfo &info,
                                                    unique_ptr<LogicalAggregate> lower_aggr) {
	auto new_join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	if (info.aggregate_side == 0) {
		new_join->children.push_back(std::move(lower_aggr));
		new_join->children.push_back(std::move(join.children[info.dimension_side]));
	} else {
		new_join->children.push_back(std::move(join.children[info.dimension_side]));
		new_join->children.push_back(std::move(lower_aggr));
	}

	for (auto &condition : join.conditions) {
		unique_ptr<Expression> *aggregate_expr;
		unique_ptr<Expression> *dimension_expr;
		GetJoinSideExpressions(condition, info, aggregate_expr, dimension_expr);
		ColumnBinding join_key;
		GetColumnBinding(**aggregate_expr, join_key);
		auto lower_binding = info.lower_group_map[join_key];
		auto lower_type = new_join->children[info.aggregate_side]->types[lower_binding.column_index.GetIndex()];
		auto lower_expr = make_uniq<BoundColumnRefExpression>(lower_type, lower_binding);
		if (info.aggregate_side == 0) {
			new_join->conditions.emplace_back(std::move(lower_expr), (*dimension_expr)->Copy(),
			                                  ExpressionType::COMPARE_EQUAL);
		} else {
			new_join->conditions.emplace_back((*dimension_expr)->Copy(), std::move(lower_expr),
			                                  ExpressionType::COMPARE_EQUAL);
		}
	}
	new_join->ResolveOperatorTypes();
	new_join->estimated_cardinality = join.estimated_cardinality;
	new_join->has_estimated_cardinality = join.has_estimated_cardinality;
	return new_join;
}

static vector<unique_ptr<Expression>> CreateUpperGroups(LogicalAggregate &aggr, LogicalComparisonJoin &new_join,
                                                        const PartialAggregatePushdownInfo &info) {
	vector<unique_ptr<Expression>> upper_groups;
	for (auto &group : aggr.groups) {
		auto &group_ref = group->Cast<BoundColumnRefExpression>();
		auto entry = info.lower_group_map.find(group_ref.binding);
		if (entry == info.lower_group_map.end()) {
			upper_groups.push_back(group->Copy());
			continue;
		}
		auto type = new_join.children[info.aggregate_side]->types[entry->second.column_index.GetIndex()];
		upper_groups.push_back(make_uniq<BoundColumnRefExpression>(type, entry->second));
	}
	return upper_groups;
}

static unique_ptr<LogicalAggregate> CreateUpperAggregate(LogicalAggregate &aggr,
                                                         unique_ptr<LogicalComparisonJoin> new_join,
                                                         const PartialAggregatePushdownInfo &info,
                                                         vector<unique_ptr<Expression>> upper_aggregates) {
	auto upper_aggr = make_uniq<LogicalAggregate>(aggr.group_index, aggr.aggregate_index, std::move(upper_aggregates));
	upper_aggr->groups = CreateUpperGroups(aggr, *new_join, info);
	// Preserve ROLLUP / CUBE / GROUPING SETS. CreateUpperGroups builds the
	// upper groups in the same order as `aggr.groups`, so the indices stored
	// in grouping_sets remain valid. Without this copy the upper aggregate
	// collapses to a single set and silently over-aggregates.
	upper_aggr->grouping_sets = aggr.grouping_sets;
	upper_aggr->grouping_functions = aggr.grouping_functions;
	upper_aggr->children.push_back(std::move(new_join));
	upper_aggr->ResolveOperatorTypes();
	upper_aggr->estimated_cardinality = aggr.estimated_cardinality;
	upper_aggr->has_estimated_cardinality = aggr.has_estimated_cardinality;
	return upper_aggr;
}

static unique_ptr<LogicalProjection> CreateFinalProjection(Optimizer &optimizer, LogicalAggregate &aggr,
                                                           unique_ptr<LogicalAggregate> upper_aggr,
                                                           const PartialAggregatePushdownInfo &info,
                                                           column_binding_map_t<ColumnBinding> &replacement_map) {
	const auto proj_index = optimizer.binder.GenerateTableIndex();
	const auto group_count = aggr.groups.size();
	vector<unique_ptr<Expression>> projection_expressions;
	projection_expressions.reserve(group_count + aggr.expressions.size());

	for (idx_t group_idx = 0; group_idx < group_count; group_idx++) {
		auto group_binding = ColumnBinding(upper_aggr->group_index, ProjectionIndex(group_idx));
		replacement_map[group_binding] = ColumnBinding(proj_index, ProjectionIndex(group_idx));
		projection_expressions.push_back(
		    make_uniq<BoundColumnRefExpression>(upper_aggr->types[group_idx], group_binding));
	}

	for (idx_t aggr_idx = 0; aggr_idx < aggr.expressions.size(); aggr_idx++) {
		auto aggregate_binding = ColumnBinding(upper_aggr->aggregate_index, ProjectionIndex(aggr_idx));
		replacement_map[aggregate_binding] = ColumnBinding(proj_index, ProjectionIndex(group_count + aggr_idx));
		auto aggregate_type = upper_aggr->types[group_count + aggr_idx];
		auto aggregate_ref = make_uniq<BoundColumnRefExpression>(aggregate_type, aggregate_binding);

		if (!info.is_fact_side_aggregate[aggr_idx]) {
			// Dimension-side aggregate (MIN/MAX): upper aggregate already holds the
			// final value — no finalize() wrapper needed.
			if (aggregate_type != aggr.expressions[aggr_idx]->GetReturnType()) {
				return nullptr;
			}
			projection_expressions.push_back(std::move(aggregate_ref));
			continue;
		}

		// Fact-side aggregate: upper aggregate holds AGGREGATE_STATE — wrap with finalize().
		auto final_expression = optimizer.BindScalarFunction("finalize", std::move(aggregate_ref));
		if (final_expression->GetReturnType() != aggr.expressions[aggr_idx]->GetReturnType()) {
			return nullptr;
		}
		projection_expressions.push_back(std::move(final_expression));
	}

	auto projection = make_uniq<LogicalProjection>(proj_index, std::move(projection_expressions));
	if (upper_aggr->has_estimated_cardinality) {
		projection->SetEstimatedCardinality(upper_aggr->estimated_cardinality);
	}
	projection->children.push_back(std::move(upper_aggr));
	projection->ResolveOperatorTypes();
	return projection;
}

void PartialAggregatePushdown::VisitOperator(unique_ptr<LogicalOperator> &op) {
	LogicalOperatorVisitor::VisitOperator(op);
	if (TryPushdownAggregate(op)) {
		modified = true;
	}
}

unique_ptr<Expression> PartialAggregatePushdown::VisitReplace(BoundColumnRefExpression &expr,
                                                              unique_ptr<Expression> *expr_ptr) {
	auto entry = replacement_map.find(expr.binding);
	if (entry != replacement_map.end()) {
		expr.binding = entry->second;
	}
	return nullptr;
}

static bool IsAcyclic(LogicalOperator &op) {
	vector<std::pair<TableIndex, TableIndex>> edges;
	struct Helper {
		static void Collect(const LogicalOperator &op, vector<std::pair<TableIndex, TableIndex>> &edges) {
			if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				auto &join = op.Cast<LogicalComparisonJoin>();
				for (auto &cond : join.conditions) {
					unordered_set<TableIndex> left_bindings;
					LogicalJoin::GetExpressionBindings(cond.GetLHS(), left_bindings);
					unordered_set<TableIndex> right_bindings;
					LogicalJoin::GetExpressionBindings(cond.GetRHS(), right_bindings);
					for (auto u : left_bindings) {
						for (auto v : right_bindings) {
							if (u != v) {
								edges.push_back({u, v});
							}
						}
					}
				}
			}
			for (auto &child : op.children) {
				if (child) {
					Collect(*child, edges);
				}
			}
		}
	};
	Helper::Collect(op, edges);

	struct TableIndexDSU {
		unordered_map<idx_t, idx_t> parent;
		idx_t Find(idx_t i) {
			if (parent.find(i) == parent.end()) {
				parent[i] = i;
				return i;
			}
			if (parent[i] == i) {
				return i;
			}
			return parent[i] = Find(parent[i]);
		}
		bool Union(idx_t x, idx_t y) {
			auto root_x = Find(x);
			auto root_y = Find(y);
			if (root_x == root_y) {
				return false;
			}
			parent[root_x] = root_y;
			return true;
		}
	} dsu;

	for (auto &edge : edges) {
		if (!dsu.Union(edge.first.index, edge.second.index)) {
			return false; // cycle detected!
		}
	}
	return true;
}

bool PartialAggregatePushdown::TryPushdownAggregate(unique_ptr<LogicalOperator> &op) {
	LogicalAggregate *aggr;
	LogicalComparisonJoin *join;
	if (!GetPushdownOperators(*op, aggr, join)) {
		return false;
	}

	if (!IsAcyclic(*op->children[0])) {
		return false;
	}

	PartialAggregatePushdownInfo info;
	if (!AnalyzePushdown(*aggr, *join, info, optimizer.context)) {
		return false;
	}
	info.lower_group_index = optimizer.binder.GenerateTableIndex();
	info.lower_aggregate_index = optimizer.binder.GenerateTableIndex();
	BuildLowerGroupMap(*aggr, *join, info);
	if (!PassesLowerGroupHeuristic(info)) {
		return false;
	}

	vector<unique_ptr<Expression>> lower_aggregates;
	vector<unique_ptr<Expression>> upper_aggregates;
	if (!BindPushdownAggregates(optimizer.context, *aggr, info, lower_aggregates, upper_aggregates)) {
		return false;
	}

	auto lower_aggr = CreateLowerAggregate(*aggr, *join, info, std::move(lower_aggregates));
	auto new_join = CreateJoin(*join, info, std::move(lower_aggr));
	auto upper_aggr = CreateUpperAggregate(*aggr, std::move(new_join), info, std::move(upper_aggregates));
	auto final_projection = CreateFinalProjection(optimizer, *aggr, std::move(upper_aggr), info, replacement_map);
	if (!final_projection) {
		return false;
	}
	op = std::move(final_projection);
	return true;
}

} // namespace duckdb
