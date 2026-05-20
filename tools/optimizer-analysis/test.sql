
CREATE OR REPLACE TABLE A AS SELECT i::INT AS a, (random() * 100)::INT AS b FROM range(1, 1001) t(i);
CREATE OR REPLACE TABLE B AS SELECT (1 + floor(random() * 1000))::INT AS a, (1 + floor(random() * 1000))::INT AS c FROM range(1, 10001) t(i);
CREATE OR REPLACE TABLE C AS SELECT (1 + floor(random() * 1000))::INT AS c, (random() * 100)::DOUBLE AS d FROM range(1, 100001) t(i);
.timer on
SELECT a.a, SUM(c.d) FROM A a JOIN B b ON b.a = a.a JOIN C c ON c.c = b.c GROUP BY a.a ORDER BY a.a;
