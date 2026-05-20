import subprocess
import hashlib
import os

with open("test.sql", "w") as f:
    f.write("""
CREATE OR REPLACE TABLE A AS SELECT i::INT AS a, (random() * 100)::INT AS b FROM range(1, 1001) t(i);
CREATE OR REPLACE TABLE B AS SELECT (1 + floor(random() * 1000))::INT AS a, (1 + floor(random() * 1000))::INT AS c FROM range(1, 10001) t(i);
CREATE OR REPLACE TABLE C AS SELECT (1 + floor(random() * 1000))::INT AS c, (random() * 100)::DOUBLE AS d FROM range(1, 100001) t(i);
.timer on
SELECT a.a, SUM(c.d) FROM A a JOIN B b ON b.a = a.a JOIN C c ON c.c = b.c GROUP BY a.a ORDER BY a.a;
""")

res1 = subprocess.run(["duckdb/build/release/duckdb", ":memory:", "-init", "test.sql"], capture_output=True, text=True)
res2 = subprocess.run(["duckdb/build/release/duckdb", ":memory:", "-init", "test.sql"], capture_output=True, text=True)

with open("out1.txt", "w") as f: f.write(res1.stdout)
with open("out2.txt", "w") as f: f.write(res2.stdout)

print("out1 length:", len(res1.stdout), "out2 length:", len(res2.stdout))
os.system("diff out1.txt out2.txt | head -n 20")
