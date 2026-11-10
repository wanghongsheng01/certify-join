"""Load the JOB / IMDB dump (Leis et al., PVLDB 2015; CSVs from event.cwi.nl/da/job/imdb.tgz) into DuckDB on the
external disk. Constraints are dropped so that the raw data are loaded as shipped (no key is enforced)."""
import duckdb, glob, os, re, sys, time
J = "<DATA_DIR>"
SCHEMA = os.path.join(J, "job", "join-order-benchmark-master", "schema.sql")
CSV = os.path.join(J, "dl", "imdb")
OUT = os.path.join(J, "data", "job2.duckdb")
if __name__ == "__main__":
    ddl = open(SCHEMA).read().replace("PRIMARY KEY", "").replace("NOT NULL", "")
    con = duckdb.connect(OUT)
    for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
        t = re.search(r"CREATE TABLE (\w+)", stmt).group(1)
        con.execute(f"DROP TABLE IF EXISTS {t}"); con.execute(stmt)
        t0 = time.time()
        con.execute(f"COPY {t} FROM '{os.path.join(CSV, t + '.csv')}' (FORMAT csv, HEADER false, ESCAPE '\\', QUOTE '\"', NULL '')")
        print(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0], round(time.time() - t0, 1), flush=True)
    con.execute("CHECKPOINT")
    con.close()
    con = duckdb.connect(OUT, read_only=True)  # full read-back: every block is checksummed on read
    for (t,) in con.execute("SHOW TABLES").fetchall():
        print("verify", t, con.execute(f'SELECT count(*), sum(hash(x)) FROM {t} AS x').fetchone(), flush=True)
    con.close()
