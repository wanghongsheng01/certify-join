"""Generate TPC-H and TPC-DS SF1 instances with DuckDB's dbgen/dsdgen."""
import duckdb, os, sys
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
for name, ext, call in [("tpch", "tpch", "CALL dbgen(sf=1)"), ("tpcds", "tpcds", "CALL dsdgen(sf=1)")]:
    p = os.path.join(D, name + ".duckdb")
    if os.path.exists(p): continue
    c = duckdb.connect(p); c.execute(f"INSTALL {ext}; LOAD {ext}; {call}"); c.close(); print(name, "done", flush=True)
