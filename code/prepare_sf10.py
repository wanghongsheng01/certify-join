"""Pools for TPC-H / TPC-DS at scale factor 10 (external disk): same schema, candidates and LLM priors as SF1 (the
prompts are identical, so the cached SF1 calls are reused), exact atoms re-probed on the SF10 data with their
measured seconds. Used for the wall-clock comparison at a larger scale (decide_live.py)."""
import os, sys
import duckdb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instances as I
import prepare

SSD = "<DATA_DIR>/data"
if __name__ == "__main__":
    for name in sys.argv[1:] or ["tpch", "tpcds"]:
        base = I.load_tpch() if name == "tpch" else I.load_tpcds()
        cand = I.candidates(base)
        con = duckdb.connect(os.path.join(SSD, f"{name}_sf10.duckdb"), read_only=True)
        inst = dict(base, name=f"{name}_sf10", con=con, tables=I._meta(con))
        if name == "tpcds":
            inst["pk"], fk = I.tpcds_gold(con)
            inst["declared"], inst["gold"] = list(fk), list(fk)
        prepare.build(inst, cand=dict(cand), prior_inst=base)
