"""Re-run only the zero-probe baselines of run_pilot.py (DDL, one LLM judge per prior model, their majority vote,
CardEst, Auto-BI-style) after new priors were added to the pools; writes the same <tag>_zero.csv files.
Usage: python code/zero_rerun.py main|job <instance>..."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pilot import load, ROOT, ddl_declared, llm_judge, llm_vote, card_est, autobi

if __name__ == "__main__":
    tag, names = sys.argv[1], sys.argv[2:]
    w = None
    with open(os.path.join(ROOT, "results", f"{tag}_zero.csv"), "w", newline="") as fh:
        for name in names:
            P, st = load(name)
            W = P.workload()
            z = [P.evaluate(W, ddl_declared(P), "DDL-declared", 0)]
            for m in P.rec["prior"]:
                if P.rec["prior"][m]:
                    z.append(P.evaluate(W, llm_judge(P, m), "LLM-judge:" + m, 0))
            z.append(P.evaluate(W, llm_vote(P), "LLM-judge:vote", 0))
            if st:
                z.append(P.evaluate(W, card_est(P, st), "CardEst(HLL stats)", 0))
                z.append(P.evaluate(W, autobi(P, st), "Auto-BI-style(reimpl.)", 0))
            for r in z:
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(r.keys()))
                    w.writeheader()
                w.writerow(r)
            print(name, flush=True)
