"""Re-run Certify-Join+S with the outcomes of executed samples (results/sampled_exec/, see sampled_exec.py) and replace
its rows in a result file of run_pilot.py (same budgets, cost models and workload); every other method's rows are
kept unchanged. Usage: [WORKLOAD=official] rerun_cjs.py <results csv> <instance>..."""
import csv, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm
from run_pilot import load, BUDGETS, SAMPLE, ROOT

if __name__ == "__main__":
    path, names = sys.argv[1], sys.argv[2:]
    rows = list(csv.reader(open(path)))
    head, body = rows[0], rows[1:]
    im, imeth = head.index("inst"), head.index("method")
    new = []
    for name in names:
        P, _ = load(name)
        m = int(math.ceil(math.log(1 / SAMPLE[1]) / SAMPLE[0]))
        n_samp = sum(k.startswith('["I"') and v["rows_R"] > 10 * m for k, v in P.A.items())
        assert n_samp == 0 or getattr(P, "exec_hits", None), name + ": no executed samples"
        W = P.workload()
        if os.environ.get("WORKLOAD") == "official":  # as run_pilot.run
            ow = json.load(open(os.path.join(ROOT, "results", "official_workload.json")))
            W = [(F, tuple(D)) for _, F, D in ow["tpcds" if name.startswith("tpcds") else "tpch"]]
        for cm in ["atom", "scan"]:
            full = P.full_cost if cm == "atom" else P.full_scan
            for f in BUDGETS:
                S = Pm.certify_join(P, W, f * full, cost_model=cm, sample=SAMPLE)
                new.append(P.evaluate(W, S.certified(), "CertifyJoin+S", S.spent,
                                      dict(cost_model=cm, budget=f, probes=S.probes, inferred=S.inferred)))
            S = Pm.certify_join(P, W, 1e30, cost_model=cm, sample=SAMPLE)
            new.append(P.evaluate(W, S.certified(), "CertifyJoin+S", S.spent, dict(cost_model=cm, budget="inf", probes=S.probes)))
        print(name, "done", flush=True)
    body = [r for r in body if not (r[im] in names and r[imeth] == "CertifyJoin+S")]
    for r in new:
        body.append([str(r.get(k, "")) for k in head])  # as csv.DictWriter in run_pilot.py (duplicated columns repeat)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(body)
