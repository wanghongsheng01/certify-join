"""RQ3 robustness: Certify-Join with priors from each LLM vendor (DeepSeek-v4-flash, GLM-4.5-flash, Kimi-k2.6).
Writes results/prior_vendor.csv (instance x prior x budget, atom cost model)."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm
from run_pilot import load, ROOT

BUD = [0.01, 0.05, 0.2, "inf"]
if __name__ == "__main__":
    names = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool")) if not f.startswith("ent"))
    w = None
    with open(os.path.join(ROOT, "results", "prior_vendor.csv"), "w", newline="") as fh:
        for name in names:
            for model in ["ds-flash", "glm", "kimi"]:
                P, st = load(name, prior=model)
                if not P.rec["prior"].get(model):
                    continue
                W = P.workload()
                for f in BUD:
                    B = 1e30 if f == "inf" else f * P.full_cost
                    S = Pm.certify_join(P, W, B)
                    r = P.evaluate(W, S.certified(), "CertifyJoin", S.spent, dict(prior=model, budget=f))
                    if w is None:
                        w = csv.DictWriter(fh, fieldnames=list(r.keys()), extrasaction="ignore"); w.writeheader()
                    w.writerow(r)
            print(name, flush=True)
