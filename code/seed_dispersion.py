"""Dispersion over seeds (review fix F9): Certify-Join+S draws the outcome of each sampled probe at random; Random
orders atoms at random. Five seeds each; pooled per family. Writes results/seed_dispersion.csv."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm
from run_pilot import load, ROOT, SAMPLE

if __name__ == "__main__":
    names = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool")) if not f.startswith("ent"))
    w = None
    with open(os.path.join(ROOT, "results", "seed_dispersion.csv"), "w", newline="") as fh:
        for name in names:
            P, st = load(name)
            W = P.workload()
            for sd in range(5):
                for f in [0.05, "inf"]:
                    B = 1e30 if f == "inf" else f * P.full_cost
                    S = Pm.certify_join(P, W, B, sample=SAMPLE, seed=sd)
                    rows = [P.evaluate(W, S.certified(), "CertifyJoin+S", S.spent, dict(seed=sd, budget=f))]
                    if f != "inf":
                        S = Pm.run_order(P, Pm.random_order(P, sd), B)
                        rows.append(P.evaluate(W, S.certified(), "Random", S.spent, dict(seed=sd, budget=f)))
                    for r in rows:
                        if w is None:
                            w = csv.DictWriter(fh, fieldnames=list(r.keys())); w.writeheader()
                        w.writerow(r)
            print(name, flush=True)
