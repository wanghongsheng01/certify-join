"""Turn raw results into the markdown tables used in 03_数据.md and 04_试点结果.md (results/summary.md)."""
import json, os, sys
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import policies as Pm

R = os.path.join(Pm.ROOT, "results")


def fam(n):
    if n.startswith("bird_"): return "BIRD (11 DBs)"
    if n.startswith("ent"): return "Enterprise"
    if "_dirty" in n: return ("TPC-H" if n.startswith("tpch") else "TPC-DS") + " dirty"
    if n.endswith("_sf10"): return ("TPC-H" if n.startswith("tpch") else "TPC-DS") + " SF10"
    return {"tpch": "TPC-H", "tpcds": "TPC-DS", "job": "JOB/IMDB"}[n]


ORDER = ["TPC-H", "TPC-H dirty", "TPC-DS", "TPC-DS dirty", "JOB/IMDB", "BIRD (11 DBs)", "Enterprise"]


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def data_table():
    rows = []
    for f in sorted(os.listdir(os.path.join(R, "pool"))):
        rec = json.load(open(os.path.join(R, "pool", f)))
        P = Pm.Pool(rec)
        W = P.workload()
        uf = sum(not P.truth[P.ea[e][0]] for e in P.edges)
        inf_ = sum(not P.truth[P.ea[e][1]] for e in P.edges)
        gbad = sum(not P.safe[e] for e in P.gold)
        nong_safe = sum(P.safe[e] for e in P.edges if e not in P.gold)
        u_atoms = len({P.ea[e][0] for e in P.edges})
        src = defaultdict(int)
        for e in P.edges:
            src[P.src[e]] += 1
        rows.append(dict(family=fam(P.name), inst=P.name, tables=len(rec["tables"]),
                         rows=sum(v["rows"] for v in rec["tables"].values()), gold=len(P.gold),
                         gold_unsafe=gbad, cand=len(P.edges), cand_ddl=src["ddl"], cand_name=src["name"],
                         cand_partial=src["partial"], unsafe=sum(not P.safe[e] for e in P.edges), U_false=uf,
                         I_false=inf_, nongold_safe=nong_safe, U_atoms=u_atoms, queries=len(W),
                         answerable=sum(P.truly_answerable(q) for q in W), full_atom_cost=P.full_cost,
                         full_scan_cost=P.full_scan, inject=len(rec.get("inject", []))))
    return pd.DataFrame(rows)


def main_tables(tag="main"):
    d = pd.read_csv(os.path.join(R, tag + ".csv"))
    d["family"] = d["inst"].map(fam)
    d["budget"] = d["budget"].astype(str)
    return d


def pooled(x, keys):
    g = x.groupby(keys)[["n_ok", "n_answerable", "n_certified", "n_card", "n_sem", "n_card_eps", "cost_frac"]].sum()
    reps = x.groupby(keys).size()
    n = x.groupby(keys)["inst"].nunique()
    g["coverage"] = g.n_ok / g.n_answerable.clip(lower=1)
    g["fcr_card"] = g.n_card / g.n_certified.clip(lower=1)
    g["fcr_card_eps"] = g.n_card_eps / g.n_certified.clip(lower=1)
    g["fcr_sem"] = g.n_sem / g.n_certified.clip(lower=1)
    g["cost_frac"] = g.cost_frac / reps  # mean over instances (and seeds)
    return g


if __name__ == "__main__":
    out = []
    D = data_table()
    D.to_csv(os.path.join(R, "data_summary.csv"), index=False)
    g = D.groupby("family").agg(instances=("inst", "count"), tables=("tables", "sum"), rows=("rows", "sum"),
                                gold=("gold", "sum"), gold_unsafe=("gold_unsafe", "sum"), cand=("cand", "sum"),
                                unsafe=("unsafe", "sum"), U_false=("U_false", "sum"), I_false=("I_false", "sum"),
                                nongold_safe=("nongold_safe", "sum"), U_atoms=("U_atoms", "sum"),
                                queries=("queries", "sum"), answerable=("answerable", "sum"))
    out.append("## Data\n" + md(g.reindex([o for o in ORDER if o in g.index]).reset_index(), "{:.0f}"))
    for tag in ["main", "official"]:
        if not os.path.exists(os.path.join(R, tag + ".csv")):
            continue
        d = main_tables(tag)
        for cm in ["atom", "scan"]:
            x = d[(d.cost_model == cm) & (d.budget != "inf")].copy()
            x["budget"] = x["budget"].astype(float)
            g = pooled(x, ["family", "method", "budget"])
            out.append(f"## {tag}: pooled coverage by budget ({cm} cost model)\n" + md(g["coverage"].unstack("budget").reset_index()))
            g2 = pooled(x, ["family", "method"])
            out.append(f"## {tag}: pooled false certification over all budgets ({cm})\n" +
                       md(g2[["n_certified", "fcr_card", "fcr_card_eps", "fcr_sem"]].reset_index()))
            y = d[(d.cost_model == cm) & (d.budget == "inf")]
            q = pooled(y, ["family", "method"])
            out.append(f"## {tag}: unbounded Certify-Join (mean cost fraction to decide the whole workload, {cm})\n"
                       + md(q[["coverage", "cost_frac", "fcr_card", "fcr_card_eps", "fcr_sem"]].reset_index()))
        z = pd.read_csv(os.path.join(R, tag + "_zero.csv"))
        z["family"] = z["inst"].map(fam)
        q = pooled(z, ["family", "method"])
        out.append(f"## {tag}: zero-probe baselines (pooled)\n" + md(q[["n_certified", "coverage", "fcr_card", "fcr_sem"]].reset_index()))
    if os.path.exists(os.path.join(R, "calib.csv")):
        c = pd.read_csv(os.path.join(R, "calib.csv"))
        out.append("## Sampled-certificate calibration\n" + md(c[["inst", "rows_R", "orphans", "rho", "m", "certified",
                                                                   "emp_rate", "theory", "rho_gt_eps"]], "{:.4g}"))
    mres = []
    for f in sorted(os.listdir(os.path.join(R, "maint"))) if os.path.exists(os.path.join(R, "maint")) else []:
        m = json.load(open(os.path.join(R, "maint", f)))
        mism = sum(l["delta_mismatch"] for l in m["log"])
        for k in ["full", "periodic", "touched", "trust", "delta"]:
            mres.append(dict(inst=m["inst"], seed=f, method=k, rows=m["cost"][k], rows_frac=m["cost"][k] / m["cost"]["full"],
                             exposure=m["exposure"][k], secs=m["secs"].get(k, float("nan")), atoms=m["n_atoms"],
                             queries=m["n_queries"], build_rows=m["build_rows"], delta_mismatch=mism,
                             events=sum(len(l["events"]) for l in m["log"])))
    if mres:
        M = pd.DataFrame(mres)
        M.to_csv(os.path.join(R, "maint_summary.csv"), index=False)
        q = M.groupby(["inst", "method"])[["rows", "rows_frac", "exposure", "secs", "delta_mismatch", "events"]].mean()
        out.append("## Maintenance (mean over seeds, 30 batches)\n" + md(q.reset_index(), "{:.4g}"))
    open(os.path.join(R, "summary.md"), "w").write("\n\n".join(out))
    print("\n\n".join(out))
