"""Figures and LaTeX tables of the paper (paper/figs/*.pdf, paper/tables/*.tex), from results/*.csv and results/maint/*.json.
Every number in the tables is computed here; nothing is typed by hand."""
import json, os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import fam, pooled, data_table, R

PAPER = os.path.join(os.path.dirname(R), "paper")
FIG, TAB = os.path.join(PAPER, "figs"), os.path.join(PAPER, "tables")
FAMS = ["TPC-H", "TPC-H dirty", "TPC-DS", "TPC-DS dirty", "JOB/IMDB", "BIRD (11 DBs)"]
SHORT = {"TPC-H": "TPC-H", "TPC-H dirty": "TPC-H$^d$", "TPC-DS": "TPC-DS", "TPC-DS dirty": "TPC-DS$^d$",
         "JOB/IMDB": "JOB", "BIRD (11 DBs)": "BIRD"}
# colour-blind-safe Okabe-Ito palette; our methods always use the two warm highlight colours, baselines muted hues
OURS = {"CertifyJoin": "#D55E00", "CertifyJoin+S": "#E69F00"}
PAL = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#999999", "#000000", "#F0E442"]
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "mathtext.fontset": "stix"})  # same face as the IEEEtran body text


def fams_present(d):
    return [f for f in FAMS if f in set(d["family"])]


def load_main(tag="main"):
    d = pd.concat([pd.read_csv(os.path.join(R, t + ".csv")) for t in (tag, "job") if os.path.exists(os.path.join(R, t + ".csv"))],
                  ignore_index=True)
    d["family"] = d["inst"].map(fam)
    d["budget"] = d["budget"].astype(str)
    d = d[d.family != "Enterprise"]
    return d


def load_zero():
    zs = [pd.read_csv(os.path.join(R, f)) for f in ("main_zero.csv", "job_zero.csv") if os.path.exists(os.path.join(R, f))]
    for f, lab in [(b + j + ".csv", None) for b in ("baseline_sji", "baseline_autobi", "baseline_hopf") for j in ("", "_job")]:
        p = os.path.join(R, f)
        if os.path.exists(p):
            zs.append(pd.read_csv(p))
    z = pd.concat(zs, ignore_index=True)
    z["family"] = z["inst"].map(fam)
    return z[z.family != "Enterprise"]


def pct(x, nd=1):
    return "--" if x is None or pd.isna(x) else f"{100 * x:.{nd}f}"


def mark(cells, lower=True):
    """Bold the best and underline the second-best printed value (ties share the rank); '--' cells are not ranked.
    Returns the marked cells and their ranks (1 best, 2 second, 3 other, 0 unranked)."""
    vals = []
    for c in cells:
        try:
            vals.append(float(c.replace(",", "").replace("\\xx", "")))
        except ValueError:
            vals.append(None)
    distinct = sorted({v for v in vals if v is not None}, reverse=not lower)
    best = distinct[0] if distinct else None
    second = distinct[1] if len(distinct) > 1 else None
    out, ranks = [], []
    for c, v in zip(cells, vals):
        if v is None:
            out.append(c); ranks.append(0)
        elif v == best:
            out.append(r"\textbf{" + c + "}"); ranks.append(1)
        elif v == second:
            out.append(r"\underline{" + c + "}"); ranks.append(2)
        else:
            out.append(c); ranks.append(3)
    return out, ranks


def count_cell(rs):
    return f"{sum(r == 1 for r in rs)}/{sum(r == 2 for r in rs)}"


def table_data():
    D = data_table()
    D = D[D.family != "Enterprise"]
    g = D.groupby("family").agg(inst=("inst", "count"), tables=("tables", "sum"), rows=("rows", "sum"),
                                cand=("cand", "sum"), unsafe=("unsafe", "sum"), U=("U_atoms", "sum"),
                                q=("queries", "sum"), ans=("answerable", "sum"))
    L = [r"\begin{table}[t]", r"\centering", r"\caption{Instances. Rows in millions; candidates proposed by one generator for all instances; unsafe = key or inclusion atom false.}",
         r"\label{tab:data}", r"\small", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Family & Inst. & Tables & Rows & Cand. & Unsafe & Queries & Answ. \\", r"\midrule"]
    for f in fams_present(D):
        r = g.loc[f]
        L.append(f"{SHORT[f]} & {r.inst} & {r.tables} & {r.rows / 1e6:.1f} & {r.cand:,} & {r.unsafe:,} & {r.q:,} & {r.ans:,} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "data.tex"), "w").write("\n".join(L) + "\n")


def table_main(d):
    a = d[d.cost_model == "atom"]
    inf = pooled(a[a.budget == "inf"], ["family", "method"])
    x = a[a.budget != "inf"].copy()
    x["budget"] = x["budget"].astype(float)
    cov = pooled(x, ["family", "method", "budget"])["coverage"]
    ex = pd.read_csv(os.path.join(R, "extra_baselines.csv"))
    ex["family"] = ex["inst"].map(fam)
    ex["budget"] = ex["budget"].astype(str)
    exc = pooled(ex[ex.budget == "0.05"], ["family", "method"])["coverage"]
    rws = pd.read_csv(os.path.join(R, "decide_seconds.csv"))
    rws["family"] = rws["inst"].map(fam)
    rws = rws.groupby("family")[["keyfirst_rows", "wlfull_rows"]].mean()
    r10 = pd.read_csv(os.path.join(R, "decide_rows_sf10.csv"))
    r10["family"] = r10["inst"].map(fam)
    r10 = r10.set_index("family")
    lv = pd.read_csv(os.path.join(R, "decide_live_quiet.csv"))
    lv["family"] = lv["inst"].map(fam)
    lv["kf"] = lv.keyfirst_plan_s + lv.keyfirst_probe_s
    lv["wl"] = lv.wlfull_plan_s + lv.wlfull_probe_s
    lv["cj"] = lv.cj_plan_s + lv.cj_probe_s
    sec = lv.groupby("family")[["full_probe_s", "kf", "wl", "cj"]].sum()

    def sx(v):
        return "--" if v is None or pd.isna(v) else f"{v:.2f}"
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Deciding the whole workload, and coverage at a 5\% budget. Rows: share of the rows read by per-edge verification of every candidate (atom cost model). Seconds: live planning time (minimum of 3 runs) plus the measured time of each exact SQL probe the method issues. KeyFirst: per-edge verification that skips the inclusion check of edges with a false key; WL-Full: KeyFirst restricted to edges on workload paths of at most 3 hops. CJ = \sys, CJ+S = sampled inclusion. SF10 rows use the SF1 workload. In each group, the best value of a row is in bold and the second best underlined (OPT, which knows every atom's truth, is not ranked); Best/2nd counts the rows.}",
         r"\label{tab:main}", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lrrrrrrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{Rows to decide (\%)} & \multicolumn{4}{c}{Seconds to decide} & \multicolumn{5}{c}{Coverage at 5\% budget (\%)} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-14}",
         r"Family & KeyFirst & WL-Full & CJ & CJ+S & Full & KeyFirst & WL-Full & CJ & Full & KeyFirst & CJ & CJ+S & OPT \\", r"\midrule"]
    fams = fams_present(d) + [f for f in ["TPC-H SF10", "TPC-DS SF10"] if f in r10.index]
    short = dict(SHORT, **{"TPC-H SF10": "TPC-H SF10", "TPC-DS SF10": "TPC-DS SF10"})
    RK = {k: [[] for _ in range(n)] for k, n in (("rows", 4), ("sec", 4), ("cov", 4))}
    for f in fams:
        def c(m):
            try:
                return cov.loc[(f, m, 0.05)]
            except KeyError:
                return None
        g = lambda ser, k: ser.loc[f, k] if f in ser.index else None
        if f in r10.index:
            rcells = [pct(g(r10, 'keyfirst_rows')), pct(g(r10, 'wlfull_rows')), pct(g(r10, 'cj_rows')), pct(g(r10, 'cjs_rows'))]
            ccells, opt = ["--"] * 4, "--"
        else:
            rcells = [pct(g(rws, 'keyfirst_rows')), pct(g(rws, 'wlfull_rows')), pct(inf.loc[(f, 'CertifyJoin'), 'cost_frac']),
                      pct(inf.loc[(f, 'CertifyJoin+S'), 'cost_frac'])]
            ccells = [pct(c('Full(catalog order)')), pct(exc.get((f, 'Full-KeyFirst'))), pct(c('CertifyJoin')), pct(c('CertifyJoin+S'))]
            opt = pct(c('OPT(clairvoyant ILP)'))
        scells = [sx(g(sec, 'full_probe_s')), sx(g(sec, 'kf')), sx(g(sec, 'wl')), sx(g(sec, 'cj'))]
        rcells, rr = mark(rcells, lower=True)
        scells, sr = mark(scells, lower=True)
        ccells, cr = mark(ccells, lower=False)
        for k, rs in (("rows", rr), ("sec", sr), ("cov", cr)):
            for i, r in enumerate(rs):
                RK[k][i].append(r)
        if f == "TPC-H SF10":
            L.append(r"\midrule")
        L.append(f"{short[f]} & " + " & ".join(rcells + scells + ccells + [opt]) + r" \\")
    L.append(r"\midrule")
    L.append(r"Best/2nd & " + " & ".join([count_cell(x) for x in RK["rows"]] + [count_cell(x) for x in RK["sec"]] +
                                         [count_cell(x) for x in RK["cov"]] + ["--"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(TAB, "main.tex"), "w").write("\n".join(L) + "\n")


ZM = [("DDL-declared", "DDL"), ("LLM-judge:gpt", "LLM GPT"), ("LLM-judge:claude", "LLM Claude"),
      ("LLM-judge:ds-flash", "LLM DeepSeek"), ("LLM-judge:glm", "LLM GLM"), ("LLM-judge:kimi", "LLM Kimi"),
      ("CardEst(HLL stats)", "CardEst"), ("Auto-BI(k-MCA, reimpl.)", "Auto-BI"), ("HoPF(reimpl.)", "HoPF"), ("SJI(2603.04176)", "SJI")]


def table_fcr(d, z):
    a = d[(d.cost_model == "atom") & (d.budget != "inf")]
    q = pooled(z, ["family", "method"])
    cj = pooled(a, ["family", "method"])
    fs = fams_present(d)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Cardinality false-certification rate (\%): certified queries whose tree fans out or mis-assigns rows, pooled per family (\sys: all budgets). Auto-BI and HoPF read every candidate column; SJI reads statistics and samples; DDL, LLM judges and CardEst read no data. Per family, the lowest rate is in bold and the second lowest underlined (ties share the mark); B/2 counts a method's best and second-best families.}",
         r"\label{tab:fcr}", r"\small", r"\setlength{\tabcolsep}{1.3pt}", r"\begin{tabular}{@{}l" + "r" * len(fs) + "r@{}}", r"\toprule",
         "Method & " + " & ".join(SHORT[f] for f in fs) + r" & B/2 \\", r"\midrule"]
    rows = []
    for m, lab in ZM:
        if m not in set(z.method):
            continue
        cells = []
        for f in fs:
            try:
                cells.append(pct(q.loc[(f, m)].fcr_card))
            except KeyError:
                cells.append("--")
        rows.append([lab, cells])
    ours = [[lab, [pct(cj.loc[(f, m)].fcr_card) for f in fs]] for m, lab in [("CertifyJoin", r"\sys"), ("CertifyJoin+S", r"\sysS")]]
    allr = rows + ours
    ranks = [[0] * len(fs) for _ in allr]
    for j in range(len(fs)):
        col, rk = mark([r[1][j] for r in allr], lower=True)
        for i in range(len(allr)):
            allr[i][1][j] = col[i]; ranks[i][j] = rk[i]
    for i, (lab, cells) in enumerate(allr):
        if i == len(rows):
            L.append(r"\midrule")
        L.append(lab + " & " + " & ".join(cells) + f" & {count_cell(ranks[i])}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "fcr.tex"), "w").write("\n".join(L) + "\n")


def maint_rows():
    rows = []
    for f in sorted(os.listdir(os.path.join(R, "maint"))):
        m = json.load(open(os.path.join(R, "maint", f)))
        tag = m.get("tag", "")
        if tag not in ("", "_sf10vol", "_sf10prop"):
            continue
        n = len(m["log"])
        rows.append(dict(inst=m["inst"], tag=tag or "_sf1", seed=f, batches=n, atoms=m["n_atoms"],
                         full_rows=m["cost"]["full"], delta_rows=m["cost"]["delta"], full_s=m["secs"]["full"],
                         delta_s=m["secs"]["delta"], exp_delta=m["exposure"].get("delta", 0),
                         exp_periodic=m["exposure"].get("periodic", 0), exp_trust=m["exposure"].get("trust", 0),
                         mism=sum(l["delta_mismatch"] for l in m["log"]), build_rows=m["build_rows"]))
    return pd.DataFrame(rows)


def table_maint():
    M = maint_rows()
    if M.empty:
        return
    g = M.groupby(["inst", "tag"]).agg(runs=("seed", "count"), batches=("batches", "first"), full_rows=("full_rows", "mean"),
                                       delta_rows=("delta_rows", "mean"), full_s=("full_s", "mean"), delta_s=("delta_s", "mean"),
                                       exp_delta=("exp_delta", "sum"), exp_periodic=("exp_periodic", "mean"),
                                       exp_trust=("exp_trust", "mean"), mism=("mism", "sum"))
    lab = {"_sf1": "SF1", "_sf10vol": "SF10, same $\\Delta$", "_sf10prop": "SF10, prop.\\ $\\Delta$"}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Maintenance per run (means over seeds). Rows: incremental / full re-verification. Time: wall-clock seconds summed over the batches (SF1: 30 batches, mean of 3 seeds; SF10: 10 batches). Same $\Delta$: the change volume per batch of SF1; prop.\ $\Delta$: batches proportional to the table sizes. The faster of the two methods is in bold.}",
         r"\label{tab:maint}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Inst. & Setting & Rows (\%) & Full (s) & Incr.\ (s) & Speedup \\", r"\midrule"]
    g = g.reindex(sorted(g.index, key=lambda k: (k[0] != "tpch", ["_sf1", "_sf10vol", "_sf10prop"].index(k[1]))))
    wins = []
    for (inst, tag), r in g.iterrows():
        tc, trk = mark([f"{r.full_s:.2f}", f"{r.delta_s:.2f}"], lower=True)
        tc = [t.replace(r"\underline{", "{") for t in tc]  # two methods: only the faster one is marked
        wins.append(trk[1] == 1)
        L.append(f"{inst.upper().replace('TPCDS', 'TPC-DS').replace('TPCH', 'TPC-H')} & {lab[tag]} & {100 * r.delta_rows / r.full_rows:.3f} & {tc[0]} & {tc[1]} & "
                 f"{r.full_s / max(r.delta_s, 1e-9):.1f}\\xx \\\\")
    L += [r"\midrule", f"\\multicolumn{{6}}{{@{{}}l}}{{Incr.\\ faster: {sum(wins)}/{len(wins)}}} \\\\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "maint.tex"), "w").write("\n".join(L) + "\n")
    g.to_csv(os.path.join(R, "maint_paper.csv"))


def fig_coverage(d):
    a = d[(d.cost_model == "atom") & (d.budget != "inf")].copy()
    ex = pd.read_csv(os.path.join(R, "extra_baselines.csv"))
    ex["family"] = ex["inst"].map(fam)
    ex["budget"] = ex["budget"].astype(str)
    a = pd.concat([a, ex[(ex.budget != "inf") & (ex.method == "Full-KeyFirst")]], ignore_index=True)
    a["budget"] = a["budget"].astype(float)
    cov = pooled(a, ["family", "method", "budget"])["coverage"]
    fs = fams_present(d)
    ms = [("CertifyJoin", "Certify-Join"), ("CertifyJoin+S", "Certify-Join+S"), ("OPT(clairvoyant ILP)", "OPT (clairvoyant)"),
          ("Full(catalog order)", "Full"), ("Full-KeyFirst", "KeyFirst"), ("KeyCoverage", "KeyCoverage"), ("Random", "Random")]
    col = {"OPT(clairvoyant ILP)": "#000000", "Full(catalog order)": "#0072B2", "Full-KeyFirst": "#56B4E9",
           "KeyCoverage": "#009E73", "Random": "#999999", **OURS}
    fig, axs = plt.subplots(1, len(fs), figsize=(7.16, 1.55), sharey=True, gridspec_kw=dict(wspace=0.3))
    for ax, f in zip(axs, fs):
        for i, (m, lab) in enumerate(ms):
            try:
                s = cov.loc[(f, m)].sort_index()
            except KeyError:
                continue
            ls = {"OPT(clairvoyant ILP)": (0, (3, 1.5)), "Random": (0, (1, 1)), "Full-KeyFirst": (0, (5, 1, 1, 1))}.get(m, "-")
            ours = m in OURS
            ax.plot(s.index, 100 * s.values, linestyle=ls, color=col[m], lw=2.0 if ours else 0.9, zorder=3 if ours else 2,
                    label=lab)
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.1, 1.0]); ax.set_xticklabels(["1%", "10%", "100%"]); ax.minorticks_off()
        ax.set_title(SHORT[f].replace("$^d$", " dirty"), fontsize=8)
        ax.grid(axis="y", lw=0.3, color="#dddddd")
        ax.set_ylim(0, 102)
    axs[0].set_ylabel("Coverage (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(ms), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.text(0.5, -0.10, "Probe budget (fraction of the cost of full verification)", ha="center", fontsize=8)
    fig.savefig(os.path.join(FIG, "coverage.pdf"), bbox_inches="tight")


def fig1(d, z):
    """Unsafe certified trees per family: one bar per baseline (the five LLM judges as one bar, median with min-max
    whisker), Certify-Join as highlighted markers at its value (0 everywhere)."""
    q = pooled(z, ["family", "method"])
    a = d[(d.cost_model == "atom") & (d.budget != "inf")]
    cj = pooled(a, ["family", "method"])
    fs = [f for f in ["BIRD (11 DBs)", "JOB/IMDB", "TPC-H dirty", "TPC-DS dirty"] if f in set(d.family)]
    judges = [m for m in ["LLM-judge:gpt", "LLM-judge:claude", "LLM-judge:ds-flash", "LLM-judge:glm", "LLM-judge:kimi"]
              if m in set(z.method)]
    groups = [("DDL", ["DDL-declared"]), (f"LLM judges ({len(judges)})", judges), ("CardEst", ["CardEst(HLL stats)"]),
              ("Auto-BI", ["Auto-BI(k-MCA, reimpl.)"]), ("SJI", ["SJI(2603.04176)"]), ("HoPF", ["HoPF(reimpl.)"])]
    cols = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#999999", "#000000"]
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    nb = len(groups) + 1
    w = 0.8 / nb
    for i, (lab, ms) in enumerate(groups):
        ys, lo, hi = [], [], []
        for f in fs:
            v = sorted(100 * q.loc[(f, m)].fcr_card for m in ms if (f, m) in q.index)
            med = v[len(v) // 2] if v else 0
            ys.append(med); lo.append(med - (v[0] if v else 0)); hi.append((v[-1] if v else 0) - med)
        xs = [j + (i - (nb - 1) / 2) * w for j in range(len(fs))]
        ax.bar(xs, ys, width=w * 0.9, color=cols[i], label=lab,
               yerr=[lo, hi] if len(ms) > 1 else None, error_kw=dict(lw=0.6, capsize=1.2, ecolor="#333333"))
        for x, y in zip(xs, ys):  # a zero-height bar stays visible as a short tick on the axis
            if y == 0:
                ax.plot([x - w * 0.4, x + w * 0.4], [0.6, 0.6], color=cols[i], lw=1.6, solid_capstyle="butt")
    xs = [j + (nb - 1 - (nb - 1) / 2) * w for j in range(len(fs))]
    ys = [100 * cj.loc[(f, "CertifyJoin")].fcr_card for f in fs]
    ax.scatter(xs, [y + 1.5 for y in ys], marker="v", s=14, color=OURS["CertifyJoin"], zorder=4, label="Certify-Join (0)")
    for x, y in zip(xs, ys):
        ax.text(x, y + 5, f"{y:.0f}", ha="center", va="bottom", fontsize=6.5, color="#000000", fontweight="bold")
    ax.set_xticks(range(len(fs))); ax.set_xticklabels([SHORT[f].replace("$^d$", " dirty") for f in fs])
    ax.set_ylabel("Unsafe certified trees (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
              columnspacing=0.8)
    fig.savefig(os.path.join(FIG, "fig1.pdf"), bbox_inches="tight")


def table_prod():
    """Production case (RQ7): deciding a workload on the withheld ERP database; aggregate figures only
    (results/erp_endtoend.json). The platform counts of RQ7 are stated in the text."""
    if not os.path.exists(os.path.join(R, "erp_endtoend.json")):  # public artifact: the production summaries are withheld
        return
    X = json.load(open(os.path.join(R, "erp_endtoend.json")))
    full = X["full_hours"]
    meth = [("Per-edge verification", full, "100.0"), ("KeyFirst", X["keyfirst_hours"], "100.0"),
            ("WL-Full", X["wlfull_hours"], "100.0"), (r"\sys", X["cj"]["hours"], f"{100 * X['cj']['coverage']:.1f}"),
            (r"\sysS", X["cjs_live"]["hours"], f"{100 * X['cjs']['coverage']:.1f}")]
    hrs, _ = mark([f"{h:.2f}" for _, h, _ in meth], lower=True)
    shr, _ = mark([f"{100 * h / full:.1f}" for _, h, _ in meth], lower=True)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Deciding a workload on the production ERP database (36 tables, 327.6M rows, 909 candidate edges; "
         f"{X['queries']} queries, {X['answerable']} answerable). " + r"\sys and \sysS need the fewest database hours. "
         r"Coverage: answerable queries certified. Fewest hours in bold, second fewest underlined.}",
         r"\label{tab:prod}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Method & DB hours & \% of per-edge & Cov.\ (\%) \\", r"\midrule"]
    for (lab, _, cov), h, sh in zip(meth, hrs, shr):
        L.append(f"{lab} & {h} & {sh} & {cov} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "prod.tex"), "w").write("\n".join(L) + "\n")


def table_scale():
    """RQ6: planning time of Certify-Join on synthetic snowflake/constellation schemas (results/synth_scale.csv)."""
    t = pd.read_csv(os.path.join(R, "synth_scale.csv"))
    t["budget"] = t["budget"].astype(str)
    g = t.groupby(["n_tables", "budget"]).agg(edges=("n_edges", "mean"), queries=("n_queries", "mean"),
                                             wall=("wall_s", "mean"), it=("iterations", "mean"),
                                             probes=("probes", "mean"), cov=("coverage", "mean")).reset_index()
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Planning time of \sys on synthetic schemas (mean of 2 seeds; probes simulated from stored truths, so the time is pure planning). Decide: unbounded budget; 5\%: budget of 5\% of per-edge verification.}",
         r"\label{tab:scale}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
         r"Tables & Edges & Queries & Decide (s) & ms/probe & 5\% (s) & Cov.\ 5\% \\", r"\midrule"]
    for n in sorted(g.n_tables.unique()):
        a = g[(g.n_tables == n) & (g.budget == "inf")].iloc[0]
        b = g[(g.n_tables == n) & (g.budget == "0.05")].iloc[0]
        L.append(f"{n:,} & {a['edges']:,.0f} & {a['queries']:,.0f} & {a['wall']:,.1f} & {1000 * a['wall'] / a['probes']:,.1f} & {b['wall']:,.1f} & {100 * b['cov']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "scale.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    d = load_main()
    z = load_zero()
    table_data(); table_main(d); table_fcr(d, z); table_maint(); table_prod(); table_scale(); fig_coverage(d); fig1(d, z)
    print("ok")
