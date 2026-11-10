"""Baseline: Scalable Join Inference (SJI) for large context graphs (Tripathi, Shetye, Qiao, Jindal; arXiv 2603.04176).

Re-implementation of the paper's pipeline as described in its Sections 4-8 and 10, restricted to what the paper states;
every unspecified choice is marked "[choice]" below and listed in 13a_基线SJI复现说明.md.

  1. Primary-key inference (Sec. 4): per-column count / distinct (exact below ~1M rows, approximate above), Eq. 1
     filter with x = 0.95 (two-sided bounds against the row count and the table's max distinct count), Eq. 2-3
     key score; a clear winner is kept, otherwise the whole candidate pool is carried forward.
  2. Inclusion dependencies (Sec. 5): sample (1e6 tuples, Sec. 10.5) all non-PK columns, drop NULLs / empty strings /
     negative values of unsigned columns, keep values inside the inter-quartile range (lengths for strings); for every
     table pair, every (non-PK column, PK candidate) pair with matching data types is an IND candidate if all sampled
     FK values occur in the PK column; Eq. 4 score (equal-weight mean of five features), threshold 0.4 (Sec. 10.2);
     LLM adjudication from table/column names and sample values.
  3. Query-history refinement (Sec. 7), BIRD only (gold dev SQL): parse equi-joins, resolve tables, validate by
     execution (valid if the join returns rows), merge with the statistically inferred joins.
  4. Output: the inferred join relationships (post-LLM INDs + validated history joins). Algorithm 1 (join paths from
     left-most fact roots) is reported as a variant ("SJI-tree"), and seeding with information-schema constraints
     (Sec. 8) as a second variant ("SJI+IS").

Only public benchmark instances (TPC-H, TPC-DS, BIRD and their dirty copies) are processed; instances whose name starts
with "ent" are refused. Data access is read-only DuckDB; the method's own rows read are counted per SQL statement.

Usage: .venv/bin/python code/baseline_sji.py [instance ...]
Writes results/baseline_sji.csv (P.evaluate columns, label "SJI(2603.04176)") and results/baseline_sji_detail.json.
"""
import csv, difflib, json, os, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm
import policies as Pm
from run_pilot import load

ROOT = Pm.ROOT
DATA = os.path.join(ROOT, "data")
LABEL = "SJI(2603.04176)"
# BIRD dev set: instances.BIRD points to a folder that no longer exists on this machine; the same dev_20240627 release
# is used from its current location (first existing path wins).
BIRD_DEV = [p for p in ["/Volumes/SSD/MacMini 桌面/CaliberGraph/CaliberGraph 投稿 AAAI 被 reject 材料/dataset/dev_20240627/dev.json",
                        os.path.join(os.path.dirname(ROOT), "20_v3_why_differ", "data", "ext", "bird_dev", "dev_20240627", "dev.json")]
            if os.path.exists(p)]

X_KEY = 0.95            # Eq. 1, value reported best in the paper
# Sec. 4 uses exact COUNT(DISTINCT) below ~1M rows and an (unnamed) approximate count above. DuckDB's
# approx_count_distinct under-counts sequential keys badly (TPC-H orders.o_orderkey: 1,279,960 for 1,500,000 rows),
# which would fail Eq. 1 and is an artefact of our engine, not of SJI. [choice, in SJI's favour] exact counts everywhere
# (the largest table has 11.7M rows; rows read are the same).
EXACT_ROWS = float("inf")
SAMPLE_ROWS = 1_000_000  # Sec. 10.5: scores stabilise at ~1e6 tuples, chosen default
IND_THR = 0.4           # Sec. 10.2
WIN_MARGIN = 0.5        # [choice] "significantly higher than the rest": >= 0.5 (smallest weight in Eq. 3)
N_VALUES = 5            # [choice] sample values per column shown to the LLM
MODEL = "ds-flash"      # [choice] the paper names no model; DeepSeek-V4-flash (thinking off) = our default prior model
PRICE = (0.30, 1.20)    # USD per 1M input / output tokens, DeepSeek official page (2026-11-03), as in 06_submission
SEED = 7

ID_SUFFIX = ("id",)                 # Eq. 3 is_named_id
KEY_SUFFIX = ("key", "nr")          # Eq. 3 is_named_key ("suffixes such as id, key, or nr")
FK_SUFFIX = ("id", "key", "nr")     # Eq. 4 is_named_id: "ends with typical suffixes like id, key, or nr"


# ---------------------------------------------------------------------------------------------- helpers
def kind(ty):
    t = str(ty).upper()
    if t.startswith("U") and re.search(r"INT", t):
        return "uint"
    if re.search(r"INT", t):
        return "int"
    if re.search(r"CHAR|TEXT|STRING", t):
        return "str"
    if re.search(r"DECIMAL|DOUBLE|FLOAT|REAL|NUMERIC", t):
        return "num"
    return t  # DATE, TIMESTAMP, BOOLEAN ...: match on the exact type


def type_match(a, b):
    ka, kb = kind(a), kind(b)
    norm = lambda k: "int" if k == "uint" else k
    return norm(ka) == norm(kb)


def longest_match(a, b):
    a, b = a.lower(), b.lower()
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


def name_aff(tn, cn):
    """Eq. 2: 1 if the column name is contained in the table name, else longest match / |table name|."""
    tn, cn = tn.lower(), cn.lower()
    if cn in tn:
        return 1.0
    return longest_match(tn, cn) / max(1, len(tn))


def ends(c, suf):
    return c.lower().endswith(suf)


def q(x):
    return '"' + x.replace('"', '""') + '"'


def db_path(name):
    if name.startswith("bird_"):
        return os.path.join(DATA, "bird", name[5:] + ".duckdb")
    return os.path.join(DATA, name + ".duckdb")


class Counter_:
    def __init__(self):
        self.rows = defaultdict(int)   # per component, rows read as issued (one SQL statement = rows it scans)
        self.tables = set()            # distinct tables touched (scan-sharing lower bound)
        self.stmts = 0

    def add(self, comp, table, rows):
        self.rows[comp] += rows
        self.tables.add(table)
        self.stmts += 1


# ---------------------------------------------------------------------------------------------- step 1: PKs
def profile(con, tables, cnt):
    prof = {}
    for t, m in tables.items():
        n = m["rows"]
        cols = list(m["cols"])
        parts = []
        for c in cols:
            d = f"count(DISTINCT {q(c)})" if n < EXACT_ROWS else f"approx_count_distinct({q(c)})"
            parts.append(f"count({q(c)}), {d}")
        r = con.execute(f"SELECT {', '.join(parts)} FROM {q(t)}").fetchone()
        cnt.add("profile", t, n)
        prof[t] = {c: dict(count=int(r[2 * i]), distinct=int(r[2 * i + 1])) for i, c in enumerate(cols)}
    return prof


def infer_pk(tables, prof):
    out = {}
    for t, m in tables.items():
        st = prof[t]
        mx = max([v["distinct"] for v in st.values()] or [0])
        cands = []
        for c, v in st.items():
            n, d = v["count"], v["distinct"]
            if n > 0 and X_KEY * n <= d <= (2 - X_KEY) * n and X_KEY * mx <= d <= (2 - X_KEY) * mx:
                s = name_aff(t, c) + d / n + (1.0 if ends(c, ID_SUFFIX) else 0.0) \
                    + (0.5 if ends(c, KEY_SUFFIX) else 0.0)
                cands.append((round(s, 4), c))
        cands.sort(key=lambda z: (-z[0], z[1]))
        if not cands:
            out[t] = dict(cands=[], chosen=[], clear=False)
        elif len(cands) == 1 or cands[0][0] - cands[1][0] >= WIN_MARGIN:
            out[t] = dict(cands=cands, chosen=[cands[0][1]], clear=True)
        else:
            out[t] = dict(cands=cands, chosen=[c for _, c in cands], clear=False)
    return out


# ---------------------------------------------------------------------------------------------- step 2: INDs
def norm_series(s, k):
    if k in ("int", "uint", "num"):
        v = pd.to_numeric(s, errors="coerce").dropna()
        return v
    if k == "str":
        v = s.dropna().astype(str)
        return v[v.str.strip() != ""]
    return s.dropna().astype(str)


CLEAN_MODE = "iqr"  # paper: keep values within the inter-quartile range; ablations: "tukey" (1.5 IQR fences), "none"


def clean(s, k):
    """NULLs, empty strings, negative values of unsigned columns removed; keep values inside [Q1, Q3]."""
    v = norm_series(s, k)
    if k == "uint":
        v = v[v >= 0]
    if len(v) == 0 or CLEAN_MODE == "none":
        return v
    if CLEAN_MODE == "tukey":
        x = v.to_numpy(dtype=float) if k in ("int", "uint", "num") else v.str.len().to_numpy(dtype=float) if k == "str" else None
        if x is None:
            return v
        q1, q3 = np.quantile(x, [0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        return v[(x >= lo) & (x <= hi)]
    if k in ("int", "uint", "num"):
        q1, q3 = np.quantile(v.to_numpy(dtype=float), [0.25, 0.75])
        v = v[(v >= q1) & (v <= q3)]
    elif k == "str":
        L = v.str.len()
        q1, q3 = np.quantile(L.to_numpy(dtype=float), [0.25, 0.75])
        v = v[(L >= q1) & (L <= q3)]
    return v


def key_of(x):
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return int(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def spread(vals, k=N_VALUES):
    vals = sorted(vals, key=lambda z: (str(type(z)), z))
    if len(vals) <= k:
        return vals
    idx = np.linspace(0, len(vals) - 1, k).round().astype(int)
    return [vals[i] for i in idx]


def sample_tables(con, tables, cnt):
    smp = {}
    for t, m in tables.items():
        n = m["rows"]
        s = f" USING SAMPLE {SAMPLE_ROWS} ROWS (reservoir, {SEED})" if n > SAMPLE_ROWS else ""
        smp[t] = con.execute(f"SELECT * FROM {q(t)}{s}").df()
        cnt.add("sample", t, n)
    return smp


def pk_values(con, t, c, k, cnt, rows):
    df = con.execute(f"SELECT DISTINCT {q(c)} AS v FROM {q(t)} WHERE {q(c)} IS NOT NULL").df()
    cnt.add("pk_values", t, rows)
    v = norm_series(df["v"], k)
    return {key_of(x) for x in v.tolist()}


def ind_candidates(tables, prof, pks, smp, pkv, extra_fk=()):
    """Return contained (A, a, B, b) pairs with Eq. 4 features. extra_fk: declared single-column FKs (SJI+IS)."""
    fkcols = {}
    for A, m in tables.items():
        for a, ty in m["cols"].items():
            if a in pks[A]["chosen"]:
                continue
            fkcols[(A, a)] = ty
    for (A, a, B, b) in extra_fk:
        fkcols.setdefault((A, a), tables[A]["cols"][a])
    clean_sets = {}
    for (A, a), ty in fkcols.items():
        v = clean(smp[A][a], kind(ty))
        clean_sets[(A, a)] = {key_of(x) for x in v.tolist()}
    pairs = []
    for (A, a), ty in fkcols.items():
        fs = clean_sets[(A, a)]
        if not fs:
            continue
        for (B, b), ps in pkv.items():
            if B == A or not type_match(ty, tables[B]["cols"][b]):
                continue
            if len(fs) > len(ps):
                continue
            if fs <= ps:
                pairs.append((A, a, B, b))
    dep = defaultdict(int)
    ref = defaultdict(int)
    for A, a, B, b in pairs:
        dep[(A, a)] += 1
        ref[(B, b)] += 1
    mref = max(ref.values()) if ref else 1
    feats = {}
    for A, a, B, b in pairs:
        da, db = prof[A][a]["distinct"], prof[B][b]["distinct"]
        f = dict(card_ratio=min(da / max(1, db), 1.0), mult_depend=1.0 / dep[(A, a)], mult_refs=ref[(B, b)] / mref,
                 name_sim=longest_match(a, b) / max(len(a), len(b)), typical_suffix=1.0 if ends(a, FK_SUFFIX) else 0.0)
        f["score"] = sum(f.values()) / 5.0
        feats[(A, a, B, b)] = f
    return feats, clean_sets


PROMPT = """You are validating join relationships inferred for a database's semantic layer.
Statistical profiling found that every sampled value of the column {A}.{a} also occurs in the column {B}.{b}, which looks
like a primary key of table {B}. Such value-domain overlap can be a real foreign key, or a coincidence (for example two
unrelated auto-increment ids, small integer codes, counts or flags that happen to fall in the same range).

Table {A} has columns: {acols}
Table {B} has columns: {bcols}
Sample values of {A}.{a}: {av}
Sample values of {B}.{b}: {bv}

Is {A}.{a} a foreign key that semantically references {B}.{b}, i.e. is joining {A}.{a} = {B}.{b} a meaningful join?
Answer ONLY with a JSON object: {{"valid": true}} or {{"valid": false}}."""


def fmt_vals(vs):
    return ", ".join(repr(str(v)[:40]) for v in vs)


def adjudicate(jobs):
    """jobs: list of (ind, prompt). Returns {ind: (verdict, usage)}."""
    def run(job):
        ind, msg = job
        r = llm.chat(MODEL, [{"role": "user", "content": msg}], max_tokens=60, thinking={"type": "disabled"})
        m = re.search(r'"valid"\s*:\s*[^a-z]*?(true|false)', r["text"], re.I) or re.search(r"\b(true|false)\b", r["text"], re.I)
        return ind, (m.group(1).lower() == "true" if m else False), r.get("usage", {}), r["text"]
    out = {}
    with ThreadPoolExecutor(8) as ex:
        for ind, v, u, txt in ex.map(run, jobs):
            out[ind] = (v, u, txt)
    return out


def usd(u):
    return (u.get("prompt_tokens", 0) * PRICE[0] + u.get("completion_tokens", 0) * PRICE[1]) / 1e6


# ---------------------------------------------------------------------------------------------- step 3: history
def _walk(node, fn):
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk(v, fn)


def _from_tables(node, out):
    """Base tables of a FROM clause (not descending into subqueries); also JOIN ... USING pairs."""
    if not isinstance(node, dict):
        return
    ty = node.get("type")
    if ty == "BASE_TABLE":
        out.append((node.get("alias") or node["table_name"], node["table_name"]))
    elif ty == "JOIN":
        _from_tables(node.get("left"), out)
        _from_tables(node.get("right"), out)


def history_joins(sqls, tables):
    """Equi-join predicates col = col between two different base tables, from FROM/ON and WHERE clauses."""
    colset = {t: {c.lower(): c for c in m["cols"]} for t, m in tables.items()}
    tl = {t.lower(): t for t in tables}
    found, n_parse_err = [], 0
    con = duckdb.connect()

    def visit(node, scope):
        if isinstance(node, list):
            for v in node:
                visit(v, scope)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "SELECT_NODE":
            ft = []
            _from_tables(node.get("from_table"), ft)
            sc = dict(scope)
            for al, tn in ft:
                if tn.lower() in tl:
                    sc[al.lower()] = tl[tn.lower()]
                    sc.setdefault(tn.lower(), tl[tn.lower()])
            local = [tl[tn.lower()] for _, tn in ft if tn.lower() in tl]
            for k, v in node.items():
                visit(v, sc)
                if k in ("from_table", "where_clause"):
                    _collect(v, sc, local)
            return
        for v in node.values():
            visit(v, scope)

    def resolve(ref, sc, local):
        names = ref["column_names"]
        if len(names) >= 2:
            t = sc.get(names[-2].lower())
            c = names[-1].lower()
            if t and c in colset[t]:
                return t, colset[t][c]
            return None
        c = names[0].lower()
        hit = [t for t in dict.fromkeys(local) if c in colset[t]]
        return (hit[0], colset[hit[0]][c]) if len(hit) == 1 else None

    def _collect(expr, sc, local):
        def fn(n):
            if n.get("type") in ("SELECT_NODE",):
                return
            if n.get("class") == "COMPARISON" and n.get("type") == "COMPARE_EQUAL":
                l, r = n.get("left"), n.get("right")
                if l and r and l.get("class") == "COLUMN_REF" and r.get("class") == "COLUMN_REF":
                    a, b = resolve(l, sc, local), resolve(r, sc, local)
                    if a and b and a[0] != b[0]:
                        found.append((a, b))
            if n.get("type") == "JOIN" and n.get("using_columns"):
                L, R = [], []
                _from_tables(n.get("left"), L)
                _from_tables(n.get("right"), R)
                for c in n["using_columns"]:
                    for _, lt in L:
                        for _, rt in R:
                            lt2, rt2 = tl.get(lt.lower()), tl.get(rt.lower())
                            if lt2 and rt2 and c.lower() in colset[lt2] and c.lower() in colset[rt2]:
                                found.append(((lt2, colset[lt2][c.lower()]), (rt2, colset[rt2][c.lower()])))
        _walk_noselect(expr, fn)

    for s in sqls:
        j = json.loads(con.execute("SELECT json_serialize_sql(?)", [s.replace("`", '"')]).fetchone()[0])
        if j.get("error"):
            n_parse_err += 1
            continue
        visit(j["statements"], {})
    con.close()
    uniq = {}
    for a, b in found:
        k = tuple(sorted([a, b]))
        uniq[k] = uniq.get(k, 0) + 1
    return uniq, n_parse_err


def _walk_noselect(node, fn):
    if isinstance(node, dict):
        if node.get("type") == "SELECT_NODE" or node.get("type") == "SUBQUERY":
            return
        fn(node)
        for v in node.values():
            _walk_noselect(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk_noselect(v, fn)


# ---------------------------------------------------------------------------------------------- Algorithm 1
def join_tree(inds):
    """Algorithm 1: roots = left-most tables (no incoming IND) sorted by reference count; for every reachable dimension
    keep the path maximising the product of hop IND scores; return the union of the chosen paths' hops."""
    out_adj = defaultdict(list)
    indeg = defaultdict(int)
    refc = defaultdict(int)
    for (A, a, B, b), s in inds.items():
        out_adj[A].append(((A, a, B, b), s))
        indeg[B] += 1
        refc[B] += 1
    nodes = set(out_adj) | set(indeg)
    roots = sorted([t for t in nodes if indeg[t] == 0 and out_adj[t]], key=lambda r: -refc[r])
    chosen = set()
    for r in roots:
        best = {}  # dim -> (score, path)

        def dfs(u, path, score, seen):
            for hop, s in out_adj[u]:
                v = hop[2]
                if v in seen:
                    continue
                ns = score * s
                if v not in best or ns > best[v][0]:
                    best[v] = (ns, path + [hop])
                if len(path) < 4:
                    dfs(v, path + [hop], ns, seen | {v})
        dfs(r, [], 1.0, {r})
        for v, (s, p) in sorted(best.items(), key=lambda z: -len(z[1][1])):
            chosen.update(p)
    return chosen


# ---------------------------------------------------------------------------------------------- driver
def sji(P, st, con, with_is=False, history=True, cache=None):
    """Run SJI on one instance. Returns (accepted_edges, detail). st (catalog stats) is unused: SJI computes its own."""
    assert not P.name.startswith("ent"), "enterprise instances are out of scope for this baseline"
    t0 = time.time()
    tables = {t: dict(rows=v["rows"], cols=v["cols"]) for t, v in P.rec["tables"].items()}
    cnt = Counter_()
    if cache is not None and "prof" in cache:  # the SJI+IS variant reuses the same profile / sample (same rows read)
        prof, smp, cnt0 = cache["prof"], cache["smp"], cache["cnt"]
        for k in ("profile", "sample"):
            cnt.rows[k] = cnt0.rows[k]
        cnt.tables |= cnt0.tables
        cnt.stmts = cnt0.stmts_ps
    else:
        prof = profile(con, tables, cnt)
    pks = infer_pk(tables, prof)
    if with_is:  # Sec. 8: information-schema PKs seed the candidates (single-column keys only are checkable here)
        for t, k in P.rec["pk"].items():
            if k and len(k) == 1 and t in pks and k[0] not in pks[t]["chosen"]:
                pks[t]["chosen"] = pks[t]["chosen"] + [k[0]]
    if cache is None or "smp" not in cache:
        smp = sample_tables(con, tables, cnt)
        if cache is not None:
            cnt.stmts_ps = cnt.stmts
            cache.update(prof=prof, smp=smp, cnt=cnt)
    pkv = {}
    for t, d in pks.items():
        for c in d["chosen"]:
            pkv[(t, c)] = pk_values(con, t, c, kind(tables[t]["cols"][c]), cnt, tables[t]["rows"])
    extra = []
    if with_is:
        extra = [(e[0], e[1][0], e[2], e[3][0]) for e in P.declared if len(e[1]) == 1 and e[4] is None]
    feats, clean_sets = ind_candidates(tables, prof, pks, smp, pkv, extra)
    kept = {k: f for k, f in feats.items() if f["score"] >= IND_THR}
    jobs = []
    for (A, a, B, b) in sorted(kept):
        msg = PROMPT.format(A=A, a=a, B=B, b=b, acols=", ".join(tables[A]["cols"]), bcols=", ".join(tables[B]["cols"]),
                            av=fmt_vals(spread(clean_sets[(A, a)])), bv=fmt_vals(spread(pkv[(B, b)])))
        jobs.append(((A, a, B, b), msg))
    verdict = adjudicate(jobs)
    inds = {k: kept[k]["score"] for k in kept if verdict[k][0]}
    usage = [verdict[k][1] for k in verdict]
    # ---- query history (BIRD only)
    hist = dict(used=False)
    hist_edges = {}
    if history and P.name.startswith("bird_") and BIRD_DEV:
        dev = json.load(open(BIRD_DEV[0]))
        sqls = [x["SQL"] for x in dev if x["db_id"] == P.name[5:]]
        joins, perr = history_joins(sqls, tables)
        valid = []
        for ((t1, c1), (t2, c2)), n in sorted(joins.items()):
            ok = con.execute(f"SELECT 1 FROM {q(t1)} a JOIN {q(t2)} b ON a.{q(c1)} = b.{q(c2)} LIMIT 1").fetchone() is not None
            cnt.add("history_validate", t1, tables[t1]["rows"])
            cnt.add("history_validate", t2, tables[t2]["rows"])
            if not ok:
                continue
            # orientation: the side that SJI holds as a PK candidate is referenced; else higher distinct ratio
            pk1 = c1 in pks[t1]["chosen"]
            pk2 = c2 in pks[t2]["chosen"]
            r1 = prof[t1][c1]["distinct"] / max(1, prof[t1][c1]["count"])
            r2 = prof[t2][c2]["distinct"] / max(1, prof[t2][c2]["count"])
            ors = []
            if pk2 and not pk1: ors = [(t1, c1, t2, c2)]
            elif pk1 and not pk2: ors = [(t2, c2, t1, c1)]
            elif pk1 and pk2: ors = [(t1, c1, t2, c2), (t2, c2, t1, c1)]
            else: ors = [(t1, c1, t2, c2)] if r2 >= r1 else [(t2, c2, t1, c1)]
            for o in ors:
                hist_edges[o] = 1.0
            valid.append(dict(join=[t1, c1, t2, c2], queries=n, oriented=[list(o) for o in ors]))
        hist = dict(used=True, n_sql=len(sqls), parse_errors=perr, n_join_pairs=len(joins), n_valid=len(valid), valid=valid)
    # ---- outputs
    edges = set(P.edges)

    def to_edge(ind):
        A, a, B, b = ind
        return (A, (a,), B, (b,), None)

    merged = dict(inds)
    for k, s in hist_edges.items():
        merged.setdefault(k, s)
    acc = sorted({to_edge(k) for k in merged if to_edge(k) in edges})
    unscored = sorted([list(k) for k in merged if to_edge(k) not in edges])
    tree = sorted({to_edge(k) for k in join_tree(merged) if to_edge(k) in edges})
    dollars = sum(usd(u) for u in usage)
    det = dict(
        inst=P.name, variant="SJI+IS" if with_is else "SJI",
        pk={t: dict(chosen=d["chosen"], clear_winner=d["clear"], candidates=d["cands"][:8],
                    declared=P.rec["pk"].get(t)) for t, d in pks.items()},
        pk_accuracy=_pk_acc(pks, P.rec["pk"]),
        n_ind_contained=len(feats), n_ind_after_threshold=len(kept), n_ind_after_llm=len(inds),
        llm=[dict(ind=list(k), score=round(kept[k]["score"], 4), valid=verdict[k][0], text=verdict[k][2][:80]) for k in sorted(kept)],
        history=hist, accepted=[Pm_list(e) for e in acc], accepted_tree=[Pm_list(e) for e in tree],
        inferred_not_in_pool=unscored, llm_calls=len(jobs), model=llm.MODELS[MODEL][1],
        prompt_tokens=sum(u.get("prompt_tokens", 0) for u in usage),
        completion_tokens=sum(u.get("completion_tokens", 0) for u in usage), dollars=round(dollars, 5),
        rows_read=sum(cnt.rows.values()), rows_read_by_step=dict(cnt.rows), sql_statements=cnt.stmts,
        rows_scan_shared=sum(tables[t]["rows"] for t in cnt.tables), secs=round(time.time() - t0, 2))
    return acc, tree, det


def Pm_list(e):
    return [e[0], list(e[1]), e[2], list(e[3]), e[4]]


def _pk_acc(pks, gold):
    """Single-column declared PKs recovered as the unique chosen key (paper's PK accuracy, for reference only)."""
    n = ok = 0
    for t, d in pks.items():
        g = gold.get(t)
        if not g:
            continue
        n += 1
        ok += d["clear"] and len(g) == 1 and d["chosen"] == g
    return dict(tables_with_declared_pk=n, exact_single_winner=ok)


def main(names):
    out_csv = os.path.join(ROOT, "results", "baseline_sji.csv")
    out_det = os.path.join(ROOT, "results", "baseline_sji_detail.json")
    rows, details = [], {}
    log0 = _usage_lines()
    for name in names:
        if name.startswith("ent"):
            print("skip enterprise instance", name)
            continue
        P, st = load(name)
        W = P.workload()
        con = duckdb.connect(db_path(name), read_only=True)
        cache = {}
        acc, tree, det = sji(P, st, con, cache=cache)
        r = P.evaluate(W, acc, LABEL, det["rows_read"])
        rows.append(r)
        det["eval"] = r
        det["eval_tree"] = P.evaluate(W, tree, "SJI-tree(Alg.1)", det["rows_read"])
        acc2, tree2, det2 = sji(P, st, con, with_is=True, cache=cache)
        det2["eval"] = P.evaluate(W, acc2, "SJI+IS", det2["rows_read"])
        det2.pop("pk", None)
        det["variant_is"] = det2
        con.close()
        details[name] = det
        print(name, "acc", len(acc), "cert", r["n_certified"], "cov", round(r["coverage"], 3), "fcr_card",
              round(r["fcr_card"], 3), "fcr_sem", round(r["fcr_sem"], 3), "llm", det["llm_calls"], "$", det["dollars"],
              "secs", det["secs"], flush=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        json.dump(details, open(out_det, "w"), indent=1, default=str)
    print("new LLM calls this run (usage.log lines):", _usage_lines() - log0)


def _usage_lines():
    p = os.path.join(llm.CACHE_DIR, "usage.log")
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def summary():
    """Pooled per family, mirroring summarize.pooled on the zero-probe baselines (counts summed across instances)."""
    import summarize as S
    det = json.load(open(os.path.join(ROOT, "results", "baseline_sji_detail.json")))
    rows = []
    for n, d in det.items():
        rows.append(d["eval"])
        rows.append(d["eval_tree"])
        rows.append(d["variant_is"]["eval"])
    z = pd.read_csv(os.path.join(ROOT, "results", "main_zero.csv"))
    z = z[z.method.isin(["DDL-declared", "CardEst(HLL stats)", "Auto-BI-style(reimpl.)", "LLM-judge:ds-flash"])
          & z.inst.isin(list(det))]
    x = pd.concat([pd.DataFrame(rows), z], ignore_index=True)
    x["family"] = x["inst"].map(S.fam)
    g = S.pooled(x, ["family", "method"])
    cost = pd.DataFrame([dict(family=S.fam(n), dollars=d["dollars"] + d["variant_is"]["dollars"], sji_dollars=d["dollars"],
                              calls=d["llm_calls"], rows_read=d["rows_read"], rows_shared=d["rows_scan_shared"],
                              secs=d["secs"]) for n, d in det.items()]).groupby("family").sum()
    order = [o for o in S.ORDER if o in set(x.family)]
    out = g[["n_certified", "n_card", "n_sem", "n_ok", "n_answerable", "coverage", "fcr_card", "fcr_sem"]].reset_index()
    out["family"] = pd.Categorical(out["family"], order)
    print(S.md(out.sort_values(["family", "method"])))
    print()
    print(S.md(cost.reindex(order).reset_index(), "{:.4g}"))
    return out, cost


def ablation(names, modes=("tukey", "none")):
    """Sensitivity of the IND cleaning rule (Sec. 5 "values within the inter-quartile range"); stored in the detail JSON."""
    global CLEAN_MODE
    out_det = os.path.join(ROOT, "results", "baseline_sji_detail.json")
    details = json.load(open(out_det))
    for name in names:
        P, st = load(name)
        W = P.workload()
        con = duckdb.connect(db_path(name), read_only=True)
        cache = {}
        details[name].setdefault("ablation_clean", {})
        for m in modes:
            CLEAN_MODE = m
            acc, tree, det = sji(P, st, con, cache=cache)
            r = P.evaluate(W, acc, "SJI-clean=" + m, det["rows_read"])
            details[name]["ablation_clean"][m] = dict(eval=r, accepted=det["accepted"], llm_calls=det["llm_calls"],
                                                      dollars=det["dollars"], n_ind_after_llm=det["n_ind_after_llm"])
            print(name, m, "acc", len(acc), "cert", r["n_certified"], "cov", round(r["coverage"], 3), "fcr_card",
                  round(r["fcr_card"], 3), "llm", det["llm_calls"], "$", det["dollars"], flush=True)
        CLEAN_MODE = "iqr"
        con.close()
        json.dump(details, open(out_det, "w"), indent=1, default=str)


if __name__ == "__main__" and "--ablation" in sys.argv:
    ablation([a for a in sys.argv[1:] if not a.startswith("--")])
    sys.exit(0)

if __name__ == "__main__" and "--summary" in sys.argv:
    summary()
    sys.exit(0)

if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool"))
                                   if f.endswith(".json") and not f.startswith("ent"))
    main(names)
