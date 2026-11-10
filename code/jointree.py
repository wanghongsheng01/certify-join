"""Certified join-tree selection (RQ8): direction, Steiner trees and path-equivalence certificates.

Motivated by the production governance platform (v4 analysis, 22_v4_prod): edges are verified child->parent
(parent key unique, match_rate >= 0.95, fanout <= 1) but routed by undirected BFS, so 40 of 41 verified edges
amplify rows when walked parent->child (up to 10,547x), and equal-length routes are ambiguous (5.6% at 2 hops,
24.9% at 3 hops). This script replays that registry rule on every pilot instance and measures

  1. reverse-direction expansion of every verified edge (the production "fanout = 1.0" statistic is child->parent);
  2. on the generated workload, three planners over the same verified edges:
       bfs      undirected shortest-hop routing, hop cap H (production before 2026-11-04);
       refuse   bfs + refuse any expanding parent->child hop (production after 2026-11-04);
       steiner  min directed Steiner arborescence over N:1 hops only (Dreyfus-Wagner, unit weights);
  3. equal-length directed ambiguity per (source, target) pair, resolved by probing path-equivalence atoms
     C(p == p'): the number of source rows on which two N:1 paths reach different target rows.

Writes results/jointree/<inst>.json and results/jointree_summary.csv. Enterprise outputs contain counts only.
"""
import itertools, json, os, sys, time
from collections import defaultdict
import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import Pool, T, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "results", "jointree")
MATCH_MIN = 0.95  # ADR-0006 of the production platform
MAX_PROBES = 300


def connect(name):
    if name == "ent_starrocks":
        p = os.path.join(ROOT, "_private", "ent.duckdb")
    elif name.startswith("bird_"):
        p = os.path.join(ROOT, "data", "bird", name[5:] + ".duckdb")
    else:
        p = os.path.join(ROOT, "data", name + ".duckdb")
    return duckdb.connect(p, read_only=True)


def q(c):
    return '"' + c.replace('"', '""') + '"'


def verified(pool):
    """Production registry rule: parent key unique (U), match_rate >= 0.95, at least one non-null child value."""
    out = []
    for e in pool.edges:
        u, i = pool.ea[e]
        I = pool.A[i]
        if pool.truth[u] and I["nonnull"] > 0 and 1 - I["orphans"] / I["nonnull"] >= MATCH_MIN:
            out.append(e)
    return out


def reverse_mult(con, e):
    """Multiplicity of child rows per matched parent key, i.e. expansion when walking S -> R."""
    R, X, S, Y, Tm = e
    xs = ", ".join(q(x) for x in X)
    nn = " AND ".join(f"{q(x)} IS NOT NULL" for x in X)
    r = con.execute(f'SELECT count(*), coalesce(sum(c), 0), coalesce(max(c), 0) FROM '
                    f'(SELECT {xs}, count(*) c FROM {q(R)} WHERE {nn} GROUP BY {xs})').fetchone()
    ndv, rows, mx = r
    return dict(ndv=int(ndv), rows=int(rows), max=int(mx), avg=(rows / ndv if ndv else 0.0))


# ----------------------------------------------------------------------------- graph helpers
def hops(edges, rev, directed):
    """Adjacency u -> [(v, edge, forward, expanding)]; forward = child->parent (N:1)."""
    adj = defaultdict(list)
    for e in edges:
        adj[e[0]].append((e[2], e, True, False))
        if not directed:
            adj[e[2]].append((e[0], e, False, rev[e]["max"] > 1))
    return adj


def hops_grain_safe(edges, rev):
    """Production main-join search as coded (uat 2b4a3a7, _grain_safe_step): N:1 hops, plus parent->child hops that
    do not expand rows; expanding hops are skipped during the search instead of refused afterwards."""
    adj = defaultdict(list)
    for e in edges:
        adj[e[0]].append((e[2], e, True, False))
        if rev[e]["max"] <= 1:
            adj[e[2]].append((e[0], e, False, False))
    return adj


def shortest_paths(adj, src, H):
    """All shortest-hop simple paths from src to every node within H hops: node -> list of hop lists."""
    dist, paths, frontier = {src: 0}, {src: [[]]}, [src]
    for d in range(1, H + 1):
        nxt = {}
        for u in frontier:
            for v, e, fw, ex in adj[u]:
                if dist.get(v, d) < d:
                    continue
                for p in paths[u]:
                    if len(paths.get(v, [])) + len(nxt.get(v, [])) >= 64:
                        break
                    nxt.setdefault(v, []).append(p + [(u, v, e, fw, ex)])
        for v, ps in nxt.items():
            dist[v] = d
            paths[v] = ps
        frontier = list(nxt)
    paths.pop(src)
    return paths


def steiner(adj, F, D):
    """Exact minimum directed Steiner arborescence (unit weights) rooted at F spanning D; returns edge set or None."""
    D = list(D)
    nodes = set(adj) | {v for u in adj for v, *_ in adj[u]} | {F} | set(D)
    INF = 10 ** 9
    # all-pairs BFS distances with parents
    dist, par = {}, {}
    for s in nodes:
        d, p, fr = {s: 0}, {}, [s]
        while fr:
            nf = []
            for u in fr:
                for v, e, *_ in adj[u]:
                    if v not in d:
                        d[v], p[v] = d[u] + 1, (u, e)
                        nf.append(v)
            fr = nf
        dist[s], par[s] = d, p
    if any(t not in dist[F] for t in D):
        return None
    k = len(D)
    full = (1 << k) - 1
    dp = {}
    back = {}
    for i, t in enumerate(D):
        for v in nodes:
            dp[(1 << i, v)] = dist[v].get(t, INF)
            back[(1 << i, v)] = ("path", t)
    for S in range(1, full + 1):
        if S & (S - 1) == 0:
            continue
        for v in nodes:
            best, arg = INF, None
            sub = (S - 1) & S
            while sub:
                if sub < (S ^ sub):
                    c = dp[(sub, v)] + dp[(S ^ sub, v)]
                    if c < best:
                        best, arg = c, ("split", sub)
                sub = (sub - 1) & S
            dp[(S, v)], back[(S, v)] = best, arg
        for v in nodes:  # extend: v reaches a junction u, then splits there
            best, arg = dp[(S, v)], back[(S, v)]
            for u in nodes:
                du = dist[v].get(u)
                if du and du + dp[(S, u)] < best:
                    best, arg = du + dp[(S, u)], ("via", u)
            dp[(S, v)], back[(S, v)] = best, arg

    tree = set()

    def walk(s, t):
        while t != s:
            u, e = par[s][t]
            tree.add(e)
            t = u

    def rec(S, v):
        kind, x = back[(S, v)]
        if kind == "path":
            walk(v, x)
        elif kind == "split":
            rec(x, v)
            rec(S ^ x, v)
        else:
            walk(v, x)
            rec(S, x)

    rec(full, F)
    return tree


# ----------------------------------------------------------------------------- path-equivalence probe
def path_sql(path, alias, start):
    js, prev = [], start
    for j, (u, v, e, fw, ex) in enumerate(path):
        R, X, S, Y, Tm = e
        a = f"{alias}{j}"
        cond = " AND ".join(f"{prev}.{q(x)} = {a}.{q(y)}" for x, y in zip(X, Y))
        if Tm:
            d, f, t = Tm
            cond += f" AND {prev}.{q(d)} BETWEEN {a}.{q(f)} AND coalesce({a}.{q(t)}, DATE '9999-12-31')"
        js.append(f"LEFT JOIN {q(S)} {a} ON {cond}")
        prev = a
    return js, prev


def probe_equiv(con, src, p1, p2):
    j1, a1 = path_sql(p1, "a", "t")
    j2, a2 = path_sql(p2, "b", "t")
    sql = (f"SELECT count(*), count(*) FILTER (WHERE {a1}.rowid IS DISTINCT FROM {a2}.rowid), "
           f"count(*) FILTER (WHERE {a1}.rowid IS NOT NULL AND {a2}.rowid IS NOT NULL AND {a1}.rowid <> {a2}.rowid) "
           f"FROM {q(src)} t " + " ".join(j1 + j2))
    n, diff, both = con.execute(sql).fetchone()
    return dict(n=int(n), diff=int(diff), diff_both=int(both))


# ----------------------------------------------------------------------------- main per instance
def run(name, H=3):
    pool = Pool(json.load(open(os.path.join(ROOT, "results", "pool", name + ".json"))))
    con = connect(name)
    V = verified(pool)
    rev = {e: reverse_mult(con, e) for e in V}
    gold = pool.gold
    res = dict(inst=name, n_tables=len(pool.rows), n_cand=len(pool.edges), n_verified=len(V),
               n_verified_gold=sum(e in gold for e in V))
    res["rev_expanding"] = sum(rev[e]["max"] > 1 for e in V)
    res["rev_max"] = max([rev[e]["max"] for e in V] or [0])
    res["rev_avg_max"] = max([rev[e]["avg"] for e in V] or [0])
    res["rev_top"] = sorted(([pool.ename[e], rev[e]["max"], round(rev[e]["avg"], 1)] for e in V),
                            key=lambda r: -r[1])[:5] if name != "ent_starrocks" else []

    und, dirg = hops(V, rev, False), hops(V, rev, True)
    # 1) pair-level route ambiguity, production style (all ordered table pairs, hop caps 2 and 3)
    for h in (2, 3):
        for lab, adj in (("und", und), ("dir", dirg)):
            n = amb = fan = 0
            for s in pool.rows:
                for t, ps in shortest_paths(adj, s, h).items():
                    n += 1
                    amb += len(ps) > 1
                    fan += all(any(hp[4] for hp in p) for p in ps)
            res[f"pairs_{lab}_h{h}"], res[f"amb_{lab}_h{h}"] = n, amb
            if lab == "und":
                res[f"fanonly_und_h{h}"] = fan

    # 2) directed ties resolved by path-equivalence atoms C(p == p'); "role" ties visit the same tables through
    #    different columns (role-playing FKs), "route" ties go through different intermediate tables
    probes, t0 = [], time.time()
    amb_pairs = []
    for s in sorted(pool.rows):
        for t, ps in shortest_paths(dirg, s, H).items():
            if len(ps) > 1:
                amb_pairs.append((s, t, ps))
    equiv = {}  # (s, t) -> True when every probed alternative reaches the same target row on every source row
    kinds = defaultdict(int)
    for s, t, ps in amb_pairs:
        route = len({tuple(hp[1] for hp in p) for p in ps}) > 1
        kinds["route" if route else "role"] += 1
        if len(probes) >= MAX_PROBES or len(ps) > 4:
            continue
        ok = True
        for p2 in ps[1:]:
            r = probe_equiv(con, s, ps[0], p2)
            r.update(src=s, dst=t, hops=len(ps[0]), kind="route" if route else "role",
                     rows_read=pool.rows[s] + sum(pool.rows[hp[1]] for hp in ps[0] + p2),
                     gold1=all(hp[2] in gold for hp in ps[0]), gold2=all(hp[2] in gold for hp in p2))
            if name == "ent_starrocks":
                r.pop("src"), r.pop("dst")
            probes.append(r)
            ok &= r["diff"] == 0
        equiv[(s, t)] = ok
    res["dir_amb_pairs"], res["dir_amb_role"], res["dir_amb_route"] = len(amb_pairs), kinds["role"], kinds["route"]
    res["n_probes"] = len(probes)
    res["probe_equiv"] = sum(p["diff"] == 0 for p in probes)
    res["probe_equiv_mod_null"] = sum(p["diff"] > 0 and p["diff_both"] == 0 for p in probes)
    res["probe_diverge"] = sum(p["diff_both"] > 0 for p in probes)
    res["probe_route"] = sum(p["kind"] == "route" for p in probes)
    res["probe_route_equiv"] = sum(p["kind"] == "route" and p["diff"] == 0 for p in probes)
    res["probe_rows"] = sum(p["rows_read"] for p in probes)
    res["probe_secs"] = round(time.time() - t0, 2)

    # 3) planners on the generated workload (seed 0)
    W = pool.workload(seed=0)
    stats = {m: defaultdict(int) for m in ("bfs3", "prod2", "prod3", "prodc2", "prodc3", "steiner")}
    safe = hops_grain_safe(V, rev)
    hop_union = hop_st = n_multi = 0
    for F, D in W:
        # production main-join planner as coded: grain-safe search, hop cap 2 (MAX_HOPS) or 3, and on equal-length
        # routes it takes the first one (find_routes(...)[0]) without refusing or certifying
        for m, h in (("prodc2", 2), ("prodc3", 3)):
            sp = shortest_paths(safe, F, h)
            st = stats[m]
            st["n"] += 1
            if any(t not in sp for t in D):
                st["unreach"] += 1
                continue
            tied = any(len(sp[t]) > 1 for t in D)
            tree = [hp for t in D for hp in sp[t][0]]
            sem = any(hp[2] not in gold for hp in tree)
            st["answered"] += 1
            st["tied_silent"] += tied
            st["nongold"] += sem
            st["correct"] += not sem
            st["correct_untied"] += not sem and not tied
        for m, h in (("bfs3", 3), ("prod2", 2), ("prod3", 3)):
            sp = shortest_paths(und, F, h)
            st = stats[m]
            st["n"] += 1
            if any(t not in sp for t in D):
                st["unreach"] += 1
                continue
            amb = any(len(sp[t]) > 1 for t in D)
            st["ambiguous"] += amb
            tree = [hp for t in D for hp in sp[t][0]]
            expand = any(hp[4] for hp in tree)
            if m.startswith("prod") and amb:
                st["refused_amb"] += 1
                continue
            if m.startswith("prod") and expand:
                st["refused_fanout"] += 1
                continue
            st["answered"] += 1
            st["fanout"] += expand
            sem = any(hp[2] not in gold for hp in tree)
            st["nongold"] += sem and not expand
            st["correct"] += (not expand and not sem)
        st = stats["steiner"]
        st["n"] += 1
        tree = steiner(dirg, F, D)
        if tree is None:
            st["unreach"] += 1
            continue
        spd = shortest_paths(dirg, F, 6)
        tied = [t for t in D if len(spd[t]) > 1]
        if tied:
            st["ambiguous"] += 1
            if not all(equiv.get((F, t), False) for t in tied):
                st["refused_amb"] += 1
                continue
            st["amb_certified"] += 1
        st["answered"] += 1
        sem = any(e not in gold for e in tree)
        st["nongold"] += sem
        st["correct"] += not sem
        if len(D) > 1:
            n_multi += 1
            hop_st += len(tree)
            hop_union += len({hp[2] for t in D for hp in spd[t][0]})
    res["n_queries"] = len(W)
    res["n_answerable"] = sum(pool.truly_answerable(x) for x in W)
    for m, st in stats.items():
        for k, v in st.items():
            res[f"{m}_{k}"] = v
    res["multi_queries"], res["hops_steiner"], res["hops_union"] = n_multi, hop_st, hop_union
    res["full_scan"] = pool.full_scan
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(summary=res, probes=probes,
                   rev={pool.ename[e]: rev[e] for e in V} if name != "ent_starrocks" else {}),
              open(os.path.join(OUT, name + ".json"), "w"), ensure_ascii=False, indent=1)
    con.close()
    print(name, {k: res[k] for k in ("n_verified", "rev_expanding", "rev_max", "dir_amb_pairs", "n_probes",
                                     "probe_equiv", "probe_diverge", "steiner_correct", "prod2_correct")}, flush=True)
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "results", "pool")))
    rows = [run(n) for n in names]
    keys = []
    for r in rows:
        keys += [k for k in r if k != "rev_top" and k not in keys]
    with open(os.path.join(ROOT, "results", "jointree_summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
