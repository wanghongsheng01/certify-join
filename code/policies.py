"""Certification policies replayed on a prepared pool (results/pool/<inst>.json).

State: each atom is unknown / True / False. An edge is certified when its U and I atoms are True.
A query q = (fact F, target tables D) is certified when every target is reachable from F through certified edges;
the planner then picks the most plausible tree (max sum of log prior) among certified edges.
"""
import heapq, json, math, os, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def T(x):
    return tuple(T(v) for v in x) if isinstance(x, list) else x


def akey(a):
    return json.dumps(a)


def atoms_of(e):
    R, X, S, Y, Tm = e
    return akey(["U", S, list(Y), [Tm[1], Tm[2]] if Tm else None]), akey(["I", R, list(X), S, list(Y), list(Tm) if Tm else None])


class Pool:
    def __init__(self, rec, prior_model="ds-flash", gate=0.0):
        self.rec = rec
        self.name = rec["name"]
        self.edges = [T(d["e"]) for d in rec["edges"]]
        self.src = {T(d["e"]): d["src"] for d in rec["edges"]}
        self.ename = {T(d["e"]): d["name"] for d in rec["edges"]}
        self.gold = {T(g) for g in rec["gold"]}
        self.declared = {T(g) for g in rec["declared"]}
        self.A = rec["atoms"]
        self.ea = {e: atoms_of(e) for e in self.edges}
        self.truth = {k: v["truth"] for k, v in self.A.items() if not k.startswith("E")}
        self.cost = {k: v["cost"] for k, v in self.A.items() if not k.startswith("E")}
        self.rows = {t: v["rows"] for t, v in rec["tables"].items()}
        self.safe = {e: self.truth[self.ea[e][0]] and self.truth[self.ea[e][1]] for e in self.edges}
        self.set_prior(prior_model)
        self.gate = gate
        # enforced keys give U for free
        self.free = set()
        for e in self.edges:
            S, Y = e[2], e[3]
            pk = rec["pk"].get(S)
            if S in rec["pk_enforced"] and pk and set(pk) <= set(Y) and e[4] is None:
                self.free.add(self.ea[e][0])
        self.full_cost = sum(self.cost[a] for a in {a for e in self.edges for a in self.ea[e]})
        self.touched = sorted({t for e in self.edges for t in (e[0], e[2])})
        self.full_scan = sum(self.rows[t] for t in self.touched)

    def set_prior(self, model):
        pri = self.rec["prior"].get(model, {}) if model else {}
        self.p = {}
        for e in self.edges:
            v = pri.get(self.ename[e])
            if v is None:
                v = 0.5
            self.p[e] = min(0.99, max(0.01, float(v)))

    # ------------------------------------------------------------------ workload
    def workload(self, seed=0, per_fact=40, decoy_frac=0.25):
        rnd = random.Random(seed)
        gold_adj = defaultdict(list)
        for e in self.gold:
            gold_adj[e[0]].append(e)
        cand_adj = defaultdict(list)
        for e in self.edges:
            cand_adj[e[0]].append(e)

        def reach(adj, F, depth=3):
            seen, fr = {F: 0}, [F]
            for d in range(depth):
                nf = []
                for u in fr:
                    for e in adj[u]:
                        if e[2] not in seen:
                            seen[e[2]] = d + 1
                            nf.append(e[2])
                fr = nf
            seen.pop(F)
            return seen

        W = []
        for F in self.rec["facts"]:
            if F not in self.rows:
                continue
            g = sorted(reach(gold_adj, F))
            if not g:
                continue
            qs = set((t,) for t in g)
            for _ in range(per_fact * 3):
                if len(qs) >= per_fact:
                    break
                k = rnd.choice([2, 3])
                if len(g) >= k:
                    qs.add(tuple(sorted(rnd.sample(g, k))))
            dec = sorted(set(reach(cand_adj, F, 2)) - set(g))
            nd = min(len(dec), int(round(decoy_frac * len(qs))))
            for t in rnd.sample(dec, nd):
                qs.add((t,))
            W += [(F, q) for q in sorted(qs)]
        return W

    def truly_answerable(self, q):
        F, D = q
        if not hasattr(self, "_ans"):
            self._ans = {}
        if F not in self._ans:
            self._ans[F] = self._reach(F, [e for e in self.gold if self.safe[e]])
        return all(t in self._ans[F] for t in D)

    def _reach(self, F, edges):
        adj = defaultdict(list)
        for e in edges:
            adj[e[0]].append(e)
        seen, st = {F}, [F]
        while st:
            u = st.pop()
            for e in adj[u]:
                if e[2] not in seen:
                    seen.add(e[2])
                    st.append(e[2])
        return seen

    def plan(self, q, cert):
        """Most plausible tree over certified edges (Dijkstra on -log p); None if some target unreachable."""
        F, D = q
        adj = defaultdict(list)
        for e in cert:
            adj[e[0]].append(e)
        dist, par, pq = {F: 0.0}, {}, [(0.0, F)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for e in adj[u]:
                nd = d - math.log(self.p[e])
                if nd < dist.get(e[2], 1e18):
                    dist[e[2]], par[e[2]] = nd, e
                    heapq.heappush(pq, (nd, e[2]))
        if not all(t in dist for t in D):
            return None
        tree = set()
        for t in D:
            while t != F:
                e = par[t]
                tree.add(e)
                t = e[0]
        return tree

    def evaluate(self, W, cert, label, cost, extra=None):
        cert = set(cert)
        n_ans = n_cert = n_ok = n_card = n_sem = n_card_eps = 0
        for q in W:
            ans = self.truly_answerable(q)
            n_ans += ans
            tree = self.plan(q, cert)
            if tree is None:
                continue
            n_cert += 1
            card = any(not self.safe[e] for e in tree)
            if card:
                # violations that an (eps=1e-3) inclusion certificate tolerates: only I atoms with orphan rate <= eps
                bad = [a for e in tree for a in self.ea[e] if not self.truth[a]]
                tol = all(a.startswith('["I"') and self.A[a]["orphans"] <= 1e-3 * max(1, self.A[a]["nonnull"]) for a in bad)
                n_card_eps += not tol
            sem = any(e not in self.gold for e in tree)
            n_card += card
            n_sem += (sem and not card)
            n_ok += (not card and not sem)
        full = self.full_scan if (extra or {}).get("cost_model") == "scan" else self.full_cost
        r = dict(inst=self.name, method=label, cost=cost, cost_frac=cost / full, n_queries=len(W),
                 n_answerable=n_ans, n_certified=n_cert, coverage=n_ok / max(1, n_ans),
                 fcr_card=n_card / max(1, n_cert), fcr_sem=n_sem / max(1, n_cert), n_card=n_card, n_sem=n_sem,
                 n_ok=n_ok, n_card_eps=n_card_eps)
        if extra:
            r.update(extra)
        return r


# ---------------------------------------------------------------------- inference closure
def atom_tables(a):
    d = json.loads(a)
    return {d[1]} if d[0] == "U" else {d[1], d[3]}


class State:
    """cost_model "atom": every probe is its own SQL query (cost = rows it reads, as in per-edge validation jobs);
    "scan": probes share table scans (cost = rows of tables not yet scanned; once all tables an atom reads are
    scanned the atom is known, which is the profiling assumption of SPIDER-style IND discovery)."""

    def __init__(self, pool, use_inference=True, cost_model="atom"):
        self.P = pool
        self.st = {}
        self.inf = use_inference
        self.cm = cost_model
        self.scanned = set()
        self.spent = 0
        self.probes = 0
        self.inferred = 0
        for a in pool.free:
            self.st[a] = True

    def cost_of(self, a):
        if self.cm == "atom":
            return self.P.cost[a]
        return sum(self.P.rows[t] for t in atom_tables(a) - self.scanned)

    def scan(self, tables):
        new = set(tables) - self.scanned
        self.spent += sum(self.P.rows[t] for t in new)
        self.scanned |= new
        if new:
            for a in self.P.truth:
                if a not in self.st and atom_tables(a) <= self.scanned:
                    self.st[a] = self.P.truth[a]

    def status(self, a):
        return self.st.get(a)

    def edge_status(self, e):
        u, i = self.P.ea[e]
        su, si = self.st.get(u), self.st.get(i)
        if su is False or si is False:
            return False
        if su and si:
            return True
        return None

    def probe(self, a, outcome=None, cost=None):
        self.probes += 1
        if self.cm == "scan" and cost is None:
            self.scan(atom_tables(a))
        else:
            self.spent += self.P.cost[a] if cost is None else cost
            self.st[a] = self.P.truth[a] if outcome is None else outcome
        if self.inf:
            self.closure()
        return self.st[a]

    def closure(self):
        """Sound rules: superkey (U(S,Y) => U(S,Y+Z)), negative subkey, IND transitivity and its contrapositive."""
        P = self.P
        if not hasattr(P, "_Ug"):
            P._Ug = defaultdict(list)
            P._I = []
            for a in P.truth:
                d = json.loads(a)
                if d[0] == "U":
                    P._Ug[(d[1], json.dumps(d[3]))].append((a, set(d[2])))
                elif d[5] is None:
                    P._I.append((a, (d[1], tuple(d[2]), d[3], tuple(d[4]))))
        if not hasattr(P, "_I_rules"):
            # Static index (same semantics as the original pass, which scans every group and every I atom):
            # only U groups with >= 2 atoms can fire, and an I atom a = (R,X,S_,Y) can only be inferred from an atom
            # b = (R,X,S1,Y1) whose partner I(S1,Y1,S_,Y) [rule True] or I(S_,Y,S1,Y1), S1 != S_ [rule False] exists.
            # Lists keep the P._I order so the first applicable rule is the same as before.
            P._Ug_multi = [grp for grp in P._Ug.values() if len(grp) >= 2]
            ikey = {k: a for a, k in P._I}
            src = defaultdict(list)
            for a, k in P._I:
                src[(k[0], k[1])].append((a, k[2], k[3]))
            P._I_rules = []
            for a, (R, X, S_, Y) in P._I:
                L = []
                for b, S1, Y1 in src[(R, X)]:
                    if b == a:
                        continue
                    ta = ikey.get((S1, Y1, S_, Y))
                    fa = ikey.get((S_, Y, S1, Y1)) if S1 != S_ else None
                    if ta is not None or fa is not None:
                        L.append((b, ta, fa))
                if L:
                    P._I_rules.append((a, L))
        st = self.st
        changed = True
        while changed:
            changed = False
            for grp in P._Ug_multi:
                for a, Y in grp:
                    if a in self.st:
                        continue
                    for b, Y2 in grp:
                        v = self.st.get(b)
                        if v is None:
                            continue
                        if v and Y2 <= Y:
                            self.st[a] = True
                        elif v is False and Y <= Y2:
                            self.st[a] = False
                        if a in self.st:
                            self.inferred += 1
                            changed = True
                            break
            # statuses are read as of the start of this pass (atoms inferred in this pass are not used until the next)
            new = set()
            for a, L in P._I_rules:
                if a in st:
                    continue
                for b, ta, fa in L:
                    if b in new:
                        continue
                    v = st.get(b)
                    if v is None:
                        continue
                    if v and ta is not None and ta not in new and st.get(ta):
                        st[a] = True
                        break
                    if v is False and fa is not None and fa not in new and st.get(fa):
                        st[a] = False
                        break
                if a in st:
                    new.add(a)
                    self.inferred += 1
                    changed = True

    def certified(self):
        return [e for e in self.P.edges if self.edge_status(e)]


# ---------------------------------------------------------------------- Certify-Join (adaptive bundle greedy)
_CJK_SRC = r'''/* Resumable single-source Dijkstra used by policies.certify_join (optional accelerator; the pure-Python path in
 * policies.py computes the same thing). Semantics mirror the Python search exactly:
 *   heap order = (dist, node id) lexicographic, node ids follow Python string order of the node names;
 *   relax v when d + w < (dist[v] if discovered else 1e30), strict; stale heap entries (d > dist[u]) skipped;
 *   stop when the last not-yet-expanded target is popped (that node is not expanded).
 * Build: cc -O2 -ffp-contract=off -shared -fPIC -o _cjk.so _cjk.c */
#include <stdint.h>

typedef struct { double d; int32_t v; } item;

static int less(const item *a, const item *b) { return a->d < b->d || (a->d == b->d && a->v < b->v); }

static void push(item *h, int64_t *n, double d, int32_t v) {
    int64_t i = (*n)++;
    h[i].d = d; h[i].v = v;
    while (i > 0) {
        int64_t p = (i - 1) >> 1;
        if (!less(&h[i], &h[p])) break;
        item t = h[i]; h[i] = h[p]; h[p] = t; i = p;
    }
}

static item pop(item *h, int64_t *n) {
    item top = h[0];
    (*n)--;
    if (*n > 0) {
        h[0] = h[*n];
        int64_t i = 0;
        for (;;) {
            int64_t l = 2 * i + 1, r = l + 1, m = i;
            if (l < *n && less(&h[l], &h[m])) m = l;
            if (r < *n && less(&h[r], &h[m])) m = r;
            if (m == i) break;
            item t = h[i]; h[i] = h[m]; h[m] = t; i = m;
        }
    }
    return top;
}

static void relax(int32_t u, double d, const int64_t *rowptr, const int32_t *tgt, const double *w,
                  const uint8_t *alive, double *dist, int32_t *par, uint8_t *has, int32_t *touched,
                  int32_t *n_touched, item *h, int64_t *nh) {
    for (int64_t e = rowptr[u]; e < rowptr[u + 1]; e++) {
        if (!alive[e]) continue;
        int32_t v = tgt[e];
        double nd = d + w[e];
        double cur = has[v] ? dist[v] : 1e30;
        if (nd < cur) {
            if (!has[v]) { has[v] = 1; touched[(*n_touched)++] = v; }
            dist[v] = nd; par[v] = (int32_t)e;
            if (h) push(h, nh, nd, v);
        }
    }
}

/* r >= 0: roll the state back to its first r expansions (replayed with the current weights, which are unchanged
 * for those nodes by construction) before continuing; r < 0: continue the current state.
 * Returns the id of the final popped target (not expanded), -1 if the heap ran empty, -2 if nothing to do. */
int advance(int32_t src, const int64_t *rowptr, const int32_t *tgt, const double *w, const uint8_t *alive,
            double *dist, int32_t *par, int32_t *idx, uint8_t *has, int32_t *order, int32_t *n_order,
            int32_t *touched, int32_t *n_touched, int32_t r, const int32_t *targets, int32_t n_targets,
            uint8_t *mark, item *h) {
    int64_t nh = 0;
    if (r >= 0) {
        for (int32_t k = r; k < *n_order; k++) idx[order[k]] = -1;
        *n_order = r;
        for (int32_t k = 0; k < *n_touched; k++) has[touched[k]] = 0;
        *n_touched = 0;
        has[src] = 1; dist[src] = 0.0; touched[(*n_touched)++] = src;
        for (int32_t k = 0; k < r; k++) {
            int32_t u = order[k];
            relax(u, dist[u], rowptr, tgt, w, alive, dist, par, has, touched, n_touched, 0, 0);
        }
    }
    int32_t left = 0;
    for (int32_t i = 0; i < n_targets; i++) {
        mark[targets[i]] = 1;
        if (idx[targets[i]] < 0) left++;
    }
    int32_t final = -2;
    if (left > 0) {
        final = -1;
        for (int32_t k = 0; k < *n_touched; k++) {
            int32_t v = touched[k];
            if (idx[v] < 0) push(h, &nh, dist[v], v);
        }
        while (nh > 0) {
            item it = pop(h, &nh);
            int32_t u = it.v;
            double d = it.d;
            if (d > dist[u]) continue;
            if (mark[u]) {
                left--;
                if (left == 0) { final = u; break; }
            }
            idx[u] = *n_order;
            order[(*n_order)++] = u;
            relax(u, d, rowptr, tgt, w, alive, dist, par, has, touched, n_touched, h, &nh);
        }
    }
    for (int32_t i = 0; i < n_targets; i++) mark[targets[i]] = 0;
    return final;
}
'''
_CJK = None


def _kernel():
    """ctypes handle of the C search used by certify_join, compiled on first use into code/_build/ (keyed by the
    source hash). None when CJ_PURE is set or no C compiler is available; certify_join then runs the same search
    in Python."""
    global _CJK
    if os.environ.get("CJ_PURE"):
        return None
    if _CJK is None:
        try:
            import ctypes, hashlib, subprocess
            h = hashlib.sha1(_CJK_SRC.encode()).hexdigest()[:12]
            d = os.path.join(ROOT, "code", "_build")
            so = os.path.join(d, "_cjk_%s.so" % h)
            if not os.path.exists(so):
                os.makedirs(d, exist_ok=True)
                c = os.path.join(d, "_cjk_%s.c" % h)
                tmp = "%s.%d.tmp" % (so, os.getpid())
                with open(c + ".%d.tmp" % os.getpid(), "w") as f:
                    f.write(_CJK_SRC)
                os.replace(c + ".%d.tmp" % os.getpid(), c)
                subprocess.run(["cc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", tmp, c],
                               check=True, capture_output=True)
                os.replace(tmp, so)
            fn = ctypes.CDLL(so).advance
            V, I = ctypes.c_void_p, ctypes.c_int32
            fn.argtypes = [I] + [V] * 12 + [I, V, I, V, V]
            fn.restype = I
            _CJK = fn
        except Exception:
            _CJK = False
    return _CJK or None


def certify_join(pool, W, budget, prior=True, inference=True, failfast=True, adaptive=True, sample=None, seed=0,
                 gate=0.05, cost_model="atom", mgreedy=False):
    """sample = (eps, delta): I atoms on large R are probed on m = ln(1/delta)/eps rows (an (eps,delta) certificate).

    Each round: cheapest-path bundle (unknown atoms on the min-weight join tree) for every open query, pick the
    bundle maximising (sum of probabilities of the bundles it contains) / cost, probe its atoms fail-first.
    Implementation (same decisions as the plain loop kept in _private/policies_before_scale.py):
    - adjacency weights are rebuilt only at nodes whose edges have an atom whose status (or cost) changed;
    - one Dijkstra per fact serves all its open queries (the per-target search of the plain loop is a prefix of
      it); when it expanded such a node it is rolled back to just before that node and resumed, and only the
      queries whose targets were settled after that point get new bundles;
    - "bundles contained in this bundle" is kept as a subset relation maintained with an inverted index
      atom -> bundles, and each bundle value is re-summed (in workload order) only when that relation or a
      member's probability changed; the arg-max over bundles is vectorised."""
    import numpy as np
    P = pool
    S = State(P, use_inference=inference, cost_model=cost_model)
    rnd = random.Random(seed)
    m_samp = None
    if sample:
        eps, delta = sample
        m_samp = int(math.ceil(math.log(1 / delta) / eps))

    sampled = {}
    for a in P.truth:
        if m_samp and a.startswith('["I"'):
            d = json.loads(a)
            if P.rows[d[1]] > 10 * m_samp:
                sampled[a] = d

    def acost(a):
        if a in sampled:
            d = sampled[a]
            return m_samp + (0 if (cost_model == "scan" and d[3] in S.scanned) else P.rows[d[3]])
        return S.cost_of(a)

    def bcost(need):
        if cost_model == "atom":
            return sum(acost(a) for a in need)
        tabs, c = set(), 0
        for a in need:
            if a in sampled:
                c += m_samp
                tabs.add(sampled[a][3])
            else:
                tabs |= atom_tables(a)
        return c + sum(P.rows[t] for t in tabs - S.scanned)
    pa = defaultdict(lambda: 0.5)
    if prior:
        for x in P.edges:
            for a in P.ea[x]:
                pa[a] = max(pa[a] if a in pa else 0.0, P.p[x])
    edges_ok = [e for e in P.edges if P.p[e] >= gate or not prior]

    # ---- adjacency (per source node, edges in edges_ok order), rebuilt per dirty node
    out_ok = defaultdict(list)
    atom_nodes = defaultdict(set)
    for e in edges_ok:
        out_ok[e[0]].append(e)
        for a in P.ea[e]:
            atom_nodes[a].add(e[0])

    def node_adj(u):
        lst = []
        for e in out_ok[u]:
            if S.edge_status(e) is not False:
                w = 1e-6 + sum(acost(a) / pa[a] for a in P.ea[e] if S.status(a) is None)
                lst.append((e, w))
        return lst

    adj = defaultdict(list)

    # Per fact F, the state of one Dijkstra from F is kept: dist, par, the expansion order and an undo log of the
    # relaxations. For every target t, the per-target search of the plain loop (which stops when t is popped) is a
    # prefix of this search, so par[t] and its ancestors are the same. The search is a deterministic function of
    # the adjacency lists of the nodes it expands: when nodes change, the state is rolled back to just before the
    # first changed node it expanded (that prefix is what a fresh search would do) and resumed from there; the heap
    # is rebuilt from the discovered-but-unexpanded nodes (stale heap entries are skipped by the search anyway).
    FS = {}
    node_f = defaultdict(set)       # node -> facts whose search state expanded it

    def fstate(F):
        if F not in FS:
            FS[F] = ({F: 0.0}, {}, {}, [], [])  # dist, par, idx (node -> expansion rank), order, undo log
        return FS[F]

    def truncate(F, r):
        dist, par, idx, order, log = fstate(F)
        while log and log[-1][0] >= r:
            k, v, od, oe = log.pop()
            if od is None:
                del dist[v]
                del par[v]
            else:
                dist[v], par[v] = od, oe
        for u in order[r:]:
            del idx[u]
            node_f[u].discard(F)
        del order[r:]

    def advance(F, targets):
        """Continue F's search until every target is settled (or the heap is exhausted); returns settled targets."""
        dist, par, idx, order, log = fstate(F)
        done = {t for t in targets if t in idx}
        left = len(targets) - len(done)
        if left == 0:
            return done
        pq = [(d, v) for v, d in dist.items() if v not in idx]
        heapq.heapify(pq)
        hpop, hpush, dget = heapq.heappop, heapq.heappush, dist.get
        while pq:
            d, u = hpop(pq)
            if d > dist[u]:
                continue
            if u in targets and u not in done:
                done.add(u)
                left -= 1
                if left == 0:
                    break
            k = len(order)
            idx[u] = k
            order.append(u)
            node_f[u].add(F)
            for e, w in adj[u]:
                v = e[2]
                od = dget(v)
                if d + w < (1e30 if od is None else od):
                    log.append((k, v, od, par.get(v)))
                    dist[v], par[v] = d + w, e
                    hpush(pq, (d + w, v))
        return done

    def derive(q, done, par):
        F, D = q
        need, prob = set(), 1.0
        for t in D:
            if t not in done:
                return None
            x = t
            while x != F:
                e = par[x]
                for a in P.ea[e]:
                    if S.status(a) is None:
                        need.add(a)
                x = e[0]
        for a in need:
            prob *= pa[a]
        return need, prob

    n = len(W)
    is_open = [True] * n            # query still open (not certified, not refuted); never reopens
    res = [None] * n                # current (need, prob) of active queries
    post = defaultdict(set)         # atom -> active queries whose bundle contains it
    canon = {}                      # active query -> canonical atom of its bundle
    cpost = defaultdict(set)        # canonical atom -> active queries
    sub = {}                        # i -> active j with need_j <= need_i  (incl. i)
    sup = {}                        # j -> active i with need_j <= need_i  (incl. j)
    val = np.zeros(n)
    cst = np.zeros(n)
    act = np.zeros(n, dtype=bool)

    fq = defaultdict(list)          # fact -> its queries (W order)
    for i, (F, D) in enumerate(W):
        fq[F].append(i)

    def compute_fact(F, r=None):
        """Roll F's search back to rank r (None: no change) and return {i: bundle} for its open queries whose
        targets were not all expanded before rank r (the others keep their bundles)."""
        qs = [i for i in fq[F] if is_open[i]]
        if not qs:
            truncate(F, 0)
            return {}
        idx = fstate(F)[2]
        if r is not None:
            qs = [i for i in qs if not all(t in idx and idx[t] < r for t in W[i][1])]
            truncate(F, r)
        if not qs:
            return {}
        targets = {t for i in fq[F] if is_open[i] for t in W[i][1]}
        done = advance(F, targets)
        par = fstate(F)[1]
        return {i: derive(W[i], done, par) for i in qs}

    def update_nodes(dn):
        for u in dn:
            adj[u] = node_adj(u)

    def affected(dn):
        dF = {}
        for u in dn:
            for F in node_f[u]:
                k = FS[F][2][u]
                if k < dF.get(F, 1 << 60):
                    dF[F] = k
        return dF

    K = _kernel()
    if K is not None:
        # Same search in C (_CJK_SRC): node ids follow Python string order so heap ties break as (d, name) does.
        names = sorted(set(P.rows) | {x for e in edges_ok for x in (e[0], e[2])} | {x for F, D in W for x in (F,) + tuple(D)})
        nid = {x: i for i, x in enumerate(names)}
        N = len(names)
        E_ok = [e for u in names for e in out_ok.get(u, ())]
        M = len(E_ok)
        rowptr = np.zeros(N + 1, dtype=np.int64)
        for u in names:
            rowptr[nid[u] + 1] = len(out_ok.get(u, ()))
        rowptr = np.cumsum(rowptr).astype(np.int64)
        tgt = np.array([nid[e[2]] for e in E_ok] or [0], dtype=np.int32)
        wt = np.zeros(max(1, M))
        alive = np.zeros(max(1, M), dtype=np.uint8)
        e_src = [nid[e[0]] for e in E_ok]
        facts = sorted(fq)
        fidx = {F: i for i, F in enumerate(facts)}
        nF = len(facts)
        A_dist = np.zeros((nF, N))
        A_par = np.zeros((nF, N), dtype=np.int32)
        A_idx = np.full((nF, N), -1, dtype=np.int32)
        A_has = np.zeros((nF, N), dtype=np.uint8)
        A_order = np.zeros((nF, N), dtype=np.int32)
        A_touch = np.zeros((nF, N), dtype=np.int32)
        A_no = np.zeros(nF, dtype=np.int32)
        A_nt = np.zeros(nF, dtype=np.int32)
        mark = np.zeros(N, dtype=np.uint8)
        heap = np.zeros(2 * (N + M + 1), dtype=np.float64)  # (double, int32) items, 16 bytes each
        started = np.zeros(nF, dtype=bool)
        fopen = np.zeros(nF, dtype=bool)

        def update_nodes(dn):
            for u in dn:
                if u not in nid:
                    continue
                x = nid[u]
                for k in range(int(rowptr[x]), int(rowptr[x + 1])):
                    e = E_ok[k]
                    if S.edge_status(e) is not False:
                        alive[k] = 1
                        wt[k] = 1e-6 + sum(acost(a) / pa[a] for a in P.ea[e] if S.status(a) is None)
                    else:
                        alive[k] = 0

        update_nodes(list(out_ok))
        B = {k: (v.ctypes.data, v.strides[0]) for k, v in dict(rowptr=rowptr, tgt=tgt, wt=wt, alive=alive, dist=A_dist,
             par=A_par, idx=A_idx, has=A_has, order=A_order, touch=A_touch, no=A_no, nt=A_nt, mark=mark,
             heap=heap).items()}
        mv_idx = [memoryview(A_idx[f]) for f in range(nF)]
        mv_par = [memoryview(A_par[f]) for f in range(nF)]

        def affected(dn):
            ids = [nid[u] for u in dn if u in nid]
            if not ids:
                return {}
            sub_ = A_idx[:, ids]
            sub_ = np.where(sub_ >= 0, sub_, np.iinfo(np.int32).max).min(axis=1)
            hit = np.flatnonzero((sub_ < np.iinfo(np.int32).max) & fopen)
            return {facts[f]: int(sub_[f]) for f in hit}

        def compute_fact(F, r=None):
            f = fidx[F]
            qs = [i for i in fq[F] if is_open[i]]
            if not qs:
                fopen[f] = False
                return {}
            fopen[f] = True
            irow = mv_idx[f]
            if r is not None:
                qs = [i for i in qs if not all(0 <= irow[nid[t]] < r for t in W[i][1])]
                if not qs:
                    return {}
            if not started[f]:
                started[f] = True
                r = 0
            targets = np.array(sorted({nid[t] for i in fq[F] if is_open[i] for t in W[i][1]}), dtype=np.int32)
            row = lambda k: B[k][0] + f * B[k][1]
            final = K(nid[F], B["rowptr"][0], B["tgt"][0], B["wt"][0], B["alive"][0], row("dist"), row("par"),
                      row("idx"), row("has"), row("order"), B["no"][0] + 4 * f, row("touch"), B["nt"][0] + 4 * f,
                      -1 if r is None else r, targets.ctypes.data, len(targets), B["mark"][0], B["heap"][0])
            prow = mv_par[f]
            ir = irow
            out = {}
            for i in qs:
                Fq, D = W[i]
                need, prob, ok = set(), 1.0, True
                for t in D:
                    x = nid[t]
                    if not (ir[x] >= 0 or x == final):
                        ok = False
                        break
                    while x != nid[Fq]:
                        k = prow[x]
                        for a in P.ea[E_ok[k]]:
                            if S.status(a) is None:
                                need.add(a)
                        x = e_src[k]
                if not ok:
                    out[i] = None
                    continue
                for a in need:
                    prob *= pa[a]
                out[i] = (need, prob)
            return out

    def deactivate(j):
        need = res[j][0]
        for i in sup.pop(j):
            if i != j:
                sub[i].discard(j)
                vdirty.add(i)
        for k in sub.pop(j):
            if k != j:
                sup[k].discard(j)
        for a in need:
            post[a].discard(j)
        cpost[canon.pop(j)].discard(j)
        res[j] = None
        act[j] = False

    def refresh(changed):
        """changed: {i: new bundle or None}; keeps post/sub/sup/val/act consistent with res."""
        for j in changed:
            if res[j] is not None:
                deactivate(j)
        newly = []
        for j, b in changed.items():
            if b is None or not b[0]:
                is_open[j] = False
                continue
            res[j] = b
            for a in b[0]:
                post[a].add(j)
            c = min(b[0])
            canon[j] = c
            cpost[c].add(j)
            act[j] = True
            newly.append(j)
        for j in newly:
            need = res[j][0]
            ps = sorted((post[a] for a in need), key=len)
            sp = set(ps[0]).intersection(*ps[1:]) if len(ps) > 1 else set(ps[0])
            sb = set()
            for a in need:
                for k in cpost[a]:
                    if res[k][0] <= need:
                        sb.add(k)
            sup[j], sub[j] = sp, sb
            for i in sp:
                if i in sub:
                    sub[i].add(j)
                vdirty.add(i)
            for k in sb:
                if k in sup:
                    sup[k].add(j)
            vdirty.add(j)
        for j in newly:  # pairs among newly activated bundles are covered by both directions above
            pass
        for i in vdirty:
            if res[i] is None:
                continue
            v = 0.0
            for k in sorted(sub[i]):
                v += res[k][1]
            val[i] = v
            if cost_model == "atom":
                cst[i] = bcost(res[i][0])
        vdirty.clear()

    vdirty = set()
    if K is None:
        update_nodes(list(out_ok))
    init = {}
    for F in fq:
        init.update(compute_fact(F))
    refresh(init)
    S.iterations = 0
    while True:
        if not act.any():
            break
        rem = budget - S.spent
        idx = np.flatnonzero(act)
        if cost_model == "atom":
            c = cst[idx]
        else:
            c = np.array([float(bcost(res[i][0])) for i in idx])
        ok = c <= rem
        if not ok.any():
            break
        s = np.where(ok, val[idx] / np.maximum(c, 1e-9), -np.inf)
        bi = int(idx[int(np.argmax(s))])
        if mgreedy:
            # modified-greedy rule (budgeted knapsack): if one affordable bundle is worth more than what the ratio
            # rule packs into the remaining budget (estimated by packing bundles in ratio order), take that bundle
            bj = int(idx[int(np.argmax(np.where(ok, val[idx], -np.inf)))])
            tot, sp = 0.0, 0.0
            for o in np.argsort(-s, kind="stable"):
                if not ok[o]:
                    break
                if sp + c[o] <= rem:
                    tot += val[idx[o]]
                    sp += c[o]
            if val[bj] > tot:
                bi = bj
        best = res[bi]
        S.iterations += 1
        n_st, scanned0 = len(S.st), len(S.scanned)
        st_vals = dict(S.st) if cost_model != "atom" else None
        need = list(best[0])
        if failfast:
            need.sort(key=lambda a: acost(a) / max(1e-6, 1 - pa[a]))
        else:
            need.sort(key=lambda a: acost(a))
        for a in need:
            if S.status(a) is not None:
                continue
            c = acost(a)
            if S.spent + c > budget:
                break
            if a in sampled and S.status(a) is None:
                st = P.A[a]
                rate = st["orphans"] / max(1, st["nonnull"])
                hit = rnd.random() < 1 - (1 - rate) ** m_samp
                if cost_model == "scan":
                    S.scan({sampled[a][3]})
                    S.spent += m_samp
                    S.probes += 1
                    S.st[a] = not hit
                    if S.inf:
                        S.closure()
                else:
                    S.probe(a, outcome=not hit, cost=c)
            else:
                S.probe(a)
            if S.status(a) is False:
                break
        # ---- invalidate: nodes with an edge whose atom changed status (atom model: statuses only grow and costs
        # are fixed); scan model: costs depend on the scanned tables and sampled atoms may be overwritten, so every
        # node and every open query is recomputed as in the plain loop.
        if cost_model == "atom":
            k = len(S.st) - n_st
            dn = set()
            if k:
                it = reversed(S.st)
                for _ in range(k):
                    dn |= atom_nodes.get(next(it), set())
        else:
            changed_st = len(S.st) != n_st or len(S.scanned) != scanned0 or any(
                S.st[a] != v for a, v in st_vals.items())
            dn = set(out_ok) if changed_st else set()
        update_nodes(dn)
        if cost_model == "atom":
            dF = affected(dn)
        else:
            dF = {F: 0 for F in fq} if dn else {}
        ch = {}
        for F in sorted(dF):
            for i, b in compute_fact(F, dF[F]).items():
                if not act[i]:
                    continue
                old = res[i]
                if b is None or not b[0] or b[0] != old[0] or b[1] != old[1]:
                    ch[i] = b
                else:
                    res[i] = b  # same atoms and probability; keep the freshly built set (its iteration order is used)
        refresh(ch)
        if not adaptive:
            pass
    return S


# ---------------------------------------------------------------------- baselines (probing)
def run_order(pool, order, budget, cost_model="atom"):
    S = State(pool, use_inference=False, cost_model=cost_model)
    for a in order:
        if S.status(a) is not None:
            continue
        c = S.cost_of(a)
        if S.spent + c > budget:
            continue
        S.probe(a)
    return S


def full_order(pool):
    seen, out = set(), []
    for e in sorted(pool.edges, key=lambda e: (pool.src[e] != "ddl", pool.ename[e])):
        for a in pool.ea[e]:
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out


def random_order(pool, seed):
    o = full_order(pool)
    random.Random(seed).shuffle(o)
    return o


def keycov_order(pool, W):
    """Key-coverage heuristic: atoms ranked by (#workload queries whose fact or target touches an edge using
    the atom) / cost, i.e. hub keys first."""
    touch = defaultdict(int)
    tabs = defaultdict(int)
    for F, D in W:
        tabs[F] += 1
        for t in D:
            tabs[t] += 1
    for e in pool.edges:
        for a in pool.ea[e]:
            touch[a] += tabs[e[0]] + tabs[e[2]]
    return sorted(touch, key=lambda a: -touch[a] / max(1, pool.cost[a]))


def profile_tables(pool, budget):
    """IND-discovery style profiling (SPIDER/HoPF-like): scan whole tables in ascending size; one pass over a
    table yields all its uniqueness and inclusion facts. An atom is known once all tables it reads are scanned."""
    S = State(pool, use_inference=False, cost_model="scan")
    for t in sorted(pool.touched, key=lambda t: pool.rows[t]):
        if S.spent + pool.rows[t] > budget:
            continue
        S.scan({t})
    return S


# ---------------------------------------------------------------------- clairvoyant optimum (ILP over enumerated paths)
def opt_ilp(pool, W, budget, max_len=3):
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    safe_edges = [e for e in pool.edges if pool.safe[e] and e in pool.gold]
    adj = defaultdict(list)
    for e in safe_edges:
        adj[e[0]].append(e)

    def paths(F, t):
        out, st = [], [(F, [])]
        while st:
            u, p = st.pop()
            if u == t and p:
                out.append(p)
                continue
            if len(p) >= max_len:
                continue
            for e in adj[u]:
                if all(e[2] != x[0] for x in p) and e[2] != F:
                    st.append((e[2], p + [e]))
        return out

    atoms = sorted({a for e in safe_edges for a in pool.ea[e]} - pool.free)
    ai = {a: i for i, a in enumerate(atoms)}
    var = len(atoms)
    pvars, qvars, qt = [], [], []
    rows, lo, hi = [], [], []
    for qi, (F, D) in enumerate(W):
        qv = var
        var += 1
        qvars.append(qv)
        for t in D:
            ps = paths(F, t)
            pv_list = []
            for p in ps:
                pv = var
                var += 1
                pv_list.append(pv)
                for e in p:
                    for a in pool.ea[e]:
                        if a in ai:
                            rows.append({pv: 1, ai[a]: -1}); lo.append(-np.inf); hi.append(0)
            # q <= sum of its paths for target t
            r = {qv: 1}
            for pv in pv_list:
                r[pv] = -1
            rows.append(r); lo.append(-np.inf); hi.append(0)
    c = np.zeros(var)
    for qv in qvars:
        c[qv] = -1
    A = np.zeros((len(rows) + 1, var))
    for k, r in enumerate(rows):
        for j, v in r.items():
            A[k, j] = v
    for a, i in ai.items():
        A[-1, i] = pool.cost[a]
    lo.append(-np.inf); hi.append(budget)
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(var), bounds=Bounds(0, 1),
               options={"time_limit": 60})
    if res.x is None:
        return None
    S = State(pool, use_inference=False)
    for a, i in ai.items():
        if res.x[i] > 0.5:
            S.probe(a)
    return S
