"""Benchmark instances for Certify-Join: schema metadata, gold FKs, candidate join edges, injected defects.

An edge is a tuple (R, X, S, Y, T) meaning "R.X -> S.Y is proposed as an N:1 join", X and Y are tuples of
column names and T is None or a temporal predicate (R.date_col, S.valid_from, S.valid_to).
"""
import json, os, re, difflib, random, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
PRIV = os.path.join(ROOT, "_private")
BIRD = os.environ.get("BIRD_DEV", "<path to BIRD dev_20240627>")

INT_T = re.compile(r"int|bigint|smallint|tinyint|hugeint|integer", re.I)
STR_T = re.compile(r"char|text|string|varchar", re.I)
NUM_T = re.compile(r"decimal|double|float|real|numeric", re.I)
KEYLIKE = {"id", "key", "sk", "code", "no", "num", "number"}


def E(R, X, S, Y, T=None):
    X = (X,) if isinstance(X, str) else tuple(X)
    Y = (Y,) if isinstance(Y, str) else tuple(Y)
    return (R, X, S, Y, T)


def ename(e):
    R, X, S, Y, T = e
    s = f"{R}.{'+'.join(X)}->{S}.{'+'.join(Y)}"
    return s + ("@" + T[1] if T else "")


# ----------------------------------------------------------------------------- gold FKs
TPCH_PK = {"region": ["r_regionkey"], "nation": ["n_nationkey"], "part": ["p_partkey"], "supplier": ["s_suppkey"],
           "partsupp": ["ps_partkey", "ps_suppkey"], "customer": ["c_custkey"], "orders": ["o_orderkey"],
           "lineitem": ["l_orderkey", "l_linenumber"]}
TPCH_FK = [E("nation", "n_regionkey", "region", "r_regionkey"), E("supplier", "s_nationkey", "nation", "n_nationkey"),
           E("customer", "c_nationkey", "nation", "n_nationkey"), E("partsupp", "ps_partkey", "part", "p_partkey"),
           E("partsupp", "ps_suppkey", "supplier", "s_suppkey"), E("orders", "o_custkey", "customer", "c_custkey"),
           E("lineitem", "l_orderkey", "orders", "o_orderkey"), E("lineitem", "l_partkey", "part", "p_partkey"),
           E("lineitem", "l_suppkey", "supplier", "s_suppkey"),
           E("lineitem", ("l_partkey", "l_suppkey"), "partsupp", ("ps_partkey", "ps_suppkey"))]
TPCH_FACTS = ["lineitem", "orders", "partsupp"]

DS_DIM = {"date_sk": "date_dim", "time_sk": "time_dim", "item_sk": "item", "customer_sk": "customer",
          "cdemo_sk": "customer_demographics", "hdemo_sk": "household_demographics", "addr_sk": "customer_address",
          "store_sk": "store", "promo_sk": "promotion", "ship_mode_sk": "ship_mode", "warehouse_sk": "warehouse",
          "call_center_sk": "call_center", "catalog_page_sk": "catalog_page", "web_page_sk": "web_page",
          "web_site_sk": "web_site", "reason_sk": "reason", "income_band_sk": "income_band"}
DS_FACTS = ["store_sales", "store_returns", "catalog_sales", "catalog_returns", "web_sales", "web_returns", "inventory",
            "sales_bk"]


def tpcds_gold(con):
    pk, fk = {}, []
    cols = {t: [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()] for (t,) in con.execute("SHOW TABLES").fetchall()}
    for t, cs in cols.items():
        if t not in DS_FACTS:
            pk[t] = [cs[0]]
    pk.update({"store_sales": ["ss_item_sk", "ss_ticket_number"], "store_returns": ["sr_item_sk", "sr_ticket_number"],
               "catalog_sales": ["cs_item_sk", "cs_order_number"], "catalog_returns": ["cr_item_sk", "cr_order_number"],
               "web_sales": ["ws_item_sk", "ws_order_number"], "web_returns": ["wr_item_sk", "wr_order_number"],
               "inventory": ["inv_date_sk", "inv_item_sk", "inv_warehouse_sk"]})
    for t, cs in cols.items():
        for c in cs:
            if not c.endswith("_sk") or (len(pk.get(t, [])) == 1 and pk[t][0] == c):
                continue
            hit = max((s for s in DS_DIM if c.endswith(s)), key=len, default=None)
            if hit and DS_DIM[hit] != t:
                d = DS_DIM[hit]
                fk.append(E(t, c, d, pk[d][0]))
    for r, s, a, b in [("store_returns", "store_sales", ("sr_item_sk", "sr_ticket_number"), ("ss_item_sk", "ss_ticket_number")),
                       ("catalog_returns", "catalog_sales", ("cr_item_sk", "cr_order_number"), ("cs_item_sk", "cs_order_number")),
                       ("web_returns", "web_sales", ("wr_item_sk", "wr_order_number"), ("ws_item_sk", "ws_order_number"))]:
        fk.append(E(r, a, s, b))
    # business-key fact: joins to the SCD dimensions are N:1 only with the validity-interval predicate
    fk.append(E("sales_bk", "sb_item_id", "item", "i_item_id", ("sb_sold_date", "i_rec_start_date", "i_rec_end_date")))
    fk.append(E("sales_bk", "sb_store_id", "store", "s_store_id", ("sb_sold_date", "s_rec_start_date", "s_rec_end_date")))
    pk["sales_bk"] = None
    return pk, fk


# ----------------------------------------------------------------------------- loaders
def _meta(con, tables=None):
    out = {}
    for (t,) in con.execute("SHOW TABLES").fetchall():
        if tables and t not in tables:
            continue
        cols = {r[0]: r[1] for r in con.execute(f'DESCRIBE "{t}"').fetchall()}
        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        out[t] = {"rows": n, "cols": cols}
    return out


def load_tpch():
    con = duckdb.connect(os.path.join(DATA, "tpch.duckdb"), read_only=True)
    m = _meta(con)
    return dict(name="tpch", con=con, tables=m, pk=TPCH_PK, pk_enforced=set(), declared=list(TPCH_FK),
                gold=list(TPCH_FK), facts=TPCH_FACTS)


def load_tpcds():
    con = duckdb.connect(os.path.join(DATA, "tpcds.duckdb"), read_only=True)
    m = _meta(con)
    pk, fk = tpcds_gold(con)
    return dict(name="tpcds", con=con, tables=m, pk=pk, pk_enforced=set(), declared=list(fk), gold=list(fk),
                facts=DS_FACTS)


def bird_to_duckdb():
    os.makedirs(os.path.join(DATA, "bird"), exist_ok=True)
    meta = json.load(open(os.path.join(BIRD, "dev_tables.json")))
    for db in meta:
        p = os.path.join(DATA, "bird", db["db_id"] + ".duckdb")
        if os.path.exists(p):
            continue
        con = duckdb.connect(p)
        con.execute("INSTALL sqlite; LOAD sqlite; SET sqlite_all_varchar=true;")
        src = os.path.join(BIRD, "dev_databases", db["db_id"], db["db_id"] + ".sqlite")
        con.execute(f"ATTACH '{src}' AS s (TYPE sqlite, READ_ONLY)")
        for t in db["table_names_original"]:
            con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM s."{t}"')
        con.execute("DETACH s")
        con.close()


def load_bird(db_id):
    meta = {d["db_id"]: d for d in json.load(open(os.path.join(BIRD, "dev_tables.json")))}[db_id]
    con = duckdb.connect(os.path.join(DATA, "bird", db_id + ".duckdb"), read_only=True)
    m = _meta(con)
    tn, cn, ct = meta["table_names_original"], meta["column_names_original"], meta["column_types"]
    for i, (ti, c) in enumerate(cn):
        if ti >= 0:  # all columns are stored as varchar; keep BIRD's declared type for compatibility checks
            m[tn[ti]]["cols"][c] = {"number": "INTEGER", "integer": "INTEGER"}.get(ct[i], "VARCHAR")
    pk = {}
    for k in meta["primary_keys"]:
        ks = k if isinstance(k, list) else [k]
        pk[tn[cn[ks[0]][0]]] = [cn[j][1] for j in ks]
    fk = []
    for a, b in meta["foreign_keys"]:
        fk.append(E(tn[cn[a][0]], cn[a][1], tn[cn[b][0]], cn[b][1]))
    # facts: tables with outgoing declared FKs
    facts = sorted({e[0] for e in fk})
    # SQLite enforces declared PRIMARY KEY uniqueness on rowid tables, treated here as an enforced key
    return dict(name="bird_" + db_id, con=con, tables=m, pk=pk, pk_enforced=set(pk), declared=fk, gold=fk, facts=facts)


# JOB / IMDB (Leis et al., PVLDB 2015): real data with real defects; the dump ships no FK constraints, only the
# FK indexes of fkindexes.sql, which we take as the declared FKs; gold adds complete_cast -> comp_cast_type.
JOB_DB = "<DATA_DIR>/data/job2.duckdb"
JOB_DECL = [E("movie_companies", "company_id", "company_name", "id"), E("movie_companies", "company_type_id", "company_type", "id"),
            E("movie_info_idx", "info_type_id", "info_type", "id"), E("movie_info", "info_type_id", "info_type", "id"),
            E("person_info", "info_type_id", "info_type", "id"), E("movie_keyword", "keyword_id", "keyword", "id"),
            E("aka_title", "kind_id", "kind_type", "id"), E("title", "kind_id", "kind_type", "id"),
            E("movie_link", "linked_movie_id", "title", "id"), E("movie_link", "link_type_id", "link_type", "id"),
            E("aka_title", "movie_id", "title", "id"), E("cast_info", "movie_id", "title", "id"),
            E("complete_cast", "movie_id", "title", "id"), E("movie_companies", "movie_id", "title", "id"),
            E("movie_info_idx", "movie_id", "title", "id"), E("movie_keyword", "movie_id", "title", "id"),
            E("movie_link", "movie_id", "title", "id"), E("movie_info", "movie_id", "title", "id"),
            E("aka_name", "person_id", "name", "id"), E("cast_info", "person_id", "name", "id"),
            E("person_info", "person_id", "name", "id"), E("cast_info", "person_role_id", "char_name", "id"),
            E("cast_info", "role_id", "role_type", "id")]
JOB_FK = JOB_DECL + [E("complete_cast", "subject_id", "comp_cast_type", "id"), E("complete_cast", "status_id", "comp_cast_type", "id")]
JOB_FACTS = ["cast_info", "movie_info", "movie_info_idx", "movie_keyword", "movie_companies", "movie_link", "complete_cast",
             "person_info", "aka_name", "aka_title"]


def load_job():
    con = duckdb.connect(JOB_DB, read_only=True)
    m = _meta(con)
    return dict(name="job", con=con, tables=m, pk={t: ["id"] for t in m}, pk_enforced=set(), declared=list(JOB_DECL),
                gold=list(JOB_FK), facts=JOB_FACTS)


def bird_ids():
    return [d["db_id"] for d in json.load(open(os.path.join(BIRD, "dev_tables.json")))]


# ----------------------------------------------------------------------------- candidate generation
def tokens(col, table):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col).lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    tt = set(re.split(r"[^a-z0-9]+", table.lower()))
    abbrev = "".join(w[0] for w in re.split(r"[^a-z0-9]+", table.lower()) if w)
    if len(toks) > 1 and (len(toks[0]) <= 3 or toks[0] in tt or toks[0] == abbrev):
        toks = toks[1:]
    return toks


def kind(t):
    t = str(t)
    if INT_T.search(t): return "int"
    if STR_T.search(t): return "str"
    if NUM_T.search(t): return "num"
    return "other"


def keylike(inst, S, y):
    if inst["pk"].get(S) and y in inst["pk"][S]:
        return True
    return bool(set(tokens(y, S)) & KEYLIKE) or y.lower().endswith(("id", "key", "sk", "code", "no"))


def name_sim(inst, R, x, S, y):
    a, b = tokens(x, R), tokens(y, S)
    jac = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
    seq = difflib.SequenceMatcher(None, "_".join(a), "_".join(b)).ratio()
    s = max(jac, seq)
    # "customer_id" -> customer.id style
    st = set(re.split(r"[^a-z0-9]+", S.lower()))
    if set(b) <= KEYLIKE and (set(a) & st) and set(a) & KEYLIKE:
        s = max(s, 0.9)
    return s


def candidates(inst, sim_min=0.6, per_col=3):
    """Uniform candidate generator: DDL-declared FKs + name/type heuristics + partial keys of composite FKs."""
    T = inst["tables"]
    cand = {}
    for e in inst["declared"]:
        cand[e] = "ddl"
        if len(e[1]) > 1:  # each component of a composite FK is a classic fan-out trap
            for x, y in zip(e[1], e[3]):
                cand.setdefault(E(e[0], x, e[2], y), "partial")
    for R in T:
        for x, tx in T[R]["cols"].items():
            kx = kind(tx)
            if kx not in ("int", "str"):
                continue
            scored = []
            for S in T:
                if S == R:
                    continue
                for y, ty in T[S]["cols"].items():
                    if kind(ty) != kx or not keylike(inst, S, y):
                        continue
                    s = name_sim(inst, R, x, S, y)
                    if s >= sim_min:
                        scored.append((s, E(R, x, S, y)))
            scored.sort(key=lambda z: -z[0])
            for s, e in scored[:per_col]:
                cand.setdefault(e, "name")
    return cand


# ----------------------------------------------------------------------------- defect injection (dirty variants)
def make_dirty(inst, out_path, seed=0, dup_frac=(1e-4, 1e-3), orphan_frac=(1e-4, 1e-3), n_dim=3, n_fact=3):
    """Copy an instance and inject realistic defects: duplicated key rows in some referenced tables and orphan
    foreign-key values in some referencing tables. Returns (inst', log)."""
    rnd = random.Random(seed)
    if os.path.exists(out_path):
        os.remove(out_path)
    con = duckdb.connect(out_path)
    src = inst["con"]
    names = list(inst["tables"])
    # copy via parquet-free attach
    src_path = src.execute("SELECT path FROM duckdb_databases() WHERE database_name = current_database()").fetchone()[0]
    con.execute(f"ATTACH '{src_path}' AS src (READ_ONLY)")
    gold = inst["gold"]
    dims = sorted({e[2] for e in gold if len(e[3]) == 1 and inst["tables"][e[2]]["rows"] >= 50})
    facts = sorted({e[0] for e in gold if len(e[1]) == 1 and inst["tables"][e[0]]["rows"] >= 50})
    ddims = rnd.sample(dims, min(n_dim, len(dims)))
    dfacts = rnd.sample(facts, min(n_fact, len(facts)))
    log = []
    for t in names:
        con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM src."{t}"')
    for t in ddims:
        n = inst["tables"][t]["rows"]
        k = max(1, int(round(n * rnd.choice(dup_frac))))
        con.execute(f'INSERT INTO "{t}" SELECT * FROM src."{t}" USING SAMPLE {k} ROWS (reservoir, {seed})')
        log.append(("dup", t, k))
    for t in dfacts:
        es = [e for e in gold if e[0] == t and len(e[1]) == 1]
        e = rnd.choice(es)
        x = e[1][0]
        n = inst["tables"][t]["rows"]
        k = max(1, int(round(n * rnd.choice(orphan_frac))))
        tx = inst["tables"][t]["cols"][x]
        bad = "-987654321" if kind(tx) == "int" else "'__orphan__'"
        cols = ", ".join((f'{bad} AS "{c}"' if c == x else f'"{c}"') for c in inst["tables"][t]["cols"])
        con.execute(f'INSERT INTO "{t}" SELECT {cols} FROM src."{t}" USING SAMPLE {k} ROWS (reservoir, {seed})')
        log.append(("orphan", t, x, k))
    con.execute("DETACH src")
    con.close()
    con = duckdb.connect(out_path, read_only=True)
    d = dict(inst)
    d.update(name=inst["name"] + "_dirty", con=con, tables=_meta(con), inject=log)
    for t in d["tables"]:  # keep declared types (BIRD)
        d["tables"][t]["cols"] = inst["tables"][t]["cols"]
    d["pk_enforced"] = set()  # copies lose enforcement
    return d
