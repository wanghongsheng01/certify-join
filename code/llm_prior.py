"""LLM priors over candidate join edges, from schema metadata only (table/column names, types, row counts).

The same call serves two purposes: (a) the prior p(e) used to order probes in Certify-Join, and (b) the
"LLM-judged" baseline that certifies every edge with p(e) >= 0.5 without probing. No data values are sent.
"""
import json, re, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm
from instances import ename

PROMPT = """You are a data-warehouse modelling expert. Below is a relational schema (table: row count; columns with types)
and a numbered list of candidate join edges written as LEFT.cols -> RIGHT.cols (optionally with a validity-interval
predicate). An edge is VALID if (1) it is a semantically meaningful foreign-key style relationship and (2) it is
many-to-one: every LEFT row matches at most one RIGHT row (RIGHT.cols identify RIGHT rows, possibly only within the
stated validity interval) and LEFT values are expected to exist in RIGHT.
For each edge give your probability (0.00-1.00) that it is VALID. Answer ONLY with a JSON object mapping the edge
number (as a string) to the probability, e.g. {"1": 0.93, "2": 0.05}.

SCHEMA
{schema}

CANDIDATE EDGES
{edges}
"""


def schema_text(inst, tables=None):
    lines = []
    for t, m in inst["tables"].items():
        if tables and t not in tables:
            continue
        cols = ", ".join(f"{c} {str(ty).split('(')[0].lower()}" for c, ty in m["cols"].items())
        lines.append(f"{t}: {m['rows']} rows; {cols}")
    return "\n".join(lines)


def edge_text(e):
    R, X, S, Y, T = e
    s = f"{R}.({', '.join(X)}) -> {S}.({', '.join(Y)})"
    if T:
        s += f" WHERE {R}.{T[0]} BETWEEN {S}.{T[1]} AND {S}.{T[2]}"
    return s


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        return {str(k): float(v) for k, v in json.loads(m.group(0)).items()}
    except Exception:
        return {}


def priors(inst, edges, model="ds-flash", batch=None, schema_tables=None):
    # reasoning models spend output tokens on thinking: smaller batches and a larger cap avoid truncated JSON
    batch = batch or (25 if model == "ds-flash" else 60)
    max_tokens = 12000 if model == "ds-flash" else 4000
    edges = list(edges)
    jobs = []
    for b in range(0, len(edges), batch):
        chunk = edges[b:b + batch]
        tabs = schema_tables or sorted({t for e in chunk for t in (e[0], e[2])})
        et = "\n".join(f"{i + 1}. {edge_text(e)}" for i, e in enumerate(chunk))
        msg = PROMPT.replace("{schema}", schema_text(inst, tabs)).replace("{edges}", et)
        jobs.append((chunk, msg))

    def run(job):
        chunk, msg = job
        r = llm.chat(model, [{"role": "user", "content": msg}], max_tokens=max_tokens)
        d = parse(r["text"])
        return chunk, d, r.get("usage", {})

    out, usage = {}, []
    with ThreadPoolExecutor(1 if model.startswith("glm") else 4) as ex:
        for chunk, d, u in ex.map(run, jobs):
            usage.append(u)
            for i, e in enumerate(chunk):
                out[ename(e)] = d.get(str(i + 1))
    return out, usage
