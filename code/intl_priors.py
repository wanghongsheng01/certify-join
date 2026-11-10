"""LLM priors of the international models (GPT-6.1 and Claude Opus 5.5 via the tokens1688 relay) for the public
benchmark pools only; production pools (ent*) are never sent through this channel. Prompts are rebuilt from the base
schema exactly as in backfill.py. Usage: TOKENS1688_API_KEY=... python code/intl_priors.py [model ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_prior as LP, policies as Pm
D = os.path.join(Pm.ROOT, "results", "pool")
models = sys.argv[1:] or ["gpt", "claude"]
for f in sorted(os.listdir(D)):
    if f.startswith("ent") or not f.endswith(".json"):
        continue
    rec = json.load(open(os.path.join(D, f)))
    base = rec["name"].split("_dirty")[0]
    brec = json.load(open(os.path.join(D, base + ".json"))) if base != rec["name"] else rec
    inst = {"tables": brec["tables"]}
    edges = [Pm.T(d["e"]) for d in rec["edges"]]
    for m in models:
        if rec["prior"].get(m) and sum(v is None for v in rec["prior"][m].values()) <= 0.02 * len(edges):
            continue
        new, usage = LP.priors(inst, edges, model=m)
        rec["prior"][m] = new
        rec.setdefault("usage", {})[m] = usage
        print(f, m, "edges", len(edges), "missing", sum(v is None for v in new.values()), flush=True)
    json.dump(rec, open(os.path.join(D, f), "w"))
