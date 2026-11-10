"""Minimal OpenAI-compatible client for the ICDE 2027 experiments, with an on-disk cache and cost log."""
import hashlib, json, os, time, threading
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ICDE2027
KEYS = json.load(open(os.environ.get("LLM_KEYS", "llm_keys.json")))  # your own API keys
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
# Endpoints verified on 2026-11-03: GLM plus-tier models return "balance insufficient", the second GLM key
# only reaches glm-4.5-air, and kimi-k2.6 accepts temperature 0.6 only.
MODELS = {"deepseek": ("deepseek", "deepseek-chat"), "ds-flash": ("deepseek", "deepseek-v4-flash"), "deepseek-r": ("deepseek", "deepseek-reasoner"),
          "glm": ("glm", "glm-4.5-flash"), "glm-air": ("glm2", "glm-4.5-air"), "kimi": ("kimi", "kimi-k2.6")}
FIXED = {"glm": {"thinking": {"type": "disabled"}}, "kimi": {"thinking": {"type": "disabled"}, "temperature": 0.6},
         "glm-air": {"thinking": {"type": "disabled"}}}
_lock = threading.Lock()

def chat(model, messages, temperature=0.0, max_tokens=512, retries=8, **kw):
    prov, name = MODELS.get(model, (model, model))
    for k, v in FIXED.get(model, {}).items():
        if k == "temperature": temperature = v
        else: kw.setdefault(k, v)
    body = {"model": kw.pop("model_name", name), "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, **kw}
    key = hashlib.sha256(json.dumps([prov, body], sort_keys=True).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key[:2], key + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    cfg = KEYS[prov]
    err = None
    for i in range(retries):
        try:
            r = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions",
                              headers={"Authorization": "Bearer " + cfg["api_key"]}, json=body, timeout=120)
            if r.status_code == 200:
                d = r.json()
                out = {"text": d["choices"][0]["message"]["content"], "usage": d.get("usage", {}), "model": body["model"]}
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with _lock:
                    json.dump(out, open(path, "w"))
                    with open(os.path.join(CACHE_DIR, "usage.log"), "a") as f:
                        f.write(json.dumps({"t": time.time(), "model": body["model"], **out["usage"]}) + "\n")
                return out
            err = f"{r.status_code} {r.text[:200]}"
        except Exception as e:
            err = repr(e)
        time.sleep(min(60, 2 ** i) + (5 if "429" in str(err) else 0))
    raise RuntimeError(err)
