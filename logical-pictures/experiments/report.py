"""Collect the numbers of all runs into results/ and print the tables used in the documents."""
import json
import os
import shutil
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])
R = "runs/"


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


os.makedirs("results", exist_ok=True)
for f in ("ablations.jsonl", "results.jsonl", "synthetic_pm.json", "synthetic_baselines.json",
          "analysed.json", "audit.json"):
    if os.path.exists(R + f):
        shutil.copy(R + f, "results/" + ("picture_machine.jsonl" if f == "results.jsonl" else f))

base = jl("baselines/results.jsonl")
extra = json.load(open("results/extra_baselines.json"))["rows"]
pm = jl(R + "results.jsonl")
abl = jl(R + "ablations.jsonl")


def b(model, n, mode):
    for r in base:
        if r["model"].startswith(model) and r["ntrain"] == n and r["mode"] == mode:
            return r["test_bpc"]
    for r in extra:
        if r["model"].startswith(model) and r["ntrain"] == n and r["mode"] == mode and r.get("slice", "A") == "A":
            return r["test_bpc"]


def p(n, mode, lo=95_000_000, tag=None):
    for r in pm:
        if r["ntrain"] == n and r["mode"] == mode and r.get("test_lo", 95_000_000) == lo and (tag is None or r["tag"] == tag):
            return r["test_bpc"]


def f(v):
    return "–" if v is None else f"{v:.3f}"


print("## headline (test slice A = text8[95.0M:95.5M])\n")
print("| Model | 100k | 1M | 10M |\n|---|---|---|---|")
rows = [("Kneser–Ney n-gram, order 8 (static)", [b("ngram", n, "static") for n in (100_000, 1_000_000, 10_000_000)]),
        ("PPM, best order (static)", [b("ppm", n, "static") for n in (100_000, 1_000_000, 10_000_000)]),
        ("PPM, best order (dynamic)", [b("ppm", n, "dynamic") for n in (100_000, 1_000_000, 10_000_000)]),
        ("Transformer, 2 layers, 254k weights, 10 / 25 CPU-minutes (static)", [None, b("transformer", 1_000_000, "static"), b("transformer", 10_000_000, "static")]),
        ("**Picture machine (static)**", [p(n, "static") for n in (100_000, 1_000_000, 10_000_000)]),
        ("**Picture machine (dynamic)**", [p(n, "dynamic") for n in (100_000, 1_000_000, 10_000_000)])]
for name, vals in rows:
    print(f"| {name} | " + " | ".join(f(v) for v in vals) + " |")

print("\n## slice B = text8[97.0M:97.5M]\n")
print("| Model | 1M | 10M |\n|---|---|---|")
for name, key, mode in (("Kneser–Ney n-gram, order 8 (static)", "ngram", "static"), ("PPM (static)", "ppm", "static"),
                        ("PPM (dynamic)", "ppm", "dynamic")):
    vals = [next((r["test_bpc"] for r in extra if r["model"].startswith(key) and r["ntrain"] == n and r["mode"] == mode
                  and r.get("slice") == "B"), None) for n in (1_000_000, 10_000_000)]
    print(f"| {name} | " + " | ".join(f(v) for v in vals) + " |")
for mode in ("static", "dynamic"):
    print(f"| **Picture machine ({mode})** | " + " | ".join(f(p(n, mode, 97_000_000)) for n in (1_000_000, 10_000_000)) + " |")

print("\n## ablations (records on 1M characters, test slice A)\n")
print("| Operations allowed | Sorts admitted | Static | Dynamic |\n|---|---|---|---|")
names = {"literal": "path only: `sign[-j]`", "classes": "+ classes of signs", "series": "+ series (`run<C>`)",
         "recency": "+ recency inside keys, sameness", "pointers": "+ pointer sorts",
         "all": "+ classes of objects, learned keys (everything)", "fine": "everything, margin 0.0003",
         "key1": "everything, keys of one determinable", "key2": "everything, keys of at most two"}
for tag, label in names.items():
    st = next((r for r in abl if r["tag"] == tag and r["mode"] == "static"), None)
    dy = next((r for r in abl if r["tag"] == tag and r["mode"] == "dynamic"), None)
    if st:
        print(f"| {label} | {len(st['sorts'])} | {st['test_bpc']:.3f} | {dy['test_bpc']:.3f} |")

print("\n## synthetic worlds: bits at the determined positions (bits per sign overall)\n")
syn = json.load(open(R + "synthetic_pm.json")) + (json.load(open(R + "synthetic_baselines.json")) if os.path.exists(R + "synthetic_baselines.json") else [])
worlds = []
for r in syn:
    if r["world"] not in worlds:
        worlds.append(r["world"])
models = []
for r in syn:
    k = (r["model"], r["mode"])
    if k not in models:
        models.append(k)
print("| Model | " + " | ".join(worlds) + " |\n|---|" + "---|" * len(worlds))
for m, mode in models:
    cells = []
    for w in worlds:
        r = next((r for r in syn if r["world"] == w and r["model"] == m and r["mode"] == mode), None)
        cells.append("–" if r is None else f"{r['bits_at_determined']:.3f} ({r['bits_per_sign']:.2f})")
    print(f"| {m}, {mode} | " + " | ".join(cells) + " |")
