"""Runner for the generator bake-off (Section 3.4, Section 4.6): the helpers a Kaggle notebook calls.

Place at scripts/bakeoff_runner.py. The notebook kaggle_bakeoff.ipynb imports it, so the logic can be tested on
a laptop with the cache backend and no GPU. It does four things:
  1. find the project files (a Kaggle dataset, or the current folder) and copy them next to the data;
  2. build the six fixed bake-off targets and their grounding (curriculum text and few-shot examples);
  3. smoke-test each backend on ONE target, so a loading problem costs minutes, not the whole run;
  4. run the full bake-off, one backend at a time (one failure never loses the others), and bundle the results.

Nothing here trains a model. The bake-off is inference only.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import sys
import zipfile
from pathlib import Path

BAKEOFF_TARGETS = [("D", "d_cp_02"), ("B", "b_cs_05"), ("D", "d_cp_05"), ("E", "e_si_02"), ("A", "a_ip_01"), ("C", "c_ia_06")]
N_PER_TARGET = 3                       # candidates per target and backend: 6 x 3 = 18 per backend, 54 in all
BACKENDS = ["tinyllama_lora_c5", "gemma4_e4b_prompted", "haiku_prompted"]
MODULE_NAMES = {"A": "Information Processing", "B": "Computer System Fundamentals", "C": "Internet and Its Applications",
                "D": "Computational Thinking and Programming", "E": "Social Implications"}


# ---------------------------------------------------------------- project files
def find_project(search_roots=("/kaggle/input", "."), marker="src/generator.py") -> Path:
    """The folder that holds src/generator.py and config/generators.json (a Kaggle dataset or the current folder)."""
    for root in map(Path, search_roots):
        if not root.exists():
            continue
        for hit in sorted(root.rglob(marker)):
            proj = hit.parent.parent
            if (proj / "config" / "generators.json").exists():
                return proj
    raise FileNotFoundError(f"no folder with {marker} and config/generators.json under {list(search_roots)}")


def find_file(name: str, search_roots=("/kaggle/input", ".")) -> Path:
    for root in map(Path, search_roots):
        if root.exists():
            for hit in sorted(root.rglob(name)):
                return hit
    raise FileNotFoundError(f"{name} not found under {list(search_roots)}")


def prepare_workdir(project: Path, work: Path, data_files=("final_dataset.csv", "taxonomy.csv", "ict_curriculum.md"),
                    search_roots=("/kaggle/input", ".")) -> Path:
    """Copy src/, config/, scripts/ and models/ to the working folder and the data files to work/data (as Capstone 5
    did on Kaggle), then put the folder on sys.path. Returns the working folder."""
    work = Path(work)
    for sub in ("src", "config", "scripts", "models"):
        if (project / sub).exists():
            shutil.copytree(project / sub, work / sub, dirs_exist_ok=True)
    (work / "data").mkdir(parents=True, exist_ok=True)
    for name in data_files:
        try:
            shutil.copy(find_file(name, search_roots), work / "data" / name)
        except FileNotFoundError:
            if name != "ict_curriculum.md":
                raise
            print("[warning] ict_curriculum.md not found: prompts will carry no curriculum text")
    sys.path.insert(0, str(work))
    return work


def environment_report() -> dict:
    """Library versions and GPUs, saved with the results so the paper can cite them."""
    rep = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "accelerate", "anthropic", "pandas"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception:                                      # noqa: BLE001 (not installed)
            rep[mod] = None
    try:
        import torch
        rep["gpus"] = [{"name": torch.cuda.get_device_name(i),
                        "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                       for i in range(torch.cuda.device_count())]
    except Exception:                                          # noqa: BLE001
        rep["gpus"] = []
    return rep


# ----------------------------------------------------------------- grounding
def curriculum_text(md: str, module: str, limit: int = 3000) -> str:
    """The part of ict_curriculum.md about one module, for the prompt's "Curriculum reference".

    In the official guide each module's section starts with a line such as "A. Information Processing" followed by
    "Introduction"; the section ends where the next module or option starts, or at "2.3.2 The Elective Part". The
    table of contents and the module list carry the same title lines but are not followed by "Introduction", so they
    are skipped. If that layout is not found, a markdown heading that names the module's official title is used,
    then the text after the first mention of the title. Returns '' if the title does not occur anywhere."""
    title = MODULE_NAMES[module.upper()]
    lines = md.splitlines()
    line_re = re.compile(r"^\s*([A-E])\.\s+(.+?)\s*$")

    def next_nonblank(i):
        return next((x.strip() for x in lines[i + 1:i + 5] if x.strip()), "")

    starts = [(i, m.group(1), m.group(2)) for i, l in enumerate(lines)
              if (m := line_re.match(l)) and next_nonblank(i) == "Introduction"]
    mine = next((i for i, letter, t in starts if letter == module.upper() and t.lower() == title.lower()), None)
    if mine is not None:
        stop = next((j for j in range(mine + 1, len(lines))
                     if any(j == i for i, _, _ in starts) or re.match(r"^\s*2\.3\.2\b", lines[j])), len(lines))
        return "\n".join(lines[mine:stop]).strip()[:limit]
    others = [t for k, t in MODULE_NAMES.items() if k != module.upper()]
    head = re.compile(r"^\s{0,3}#{1,6}\s")
    start = next((i for i, l in enumerate(lines) if head.match(l) and title.lower() in l.lower()), None)
    if start is None:
        pos = md.lower().find(title.lower())
        return md[pos:pos + limit] if pos >= 0 else ""
    end = next((j for j in range(start + 1, len(lines))
                if head.match(lines[j]) and any(o.lower() in lines[j].lower() for o in others)), len(lines))
    return "\n".join(lines[start:end])[:limit]


def build_targets(GEN) -> list:
    return [GEN.Target(module, sub, "medium", N_PER_TARGET) for module, sub in BAKEOFF_TARGETS]


def make_grounding_for(GEN, raw_bank, curriculum_md: str, k: int = 3, seed: int = 0):
    """grounding_for(target) for bake_off: the module's curriculum text and k real examples (the same for every backend)."""
    cache = {}

    def grounding_for(t):
        key = (t.module, t.subtopic)
        if key not in cache:
            cache[key] = GEN.Grounding(curriculum_text(curriculum_md, t.module), GEN.select_examples(raw_bank, t, k=k, seed=seed))
        return cache[key]
    return grounding_for


# ---------------------------------------------------------------- the runs
def smoke_test(GEN, configs: dict, names: list, grounding_for, target=None, verbose=True) -> dict:
    """Load each backend, write ONE candidate for one target, show it, free the memory. A failure is reported, not raised."""
    target = target or GEN.Target("D", "d_cp_02", "medium", 1)
    out = {}
    for name in names:
        try:
            gen = GEN.make_generator(configs[name])
            cands = gen.generate(GEN.Target(target.module, target.subtopic, target.difficulty_band, 1), grounding_for(target))
            gen.close()
            c = cands[0] if cands else None
            out[name] = {"ok": c is not None, "valid_format": bool(c and c.valid_format), "parse_error": c.parse_error if c else "no output",
                         "raw": (c.raw[:600] if c else ""), "sampling": GEN.SAMPLING_LOG.get(name)}
            if verbose:
                print(f"--- {name}: ok={out[name]['ok']} valid_format={out[name]['valid_format']} {out[name]['parse_error']}"
                      f"\n    sampling used: {out[name]['sampling']}\n{out[name]['raw']}\n")
        except Exception as e:                                 # noqa: BLE001 (report and carry on with the next backend)
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            if verbose:
                print(f"--- {name}: FAILED {out[name]['error']}\n")
    return out


def run_bakeoff(GEN, configs: dict, names: list, grounding_for, out_dir, run_id=None) -> dict:
    """The full run, one backend at a time. Writes <out_dir>/<name>.jsonl per backend and returns {name: n candidates or error}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        try:
            res = GEN.bake_off(configs, [name], build_targets(GEN), grounding_for, out_dir, run_id=run_id)
            cands = res[name]
            summary[name] = {"candidates": len(cands), "valid_format": sum(c.valid_format for c in cands),
                             "sampling": GEN.SAMPLING_LOG.get(name)}
        except Exception as e:                                 # noqa: BLE001
            summary[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(name, summary[name])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def bundle(out_dir, zip_path) -> Path:
    """Zip the JSONL files, the summary and the environment report: the small files that come back to the project."""
    out_dir, zip_path = Path(out_dir), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.glob("*.jsonl")) + sorted(out_dir.glob("*.json")):
            z.write(f, f.name)
    return zip_path
