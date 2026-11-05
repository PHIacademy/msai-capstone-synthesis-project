"""Tests for scripts/bakeoff_runner.py with the cache backend: no GPU, no API, no Kaggle."""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

import bakeoff_runner as BR  # noqa: E402
from src import data as D  # noqa: E402
from src import generator as GEN  # noqa: E402

tax = GEN.load_taxonomy(ROOT / "data/taxonomy.csv")
raw = D.load_bank(ROOT / "data/final_dataset.csv", tax)
JSON_OK = json.dumps({"stem": "Which device forwards data packets between two separate networks in a school?",
                      "options": {"A": "Router", "B": "Keyboard", "C": "Scanner", "D": "Monitor"}, "key": "A"})
MD = """# Compulsory Part
## Module A: Information Processing
Data, files and spreadsheets.
## Module B: Computer System Fundamentals
Hardware and software.
### Storage
Disks.
## Module C: Internet and Its Applications
Networks.
"""


def test_the_six_targets_and_the_arithmetic_of_the_run():
    t = BR.build_targets(GEN)
    assert [(x.module, x.subtopic) for x in t] == BR.BAKEOFF_TARGETS and {x.n for x in t} == {3}
    assert len(t) * BR.N_PER_TARGET * len(BR.BACKENDS) == 54
    assert set(BR.BACKENDS) <= set(GEN.load_configs(ROOT / "config/generators.json"))
    assert all(sub[0].upper() == mod for mod, sub in BR.BAKEOFF_TARGETS)


def test_curriculum_text_cuts_one_module_and_says_nothing_found_as_empty():
    b = BR.curriculum_text(MD, "B")
    assert "Hardware and software." in b and "Disks." in b and "Networks." not in b and "Data, files" not in b
    assert BR.curriculum_text(MD, "D") == ""                                # no heading names Module D
    assert BR.curriculum_text("no headings, but Social Implications are here", "E").startswith("Social Implications")
    assert len(BR.curriculum_text(MD + "x" * 5000, "C", limit=100)) <= 100


OFFICIAL_LAYOUT = """The Compulsory Part comprises five modules:

A. Information Processing

Introduction

About data and files.

Learning Objectives

Students will learn about information.

B. Computer System Fundamentals

Introduction

About hardware.

C. Internet and its Applications

Introduction

About networks.

2.3.2 The Elective Part comprises three options.

A. Databases

Introduction

About SQL.
"""
CONTENTS_FIRST = "Module / Option\nA. Information Processing \n  a. Introduction to Information Processing \nB. Computer System Fundamentals \n  a. Basic Machine Organisation \n\n" + OFFICIAL_LAYOUT


def test_curriculum_text_reads_the_official_layout_and_skips_the_contents_table():
    a = BR.curriculum_text(CONTENTS_FIRST, "A")
    assert a.startswith("A. Information Processing") and "About data and files." in a and "About hardware" not in a
    c = BR.curriculum_text(CONTENTS_FIRST, "C")                                  # the title is written "its" in the guide
    assert "About networks." in c and "About SQL" not in c and "Elective" not in c
    assert BR.curriculum_text(CONTENTS_FIRST, "D") == ""


@pytest.mark.skipif(not (ROOT / "data/ict_curriculum.md").exists(), reason="curriculum guide not in data/")
def test_every_module_is_found_in_the_real_guide():
    md = (ROOT / "data/ict_curriculum.md").read_text(encoding="utf-8")
    got = {m: BR.curriculum_text(md, m, limit=10**6) for m in "ABCDE"}
    assert all(len(v) > 1500 for v in got.values())
    for m, v in got.items():
        assert v.startswith(f"{m}. ") and "Introduction" in v[:80]
        assert all(f"\n{o}. " not in v[:400] for o in "ABCDE" if o != m)
    assert "Technological Innovations" in got["E"] and "Elective" not in got["E"][-300:]


def test_grounding_is_the_same_for_every_backend_and_has_real_examples():
    g = BR.make_grounding_for(GEN, raw, MD)
    t = GEN.Target("B", "b_cs_05")
    a, b = g(t), g(t)
    assert a is b and len(a.examples) == 3 and "Hardware" in a.curriculum_text


def test_find_project_and_prepare_workdir(tmp_path):
    proj = BR.find_project(search_roots=(str(ROOT),))
    assert (proj / "src/generator.py").exists()
    work = BR.prepare_workdir(proj, tmp_path / "w", search_roots=(str(ROOT / "data"),))
    assert (work / "src/generator.py").exists() and (work / "data/final_dataset.csv").exists() and (work / "data/taxonomy.csv").exists()
    with pytest.raises(FileNotFoundError):
        BR.find_project(search_roots=(str(tmp_path / "nothing"),))


def test_environment_report_works_without_a_gpu_or_the_libraries():
    rep = BR.environment_report()
    assert rep["python"] and isinstance(rep["gpus"], list) and "transformers" in rep


def cache_configs(tmp_path):
    f = tmp_path / "cache.jsonl"
    f.write_text("\n".join(json.dumps({"module": m, "subtopic": s, "raw": JSON_OK}) for m, s in BR.BAKEOFF_TARGETS for _ in range(3)), encoding="utf-8")
    cfg = GEN.GeneratorConfig(name="offline_cache", kind="cache", cache_path=str(f))
    bad = GEN.GeneratorConfig(name="broken", kind="nope")
    return {"offline_cache": cfg, "broken": bad}


def test_smoke_test_reports_a_failing_backend_and_carries_on(tmp_path):
    cfgs = cache_configs(tmp_path)
    res = BR.smoke_test(GEN, cfgs, ["broken", "offline_cache"], BR.make_grounding_for(GEN, raw, MD), verbose=False)
    assert res["broken"]["ok"] is False and "unknown generator kind" in res["broken"]["error"]
    assert res["offline_cache"]["ok"] and res["offline_cache"]["valid_format"]
    assert res["offline_cache"]["sampling"] == {"replay": True}


def test_full_run_writes_one_file_per_backend_and_a_bundle(tmp_path):
    cfgs = cache_configs(tmp_path)
    out = tmp_path / "out"
    summary = BR.run_bakeoff(GEN, cfgs, ["broken", "offline_cache"], BR.make_grounding_for(GEN, raw, MD), out, run_id="r1")
    assert summary["offline_cache"] == {"candidates": 18, "valid_format": 18, "sampling": {"replay": True}} and "error" in summary["broken"]
    assert len((out / "offline_cache.jsonl").read_text(encoding="utf-8").splitlines()) == 18
    assert not (out / "broken.jsonl").exists()
    z = BR.bundle(out, tmp_path / "b.zip")
    assert sorted(zipfile.ZipFile(z).namelist()) == ["offline_cache.jsonl", "summary.json"]
