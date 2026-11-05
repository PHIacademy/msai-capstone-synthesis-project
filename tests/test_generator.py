"""Tests for src/generator.py: prompts, parser, example selection, cache replay and run-unique ids.

No model is loaded and no API is called: the heavy imports are lazy, and the cache backend replays text."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src import data as D  # noqa: E402
from src import gate as G  # noqa: E402

heavy_before = {m for m in ("torch", "transformers", "peft", "litellm") if m in sys.modules}
from src import generator as GEN  # noqa: E402
heavy_loaded_by_import = {m for m in ("torch", "transformers", "peft", "litellm") if m in sys.modules} - heavy_before

tax = GEN.load_taxonomy(ROOT / "data/taxonomy.csv")
raw = D.load_bank(ROOT / "data/final_dataset.csv", tax)
df = D.derive(D.prepare_bank(raw))
CFG = GEN.load_configs(ROOT / "config/generators.json")
JSON_OK = json.dumps({"stem": "Which device connects networks and forwards packets between them?",
                      "options": {"A": "Router", "B": "Keyboard", "C": "Scanner", "D": "Monitor"},
                      "key": "A", "rationales": {"A": "r"}})


def test_importing_the_generator_loads_no_heavy_library():
    assert not heavy_loaded_by_import


def test_config_file_loads_and_names_the_backends():
    assert list(CFG) == ["tinyllama_lora_c5", "gemma4_e4b_prompted", "haiku_prompted", "offline_cache"]
    assert [c.kind for c in CFG.values()] == ["hf", "hf", "anthropic", "cache"]
    assert CFG["tinyllama_lora_c5"].prompt_style == "legacy_c5" and CFG["gemma4_e4b_prompted"].n_shots == 3
    assert not any("qwen" in c.model_id.lower() for c in CFG.values())


def test_bank_rules_come_from_data_py():
    assert GEN.MODULE_NAMES is D.OFFICIAL_MODULE_NAMES
    assert GEN.prepare_bank is D.prepare_bank and GEN.primary_module is D.primary_module
    src = (ROOT / "src/generator.py").read_text(encoding="utf-8")
    assert "def prepare_bank" not in src and "def primary_module" not in src


# ---------------------------------------------------------------- prompts
def test_legacy_c5_prompt_is_the_header_the_adapter_was_trained_on():
    p = GEN.build_prompt(GEN.Target("C", "c_ia_03"), GEN.Grounding(), CFG["tinyllama_lora_c5"])
    assert p == "Module: Internet and Its Applications\nType: c_ia_03\n\nQuestion:"
    with pytest.raises(ValueError, match="needs target.subtopic"):
        GEN.build_prompt(GEN.Target("C"), GEN.Grounding(), CFG["tinyllama_lora_c5"])


def test_structured_prompt_names_module_subtopic_and_examples():
    t = GEN.Target("D", "d_cp_03", "easy")
    ex = GEN.select_examples(raw, t, k=3, seed=1)
    msgs = GEN.build_prompt(t, GEN.Grounding("curriculum text", ex), CFG["gemma4_e4b_prompted"])
    assert [m["role"] for m in msgs] == ["system", "user"]
    u = msgs[1]["content"]
    assert "D (Computational Thinking and Programming)" in u and "d_cp_03 (Control Structures)" in u
    assert u.count("Example ") == 3 and u.count("Answer: ") == 3 and "curriculum text" in u


# ------------------------------------------------------- example selection
def test_examples_are_eligible_standalone_text_only_with_four_plain_options():
    ids = set(df[df.example_eligible].qn_id)
    for mod, sub in [("A", None), ("D", "d_cp_02"), ("D", "d_cp_03"), ("B", "b_cs_05"), ("E", "e_si_02")]:
        ex = GEN.select_examples(raw, GEN.Target(mod, sub), k=3, seed=0)
        assert len(ex) == 3
        for e in ex:
            assert e["qn_id"] in ids and e["cluster"] != 1
            assert len(D.parse_options(e["answer_options"])) == 4
            assert e["primary_module"] == mod or sub and sub in e["tags"]


def test_selection_is_reproducible_respects_max_year_and_widens_a_thin_subtopic():
    t = GEN.Target("D", "d_cp_02")
    a, b = GEN.select_examples(raw, t, seed=7), GEN.select_examples(raw, t, seed=7)
    assert [e["qn_id"] for e in a] == [e["qn_id"] for e in b]
    assert all(e["year"] <= 2020 for e in GEN.select_examples(raw, GEN.Target("A"), k=5, max_year=2020))
    thin = GEN.select_examples(raw, GEN.Target("E", "e_si_01"), k=3)        # no question carries e_si_01
    assert len(thin) == 3 and all(e["primary_module"] == "E" for e in thin)
    few = GEN.select_examples(raw, GEN.Target("D", "d_cp_02"), k=3)         # one usable real example exists: it comes first
    assert len(few) == 3 and "d_cp_02" in few[0]["tags"] and sum("d_cp_02" in e["tags"] for e in few) == 1
    assert len({e["qn_id"] for e in few}) == 3


def test_question_numbers_are_not_shown_in_examples():
    ex = GEN.select_examples(raw, GEN.Target("A"), k=3)
    assert all(re.match(r"\s*\*\*\d+\.\*\*", e["question_text"]) for e in ex)            # the bank prints them
    u = GEN.build_prompt(GEN.Target("A"), GEN.Grounding("", ex), CFG["gemma4_e4b_prompted"])[1]["content"]
    assert not re.search(r"\*\*\d+\.\*\*", u)


def test_the_paper_end_banner_never_reaches_a_prompt():
    banner = df[df.answer_options.astype(str).str.contains(r"\*\*END OF", regex=True)]
    assert len(banner) == 14
    for _, row in banner.iterrows():                                         # every banner question, forced in as an example
        msgs = GEN.build_prompt(GEN.Target("A"), GEN.Grounding("", [row.to_dict()]), CFG["gemma4_e4b_prompted"])
        assert "END OF" not in msgs[1]["content"]
    for sub in tax.qn_type_id:                                               # and no selected example carries it
        t = GEN.Target(sub[0].upper(), sub)
        ex = GEN.select_examples(raw, t, k=3)
        assert "END OF" not in GEN.build_prompt(t, GEN.Grounding("", ex), CFG["gemma4_e4b_prompted"])[1]["content"]


# ----------------------------------------------------------------- parser
T = GEN.Target("C", "c_ia_02")


def parse(text):
    return GEN.parse_candidate(text, "x", T)


def test_parser_reads_json_and_ignores_thinking_blocks():
    c = parse("<think>hmm {not json}</think>\n" + JSON_OK)
    assert c.valid_format and c.key == "A" and c.options["B"] == "Keyboard" and c.rationales == {"A": "r"}


@pytest.mark.parametrize("layout", ["A. Router\nB. Keyboard\nC. Scanner\nD. Monitor",
                                    "(A) Router\n(B) Keyboard\n(C) Scanner\n(D) Monitor",
                                    "| A. | Router |\n| B. | Keyboard |\n| C. | Scanner |\n| D. | Monitor |"])
def test_parser_text_fallback_accepts_the_option_layouts(layout):
    c = parse("Question: Which device forwards packets?\n" + layout + "\nAnswer: A")
    assert c.valid_format and c.stem == "Which device forwards packets?" and c.key == "A" and c.options["D"] == "Monitor"


def test_parser_rejects_bad_candidates_with_a_reason():
    assert not parse("Which device?\nA. a\nB. b\nC. c").valid_format                          # three options
    dup = parse("Which device?\nA. a\nB. b\nC. c\nD. d\nA. again")
    assert not dup.valid_format and "duplicate option letter" in dup.parse_error
    assert not parse(json.dumps({"stem": "s", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "key": "E"})).valid_format
    assert not parse('{"stem": "broken", "options": {').valid_format
    trailing = parse('{"stem": "s", "options": {"A": "a", "B": "b", "C": "c", "D": "d",}, "key": "A"}')     # seen from Gemma on Kaggle
    assert not trailing.valid_format and trailing.parse_error.startswith("bad json:")
    assert parse("Which device?\nA. a\nB. b\nC. c\nD. d\n\nModule: next sample\nA. x").valid_format   # runaway sample cut


# ------------------------------------------------------ cache and run ids
def cache_cfg(tmp_path, rows):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return GEN.GeneratorConfig(name="offline_cache", kind="cache", cache_path=str(p))


def test_cache_replays_only_matching_targets(tmp_path):
    cfg = cache_cfg(tmp_path, [{"module": "C", "subtopic": "c_ia_02", "raw": JSON_OK},
                               {"module": "C", "subtopic": "c_ia_03", "raw": "other"},
                               {"module": "D", "raw": "wrong module"}])
    gen = GEN.make_generator(cfg)
    assert [c.raw for c in gen.generate(GEN.Target("C", "c_ia_02"), GEN.Grounding())] == [JSON_OK]
    assert gen.generate(GEN.Target("E", "e_si_02"), GEN.Grounding()) == []
    assert len(gen.generate(GEN.Target("C", None, n=1), GEN.Grounding())) == 1


def test_candidate_ids_are_unique_across_runs_and_stable_within_one(tmp_path):
    cfg = cache_cfg(tmp_path, [{"module": "C", "raw": JSON_OK}, {"module": "C", "raw": JSON_OK}])
    a, b = GEN.make_generator(cfg), GEN.make_generator(cfg)
    ids_a = [c.cand_id for c in a.generate(GEN.Target("C"), GEN.Grounding())]
    ids_b = [c.cand_id for c in b.generate(GEN.Target("C"), GEN.Grounding())]
    assert len(set(ids_a + ids_b)) == 4                                       # two runs of one backend never collide
    assert all(re.fullmatch(r"GEN_offline_cache_[0-9a-f]{8}_000[12]", i) for i in ids_a + ids_b)
    fixed = GEN.make_generator(cfg, run_id="r1")
    assert [c.cand_id for c in fixed.generate(GEN.Target("C"), GEN.Grounding())] == ["GEN_offline_cache_r1_0001", "GEN_offline_cache_r1_0002"]


def test_the_anthropic_backend_calls_the_sdk_like_the_key_test(monkeypatch):
    """A stub stands in for the anthropic package: the base URL, key and message split are checked, no network is used."""
    calls = {}

    class Block:
        type, text = "text", JSON_OK

    class Messages:
        def create(self, **kw):
            calls["create"] = kw
            return type("R", (), {"content": [Block()]})()

    class Client:
        def __init__(self, **kw):
            calls["client"] = kw
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", type(sys)("anthropic"))
    sys.modules["anthropic"].Anthropic = Client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-test")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    gen = GEN.make_generator(CFG["haiku_prompted"], run_id="r")
    c = gen.generate(GEN.Target("C", "c_ia_02", n=2), GEN.Grounding("text", []))
    assert [x.cand_id for x in c] == ["GEN_haiku_prompted_r_0001", "GEN_haiku_prompted_r_0002"] and c[0].valid_format
    assert calls["client"] == {"base_url": "https://claude.vocareum.com", "api_key": "k-test"}
    kw = calls["create"]
    assert kw["model"] == "claude-haiku-4-5-20251001" and kw["messages"][0]["role"] == "user" and "JSON" in kw["system"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr(GEN, "load_env", lambda: None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        gen.generate(GEN.Target("C", "c_ia_02"), GEN.Grounding())


def stub_anthropic(monkeypatch, create):
    class Block:
        type, text = "text", JSON_OK

    class Messages:
        def __init__(self):
            self.create = create

    class Client:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", type(sys)("anthropic"))
    sys.modules["anthropic"].Anthropic = Client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-test")
    return Block


def test_temperature_is_applied_as_an_argument_when_the_sdk_takes_it(monkeypatch):
    seen = []
    block = stub_anthropic(monkeypatch, lambda **kw: seen.append(kw) or type("R", (), {"content": [block()]})())
    gen = GEN.make_generator(CFG["haiku_prompted"], run_id="r")
    gen.generate(GEN.Target("C", "c_ia_02", n=2), GEN.Grounding())
    assert [kw["temperature"] for kw in seen] == [0.7, 0.7]
    assert GEN.SAMPLING_LOG["haiku_prompted"] == {"temperature": 0.7, "applied_by": "argument", "max_tokens": 700}


def test_an_sdk_that_rejects_the_argument_gets_the_temperature_through_extra_body(monkeypatch):
    """Seen on Kaggle: Messages.create() got an unexpected keyword argument 'temperature' (anthropic 2.3.0)."""
    seen = []

    def create(*, model, max_tokens, messages, system=None, extra_body=None):
        seen.append(extra_body)
        return type("R", (), {"content": [block()]})()

    block = stub_anthropic(monkeypatch, create)
    gen = GEN.make_generator(CFG["haiku_prompted"], run_id="r")
    c = gen.generate(GEN.Target("C", "c_ia_02", n=3), GEN.Grounding())
    assert len(c) == 3 and all(x.valid_format for x in c) and seen == [{"temperature": 0.7}] * 3
    assert GEN.SAMPLING_LOG["haiku_prompted"] == {"temperature": 0.7, "applied_by": "extra_body", "max_tokens": 700}


def test_when_neither_way_works_the_api_default_is_used_and_recorded(monkeypatch):
    seen = []

    def create(*, model, max_tokens, messages, system=None):
        seen.append(model)
        return type("R", (), {"content": [block()]})()

    block = stub_anthropic(monkeypatch, create)
    gen = GEN.make_generator(CFG["haiku_prompted"], run_id="r")
    with pytest.warns(UserWarning, match="API default is used"):
        c = gen.generate(GEN.Target("C", "c_ia_02", n=3), GEN.Grounding())
    assert len(c) == 3 and len(seen) == 3
    assert GEN.SAMPLING_LOG["haiku_prompted"]["temperature"] == "API default"
    assert GEN.SAMPLING_LOG["haiku_prompted"]["applied_by"] == "api_default"


def test_every_backend_records_the_sampling_it_used(tmp_path):
    cfg = cache_cfg(tmp_path, [{"module": "C", "raw": JSON_OK}])
    GEN.make_generator(cfg).generate(GEN.Target("C"), GEN.Grounding())
    assert GEN.SAMPLING_LOG["offline_cache"] == {"replay": True}
    base = GEN.make_generator(GEN.GeneratorConfig(name="x", kind="cache", cache_path=cfg.cache_path))
    assert GEN.Generator.sampling_info(base) == {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 700, "seed": None}


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown generator kind"):
        GEN.make_generator(GEN.GeneratorConfig(name="x", kind="nope"))


def test_bake_off_writes_one_file_per_backend_and_does_not_duplicate_on_rerun(tmp_path):
    cfg = cache_cfg(tmp_path, [{"module": "C", "raw": JSON_OK}])
    for _ in range(2):
        out = GEN.bake_off({"offline_cache": cfg}, ["offline_cache"], [GEN.Target("C")], lambda t: GEN.Grounding(), tmp_path / "o")
    lines = (tmp_path / "o" / "offline_cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["cand_id"] == out["offline_cache"][0].cand_id


def test_a_cached_candidate_goes_through_the_gate_and_into_the_store(tmp_path):
    from src import store as S
    c = GEN.make_generator(cache_cfg(tmp_path, [{"module": "C", "raw": JSON_OK}])).generate(GEN.Target("C"), GEN.Grounding())[0]
    res = G.evaluate(c.stem, c.options, c.key, valid_format=c.valid_format, parse_error=c.parse_error)
    assert res["verdict"] in {"pass", "flag", "reject"} and set(res) == {"checks", "verdict", "reasons", "not_run"}
    con = S.init_db(tmp_path / "s.sqlite", tax)
    with con:
        con.execute("INSERT INTO candidate(cand_id,backend,target_module,stem,valid_format) VALUES (?,?,?,?,?)",
                    (c.cand_id, c.backend, "C", c.stem, int(c.valid_format)))
    G.write_gate_result(con, c.cand_id, res)
    assert con.execute("SELECT COUNT(*) FROM gate_result").fetchone()[0] == 1
