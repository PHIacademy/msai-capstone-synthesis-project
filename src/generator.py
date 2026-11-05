"""Backend-agnostic candidate generator for the integrated HKDSE ICT system.

Place at src/generator.py. Every backend (a LoRA-tuned small model, a prompted
open model, an API model, or a replay cache) implements the same interface, so
a bake-off is a loop over config entries rather than new code.

Design rules
- One shared prompt builder, so models are compared on identical inputs.
- One shared parser, so "valid format" means the same thing for every backend.
- Heavy imports (transformers, peft, litellm, anthropic) are lazy: importing this module
  never needs a GPU or an API key.
- The 'cache' backend replays pre-generated JSONL, filtered by target, which is
  the offline mode for the Streamlit demo and the defence.
- This module only PRODUCES candidates. Acceptance is decided by gate.py and
  the teacher review in review.py. A model's answer key is never authoritative.
- The bank rules (which questions are text-only, which module is primary, how an
  options block is split and its paper-end banner removed) are defined once in
  src/data.py and imported here, never copied.

Bank columns used (schema v1.7, final_dataset.csv, 13 columns): qn_id, paper_year,
qn_type_id (semicolon-delimited, ascending, or 'unclassified'), question_text,
answer_options ('[IMAGE]' sentinel), correct_answer ('*' = withdrawn), cohort_pct,
qn_image_filename, qn_image_type, qn_image_context, ans_image_filename, cluster,
cluster_desc. There is no core_type / cat_* / subcat_* column: module and sub-topic
come from qn_type_id, and names come from data/taxonomy.csv.

Few-shot examples: select_examples keeps only real standalone text-only questions
(data.derive's example_eligible) whose options split into exactly four plain-text
choices (data.parse_options). The prompt shows the options in the layout produced by
data.format_options, so the printed paper-end banner ('**END OF PAPER**' or
'**END OF SECTION A**', inside option D of the last question of each paper) can never
reach a prompt.

Images: a body image of type exactly 'code' has been OCR-converted to a text block in
question_text, so data.prepare_bank treats it as text-only. Figure and table images were
NOT converted and stay excluded. Cluster members stay excluded (their code, if any, sits
in cluster_desc).

Credentials: the API backends read the key and the base URL from environment variables named in the
config (never from the config itself); a .env file is loaded if python-dotenv is installed.

Candidate ids: GEN_<backend>_<run>_<seq>. The run part is unique to each generator
object (a random 8-character id unless the caller passes one), so two runs of the same
backend never produce the same cand_id and can be stored together (candidate.cand_id is
the table's primary key).

NOT YET RUN against a model: no model has been loaded and no API call made.
"""
from __future__ import annotations

import json
import os
import re
import uuid
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from src import data as D

LETTERS = D.LETTERS

# Official module titles (the single table is in data.py; this name is kept for callers).
MODULE_NAMES = D.OFFICIAL_MODULE_NAMES
prepare_bank = D.prepare_bank          # kept so older callers of generator.prepare_bank still work
primary_module = D.primary_module      # same for generator.primary_module
SAMPLING_LOG: dict = {}      # backend name -> the sampling settings actually used in its last generate() call
_SUBTOPIC_NAMES: dict = {}   # filled by load_taxonomy(): {"d_cp_02": "Data Type", ...}


def load_taxonomy(path: str | Path = "data/taxonomy.csv"):
    """Read and check the static taxonomy (data.load_taxonomy) and remember the sub-topic
    names for the prompt. Returns the DataFrame."""
    t = D.load_taxonomy(path)
    _SUBTOPIC_NAMES.update(dict(zip(t["qn_type_id"], t["subtopic_name"])))
    return t


# ----------------------------------------------------------------- data types
@dataclass
class Target:
    module: str                       # e.g. "D" (curriculum module)
    subtopic: Optional[str] = None    # e.g. "d_cp_02"; None = whole module
    difficulty_band: str = "medium"   # easy | medium | hard (a prompt hint only)
    n: int = 5                        # candidates requested


@dataclass
class Grounding:
    curriculum_text: str = ""         # module text from ict_curriculum.md
    examples: list = field(default_factory=list)  # real few-shot items (dicts from select_examples)


@dataclass
class Candidate:
    stem: str
    options: dict                     # {"A": ..., "B": ..., "C": ..., "D": ...}
    key: Optional[str]                # generator's intended answer (NOT authoritative)
    rationales: dict                  # optional per-option rationale
    backend: str
    target: dict
    valid_format: bool
    parse_error: str = ""
    raw: str = ""
    provisional: bool = True          # always True until a teacher approves
    cand_id: str = ""                 # GEN_<backend>_<run>_<seq>; never collides with qn_id or another run


@dataclass
class GeneratorConfig:
    name: str
    kind: str                         # hf | anthropic | litellm | cache
    model_id: str = ""
    adapter_path: Optional[str] = None  # LoRA adapter dir (hf only)
    prompt_style: str = "structured"  # structured | legacy_c5
    legacy_template: str = ""         # C5 header; fields: module_name, qn_type (also module, subtopic, difficulty)
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 700
    n_shots: int = 3
    seed: Optional[int] = None        # torch seed set once at load (hf only)
    dtype: str = "float16"            # torch dtype name; T4 has no bf16
    auto_class: str = "AutoModelForCausalLM"  # or AutoModelForMultimodalLM if a model needs it
    chat_template_kwargs: dict = field(default_factory=dict)  # e.g. {"enable_thinking": false}
    pad_with_unk: bool = False        # C5 set pad token = unk
    load_in_4bit: bool = False        # needs bitsandbytes: outside coursework, off by default
    api_key_env: str = "ANTHROPIC_API_KEY"  # litellm: name of the env var holding the key
    api_base_env: Optional[str] = None  # litellm: name of the env var holding the base URL, if any
    api_base_default: Optional[str] = None  # litellm: used when that env var is unset (C6 used setdefault)
    cache_path: Optional[str] = None  # cache backend only
    notes: str = ""


def load_configs(path: str | Path) -> dict[str, GeneratorConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c["name"]: GeneratorConfig(**c) for c in data["generators"]}


# -------------------------------------------------------------------- grounding
def select_examples(bank, target: Target, k: int = 3, seed: int = 0, max_year: Optional[int] = None) -> list:
    """Pick k real standalone text-only items from the same module/sub-topic, nearest
    the requested difficulty band (cohort_pct: higher = easier). `bank` is the raw bank
    DataFrame (data.load_bank) or a derived one.

    The pool is data.derive's example_eligible (labelled, not withdrawn, not a cluster
    member, no figure or table image, no option image, no '[IMAGE]' options; code-only
    images count as text) restricted to items whose options split into exactly four
    choices (table-formatted options are left out). Years above max_year are dropped if
    given. With a sub-topic, any tag may match it; if fewer than k match, all of them are
    used and the rest are filled from the module (so a thin sub-topic gets every real
    example there is). Without a sub-topic, the question's primary module must match."""
    df = D.derive(D.prepare_bank(bank))
    df = df[df["example_eligible"] & df["answer_options"].apply(lambda o: len(D.parse_options(o)) == len(LETTERS))]
    if max_year is not None:
        df = df[df["year"] <= max_year]
    in_module = df[df["primary_module"] == target.module.upper()]
    goal = {"easy": 0.75, "medium": 0.55, "hard": 0.35}.get(target.difficulty_band, 0.55)

    def nearest(frame):
        """Rows ordered by closeness of cohort_pct to the band's goal (ties by qn_id, so the order is stable)."""
        pct = pd_num(frame["cohort_pct"])
        scale = 100.0 if pct.max() > 1.5 else 1.0
        return frame.assign(_d=(pct / scale - goal).abs().fillna(9.0)).sort_values(["_d", "qn_id"])

    def pick(frame, n):
        pool = nearest(frame).head(max(n * 3, n))
        return pool.sample(n=min(n, len(pool)), random_state=seed)

    if target.subtopic:
        sub = df[df["tags"].apply(lambda tags: target.subtopic in tags)]
        if len(sub) >= k:
            chosen = pick(sub, k)
        else:                                       # too few: every real example of the sub-topic, then fill from the module
            rest = in_module[~in_module["qn_id"].isin(sub["qn_id"])]
            chosen = pd_concat([sub, pick(rest, k - len(sub))]) if len(rest) else sub
    else:
        chosen = pick(in_module, k) if len(in_module) else in_module
    return chosen.drop(columns=["_d"], errors="ignore").head(k).to_dict("records")


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames)


def pd_num(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


# ----------------------------------------------------------------------- prompt
_SCHEMA = (
    '{"stem": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
    '"key": "A|B|C|D", "rationales": {"A": "...", "B": "...", "C": "...", "D": "..."}}'
)


_QN_NUMBER = re.compile(r"^\s*\*\*\d+\.\*\*[ \t]*")   # the printed question number, e.g. '**31.** '


def format_example(ex: dict, number: int) -> str:
    """One few-shot example as it appears in a prompt. The options go through
    data.parse_options and data.format_options, so the paper-end banner is removed and the
    layout is the same for every example. The printed question number is dropped: C5 found
    that a model shown numbers learns to write them."""
    opts = D.parse_options(ex.get("answer_options", ""))
    stem = _QN_NUMBER.sub("", str(ex.get("question_text", ""))).strip()
    return (f"Example {number}:\n{stem}\n{D.format_options(opts)}\n"
            f"Answer: {ex.get('correct_answer', '')}")


def build_prompt(target: Target, g: Grounding, cfg: GeneratorConfig):
    """Return a chat message list (structured) or a plain string (legacy_c5)."""
    if cfg.prompt_style == "legacy_c5":
        if not target.subtopic:
            raise ValueError("legacy_c5 prompt needs target.subtopic (the C5 question type id)")
        # Module text is the official curriculum title; qn_type is the tag passed to C5's module_name(qn_type).
        return cfg.legacy_template.format(
            module_name=MODULE_NAMES.get(target.module.upper(), target.module),
            qn_type=target.subtopic,
            module=target.module, subtopic=target.subtopic,
            difficulty=target.difficulty_band)
    shots = "\n\n".join(format_example(ex, i + 1) for i, ex in enumerate(g.examples[: cfg.n_shots]))
    system = ("You write HKDSE ICT Paper 1A multiple-choice questions. Each question has "
              "exactly four options, exactly one defensible correct answer, and plausible, "
              "mutually distinct distractors. Text only: never refer to a figure or image. "
              "Do not copy the examples. Reply with a single JSON object and nothing else.")
    mod = f"{target.module} ({MODULE_NAMES.get(target.module.upper(), '')})"
    sub = (f"{target.subtopic} ({_SUBTOPIC_NAMES[target.subtopic]})"
           if target.subtopic in _SUBTOPIC_NAMES else target.subtopic)
    user = (f"Curriculum module: {mod}"
            + (f", sub-topic: {sub}" if target.subtopic else "")
            + f"\nTarget difficulty: {target.difficulty_band}\n\n"
            f"Curriculum reference:\n{g.curriculum_text[:3000]}\n\n"
            f"Real past-paper examples (style only):\n{shots}\n\n"
            f"Write ONE new question. JSON schema:\n{_SCHEMA}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------------------------------------------------- parser
# Accepts "A. text", "(A) text", "A: text" and markdown-table rows "| A. | text |".
_OPT = re.compile(r"^[ \t]*\|?[ \t]*\(?([ABCD])[\).:][ \t]*\|?[ \t]*(.+?)[ \t]*\|?[ \t]*$", re.M)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_candidate(raw: str, backend: str, target: Target) -> Candidate:
    """Shared parser. JSON first; fall back to 'stem + A-D lines' (Capstone 5 style).

    Written from the handoff description of C5's parse_mcq (tables, cut at the next
    '\\nModule:'). It has not been compared line by line with C5's function; swap
    that in if exact reuse matters for the defence."""
    tgt = asdict(target)
    text = _THINK.sub("", raw)
    text = re.split(r"\n[ \t]*Module:", text, maxsplit=1)[0]   # drop a runaway next sample
    stem, options, key, rats, err, json_err = "", {}, None, {}, "", ""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            stem = str(obj.get("stem", "")).strip()
            options = {str(k).upper(): str(v).strip() for k, v in obj.get("options", {}).items()}
            key = str(obj.get("key", "")).strip().upper() or None
            rats = obj.get("rationales", {}) or {}
        except json.JSONDecodeError as e:
            err, json_err = "bad json", f"bad json: {e.msg} at line {e.lineno}"
        except AttributeError:
            err, json_err = "bad json", "bad json: unexpected structure"
    if not options:
        err = ""                                               # JSON failed or absent: judge the text fallback on its own
        dup = []
        for letter, body in _OPT.findall(text):
            if letter in options:
                dup.append(letter)                             # never overwrite silently
            else:
                options[letter] = body.strip()
        first = _OPT.search(text)
        stem = text[: first.start()].strip() if first else text.strip()
        stem = re.sub(r"^\s*Question:\s*", "", stem)
        km = re.search(r"answer\s*[:\-]\s*\(?([ABCD])\b", text, re.I)
        key = km.group(1).upper() if km else None
        if dup:
            err = "duplicate option letter(s): " + ",".join(sorted(set(dup)))
    ok = (not err and bool(stem) and set(options) == set(LETTERS)
          and all(options[l] for l in LETTERS)
          and (key is None or key in LETTERS))
    if not ok and not err:
        err = json_err or "missing stem/options or invalid key"
    return Candidate(stem, options, key, rats, backend, tgt, ok, err, raw)


# --------------------------------------------------------------------- backends
class Generator(ABC):
    def __init__(self, cfg: GeneratorConfig, run_id: Optional[str] = None):
        self.cfg = cfg
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._seq = 0

    def close(self) -> None:
        """Free what the backend holds (model weights, GPU memory). A no-op for API and cache backends."""

    def sampling_info(self) -> dict:
        """The sampling settings this backend actually used, saved with the results (SAMPLING_LOG)."""
        c = self.cfg
        return {"temperature": c.temperature, "top_p": c.top_p, "max_new_tokens": c.max_new_tokens, "seed": c.seed}

    @abstractmethod
    def _complete(self, prompt, n: int, target: Target) -> list[str]:
        """Return up to n raw completions for the prompt (target is for backends that filter)."""

    def generate(self, target: Target, grounding: Grounding) -> list[Candidate]:
        prompt = build_prompt(target, grounding, self.cfg)
        raws = self._complete(prompt, target.n, target)
        SAMPLING_LOG[self.cfg.name] = self.sampling_info()
        cands = []
        for r in raws:
            c = parse_candidate(r, self.cfg.name, target)
            self._seq += 1
            c.cand_id = f"GEN_{self.cfg.name}_{self.run_id}_{self._seq:04d}"
            cands.append(c)
        return cands


class HFGenerator(Generator):
    """Local Transformers model, optionally with a PEFT/LoRA adapter (Capstone 5 stack)."""

    def __init__(self, cfg, run_id=None):
        super().__init__(cfg, run_id)
        import torch
        import transformers
        from transformers import AutoTokenizer
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        kw = {"device_map": "auto"}
        dtype = getattr(torch, cfg.dtype)
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        if cfg.pad_with_unk and self.tok.pad_token is None:
            self.tok.pad_token = self.tok.unk_token             # as in Capstone 5
        model_cls = getattr(transformers, cfg.auto_class)
        try:                                                       # newer transformers name the argument dtype ...
            self.model = model_cls.from_pretrained(cfg.model_id, dtype=dtype, **kw)
        except TypeError:                                          # ... older ones torch_dtype (as in Capstone 5)
            self.model = model_cls.from_pretrained(cfg.model_id, torch_dtype=dtype, **kw)
        if cfg.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)
        self.model.eval()

    def close(self) -> None:
        import gc
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _complete(self, prompt, n, target):
        import torch
        if isinstance(prompt, list) and self.tok.chat_template:
            text = self.tok.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, **self.cfg.chat_template_kwargs)
        else:
            text = prompt if isinstance(prompt, str) else "\n\n".join(m["content"] for m in prompt)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id  # id 0 is valid
        outs = []
        for _ in range(n):
            with torch.no_grad():
                ids = self.model.generate(
                    **enc, do_sample=True, temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p, max_new_tokens=self.cfg.max_new_tokens,
                    pad_token_id=pad)
            outs.append(self.tok.decode(ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs


def load_env() -> None:
    """Read a .env file if python-dotenv is installed (as in the Capstone 6 key test); otherwise do nothing."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


class AnthropicGenerator(Generator):
    """Claude through the Anthropic SDK, with the base URL and key arrangement of the key test:
    anthropic.Anthropic(base_url=..., api_key=os.environ[...]) and client.messages.create(...).

    The configured temperature is applied in the best way the installed SDK and the API allow, and the way used is
    recorded (sampling_info, SAMPLING_LOG, summary.json): first as the temperature argument; if this SDK version does
    not accept it (seen with anthropic 2.3.0), through extra_body; if the API refuses that too, the API default."""

    _temp_mode = None            # None (not tried yet) | "argument" | "extra_body" | "api_default"

    def sampling_info(self) -> dict:
        mode = self._temp_mode or "argument"
        return {"temperature": "API default" if mode == "api_default" else self.cfg.temperature,
                "applied_by": mode, "max_tokens": self.cfg.max_new_tokens}

    def _create(self, client, base_args):
        t = self.cfg.temperature
        if self._temp_mode in (None, "argument"):
            try:
                r = client.messages.create(temperature=t, **base_args)
                self._temp_mode = "argument"
                return r
            except TypeError as e:                          # this SDK version does not take the argument
                if "temperature" not in str(e):
                    raise
                self._temp_mode = "extra_body"
        if self._temp_mode == "extra_body":
            try:
                return client.messages.create(extra_body={"temperature": t}, **base_args)
            except Exception as e:                          # noqa: BLE001 (the API or the proxy refused it)
                self._temp_mode = "api_default"
                warnings.warn(f"temperature could not be applied ({type(e).__name__}); the API default is used")
        return client.messages.create(**base_args)


    def _complete(self, prompt, n, target):
        import anthropic
        load_env()
        key = os.getenv(self.cfg.api_key_env) if self.cfg.api_key_env else None
        if not key:
            raise RuntimeError(f"environment variable {self.cfg.api_key_env} is not set")
        base = (os.getenv(self.cfg.api_base_env) if self.cfg.api_base_env else None) or self.cfg.api_base_default
        client = anthropic.Anthropic(base_url=base, api_key=key) if base else anthropic.Anthropic(api_key=key)
        if isinstance(prompt, str):
            system, msgs = "", [{"role": "user", "content": prompt}]
        else:
            system = "\n\n".join(m["content"] for m in prompt if m["role"] == "system")
            msgs = [m for m in prompt if m["role"] != "system"]
        extra = {"system": system} if system else {}
        base_args = dict(model=self.cfg.model_id, max_tokens=self.cfg.max_new_tokens, messages=msgs, **extra)
        outs = []
        for _ in range(n):
            r = self._create(client, base_args)
            outs.append("".join(b.text for b in r.content if getattr(b, "type", "") == "text"))
        return outs


class LiteLLMGenerator(Generator):
    """API model through LiteLLM (as in Capstone 6). Credentials come from the environment."""

    def _complete(self, prompt, n, target):
        import litellm
        msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        key = os.getenv(self.cfg.api_key_env) if self.cfg.api_key_env else None
        base = (os.getenv(self.cfg.api_base_env) if self.cfg.api_base_env else None) or self.cfg.api_base_default
        outs = []
        for _ in range(n):
            r = litellm.completion(
                model=self.cfg.model_id, messages=msgs,
                temperature=self.cfg.temperature, max_tokens=self.cfg.max_new_tokens,
                api_key=key, api_base=base)
            outs.append(r.choices[0].message.content)
        return outs


class CacheGenerator(Generator):
    """Offline replay of pre-generated completions, filtered by target.

    JSONL, one row per completion: {"module": "D", "raw": "...", optional "subtopic",
    optional "difficulty_band"}. Returns an empty list when nothing matches, so a
    demo never shows an item for the wrong target."""

    def __init__(self, cfg, run_id=None):
        super().__init__(cfg, run_id)
        lines = Path(cfg.cache_path).read_text("utf-8").splitlines()
        self._rows = [json.loads(l) for l in lines if l.strip()]

    @staticmethod
    def _matches(row, t: Target) -> bool:
        if str(row.get("module", "")).upper() != t.module.upper():
            return False
        for key, want in (("subtopic", t.subtopic), ("difficulty_band", t.difficulty_band)):
            if row.get(key) and want and str(row[key]).lower() != str(want).lower():
                return False
        return True

    def sampling_info(self) -> dict:
        return {"replay": True}

    def _complete(self, prompt, n, target):
        return [r["raw"] for r in self._rows if self._matches(r, target)][:n]


_BACKENDS = {"hf": HFGenerator, "anthropic": AnthropicGenerator, "litellm": LiteLLMGenerator, "cache": CacheGenerator}


def make_generator(cfg: GeneratorConfig, run_id: Optional[str] = None) -> Generator:
    if cfg.kind not in _BACKENDS:
        raise ValueError(f"unknown generator kind: {cfg.kind}")
    return _BACKENDS[cfg.kind](cfg, run_id)


# ---------------------------------------------------------------------- bake-off
def save_candidates(cands: list[Candidate], path: str | Path, append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def bake_off(configs: dict[str, GeneratorConfig], names: list[str],
             targets: list[Target], grounding_for, out_dir: str | Path,
             run_id: Optional[str] = None) -> dict:
    """Run the same targets through each named backend; save raw candidates per model.
    grounding_for(target) -> Grounding. Each run overwrites that model's file, so a
    re-run never duplicates candidates. All backends of one call share run_id (random if
    not given). Returns {name: [Candidate, ...]}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or uuid.uuid4().hex[:8]
    results = {}
    for name in names:
        gen = make_generator(configs[name], run_id)
        cands = []
        for t in targets:
            cands.extend(gen.generate(t, grounding_for(t)))
        gen.close()                                   # free the weights before the next backend loads
        save_candidates(cands, out_dir / f"{name}.jsonl")
        results[name] = cands
    return results
