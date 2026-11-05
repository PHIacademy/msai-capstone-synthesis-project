"""The tutor's tool layer: the deterministic functions of Capstone 6, now over the effective bank.

What is reused from Capstone 6 (Sections 3.3 to 3.6): the session state, serving a whole cluster
as one unit, single scoring, the answer-reveal gate (the key leaves this class only after an
attempt is recorded), and feedback grounded in the curriculum text.

What is new here:
- The source is the effective bank of the governed store (official questions, approved labels and
  admitted items). The tutor keeps no database connection; Tutor.from_store reads once, read-only.
- Admitted generated items are provisional. They are served only when no unseen past-paper unit is
  left for the requested topic, and every one is shown with PROVENANCE_LABEL.
- A request that nothing can serve returns status "exhausted" with alternatives; it never changes
  topic silently (the open gap of Capstone 6, Limitation 2).
- Mastery is kept per module and per sub-topic. A module-only label credits the module only.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data import IMAGE_SENTINEL, strip_paper_end
from src.tutor.session import SessionState

PROVENANCE_LABEL = "AI-drafted, teacher-approved, not an HKEAA question"   # same text as store.PROVENANCE_LABEL
LETTERS = ("A", "B", "C", "D")
MODULES = ("A", "B", "C", "D", "E")
IMAGE_ROOT = Path("data")
REQUIRED = ("item_id", "source", "question_text", "answer_options", "correct_answer", "tags",
            "primary_module", "label_level", "unit", "provisional", "cluster_desc",
            "qn_image_filename", "year")
FORBIDDEN = ("cand_id", "gen_key", "gate_verdict", "backend")      # candidate-side columns must never arrive
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)[ \t]*\n?")
_TAG = re.compile(r"\b([a-e]_(?:ip|cs|ia|cp|si)_\d{2})\b", re.I)


def _blank(v) -> bool:
    return v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)) or str(v).strip() == ""


class Tutor:
    def __init__(self, eb: pd.DataFrame, tax: pd.DataFrame | None = None, curriculum: dict | None = None,
                 label: str = PROVENANCE_LABEL):
        missing = [c for c in REQUIRED if c not in eb.columns]
        if missing:
            raise ValueError(f"effective bank lacks columns {missing}")
        leaked = [c for c in FORBIDDEN if c in eb.columns]
        if leaked:
            raise ValueError(f"effective bank carries candidate-side columns {leaked}; the tutor never sees candidates")
        if (eb["source"] == "generated").any() and not eb.loc[eb["source"] == "generated", "provisional"].astype(bool).all():
            raise ValueError("a generated item is not provisional; the tutor serves generated items only as provisional")
        self.eb = eb.reset_index(drop=True)
        self.by_id = self.eb.set_index("item_id", drop=False)
        self.tax = tax
        self.curriculum = curriculum or {}
        self.label = label
        self.tag_names = ({} if tax is None else dict(zip(tax["qn_type_id"], tax["subtopic_name"])))
        self.tag_module = ({} if tax is None else dict(zip(tax["qn_type_id"], tax["module"])))
        observed = sorted({t for ts in self.eb["tags"] for t in ts})
        self.all_tags = sorted(self.tag_names) if self.tag_names else observed
        self.units: dict = {}
        for order, (unit, g) in enumerate(self.eb.groupby("unit", sort=False)):
            years = [int(y) for y in g["year"] if not _blank(y)]
            self.units[unit] = {
                "order": order, "items": list(g["item_id"]),
                "provisional": bool(g["provisional"].astype(bool).all()),
                "year": min(years) if years else 9999,
                "tags": sorted({t for ts in g["tags"] for t in ts}),
                "modules": sorted({m for m in g["primary_module"] if not _blank(m)}),
            }
        self.state = SessionState()

    # ------------------------------------------------------------------ set-up
    @classmethod
    def from_store(cls, qdf: pd.DataFrame, db_path, tax: pd.DataFrame | None = None,
                   curriculum_path=None) -> "Tutor":
        """Read the effective bank once through the read-only connection, then close it.
        qdf is the derived bank (data.derive(data.prepare_bank(data.load_bank(...))))."""
        from src import store as S
        from src.tutor.curriculum import load_curriculum
        con = S.connect_ro(db_path)
        try:
            eb = S.build_effective_bank(qdf, con)
        finally:
            con.close()
        return cls(eb, tax, load_curriculum(curriculum_path))

    def start(self, session_id: str = "session") -> SessionState:
        self.state = SessionState(session_id=session_id)
        return self.state

    # ------------------------------------------------------------------ catalogue
    def coverage(self) -> pd.DataFrame:
        """Units per sub-topic: real (past paper) and provisional (admitted), over the whole bank."""
        rows = []
        for t in self.all_tags:
            us = [m for m in self.units.values() if t in m["tags"]]
            rows.append({"tag": t, "name": self.tag_names.get(t, ""),
                         "real_units": sum(not m["provisional"] for m in us),
                         "provisional_units": sum(m["provisional"] for m in us)})
        return pd.DataFrame(rows)

    def resolve_topic(self, text: str):
        """('tag', code) | ('module', letter) | None, from a tag code, 'module D' or a sub-topic name."""
        if not text:
            return None
        m = _TAG.search(text)
        if m and m.group(1).lower() in self.all_tags:
            return ("tag", m.group(1).lower())
        low = text.lower()
        for name, tag in sorted(((n.lower(), t) for t, n in self.tag_names.items()), key=lambda x: -len(x[0])):
            if name and name in low:
                return ("tag", tag)
        m = re.search(r"\bmodule\s+([a-e])\b", low)
        if m:
            return ("module", m.group(1).upper())
        return None

    # ------------------------------------------------------------------ selection
    def _unseen(self, kind: str, target: str, provisional: bool) -> list:
        out = []
        for u, m in self.units.items():
            if u in self.state.seen_units or m["provisional"] != provisional:
                continue
            if (kind == "tag" and target in m["tags"]) or (kind == "module" and target in m["modules"]):
                out.append(u)
        return sorted(out, key=lambda u: (self.units[u]["year"], self.units[u]["order"]))

    def mastery_summary(self) -> dict:
        rows = []
        for t in self.all_tags:
            m = self.state.mastery_tag.get(t, {"attempts": 0, "correct": 0})
            rows.append({"tag": t, "attempts": m["attempts"], "correct": m["correct"],
                         "accuracy": (m["correct"] / m["attempts"]) if m["attempts"] else None})
        rows.sort(key=lambda r: (r["accuracy"] is None, r["accuracy"] if r["accuracy"] is not None else 0, r["tag"]))
        by_module = [{"module": k, **v, "accuracy": v["correct"] / v["attempts"] if v["attempts"] else None}
                     for k, v in sorted(self.state.mastery_module.items())]
        return {"by_tag": rows, "by_module": by_module,
                "weak_tags": [r["tag"] for r in rows if r["accuracy"] is not None][:3],
                "untried_tags": [r["tag"] for r in rows if r["accuracy"] is None]}

    def _auto_target(self):
        """Weakest attempted sub-topic that still has something to serve, else the first untried one."""
        ms = self.mastery_summary()["by_tag"]
        for r in ms:
            if r["accuracy"] is not None and (self._unseen("tag", r["tag"], False) or self._unseen("tag", r["tag"], True)):
                return ("tag", r["tag"])
        for r in ms:
            if r["accuracy"] is None and (self._unseen("tag", r["tag"], False) or self._unseen("tag", r["tag"], True)):
                return ("tag", r["tag"])
        return None

    def next_question(self, topic=None) -> dict:
        """Serve one unit. topic: None, ('tag', code) or ('module', letter).

        status: 'served' | 'pending_open' | 'exhausted'. Real units come first; an admitted
        (provisional) unit is served only when no unseen real unit matches, and the reply says so."""
        s = self.state
        if s.pending_unit:
            return {"status": "pending_open", "unit": s.pending_unit, "item_ids": list(s.pending),
                    "text": self.render(s.pending_unit)}
        target = topic or self._auto_target()
        if target is None:
            return {"status": "exhausted", "target": None, "alternatives": [],
                    "message": "Every question in the bank has been served in this session."}
        kind, tgt = target
        real, prov = self._unseen(kind, tgt, False), self._unseen(kind, tgt, True)
        if real:
            unit, notice = real[0], None
        elif prov:
            unit = prov[0]
            notice = (f"No unseen past-paper question is left for {self._name(kind, tgt)}, so this one is an "
                      f"admitted item: {self.label}.")
        else:
            alts = []
            if kind == "tag":
                mod = self.tag_module.get(tgt) or tgt[0].upper()
                for t in self.all_tags:
                    if t != tgt and (self.tag_module.get(t) or t[0].upper()) == mod and self._unseen("tag", t, False):
                        alts.append({"tag": t, "name": self.tag_names.get(t, ""), "unseen_real_units": len(self._unseen("tag", t, False))})
            return {"status": "exhausted", "target": target, "alternatives": alts[:3],
                    "message": f"No unseen question is left for {self._name(kind, tgt)}."}
        m = self.units[unit]
        s.seen_units.add(unit)
        s.pending_unit, s.pending = unit, list(m["items"])
        s.served.append({"unit": unit, "source": "generated" if m["provisional"] else "hkeaa", "provisional": m["provisional"]})
        return {"status": "served", "unit": unit, "item_ids": list(m["items"]), "provisional": m["provisional"],
                "notice": notice, "text": self.render(unit)}

    def _name(self, kind, tgt):
        return f"{tgt} ({self.tag_names[tgt]})" if kind == "tag" and tgt in self.tag_names else (f"module {tgt}" if kind == "module" else tgt)

    # ------------------------------------------------------------------ display
    def render(self, unit: str) -> str:
        """The unit as the student sees it. Never contains the key. A provisional item carries the label."""
        m = self.units[unit]
        rows = [self.by_id.loc[i] for i in m["items"]]
        parts = []
        first = rows[0]
        if not _blank(first["cluster_desc"]):
            parts.append(_MD_IMAGE.sub("", str(first["cluster_desc"])).strip())
        for n, r in enumerate(rows, 1):
            block = []
            if len(rows) > 1:
                block.append(f"**Question {n} of {len(rows)}**")
            block.append(_MD_IMAGE.sub("", str(r["question_text"])).strip())
            if not _blank(r["qn_image_filename"]):
                for f in str(r["qn_image_filename"]).split(";"):
                    block.append(f"*(image: `{IMAGE_ROOT / f.strip()}`)*")
            block.extend(self._options_block(r))
            parts.append("\n\n".join(block))
        text = "\n\n---\n\n".join(parts)
        if m["provisional"]:
            text += f"\n\n> *{self.label}*"
        return text

    @staticmethod
    def _options_block(r) -> list:
        """The options as the student sees them. The printed paper-end banner (kept in the bank's text of the last
        question of each paper) is removed with the same function the generator and the gate use. Option pictures
        are shown too: when the options are only the picture the text says so, otherwise the picture follows the options."""
        opts = strip_paper_end(str(r["answer_options"])).strip()
        v = r.get("ans_image_filename")
        pics = [] if _blank(v) else [f.strip() for f in str(v).split(";") if f.strip()]
        parts = ["The options are in the picture below." if (pics and opts == IMAGE_SENTINEL) else opts]
        parts += [f"*(image: `{IMAGE_ROOT / f}`)*" for f in pics]
        return parts

    # ------------------------------------------------------------------ answering
    def answer(self, item_id: str, letter: str) -> dict:
        """Grade one pending item by letter (single scoring). Raises ValueError otherwise."""
        s = self.state
        letter = str(letter).strip().upper()
        if letter not in LETTERS:
            raise ValueError(f"{letter!r} is not an option letter (A-D)")
        if item_id not in s.pending:
            raise ValueError(f"{item_id} is not a pending question - nothing to score")
        row = self.by_id.loc[item_id]
        key = str(row["correct_answer"]).strip().upper()
        correct = letter == key
        credit = row["primary_module"] if not _blank(row["primary_module"]) else "general"
        mm = s._row(s.mastery_module, credit)
        mm["attempts"] += 1; mm["correct"] += int(correct)
        for t in row["tags"]:                              # a module-only label has no tags, so no tag credit
            tm = s._row(s.mastery_tag, t)
            tm["attempts"] += 1; tm["correct"] += int(correct)
        s.pending.remove(item_id)
        if not s.pending:
            s.pending_unit = None
        s.turn += 1
        rec = {"turn": s.turn, "item_id": item_id, "unit": row["unit"], "tags": list(row["tags"]),
               "module": credit, "chosen": letter, "correct_answer": key, "correct": correct,
               "provisional": bool(row["provisional"]), "label_level": row["label_level"]}
        s.history.append(rec)
        return rec

    def explain(self, item_id: str) -> dict:
        """Feedback for an item that already has a recorded attempt, with module-level curriculum text."""
        rec = next((h for h in reversed(self.state.history) if h["item_id"] == item_id), None)
        if rec is None:
            raise ValueError(f"No recorded attempt for {item_id} this session - answer it first.")
        mods = sorted({t[0].upper() for t in rec["tags"]} | ({rec["module"]} if rec["module"] in MODULES else set()))
        refs = [{"module": m, "grounded": m in self.curriculum, "text": self.curriculum.get(m)} for m in mods]
        if not refs:
            refs = [{"module": None, "grounded": False, "text": None}]
        return {**rec, "references": refs, "fully_grounded": all(r["grounded"] for r in refs),
                "note": (f"This is an admitted item ({self.label}); the teacher confirmed its key." if rec["provisional"] else None)}

    def progress(self) -> dict:
        h = self.state.history
        return {"attempts": len(h), "correct": sum(x["correct"] for x in h),
                "accuracy": (sum(x["correct"] for x in h) / len(h)) if h else None,
                "units_served": len(self.state.served),
                "provisional_units_served": sum(x["provisional"] for x in self.state.served),
                **self.mastery_summary()}
