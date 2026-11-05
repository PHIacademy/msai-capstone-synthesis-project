"""Student-side chat page for the tutor (Sections 4.12 and 4.13 of the notebook).

Run from the project root:  streamlit run app/streamlit_app.py

The page contains no logic of its own. It builds a Tutor from the governed store through the read-only
connection, passes every typed message to the router chosen in the sidebar (RuleRouter, or the LLM Orchestrator
when a key is available) and shows the reply text. The sidebar buttons always use the rules: the page writes
those messages itself, so nothing needs a model and nothing is sent out. It never shows the stored
`provenance` of an admitted item (that field names the generating backend); an admitted item carries the fixed
label that the tutor adds when it renders it. Nothing is stored beyond the browser session.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                    # the bank, the store and the question images are found relative to the root

from src.tutor import Orchestrator, RuleRouter, Tutor, llm_available    # noqa: E402

DB_PATH = ROOT / "data" / "state.sqlite"
WELCOME = ("Hello. I serve HKDSE ICT Paper 1A practice questions. Say *next question*, name a topic "
           "(for example *module D* or *d_cp_02*), answer with a letter, and ask *why?* after an answer.")
IMAGE = re.compile(r"\*\(image: `([^`]+)`\)\*")                    # the marker the tutor writes where a picture belongs
IMAGE_WIDTH = 500                                                  # pixels; a figure should not fill the whole chat
LEADING_NUMBER = re.compile(r"^(\*\*Question \d+ of \d+\*\*\n\n)?(?:\*\*)?\d{1,2}\.(?:\*\*)?\s+")   # the number printed in the paper: "7." or, as the bank stores it, "**7.**"
PASSAGE_NUMBERS = re.compile(r"(?m)^Answer Questions? \d+\s*(?:and|to|-)\s*\d+ with reference to")   # "Answer Questions 30 and 31 with reference to ..."
MODES = ("Rules", "Model")


@st.cache_resource(show_spinner="Loading the question bank...")
def load_bank(fingerprint):
    """Read the effective bank once per store state, through the read-only connection."""
    from src import data as D, store as S
    tax = D.load_taxonomy(ROOT / "data" / "taxonomy.csv")
    qdf = D.derive(D.prepare_bank(D.load_bank(ROOT / "data" / "final_dataset.csv", tax)))
    curriculum = next((p for p in (ROOT / "data" / "ict_curriculum.md", ROOT / "ict_curriculum.md") if p.exists()), None)
    t = Tutor.from_store(qdf, DB_PATH, tax, curriculum)
    return t.eb, tax, t.curriculum


def store_fingerprint():
    from src import store as S
    con = S.connect_ro(DB_PATH)
    try:
        return S.state_fingerprint(con)
    finally:
        con.close()


def start_session():
    fp = store_fingerprint()
    eb, tax, curriculum = load_bank(fp)
    st.session_state.tutor = Tutor(eb, tax, curriculum)          # one tutor, so one session state, per browser session
    st.session_state.db_fp = fp


def refresh_if_store_changed():
    """When the staff side commits something new, rebuild the bank and keep this session's progress."""
    fp = st.session_state.get("db_fp")
    if fp is None:                                                # a tutor supplied by a test: nothing to reload
        return
    now = store_fingerprint()
    if now != fp:
        eb, tax, curriculum = load_bank(now)
        old = st.session_state.tutor
        new = Tutor(eb, tax, curriculum)
        new.state = old.state
        st.session_state.tutor, st.session_state.db_fp = new, now


def strip_paper_numbers(text: str) -> str:
    """Remove the number the paper printed before a question ("7. What ..." or the bank's "**7.** What ..."), which
    could be read as the number of the question in this session, and reword the printed question numbers of a shared
    passage ("Answer Questions 30 and 31 with reference to ..." becomes "Answer the questions below with reference
    to ..."), which would otherwise disagree with "Question 1 of 2". Only the start of each block or line is touched."""
    blocks = text.split("\n\n---\n\n")
    text = "\n\n---\n\n".join(LEADING_NUMBER.sub(lambda m: m.group(1) or "", b, count=1) for b in blocks)
    return PASSAGE_NUMBERS.sub("Answer the questions below with reference to", text)


def show_message(text: str):
    """Text and pictures in the order the tutor wrote them (stem, picture, options); the file path is never shown."""
    root = Path(st.session_state.get("image_root", ROOT))
    text, pos, shown = strip_paper_numbers(text), 0, set()
    for m in IMAGE.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            st.markdown(before)
        path = root / Path(m.group(1))
        if path in shown:                                          # members of a cluster can share one picture
            pass
        elif path.exists():
            st.image(str(path), width=IMAGE_WIDTH)
            shown.add(path)
        else:
            st.warning("The picture for this question could not be found.")
        pos = m.end()
    rest = text[pos:].strip()
    if rest:
        st.markdown(rest)


def reset():
    st.session_state.tutor.start()
    st.session_state.messages = [("assistant", WELCOME)]


def make_router(rules_only: bool = False):
    """The router for this message. Model mode is used only for what the student types; an llm_client in the
    session (a test's stub) replaces the real client."""
    tutor = st.session_state.tutor
    if not rules_only and st.session_state.get("router_mode") == "Model":
        return Orchestrator(tutor, client=st.session_state.get("llm_client"))
    return RuleRouter(tutor)


def send(text: str, rules_only: bool = False):
    reply = make_router(rules_only).handle(text)
    st.session_state.messages.append(("user", text))
    st.session_state.messages.append(("assistant", reply.text))     # only the reply text is shown, never reply.data


st.set_page_config(page_title="HKDSE ICT practice tutor", page_icon=None)
st.title("HKDSE ICT Paper 1A practice")
st.caption("Past-paper questions, and AI-drafted items approved by a teacher (each is labelled). "
           "Nothing you do here is stored beyond this session.")

if "tutor" not in st.session_state:
    start_session()
else:
    refresh_if_store_changed()
if "messages" not in st.session_state:
    st.session_state.messages = [("assistant", WELCOME)]
tutor = st.session_state.tutor

with st.sidebar:
    st.header("Your session")
    model_ok = "llm_client" in st.session_state or llm_available()
    st.radio("Tutor mode", MODES if model_ok else MODES[:1], key="router_mode", horizontal=True)
    st.caption("Model mode sends what you type to an external AI service (Anthropic)." if model_ok
               else "Model mode is off: no API key is set.")
    prog = tutor.progress()
    c1, c2 = st.columns(2)
    c1.metric("Answered", prog["attempts"])
    c2.metric("Correct", prog["correct"])
    st.caption(f"Admitted (AI-drafted) items served: {prog['provisional_units_served']} of {prog['units_served']}")
    names = [f"{t} - {tutor.tag_names.get(t, '')}" for t in tutor.all_tags]
    choice = st.selectbox("Topic", ["Any topic"] + names, key="topic")
    if st.button("Next question", key="next_btn"):
        send("next question" + ("" if choice == "Any topic" else f" on {choice.split(' - ')[0]}"), rules_only=True)
    if st.button("Start again", key="reset_btn"):
        reset()
        st.rerun()
    attempted = [r for r in prog["by_tag"] if r["attempts"]]
    if attempted:
        st.subheader("By sub-topic")
        st.dataframe([{"sub-topic": r["tag"], "answered": r["attempts"], "correct": r["correct"]} for r in attempted],
                     hide_index=True)

for role, text in st.session_state.messages:
    with st.chat_message(role):
        show_message(text)

if prompt := st.chat_input("Your answer or request"):
    send(prompt)
    st.rerun()
