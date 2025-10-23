
import re
import difflib
import unicodedata
from typing import List, Optional
import pandas as pd
import streamlit as st

st.set_page_config(page_title="First Name Guesser", page_icon="🧠", layout="wide")

# ====== Configs (editable via UI as well) ======
import streamlit as st
import os

st.set_page_config(page_title="First Name Guesser", page_icon="🧠", layout="wide")

# ====== Configs (editable via UI as well) ======
@st.cache_data(show_spinner=False)
def load_english_names(path: str = "english_given_names.txt") -> set[str]:
    if not os.path.isfile(path):
        # Small default fallback so the app still runs
        return {
            "john","mary","michael","michelle","david","sarah","jason","jennifer",
            "ryan","samantha","jiaen","weiling","weijie","minghao","yuxuan","yuxin",
            "noel","yvonne","lydia","ashley","donald","stuart","hilary"
        }
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

ENGLISH_FIRST_NAMES = load_english_names()

MALAY_PARTICLES = {"BIN","BINTE","BINTI","BT","BN","ABD","ALA","BTE"}
COMMON_SURNAMES = {
    "TAN","LIM","NG","LEE","CHUA","GOH","ONG","TEO","TAY","CHAN","KOH","WONG","YEO","HENG","CHEE","SIM",
    "ANG","TOH","PEH","MOK","KOU","CHEN","TEH","LOW","LOH","PEK"
}
AMBIGUOUS_SHORT = {"ian", "ann", "lee"}  # extend if needed
SUSPICIOUS_SUFFIXES = {"kumar", "singh", "preet", "jit", "deep", "raj"}




# ====== Logic ======

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize_for_tokens(name: str) -> list:
    s = strip_accents(str(name)).upper()
    s = re.sub(r"[,/.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split(" ") if t]
    toks = [t for t in toks if t not in MALAY_PARTICLES]
    return toks

def join_no_space(tokens: list) -> str:
    return "".join(tokens)

def fuzzy_find_english(tokens: list, joined: str, original: str, english_names: set, cutoff: float):
    # 1) direct token match (prefer last)
    for t in reversed(tokens):
        if t.lower() in english_names:
            return t.capitalize()

    # 1b) if original has comma, bias rightmost segment
    if "," in original:
        parts = [p for p in original.split(",") if p.strip()]
        right = normalize_for_tokens(parts[-1]) if parts else []
        for t in reversed(right):
            if t.lower() in english_names:
                return t.capitalize()

    # 2) exact prefix/suffix on the joined blob
    for en in english_names:
        U = en.upper()
        if joined.endswith(U) or joined.startswith(U):
            # guard ambiguous short names unless exact token or true prefix
            if en.lower() in AMBIGUOUS_SHORT:
                if any(tok.lower() == en for tok in tokens) or joined.startswith(U):
                    return en.capitalize()
                else:
                    continue
            return en.capitalize()

    # 3) fuzzy candidates from windows + last token
    import difflib
    candidates = set()
    L = len(joined)
    for k in range(3, min(12, L) + 1):
        candidates.add(joined[-k:])
        candidates.add(joined[:k])
    if tokens:
        candidates.add(tokens[-1])

    best = None
    best_score = 0.0
    en_list = list(english_names)
    for c in candidates:
        c_l = c.lower()

        # Skip candidates that are entirely inside suspicious non-English suffixes
        if any(c_l.endswith(suf) or c_l == suf for suf in SUSPICIOUS_SUFFIXES):
            continue

        match = difflib.get_close_matches(c_l, en_list, n=1, cutoff=cutoff)
        if not match:
            continue

        en = match[0]                 # matched english name (lowercase)
        # **First-letter agreement**: reduce false positives like umar -> mary
        if c_l[0] != en[0]:
            continue

        score = difflib.SequenceMatcher(None, c_l, en).ratio()

        # Block ambiguous short names unless exact token or true prefix
        if en in AMBIGUOUS_SHORT and not (
            any(tok.lower() == en for tok in tokens) or joined.startswith(en.upper())
        ):
            continue

        if score > best_score or (abs(score - best_score) < 1e-9 and len(en) > len(best or "")):
            best = en
            best_score = score

    if best:
        return best.capitalize()
    return None


def first_name_best_guess(name: str, english_names: set, cutoff: float = 0.82, prefer_end: bool = True) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    original = name
    tokens = normalize_for_tokens(name)
    if not tokens:
        return ""

    joined = "".join(tokens)

    # Indian fused-name rule: VINODKUMAR -> Vinod, AMANSINGH -> Aman
    if len(tokens) == 1:
        t = tokens[0]
        for suf in ("KUMAR", "SINGH"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                return t[:-len(suf)].capitalize()

    # Single-token quick return (surname-like or short)
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in COMMON_SURNAMES or len(tok) <= 6:
            return tok.capitalize()

    # Malay BIN/BINTI handling
    raw_upper = strip_accents(original).upper()
    raw_tokens = [t for t in re.split(r"\s+|[,/.-]", raw_upper) if t]
    if any(p in raw_tokens for p in MALAY_PARTICLES):
        return tokens[0].capitalize()

    # English detection (exact + fuzzy)
    en = fuzzy_find_english(tokens, joined, original, english_names, cutoff=cutoff)
    if en:
        return en

    # Fallback: prefer end (Chinese fused names) or start
    return (tokens[-1] if prefer_end else tokens[0]).capitalize()



# ====== UI ======
st.title("🧠 First Name Guesser")
st.write("Upload a CSV, select the name column, and get a `FirstNameGuess` column with best-guess first names.")

with st.sidebar:
    st.header("Settings")
    cutoff = st.slider("Fuzzy match cutoff", 0.60, 0.95, 0.82, 0.01, help="Higher = stricter, lower = more permissive")
    prefer_end = st.toggle("Prefer end segment when uncertain", value=True, help="If no English name found, choose last token (common for fused CN names).")
    allow_custom_dict = st.toggle("Provide custom English given-name list", value=False)
    custom_dict_text = st.text_area(
        "Custom names (comma-separated)",
        placeholder="e.g., aidan, hilary, yvonne, stuart, noel"
    ) if allow_custom_dict else ""

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows.")
    cols = list(df.columns)
    if not cols:
        st.error("No columns detected in the uploaded CSV.")
        st.stop()

    name_col = st.selectbox("Column containing names", options=cols, index=0)

    # Build the effective dictionary:
    # start from the file-loaded ENGLISH_FIRST_NAMES, then extend with any custom names
    english_names = set(ENGLISH_FIRST_NAMES)
    if allow_custom_dict and custom_dict_text.strip():
        extras = [w.strip().lower() for w in custom_dict_text.split(",") if w.strip()]
        english_names |= set(extras)

    if name_col:
        with st.spinner("Processing names..."):
            df["FirstNameGuess"] = df[name_col].apply(
                lambda x: first_name_best_guess(x, english_names, cutoff=cutoff, prefer_end=prefer_end)
            )

        st.subheader("Preview")
        st.dataframe(df.head(50))

        # Download button
        out_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV with FirstNameGuess",
            out_csv,
            file_name="names_with_firstname.csv",
            mime="text/csv"
        )

        with st.expander("Diagnostics (random sample)", expanded=False):
            st.write(df.sample(min(100, len(df)), random_state=1337)[[name_col, "FirstNameGuess"]])
else:
    st.info("Upload a CSV to begin.")

