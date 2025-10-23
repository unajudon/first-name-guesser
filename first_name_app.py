# first_name_app.py
import os
import re
import difflib
import unicodedata
from typing import List, Optional

import pandas as pd
import streamlit as st

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="First Name Guesser", page_icon="🧠", layout="wide")

# ---------------- CONFIG / DICTS ----------------
@st.cache_data(show_spinner=False)
def load_english_names(path: str = "english_given_names.txt") -> set[str]:
    """Load the main dictionary (one name per line, lowercase recommended)."""
    if not os.path.isfile(path):
        # Small fallback so the app can still run
        return {
            "john","mary","michael","michelle","david","sarah","jason","jennifer",
            "ryan","samantha","jiaen","weiling","weijie","minghao","yuxuan","yuxin",
            "noel","yvonne","lydia","ashley","donald","stuart","hilary","mark"
        }
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}

ENGLISH_FIRST_NAMES = load_english_names()

# Malay / Arabic particles and generics
PARTICLES_MALAY = {"BIN","BINTI","BINTE","BT","BTE","BN"}
PARTICLES_PATRONYMIC = {"S/O","D/O","A/L","A/P"}  # Indian/Malay patronymics
GENERIC_ARABIC_PREFIXES = {
    "MUHAMMAD","MOHAMMAD","MOHAMED","MOHD","MD","MUHAMAD","MUHAMED",
    "ABDUL","ABDULLAH","ABDEL","ABDULLA","ABD","ABDULRAHMAN","HAJI","HAJJI","HAJ"
}

# Common Singapore surnames (used for short single-token fallback)
COMMON_SURNAMES = {
    "TAN","LIM","NG","LEE","CHUA","GOH","ONG","TEO","TAY","CHAN","KOH","WONG","YEO","HENG","CHEE","SIM",
    "ANG","TOH","PEH","MOK","KOU","CHEN","TEH","LOW","LOH","PEK"
}

# Guards
AMBIGUOUS_SHORT = {"ian", "ann"}  # only allow if exact token or true prefix of whole string
SUSPICIOUS_SUFFIXES = {"kumar", "singh", "preet", "jit", "deep", "raj"}  # ignore as fuzzy suffix candidates

# ---------------- CORE HELPERS ----------------
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

# Build a regex to insert spaces around known tokens even when fused
_SPLIT_TOKENS = sorted(
    list(PARTICLES_MALAY | PARTICLES_PATRONYMIC | GENERIC_ARABIC_PREFIXES),
    key=len, reverse=True
)
_SPLIT_PATTERN = re.compile(r"(" + "|".join(map(re.escape, _SPLIT_TOKENS)) + r")")

def normalize_for_tokens(name: str) -> List[str]:
    """
    Uppercase, strip accents, insert spaces around particles/prefixes,
    normalize punctuation to spaces, split into tokens.
    (We DO NOT drop the particles here; the resolver uses them.)
    """
    s = strip_accents(str(name)).upper()
    s = re.sub(r"[.,/\\\-]+", " ", s)          # punctuation -> space
    s = _SPLIT_PATTERN.sub(r" \1 ", s)         # space around special tokens even when fused
    s = re.sub(r"\s+", " ", s).strip()
    toks = [t for t in s.split(" ") if t]
    return toks

def join_no_space(tokens: List[str]) -> str:
    return "".join(tokens)

# ---------------- SPECIAL RESOLVERS ----------------
def resolve_malay_arabic_patronymic(tokens: List[str]) -> Optional[str]:
    """
    Handle Malay/Arabic/Patronymic patterns.
    Rules:
      - If tokens contain BIN/BINTI/BINTE: pick first non-generic BEFORE the particle.
        If the name starts with the particle, pick first non-generic AFTER it.
      - If tokens contain S/O, D/O, A/L, A/P: prefer token BEFORE; if none, take AFTER.
      - If no particles but generic Arabic prefixes at start: return first non-generic token.
    """
    if not tokens:
        return None

    T = tokens

    # Patronymics S/O, D/O, A/L, A/P
    for i, tok in enumerate(T):
        if tok in PARTICLES_PATRONYMIC:
            if i > 0:
                return T[i-1].capitalize()
            if i + 1 < len(T):
                j = i + 1
                while j < len(T) and T[j] in GENERIC_ARABIC_PREFIXES:
                    j += 1
                if j < len(T):
                    return T[j].capitalize()
            return None

    # Malay BIN/BINTI/BINTE
    for i, tok in enumerate(T):
        if tok in PARTICLES_MALAY:
            if i > 0:
                # pick first non-generic from the head segment
                j = 0
                while j < i and T[j] in GENERIC_ARABIC_PREFIXES:
                    j += 1
                if j < i:
                    return T[j].capitalize()
                return T[0].capitalize()
            # starts with BIN/BINTI → take first non-generic after it
            if i + 1 < len(T):
                j = i + 1
                while j < len(T) and T[j] in GENERIC_ARABIC_PREFIXES:
                    j += 1
                if j < len(T):
                    return T[j].capitalize()
            return None

    # No particles but starts with generic Arabic prefixes → first non-generic
    if T and T[0] in GENERIC_ARABIC_PREFIXES:
        j = 0
        while j < len(T) and T[j] in GENERIC_ARABIC_PREFIXES:
            j += 1
        if j < len(T):
            return T[j].capitalize()

    return None

# ---------------- MATCHERS ----------------
def fuzzy_find_english(tokens: List[str], joined: str, original: str, english_names: set, cutoff: float) -> Optional[str]:
    """Find an English given name by exact token, exact prefix/suffix, then fuzzy (with guards)."""
    # 1) Direct token match (prefer last)
    for t in reversed(tokens):
        if t.lower() in english_names:
            return t.capitalize()

    # 1b) If original has a comma, bias the rightmost comma part
    if "," in original:
        parts = [p for p in original.split(",") if p.strip()]
        right = normalize_for_tokens(parts[-1]) if parts else []
        for t in reversed(right):
            if t.lower() in english_names:
                return t.capitalize()

    # 2) Exact prefix/suffix on the joined blob — prefer the LONGEST exact hit
    exact_hit = None
    for en in english_names:
        U = en.upper()
        if joined.startswith(U) or joined.endswith(U):
            # guard ambiguous very-short names unless true prefix or exact token
            if en in AMBIGUOUS_SHORT:
                if any(tok.lower() == en for tok in tokens) or joined.startswith(U):
                    pass
                else:
                    continue
            if exact_hit is None or len(en) > len(exact_hit):
                exact_hit = en
    if exact_hit:
        return exact_hit.capitalize()

    # 3) Fuzzy windows + last token, with stronger guards
    candidates = set()
    L = len(joined)
    # Require windows of at least 4 chars (reduces MAR->MARY style errors)
    for k in range(4, min(12, L) + 1):
        candidates.add(joined[-k:])
        candidates.add(joined[:k])
    if tokens:
        candidates.add(tokens[-1])

    best = None
    best_score = 0.0
    en_list = list(english_names)
    for c in candidates:
        c_l = c.lower()

        # Skip suspicious suffixes fully (e.g., 'kumar', 'singh')
        if any(c_l.endswith(suf) or c_l == suf for suf in SUSPICIOUS_SUFFIXES):
            continue

        match = difflib.get_close_matches(c_l, en_list, n=1, cutoff=cutoff)
        if not match:
            continue

        en = match[0]

        # First-letter agreement (blocks umar->mary etc.)
        if not c_l or c_l[0] != en[0]:
            continue

        score = difflib.SequenceMatcher(None, c_l, en).ratio()

        # If this is a prefix window and last char differs, require very high score
        is_prefix_window = joined.startswith(c)
        if is_prefix_window and len(c) >= 4 and c_l[-1:] != en[-1:]:
            if score < 0.93:
                continue

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

# ---------------- MAIN GUESSER ----------------
def first_name_best_guess(name: str, english_names: set, cutoff: float = 0.90, prefer_end: bool = True) -> str:
    """
    Heuristic first-name guesser tuned for SG/MY patterns.
    Order:
      1) Malay/Arabic/Patronymic resolver
      2) Indian fused-name rule (…KUMAR / …SINGH)
      3) Single-token quick return (surname-like or short)
      4) English detection (exact + fuzzy)
      5) Fallback (end or start)
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    original = name
    tokens = normalize_for_tokens(name)
    if not tokens:
        return ""

    joined = join_no_space(tokens)

    # 1) Malay/Arabic/Patronymic specialized rules
    special = resolve_malay_arabic_patronymic(tokens)
    if special:
        return special

    # 2) Indian fused-name rule: VINODKUMAR -> Vinod, AMANSINGH -> Aman
    if len(tokens) == 1:
        t = tokens[0]
        for suf in ("KUMAR", "SINGH"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                return t[:-len(suf)].capitalize()

    # 3) Single-token quick return (surname-like or short)
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in COMMON_SURNAMES or len(tok) <= 6:
            return tok.capitalize()

    # 4) English detection (exact + fuzzy)
    en = fuzzy_find_english(tokens, joined, original, english_names, cutoff=cutoff)
    if en:
        return en

    # 5) Fallback (Chinese fused names often end with given; else prefer start)
    return (tokens[-1] if prefer_end else tokens[0]).capitalize()

# ---------------- UI ----------------
st.title("🧠 First Name Guesser")
st.write("Upload a CSV, select the name column, and get a `FirstNameGuess` column with best-guess first names.")

with st.sidebar:
    st.header("Settings")
    cutoff = st.slider("Fuzzy match cutoff", 0.60, 0.95, 0.90, 0.01, help="Higher = stricter, lower = more permissive")
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

    # Build the effective dictionary (file-loaded + optional custom additions)
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
