# =========================
# Resume Scorer (DEMO ONLY)
# =========================
# Runs without SharePoint. It reads .pdf/.docx from a local `data/` folder
# and/or from files the user uploads in the UI.

import os, io, re, sys
from datetime import date

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

# Similarity scoring (optional)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------- Page setup ----------
st.set_page_config(page_title="Resume Scorer (Demo)", page_icon="📄", layout="wide")
st.title("📄 Resume Scorer (Demo Mode)")
st.caption("This demo reads local / uploaded resumes only — no SharePoint connection.")
st.sidebar.write("🐍 Python:", sys.version.split()[0])

# ========================
# Helpers: text extraction
# ========================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return text from a PDF (best effort)."""
    tmp = "tmp_demo.pdf"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    text = []
    reader = PdfReader(tmp)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Return text from a DOCX."""
    tmp = "tmp_demo.docx"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    doc = Document(tmp)
    return "\n".join(p.text for p in doc.paragraphs)

# ==================================
# Helpers: years-of-experience logic
# (your original logic, simplified)
# ==================================
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

def _mk_date(y: int, m: int) -> date:
    m = min(max(1, m), 12)
    return date(int(y), int(m), 15)

def _parse_month(tok: str):
    return MONTHS.get((tok or "").strip().lower())

def _parse_year(tok: str):
    if not tok: return None
    m = re.match(r"(19|20)\d{2}$", tok.strip())
    return int(m.group(0)) if m else None

def _present_to_date() -> date:
    t = date.today()
    return date(t.year, t.month, 15)

def _extract_date_ranges(text: str):
    t = text.replace("\u2013", "-").replace("\u2014", "-")
    ranges = []

    pat_month_year = re.compile(
        r"\b(?P<m1>[A-Za-z]{3,9})\s+(?P<y1>(?:19|20)\d{2})\s*[-to]+\s*(?P<m2>Present|Current|[A-Za-z]{3,9})\s*(?P<y2>(?:19|20)\d{2})?\b",
        flags=re.I
    )
    for m in pat_month_year.finditer(t):
        m1 = _parse_month(m.group("m1"))
        y1 = _parse_year(m.group("y1"))
        m2tok = m.group("m2")
        y2tok = m.group("y2")
        if m1 and y1:
            start = _mk_date(y1, m1)
            if m2tok and m2tok.lower() in ("present", "current"):
                end = _present_to_date()
            else:
                m2 = _parse_month(m2tok); y2 = _parse_year(y2tok) if y2tok else None
                if not (m2 and y2): continue
                end = _mk_date(y2, m2)
            if end > start:
                ranges.append((start, end))

    pat_year_year = re.compile(
        r"\b(?P<y1>(?:19|20)\d{2})\s*[-to]+\s*(?P<y2>Present|Current|(?:19|20)\d{2})\b",
        flags=re.I
    )
    for m in pat_year_year.finditer(t):
        y1 = _parse_year(m.group("y1"))
        if not y1: continue
        start = _mk_date(y1, 6)
        y2tok = m.group("y2")
        if y2tok.lower() in ("present", "current"):
            end = _present_to_date()
        else:
            y2 = _parse_year(y2tok)
            if not y2: continue
            end = _mk_date(y2, 6)
        if end > start:
            ranges.append((start, end))

    pat_mmyyyy = re.compile(
        r"\b(?P<m1>0?[1-9]|1[0-2])/(?P<y1>(?:19|20)\d{2})\s*[-to]+\s*(?P<m2>0?[1-9]|1[0-2])/(?P<y2>(?:19|20)\d{2}|Present|Current)\b",
        flags=re.I
    )
    for m in pat_mmyyyy.finditer(t):
        m1 = int(m.group("m1")); y1 = _parse_year(m.group("y1"))
        if not (y1 and 1 <= m1 <= 12): continue
        start = _mk_date(y1, m1)
        y2raw = m.group("y2")
        if y2raw.lower() in ("present", "current"):
            end = _present_to_date()
        else:
            m2 = int(m.group("m2")); y2 = _parse_year(y2raw)
            if not (y2 and 1 <= m2 <= 12): continue
            end = _mk_date(y2, m2)
        if end > start:
            ranges.append((start, end))

    if not ranges:
        return []
    ranges.sort(key=lambda r: r[0])
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def _years_from_ranges(text: str) -> float:
    months = 0
    for s, e in _extract_date_ranges(text):
        diff = (e.year - s.year) * 12 + (e.month - s.month)
        months += max(0, diff)
    return round(months / 12.0, 1)

def _years_from_phrases(text: str) -> int:
    best = 0
    for m in re.finditer(r"\b([1-4]?\d)\s*\+?\s*[- ]?\s*(?:years?|yrs?)\b", text, flags=re.I):
        best = max(best, int(m.group(1)))
    return best

def estimate_years_experience(text: str):
    yrs_ranges = _years_from_ranges(text)
    yrs_phrases = _years_from_phrases(text)
    return (yrs_ranges, "ranges") if yrs_ranges >= 0.5 else (float(yrs_phrases), "phrases")

def classify_level(years: float, jr_max: int, mid_max: int) -> str:
    return "Junior" if years <= jr_max else ("Mid" if years <= mid_max else "Senior")



# ================================
# Similarity scoring (TF-IDF cosine)
# ================================
def similarity_to_reference(reference_text: str, candidate_texts: list[str]) -> list[float]:
    """Return cosine similarity (0–1) comparing each candidate to reference_text."""
    if not SKLEARN_OK:
        return [0.0 for _ in candidate_texts]
    docs = [reference_text] + candidate_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(docs)
    ref_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]
    sims = cosine_similarity(cand_vecs, ref_vec).ravel()
    return [float(s) for s in sims]

# ===================
# Requirements input
# ===================
st.subheader("📥 Job Requirements (.txt)")
uploaded_req_file = st.file_uploader("Upload a .txt with keywords (one per line)", type=["txt"])
KEYWORDS = []
if uploaded_req_file:
    for line in uploaded_req_file.read().decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.endswith(":"):
            KEYWORDS.append(line)
    st.success(f"✅ Loaded {len(KEYWORDS)} keyword(s).")
else:
    st.warning("Upload a requirements .txt to enable scoring.")
    st.stop()

# =========================
# Scoring controls (knobs)
# =========================
st.subheader("⚙️ Scoring & Filters")
colA, colB, colC, colD = st.columns(4)
with colA:
    exp_points_per_year = st.number_input("Pts / year", 0, 50, 5, 1)
with colB:
    jr_max = st.number_input("Max years: Junior", 0, 10, 2, 1)
with colC:
    # Fix: default must be >= jr_max to avoid ValueBelowMinError
    mid_max = st.number_input(
        "Max years: Mid",
        min_value=jr_max, max_value=25,
        value=max(jr_max + 1, 6), step=1
    )
with colD:
    enforce_min = st.checkbox("Enforce minimum YOE", value=False)
min_years_required = st.number_input("Minimum years to include", 0, 30, 3, 1)

# ===================
# Scoring function
# ===================
def score_resume(text: str):
    lower_text = text.lower()

    # Keyword scoring
    found_keywords = [kw for kw in KEYWORDS if kw.lower() in lower_text]
    kw_score = 10 * len(found_keywords)

    # YOE
    years, src = estimate_years_experience(text)
    exp_score = years * exp_points_per_year
    total = kw_score + exp_score

    # Clearance check (case-insensitive)
    clearance_phrases = ["active secret clearance", "secret clearance"]
    found_clearances = [p for p in clearance_phrases if p in lower_text]
    has_clearance = "Yes" if found_clearances else "No"

    return {
        "years": years,
        "years_source": src,
        "level": classify_level(years, jr_max, mid_max),
        "kw_score": kw_score,
        "exp_score": exp_score,
        "total": total,
        "keywords_found": ", ".join(found_keywords),
        "clearance": has_clearance,
        "clearance_phrases": ", ".join(found_clearances),
    }

# ================================
# Choose resumes to analyze (demo)
# ================================
st.subheader("📂 Pick resumes")
left, right = st.columns(2)

# Option 1: local data/ folder
with left:
    st.markdown("**Local folder: `data/`**")
    local_files = []
    if os.path.isdir("data"):
        for fn in os.listdir("data"):
            if fn.lower().endswith((".pdf", ".docx")):
                with open(os.path.join("data", fn), "rb") as f:
                    local_files.append((fn, f.read()))
    st.write([n for n, _ in local_files] or "(no files found)")

# Option 2: upload now
with right:
    st.markdown("**Or upload resumes now**")
    uploads = st.file_uploader(
        "Drop .pdf/.docx here", type=["pdf", "docx"], accept_multiple_files=True
    )
    uploaded_files = []
    if uploads:
        for f in uploads:
            uploaded_files.append((f.name, f.read()))

# Combine sources
all_files = local_files + uploaded_files
if not all_files:
    st.info("Add .pdf/.docx to `data/` or upload here, then click **Score resumes**.")

# =====================
# Similarity benchmark UI
# =====================
st.subheader("🎯 Similarity Benchmark (optional)")
use_similarity = st.checkbox("Compare resumes to a benchmark resume", value=True)
benchmark_name = None
sim_weight = 0.30
if use_similarity:
    if not SKLEARN_OK:
        st.warning("Similarity scoring requires scikit-learn. If you deploy this, add 'scikit-learn' to requirements.txt.")
        use_similarity = False
    elif all_files:
        benchmark_name = st.selectbox(
            "Select the benchmark resume (others will be compared to this)",
            options=[n for n, _ in all_files]
        )
        sim_weight = st.slider("Similarity weight (used in Final Score)", 0.0, 1.0, 0.30, 0.05)

# ===================
# Run scoring
# ===================
if st.button("▶️ Score resumes"):
    rows = []
    texts = {}

    # 1) Extract text for all files once (reuse for similarity + scoring)
    for name, data in all_files:
        try:
            if name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(data)
            elif name.lower().endswith(".docx"):
                text = extract_text_from_docx(data)
            else:
                continue
            texts[name] = text
        except Exception:
            texts[name] = ""

    # 2) Compute similarity map (name -> 0..1) if enabled
    sim_map = {}
    if use_similarity and benchmark_name and benchmark_name in texts:
        reference_text = texts.get(benchmark_name, "")
        candidate_names = [n for n in texts.keys() if n != benchmark_name]
        candidate_texts = [texts[n] for n in candidate_names]
        sims = similarity_to_reference(reference_text, candidate_texts)
        sim_map = {n: float(s) for n, s in zip(candidate_names, sims)}

    # 3) Score resumes (existing logic) + attach similarity
    for name, text in texts.items():
        if not text:
            continue

        result = score_resume(text)
        if enforce_min and result["years"] < float(min_years_required):
            continue

        sim = sim_map.get(name, 1.0 if (use_similarity and name == benchmark_name) else 0.0)

        # Final Score: blend your existing total with similarity (scaled to 0..100)
        # If similarity is off, Final Score = Total Score
        final_score = result["total"]
        if use_similarity:
            final_score = (1.0 - sim_weight) * float(result["total"]) + sim_weight * (sim * 100.0)

        rows.append({
            "File Name": name,
            "Est. Years": result["years"],
            "Level (Jr/Mid/Sr)": result["level"],
            "Experience Source": result["years_source"],
            "Keyword Score": result["kw_score"],
            "Experience Score": result["exp_score"],
            "Total Score": result["total"],
            "Similarity to Benchmark": round(sim, 4) if use_similarity else "",
            "Final Score": round(final_score, 2),
            "Keywords Found": result["keywords_found"],
            "Clearance Mentioned": result["clearance"],
            "Clearance Phrases": result["clearance_phrases"],
        })

    if not rows:
        st.warning("No qualifying resumes to show.")
    else:
        df = pd.DataFrame(rows)
        # Sort primarily by Final Score (then your original Total/YOE fields)
        sort_cols = ["Final Score"]
        for c in ["Total Score", "Est. Years"]:
            if c in df.columns:
                sort_cols.append(c)
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

        st.success(f"Scored {len(df)} resume(s).")
        st.dataframe(df, use_container_width=True)

        # Download results
        out = io.BytesIO()
        df.to_excel(out, index=False)
        out.seek(0)
        st.download_button("📥 Download Excel Report", out, file_name="resume_scores_demo.xlsx")


        # Download results
        out = io.BytesIO()
        df.to_excel(out, index=False)
        out.seek(0)
        st.download_button("📥 Download Excel Report", out, file_name="resume_scores_demo.xlsx")
