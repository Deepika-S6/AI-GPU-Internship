import re

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


SUSPICIOUS_PATTERNS = {
    "payment request": r"\b(pay|payment|registration fee|security deposit|processing fee|refundable fee)\b",
    "unrealistic promise": r"\b(guaranteed job|100% placement|earn daily|easy money|high salary)\b",
    "pressure wording": r"\b(urgent|limited seats|apply immediately|today only|last chance)\b",
    "unprofessional contact": r"\b(whatsapp only|telegram|personal gmail|dm me|no interview)\b",
    "vague role": r"\b(work from mobile|simple typing|copy paste|no skills required)\b",
}


EXAMPLES = {
    "Suspicious sample": """Urgent hiring for data science internship. No interview required.
Pay registration fee of 799 today and get guaranteed certificate, stipend, and job offer.
Contact only on WhatsApp.""",
    "Genuine sample": """Software engineering internship for final-year students.
Responsibilities include building APIs, writing tests, and joining weekly mentor reviews.
Applicants should know Python, Git, and SQL. Apply through the company careers portal.""",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("internship_offers.csv")


@st.cache_resource
def train_model(data: pd.DataFrame) -> Pipeline:
    x_train, _x_test, y_train, _y_test = train_test_split(
        data["offer_text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    return model


def find_warning_signals(text: str) -> list[str]:
    matches = []
    for label, pattern in SUSPICIOUS_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(label)
    return matches


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def get_prediction(model: Pipeline, text: str) -> tuple[str, float]:
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    class_index = list(model.classes_).index(prediction)
    return prediction, float(probabilities[class_index])


st.set_page_config(page_title="InternGuard", page_icon="IG", layout="wide")

# ---------------------------------------------------------------------------
# Professional theme styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background: #f7f8fa;
            color: #1f2937;
        }

        [data-testid="stHeader"] {
            background: rgba(247, 248, 250, 0.9);
            border-bottom: 1px solid #e2e5ea;
        }

        .block-container {
            max-width: 1140px;
            padding-top: 1.6rem;
        }

        h1, h2, h3 { color: #12213b; font-weight: 700; }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: #4b5566;
        }

        /* ---------- Header bar ---------- */
        .masthead {
            background: #12213b;
            border-radius: 10px;
            padding: 1.7rem 2.1rem;
            margin-bottom: 1.5rem;
        }
        .masthead-eyebrow {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #9fb3d9;
            margin-bottom: 0.55rem;
        }
        .masthead-title {
            font-family: 'Source Serif 4', serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0 0 0.25rem 0;
        }
        .masthead-subtitle {
            font-size: 0.98rem;
            color: #c3ceE0;
            color: #c3cee0;
            margin: 0;
        }

        /* ---------- Panels ---------- */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stMetric"],
        [data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e2e5ea;
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(18, 33, 59, 0.03);
        }
        [data-testid="stMetric"] { padding: 0.9rem 1rem; }
        [data-testid="stMetricLabel"] { color: #6b7686; font-size: 0.82rem; }
        [data-testid="stMetricValue"] { color: #12213b; font-weight: 700; }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background: #eef0f3;
            padding: 4px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            color: #5b6474;
            font-weight: 600;
            font-size: 0.92rem;
            padding: 0.5rem 1.1rem;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: #12213b !important;
            box-shadow: 0 1px 2px rgba(18, 33, 59, 0.1);
        }

        /* ---------- Inputs ---------- */
        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stFileUploader"] section {
            background: #ffffff;
            color: #1f2937;
            border-color: #d7dbe2 !important;
            border-radius: 8px !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder { color: #97a0ae; }
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: #12213b !important;
            box-shadow: 0 0 0 3px rgba(18, 33, 59, 0.08) !important;
        }
        [data-testid="stFileUploader"] section {
            border: 1.5px dashed #c7cdd7 !important;
        }

        /* ---------- Buttons ---------- */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 0;
            background: #12213b;
            color: #ffffff;
            font-weight: 600;
            padding: 0.7rem 1rem;
            letter-spacing: 0.01em;
        }
        .stButton > button:hover {
            background: #1c3358;
            color: #ffffff;
        }

        /* ---------- Alerts ---------- */
        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid #e2e5ea;
        }

        /* ---------- Dataframe ---------- */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e2e5ea;
        }

        div[data-baseweb="select"] span,
        label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] div {
            color: #3c4454 !important;
        }

        hr { border-color: #e2e5ea; }

        /* ---------- Signal chips ---------- */
        .signal-chip {
            display: inline-block;
            background: #f2f4f7;
            border: 1px solid #dde1e8;
            border-radius: 6px;
            padding: 0.42rem 0.7rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            font-size: 0.86rem;
            font-weight: 500;
            color: #3c4454;
        }
        .signal-chip::before {
            content: "—";
            color: #12213b;
            margin-right: 0.4rem;
        }

        /* ---------- Result banner ---------- */
        .result-banner {
            border-radius: 8px;
            padding: 1rem 1.2rem;
            font-weight: 700;
            font-size: 1.02rem;
            border: 1px solid;
            margin-bottom: 0.6rem;
        }
        .result-suspicious {
            background: #fdf2f2;
            border-color: #f3caca;
            color: #9b2c2c;
        }
        .result-genuine {
            background: #f1f8f4;
            border-color: #c3e0cd;
            color: #1e5c37;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e5ea;
        }
        [data-testid="stSidebar"] h3 { color: #12213b; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

data = load_data()
model = train_model(data)

# ---------------------------------------------------------------------------
# Sidebar: stats + reference info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model overview")
    st.metric("Training samples", len(data))
    st.metric("Model", "TF-IDF + LogReg")
    st.metric("Input types", "Text / PDF")

    st.markdown("---")
    st.markdown("### Common warning signs")
    for signal in SUSPICIOUS_PATTERNS:
        st.markdown(f'<span class="signal-chip">{signal.title()}</span>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("View sample dataset"):
        st.dataframe(data, use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <span class="masthead-eyebrow">Student Safety · ML Tool</span>
        <p class="masthead-title">InternGuard</p>
        <p class="masthead-subtitle">AI-Based Fake Internship Offer Detection System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.markdown("### Submit an internship offer")

selected_example = st.selectbox(
    "Try a quick example",
    ["Write my own"] + list(EXAMPLES.keys()),
)
default_text = "" if selected_example == "Write my own" else EXAMPLES[selected_example]

extracted_sources = []

tab_text, tab_pdf = st.tabs(["Paste text", "PDF upload"])

with tab_text:
    manual_text = st.text_area(
        "Paste internship offer or job post text",
        value=default_text,
        height=220,
        placeholder="Paste the internship description, message, or email here...",
    )
    if manual_text.strip():
        extracted_sources.append(("Manual", manual_text))

with tab_pdf:
    uploaded_pdf = st.file_uploader(
        "Upload internship offer PDF",
        type=["pdf"],
        help="Upload an offer letter, internship notice, or job post saved as PDF.",
    )
    if uploaded_pdf is not None:
        try:
            pdf_text = extract_pdf_text(uploaded_pdf)
            if pdf_text:
                extracted_sources.append(("PDF", pdf_text))
                st.success("PDF text extracted successfully.")
            else:
                st.warning("No readable text found in this PDF. Try a text-based PDF or paste the content manually.")
        except Exception as error:
            st.error(f"Could not read this PDF: {error}")

# Combine whatever was captured across the tabs (manual text takes priority if present,
# otherwise fall back to extracted file content), preserving original behavior.
non_manual_text = "\n\n".join(text for source, text in extracted_sources if source != "Manual")
offer_text = (manual_text.strip() if manual_text.strip() else "") or non_manual_text

st.markdown("")
analyze = st.button("Analyze offer", type="primary")

# ---------------------------------------------------------------------------
# Result section
# ---------------------------------------------------------------------------
if analyze:
    cleaned_text = offer_text.strip()
    if not cleaned_text:
        st.warning("Please paste text or upload a readable PDF before analyzing.")
    else:
        prediction, confidence = get_prediction(model, cleaned_text)
        warning_signals = find_warning_signals(cleaned_text)

        st.markdown("## Result")
        result_col, confidence_col = st.columns(2)
        with result_col:
            if prediction == "suspicious":
                st.markdown(
                    '<div class="result-banner result-suspicious">⚠ Suspicious internship offer</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-banner result-genuine">✓ Likely genuine internship offer</div>',
                    unsafe_allow_html=True,
                )
        with confidence_col:
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        st.markdown("### Explainability")
        if warning_signals:
            st.write("The post contains these suspicious signals:")
            signal_chips = "".join(
                f'<span class="signal-chip">{signal.title()}</span>' for signal in warning_signals
            )
            st.markdown(signal_chips, unsafe_allow_html=True)
        else:
            st.write("No strong rule-based warning signals were found.")

        st.info(
            "This is an educational ML demo. Final decisions should also check company website, email domain, "
            "official careers page, reviews, and whether money is requested."
        )