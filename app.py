# app.py

import os
import pickle
import numpy as np
import streamlit as st

from scipy.sparse import hstack

from preprocess import preprocess
from score import score_label, sophistication_score

# -------------------------------------------------
# TRUSTED DOMAINS
# -------------------------------------------------

TRUSTED_DOMAINS = [
    "google.com",
    "microsoft.com",
    "apple.com",
    "github.com",
    "amazon.com"
]

def trust_score(text: str) -> float:
    text = str(text).lower()

    for domain in TRUSTED_DOMAINS:
        if domain in text:
            return 0.9

    return 0.3


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Phishing Detector",
    page_icon="🎣",
    layout="centered"
)

# -------------------------------------------------
# STYLE
# -------------------------------------------------

st.markdown("""
<style>
.risk-badge {
    display:inline-block;
    padding:4px 14px;
    border-radius:20px;
    font-weight:600;
    font-size:0.85rem;
}
.risk-low      { background:#d1fae5; color:#065f46; }
.risk-moderate { background:#fef3c7; color:#92400e; }
.risk-high     { background:#fee2e2; color:#991b1b; }
.risk-critical { background:#7f1d1d; color:#fef2f2; }

.technique-tag {
    display:inline-block;
    margin:3px 4px;
    padding:3px 10px;
    border-radius:12px;
    font-size:0.8rem;
    font-weight:500;
}

.tag-urgency   { background:#fef9c3; color:#713f12; }
.tag-authority { background:#dbeafe; color:#1e3a5f; }
.tag-fear      { background:#fce7f3; color:#831843; }
.tag-scarcity  { background:#ede9fe; color:#4c1d95; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🎣 Phishing Email Detector")
st.caption("ML classifier with trust-domain context scoring")
st.divider()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        return None, None

    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

if model is None:
    st.error("Model files not found. Train first.")
    st.stop()

# -------------------------------------------------
# INPUT
# -------------------------------------------------

email_input = st.text_area(
    "Paste email content below:",
    height=220,
    placeholder="Paste subject + body here..."
)

analyse_btn = st.button(
    "🔍 Analyse Email",
    type="primary",
    use_container_width=True
)

# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------

if analyse_btn:

    if not email_input.strip():
        st.warning("Paste an email first.")
        st.stop()

    with st.spinner("Analysing... because apparently trust issues need automation."):

        clean_text = preprocess(email_input)

        if not clean_text.strip():
            st.error("No usable text after preprocessing.")
            st.stop()

        # -----------------------------------------
        # TEXT FEATURES
        # -----------------------------------------

        vec_text = vectorizer.transform([clean_text])

        # -----------------------------------------
        # TRUST FEATURE (MUST MATCH TRAINING)
        # -----------------------------------------

        score_trust = trust_score(email_input)
        vec_trust = np.array([[score_trust]])

        vec = hstack([vec_text, vec_trust])

        # -----------------------------------------
        # PREDICT
        # -----------------------------------------

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(vec)[0][1])
        else:
            prob = float(model.predict(vec)[0])

        label = "PHISHING" if prob > 0.5 else "LEGITIMATE"
        is_phish = label == "PHISHING"

        # -----------------------------------------
        # SOPHISTICATION SCORE
        # -----------------------------------------

        score, found, explanation = sophistication_score(email_input, prob)
        risk = score_label(score)

# -------------------------------------------------
# RESULTS
# -------------------------------------------------

    st.divider()
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Classification", label)

        if is_phish:
            st.error("⚠️ Likely phishing")
        else:
            st.success("✅ Likely legitimate")

    with col2:
        st.metric("Phishing Probability", f"{prob:.1%}")
        st.progress(prob)

    with col3:
        st.metric("Sophistication Score", f"{score}%")

        badge_class = {
            "Low Risk": "risk-low",
            "Moderate Risk": "risk-moderate",
            "High Risk": "risk-high",
            "Critical Risk": "risk-critical"
        }.get(risk, "risk-low")

        st.markdown(
            f'<span class="risk-badge {badge_class}">{risk}</span>',
            unsafe_allow_html=True
        )

    st.divider()

# -------------------------------------------------
# TECHNIQUES
# -------------------------------------------------

    st.subheader("Manipulation Techniques Detected")

    tag_classes = {
        "urgency": "tag-urgency",
        "authority": "tag-authority",
        "fear": "tag-fear",
        "scarcity": "tag-scarcity"
    }

    any_found = any(v for v in found.values())

    if any_found:
        for technique, hits in found.items():
            if hits:
                cls = tag_classes.get(technique, "")
                tags = " ".join(
                    f'<span class="technique-tag {cls}">{x}</span>'
                    for x in hits
                )

                st.markdown(
                    f"**{technique.capitalize()}** &nbsp; {tags}",
                    unsafe_allow_html=True
                )
    else:
        st.info("No manipulation techniques detected.")

# -------------------------------------------------
# DEBUG TOKENS
# -------------------------------------------------

    with st.expander("🔬 View preprocessed tokens"):
        st.code(clean_text)