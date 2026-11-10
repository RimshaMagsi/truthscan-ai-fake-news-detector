from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="TRUTHSCAN AI | Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 10% 5%, rgba(42,112,255,.18), transparent 28%),
      radial-gradient(circle at 90% 12%, rgba(145,70,255,.16), transparent 25%),
      linear-gradient(135deg, #07111f 0%, #0b1730 55%, #10152b 100%);
    color: #edf4ff;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1830, #10152b);
    border-right: 1px solid rgba(130,170,255,.16);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
.hero {
    padding: 28px 30px; border-radius: 22px; margin-bottom: 22px;
    background: linear-gradient(110deg, rgba(25,91,190,.42), rgba(105,55,190,.30));
    border: 1px solid rgba(128,174,255,.28);
    box-shadow: 0 14px 45px rgba(0,0,0,.22);
}
.hero h1 { font-size: 2.25rem; margin: 0; color: #fff; font-weight: 800; }
.hero p { color: #c7d8f5; margin: 8px 0 0 0; font-size: 1rem; }
.panel {
    background: rgba(16,31,57,.78); border: 1px solid rgba(129,167,230,.18);
    border-radius: 18px; padding: 20px 22px; margin-bottom: 16px;
}
.metric {
    background: linear-gradient(145deg, rgba(24,54,94,.88), rgba(30,35,76,.88));
    border: 1px solid rgba(120,168,255,.22); border-radius: 16px;
    padding: 18px; min-height: 112px;
}
.metric .label { color: #a9bfdf; font-size: .85rem; }
.metric .value { color: #fff; font-size: 1.7rem; font-weight: 800; margin-top: 7px; }
.small-note { color: #a9bbd8; font-size: .86rem; }
div.stButton > button {
    background: linear-gradient(90deg, #2378f6, #7954e8);
    color: white; border: 0; border-radius: 11px; padding: .65rem 1.2rem;
    font-weight: 700; transition: .2s ease;
}
div.stButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); }
.stTextArea textarea, .stTextInput input {
    background: #101f38 !important; color: #f3f7ff !important;
    border: 1px solid #38527b !important; border-radius: 12px !important;
}
h1, h2, h3 { color: #f4f7ff; }
[data-testid="stFileUploader"] {
    background: rgba(20,39,70,.55); border: 1px dashed #5273a8; border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
metrics = load_metrics()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("## 📰 TRUTHSCAN AI")
    st.caption("AI-powered news text pattern analysis")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Analyze News", "Batch Analysis", "Scan History", "About Project"]
    )
    st.markdown("---")
    st.caption("Machine Learning • NLP • TF-IDF")
    st.caption("Educational project")

st.markdown("""
<div class="hero">
  <h1>TRUTHSCAN <span style="color:#62c7ff">AI</span></h1>
  <p>Explore news text patterns with machine learning and natural language processing.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("The trained model is not available yet. Run `python train_model.py` in the VS Code terminal first.")
    st.stop()

def analyse(text):
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    confidence = float(max(probabilities)) * 100
    return prediction, confidence

if page == "Dashboard":
    st.subheader("Project Dashboard")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            ("TEST ACCURACY", f"{metrics['accuracy'] * 100:.1f}%"),
            ("PRECISION (FAKE)", f"{metrics['precision_fake'] * 100:.1f}%"),
            ("RECALL (FAKE)", f"{metrics['recall_fake'] * 100:.1f}%"),
            ("TESTED DATASET", f"{metrics['dataset_size']:,}")
        ]
        for col, (label, value) in zip([c1, c2, c3, c4], cards):
            with col:
                st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f"**Selected model:** {metrics['model_name']}")
        st.markdown("**Model comparison (held-out test split)**")
        comparison = pd.DataFrame(metrics["comparison"]).set_index("model")
        st.dataframe((comparison * 100).round(2), use_container_width=True)
        st.markdown("**Confusion matrix**")
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual FAKE", "Actual REAL"],
            columns=["Predicted FAKE", "Predicted REAL"]
        )
        st.dataframe(cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Training metrics will appear here after model training.")

elif page == "Analyze News":
    st.subheader("Analyze a News Article")
    st.markdown('<div class="panel">Paste the article title and/or body text below. Longer text usually gives the model more information.</div>', unsafe_allow_html=True)
    news_text = st.text_area("News article text", height=250, placeholder="Paste a news headline and article text here...")
    if st.button("🔍 Analyze Article", use_container_width=False):
        if len(news_text.strip()) < 40:
            st.error("Please enter at least 40 characters so the model has enough text to analyze.")
        else:
            label, confidence = analyse(news_text.strip())
            st.session_state.history.insert(0, {
                "label": label,
                "confidence": round(confidence, 2),
                "preview": news_text.strip()[:110]
            })
            if label == "FAKE":
                st.error(f"Model prediction: {label}")
            else:
                st.success(f"Model prediction: {label}")
            st.metric("Model score", f"{confidence:.1f}%")
            st.caption("This score reflects the model's output for the two dataset labels. It is not proof that the article is true or false.")
            st.info("Always verify important claims using reliable independent sources.")

elif page == "Batch Analysis":
    st.subheader("Batch News Analysis")
    st.write("Upload a CSV containing a `text` column, or both `title` and `text` columns.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            if "text" not in frame.columns:
                st.error("Your CSV needs a column named `text`.")
            else:
                if "title" in frame.columns:
                    combined = (frame["title"].fillna("").astype(str) + " " + frame["text"].fillna("").astype(str)).str.strip()
                else:
                    combined = frame["text"].fillna("").astype(str).str.strip()
                valid = combined.str.len() >= 40
                if valid.sum() == 0:
                    st.warning("No rows contain at least 40 characters of text.")
                else:
                    if st.button("Analyze CSV"):
                        rows = combined[valid].tolist()[:2000]
                        predictions = model.predict(rows)
                        probabilities = model.predict_proba(rows)
                        confidence = probabilities.max(axis=1) * 100
                        result = pd.DataFrame({
                            "article_text": rows,
                            "prediction": predictions,
                            "model_score_percent": confidence.round(2)
                        })
                        st.dataframe(result.head(100), use_container_width=True)
                        st.download_button(
                            "Download results as CSV",
                            data=result.to_csv(index=False).encode("utf-8"),
                            file_name="truthscan_batch_results.csv",
                            mime="text/csv"
                        )
                        st.caption("For performance, this demo processes up to 2,000 valid rows per upload.")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")

elif page == "Scan History":
    st.subheader("Recent Scans")
    if not st.session_state.history:
        st.info("Your scans from this current app session will appear here.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()

elif page == "About Project":
    st.subheader("About TRUTHSCAN AI")
    st.markdown("""
    <div class="panel">
    <h3>Project objective</h3>
    This semester project demonstrates a supervised machine-learning approach to classifying news article text into the FAKE and REAL labels found in a training dataset.
    <h3>Technologies</h3>
    Python, Streamlit, Pandas, Scikit-learn, TF-IDF, Logistic Regression and Multinomial Naive Bayes.
    <h3>How it works</h3>
    Article text is converted into numerical TF-IDF features. Two classifiers are trained and compared using a held-out test split. The better F1-scoring pipeline is saved and used by this app.
    <h3>Important limitations</h3>
    This is an educational text classifier, not a live fact-checking service. It learns patterns in its training dataset and can be wrong, especially on recent events, other languages, satire, or topics unlike its training data. The displayed model score is not the probability that a claim is objectively true.
    </div>
    """, unsafe_allow_html=True)
