import streamlit as st
import joblib
import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── NLTK downloads ────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "MindScan AI",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px; font-weight: 600; padding: 8px 20px;
    }
    .metric-card {
        background: #1E2130; border-radius: 10px;
        padding: 16px; text-align: center; border: 1px solid #2E3350;
    }
    .metric-card h3 { color: #7EB3FF; font-size: 28px; margin: 0; }
    .metric-card p  { color: #AAB4C8; font-size: 13px; margin: 4px 0 0 0; }
    .result-box {
        padding: 20px; border-radius: 12px;
        margin: 16px 0; border-left: 6px solid;
    }
    .word-positive { background:#1a3a1a; color:#66ff66;
        padding:4px 8px; border-radius:4px; margin:2px;
        display:inline-block; font-weight:bold; font-size:14px; }
    .word-negative { background:#3a1a1a; color:#ff6666;
        padding:4px 8px; border-radius:4px; margin:2px;
        display:inline-block; font-weight:bold; font-size:14px; }
    .word-neutral  { background:#2a2a2a; color:#aaaaaa;
        padding:4px 8px; border-radius:4px; margin:2px;
        display:inline-block; font-size:14px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'model')

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_PATH, 'best_model.pkl'))
    tfidf = joblib.load(os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl'))
    return model, tfidf

model, tfidf = load_model()
CLASSES = list(model.classes_)

# ── Text cleaning ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

# ── Class info ────────────────────────────────────────────────────────────────
CLASS_INFO = {
    'Normal':               {'color':'#4CAF50','emoji':'😊','risk':'Low',
        'desc':'No mental health concern detected. Post appears positive or neutral.'},
    'Depression':           {'color':'#1565C0','emoji':'😔','risk':'Medium',
        'desc':'Signs of depression — persistent sadness, emptiness, hopelessness.'},
    'Anxiety':              {'color':'#FFA000','emoji':'😰','risk':'Medium',
        'desc':'Signs of anxiety — excessive worry, fear, restlessness, panic attacks.'},
    'Suicidal':             {'color':'#C62828','emoji':'🚨','risk':'HIGH',
        'desc':'URGENT: Signs of suicidal ideation detected. Immediate support recommended.'},
    'Stress':               {'color':'#6A1B9A','emoji':'😤','risk':'Low-Medium',
        'desc':'Signs of stress — feeling overwhelmed, under pressure, burned out.'},
    'Bipolar':              {'color':'#00838F','emoji':'🔄','risk':'Medium',
        'desc':'Signs of bipolar disorder — extreme mood swings between highs and lows.'},
    'Personality disorder': {'color':'#4E342E','emoji':'🌀','risk':'Medium',
        'desc':'Signs of personality disorder — unstable emotions, impulsive behavior.'},
}

# ── LIME explainer ────────────────────────────────────────────────────────────
@st.cache_resource
def get_lime_explainer():
    try:
        from lime.lime_text import LimeTextExplainer
        return LimeTextExplainer(class_names=CLASSES)
    except ImportError:
        return None

def predict_proba_for_lime(texts):
    vecs = tfidf.transform([clean_text(t) for t in texts])
    return model.predict_proba(vecs)

def get_lime_explanation(text, pred_class):
    explainer = get_lime_explainer()
    if explainer is None:
        return None
    try:
        exp = explainer.explain_instance(
            text, predict_proba_for_lime,
            num_features=10,
            labels=[CLASSES.index(pred_class)]
        )
        return exp.as_list(label=CLASSES.index(pred_class))
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px;'>
        <div style='font-size:56px; margin-bottom:8px;'>🧬</div>
        <h2 style='color:#7EB3FF; margin:8px 0 4px 0;'>MindScan AI</h2>
        <p style='color:#AAB4C8; font-size:13px; margin:0;'>
            Mental Health Text Classifier
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model stats ──
    st.markdown("### 📊 Model Stats")
    stats = [
        ("🎯 Accuracy",  "74.92%"),
        ("📈 AUC Score", "0.921"),
        ("📝 Dataset",   "53,043 posts"),
        ("🏷️ Classes",   "7 categories"),
        ("🔤 Features",  "8,000 TF-IDF"),
        ("🤖 Algorithm", "Logistic Reg."),
    ]
    for label, value in stats:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    padding:6px 0; border-bottom:1px solid #2E3350;
                    font-size:13px;'>
            <span style='color:#AAB4C8;'>{label}</span>
            <span style='color:#7EB3FF; font-weight:bold;'>{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tech stack ──
    st.markdown("### 🛠️ Tech Stack")
    techs = ["Python 3.10", "scikit-learn", "NLTK", "TF-IDF",
             "VADER Sentiment", "Streamlit", "Tableau Public"]
    for t in techs:
        st.markdown(f"&nbsp;&nbsp;`{t}`")

    st.markdown("---")
    st.markdown("""
    <p style='color:#555; font-size:11px; text-align:center;'>
    ⚠️ For educational purposes only.<br>
    Not a medical diagnostic tool.
    </p>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding:20px 0 10px 0;'>
    <div style='font-size:52px;'>🧬</div>
    <h1 style='color:#7EB3FF; font-size:42px; margin:8px 0 0 0;'>
        MindScan AI
    </h1>
    <p style='color:#AAB4C8; font-size:16px; margin:8px 0 0 0;'>
        NLP-powered Mental Health Text Classifier
        &nbsp;|&nbsp; Logistic Regression &nbsp;|&nbsp; 74.92% Accuracy
    </p>
</div>
<hr style='border-color:#2E3350; margin:16px 0;'>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
for col, val, label in zip(
    [k1, k2, k3, k4],
    ["74.92%", "0.921", "53,043", "7"],
    ["Test Accuracy", "AUC Score", "Training Posts", "Mental Health Classes"]
):
    col.markdown(f"""
    <div class='metric-card'>
        <h3>{val}</h3>
        <p>{label}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Predict",
    "🧬 Explain (LIME)",
    "📊 Batch Predict",
    "ℹ️ How It Works"
])

# ═══════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════
with tab1:
    st.subheader("Single Text Prediction")

    # Example buttons
    st.markdown("**Quick examples — click to load:**")
    c1, c2, c3, c4 = st.columns(4)
    examples = {
        "dep": "I feel completely empty and hopeless. Nothing makes me happy anymore. I just want to disappear from everything.",
        "anx": "I can't stop worrying about everything. My heart races constantly and I feel like something terrible is always about to happen.",
        "nor": "Had a really productive and wonderful day today! Feeling grateful, happy and positive about everything in life.",
        "sui": "I don't want to be here anymore. Everything feels completely pointless and I see absolutely no reason to continue."
    }
    if c1.button("😔 Depression", use_container_width=True):
        st.session_state['txt'] = examples['dep']
    if c2.button("😰 Anxiety",    use_container_width=True):
        st.session_state['txt'] = examples['anx']
    if c3.button("😊 Normal",     use_container_width=True):
        st.session_state['txt'] = examples['nor']
    if c4.button("🚨 Suicidal",   use_container_width=True):
        st.session_state['txt'] = examples['sui']

    user_input = st.text_area(
        "Enter your text here:",
        value=st.session_state.get('txt', ''),
        height=140,
        placeholder="Type or paste any text here — Reddit post, message, diary entry..."
    )

    col_btn, col_clear = st.columns([4, 1])
    predict_btn = col_btn.button("🔍 Predict Now", type="primary", use_container_width=True)
    if col_clear.button("🗑️ Clear", use_container_width=True):
        st.session_state['txt'] = ''
        st.rerun()

    if predict_btn and user_input.strip():
        with st.spinner("Analyzing text with NLP pipeline..."):
            cleaned    = clean_text(user_input)
            vec        = tfidf.transform([cleaned])
            pred       = model.predict(vec)[0]
            proba      = model.predict_proba(vec)[0]
            confidence = proba.max() * 100
            info       = CLASS_INFO.get(pred, {'color':'#999','emoji':'❓',
                                               'risk':'Unknown','desc':''})

        # ── Result card ──
        st.markdown(f"""
        <div class='result-box' style='background:{info["color"]}18;
             border-left-color:{info["color"]};'>
            <div style='display:flex; align-items:center; gap:16px;'>
                <span style='font-size:48px;'>{info["emoji"]}</span>
                <div>
                    <h2 style='color:{info["color"]}; margin:0; font-size:28px;'>
                        {pred}
                    </h2>
                    <p style='color:#AAB4C8; margin:4px 0;'>
                        Confidence: <b style='color:{info["color"]};'>{confidence:.1f}%</b>
                        &nbsp;|&nbsp; Risk Level:
                        <b style='color:{info["color"]};'>{info["risk"]}</b>
                    </p>
                    <p style='color:#888; margin:0; font-size:14px;'>{info["desc"]}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Crisis alert
        if pred == 'Suicidal' and confidence > 65:
            st.error("""
            🚨 **HIGH RISK — IMMEDIATE SUPPORT RECOMMENDED**

            This text shows strong indicators of suicidal ideation.

            **Crisis Helplines (India):**
            - **iCall:** 9152987821
            - **Vandrevala Foundation:** 1860-2662-345 *(24/7)*
            - **AASRA:** 9820466627
            - **iCall Online Chat:** icallhelpline.org
            """)

        # ── Two column layout ──
        col_chart, col_stats = st.columns([3, 2])

        with col_chart:
            st.markdown("#### 📊 Confidence per Class")
            sorted_idx = np.argsort(proba)[::-1]
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0F1117')
            ax.set_facecolor('#1E2130')
            bar_colors = [CLASS_INFO.get(CLASSES[i], {}).get('color', '#999')
                          for i in sorted_idx]
            bars = ax.barh(
                [CLASSES[i] for i in sorted_idx],
                [proba[i] * 100 for i in sorted_idx],
                color=bar_colors, edgecolor='none', height=0.6
            )
            for bar, idx in zip(bars, sorted_idx):
                ax.text(bar.get_width() + 0.8,
                        bar.get_y() + bar.get_height() / 2,
                        f'{proba[idx]*100:.1f}%',
                        va='center', color='white', fontsize=10)
            ax.set_xlabel('Probability (%)', color='#AAB4C8')
            ax.set_xlim(0, 118)
            ax.tick_params(colors='#AAB4C8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#2E3350')
            ax.spines['left'].set_color('#2E3350')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_stats:
            st.markdown("#### 📝 Text Statistics")
            words  = user_input.split()
            tokens = cleaned.split()
            st.markdown(f"""
            <div style='background:#1E2130; padding:16px; border-radius:10px;
                        border:1px solid #2E3350;'>
                <div style='display:flex; justify-content:space-between;
                            padding:8px 0; border-bottom:1px solid #2E3350;'>
                    <span style='color:#AAB4C8;'>Total Words</span>
                    <span style='color:#7EB3FF; font-weight:bold;'>{len(words)}</span>
                </div>
                <div style='display:flex; justify-content:space-between;
                            padding:8px 0; border-bottom:1px solid #2E3350;'>
                    <span style='color:#AAB4C8;'>Characters</span>
                    <span style='color:#7EB3FF; font-weight:bold;'>{len(user_input)}</span>
                </div>
                <div style='display:flex; justify-content:space-between;
                            padding:8px 0; border-bottom:1px solid #2E3350;'>
                    <span style='color:#AAB4C8;'>Clean Tokens</span>
                    <span style='color:#7EB3FF; font-weight:bold;'>{len(tokens)}</span>
                </div>
                <div style='display:flex; justify-content:space-between;
                            padding:8px 0; border-bottom:1px solid #2E3350;'>
                    <span style='color:#AAB4C8;'>Predicted Class</span>
                    <span style='color:{info["color"]}; font-weight:bold;'>{pred}</span>
                </div>
                <div style='display:flex; justify-content:space-between;
                            padding:8px 0; border-bottom:1px solid #2E3350;'>
                    <span style='color:#AAB4C8;'>Confidence</span>
                    <span style='color:{info["color"]}; font-weight:bold;'>{confidence:.1f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between;
                            padding:8px 0;'>
                    <span style='color:#AAB4C8;'>Risk Level</span>
                    <span style='color:{info["color"]}; font-weight:bold;'>{info["risk"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Go to **Explain (LIME)** tab to see which words caused this prediction!")

    elif predict_btn:
        st.warning("⚠️ Please enter some text before predicting.")

# ═══════════════════════════════════════════════════════
# TAB 2 — LIME EXPLAINABILITY
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("🧬 LIME — Why did the model predict this?")
    st.markdown("""
    LIME (Local Interpretable Model-agnostic Explanations) shows you **exactly which words
    pushed the model toward or away from each prediction**.

    - 🟢 **Green words** — pushed the prediction TOWARD the predicted class
    - 🔴 **Red words** — pushed the prediction AWAY from the predicted class
    """)

    lime_input = st.text_area(
        "Enter text to explain:",
        height=120,
        placeholder="Paste any text here to see word-level explanation...",
        key="lime_input"
    )

    explain_btn = st.button("🧬 Explain Prediction", type="primary", use_container_width=True)

    if explain_btn and lime_input.strip():
        with st.spinner("Running LIME analysis... (this takes 10-15 seconds)"):
            # Get prediction first
            cleaned    = clean_text(lime_input)
            vec        = tfidf.transform([cleaned])
            pred       = model.predict(vec)[0]
            proba      = model.predict_proba(vec)[0]
            confidence = proba.max() * 100
            info       = CLASS_INFO.get(pred, {'color':'#999','emoji':'❓',
                                               'risk':'Unknown','desc':''})

            # Show prediction
            st.markdown(f"""
            <div class='result-box' style='background:{info["color"]}18;
                 border-left-color:{info["color"]}; margin-bottom:16px;'>
                <b style='color:{info["color"]}; font-size:20px;'>
                    {info["emoji"]} Predicted: {pred}
                </b>
                <span style='color:#AAB4C8; margin-left:16px;'>
                    Confidence: {confidence:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Get LIME explanation
            explanation = get_lime_explanation(lime_input, pred)

            if explanation:
                st.markdown(f"#### Why **{pred}**? — Top influential words:")

                # Sort by absolute importance
                pos_words = [(w, s) for w, s in explanation if s > 0]
                neg_words = [(w, s) for w, s in explanation if s < 0]

                col_pos, col_neg = st.columns(2)

                with col_pos:
                    st.markdown(f"##### 🟢 Pushed TOWARD {pred}")
                    if pos_words:
                        for word, score in sorted(pos_words, key=lambda x: -x[1])[:6]:
                            bar_len = int(abs(score) * 200)
                            bar_len = min(bar_len, 100)
                            st.markdown(f"""
                            <div style='display:flex; align-items:center;
                                        gap:10px; margin:6px 0;'>
                                <span class='word-positive'>{word}</span>
                                <div style='background:#1a3a1a; height:12px;
                                            border-radius:6px; flex:1;'>
                                    <div style='background:#4CAF50; height:12px;
                                                border-radius:6px;
                                                width:{bar_len}%;'></div>
                                </div>
                                <span style='color:#4CAF50; font-size:12px;
                                             min-width:50px;'>+{score:.3f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No strong positive words found.")

                with col_neg:
                    st.markdown(f"##### 🔴 Pushed AWAY from {pred}")
                    if neg_words:
                        for word, score in sorted(neg_words, key=lambda x: x[1])[:6]:
                            bar_len = int(abs(score) * 200)
                            bar_len = min(bar_len, 100)
                            st.markdown(f"""
                            <div style='display:flex; align-items:center;
                                        gap:10px; margin:6px 0;'>
                                <span class='word-negative'>{word}</span>
                                <div style='background:#3a1a1a; height:12px;
                                            border-radius:6px; flex:1;'>
                                    <div style='background:#C62828; height:12px;
                                                border-radius:6px;
                                                width:{bar_len}%;'></div>
                                </div>
                                <span style='color:#C62828; font-size:12px;
                                             min-width:50px;'>{score:.3f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No strong negative words found.")

                # Highlighted text view
                st.markdown("#### 📝 Highlighted Text View")
                pos_set = {w for w, _ in pos_words}
                neg_set = {w for w, _ in neg_words}

                highlighted = []
                for word in lime_input.split():
                    clean_w = re.sub(r'[^a-z]', '', word.lower())
                    lem_w   = LEMMATIZER.lemmatize(clean_w)
                    if lem_w in pos_set or clean_w in pos_set:
                        highlighted.append(f"<span class='word-positive'>{word}</span>")
                    elif lem_w in neg_set or clean_w in neg_set:
                        highlighted.append(f"<span class='word-negative'>{word}</span>")
                    else:
                        highlighted.append(f"<span class='word-neutral'>{word}</span>")

                st.markdown(
                    "<div style='background:#1E2130; padding:16px; border-radius:10px;"
                    "line-height:2.2; border:1px solid #2E3350;'>"
                    + ' '.join(highlighted) + "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ LIME not available. Install it with: `pip install lime`")
                st.code("pip install lime")

    elif explain_btn:
        st.warning("⚠️ Please enter some text to explain.")

# ═══════════════════════════════════════════════════════
# TAB 3 — BATCH PREDICT
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Batch Prediction — Predict Multiple Texts at Once")
    st.markdown("""
    Upload a CSV file with a column named **`text`** or **`statement`**.
    The model will predict the mental health category for every row instantly.
    """)

    uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"✅ File loaded: **{df_batch.shape[0]} rows**, {df_batch.shape[1]} columns")
        st.dataframe(df_batch.head(3), use_container_width=True)

        text_col = None
        for col in ['text', 'statement', 'content', 'post', 'message']:
            if col in df_batch.columns:
                text_col = col
                break
        if text_col is None:
            text_col = df_batch.columns[0]
            st.info(f"ℹ️ Using first column as text: **{text_col}**")

        if st.button("🔍 Predict All Rows", type="primary", use_container_width=True):
            with st.spinner(f"Predicting {df_batch.shape[0]} rows..."):
                df_batch['cleaned']    = df_batch[text_col].apply(clean_text)
                vecs                   = tfidf.transform(df_batch['cleaned'])
                df_batch['prediction'] = model.predict(vecs)
                df_batch['confidence'] = (model.predict_proba(vecs)
                                          .max(axis=1) * 100).round(2)
                df_batch['confidence'] = df_batch['confidence'].astype(str) + '%'
                df_batch               = df_batch.drop(columns=['cleaned'])

            st.success(f"✅ Done! Predicted **{df_batch.shape[0]} rows**.")
            st.dataframe(
                df_batch[['prediction','confidence', text_col]],
                use_container_width=True
            )

            col_chart2, col_pie = st.columns(2)

            with col_chart2:
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor('#0F1117')
                ax.set_facecolor('#1E2130')
                counts = df_batch['prediction'].value_counts()
                colors = [CLASS_INFO.get(c,{}).get('color','#999') for c in counts.index]
                ax.bar(counts.index, counts.values, color=colors, edgecolor='none')
                ax.set_title('Prediction Distribution', color='white', fontweight='bold')
                ax.tick_params(colors='#AAB4C8', rotation=30)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#2E3350')
                ax.spines['left'].set_color('#2E3350')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_pie:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor('#0F1117')
                ax2.set_facecolor('#1E2130')
                ax2.pie(counts.values, labels=counts.index,
                        colors=colors, autopct='%1.1f%%',
                        textprops={'color': 'white', 'fontsize': 9},
                        wedgeprops={'edgecolor': '#0F1117', 'linewidth': 2})
                ax2.set_title('Class Proportions', color='white', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

            csv_out = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results as CSV",
                csv_out, "predictions_output.csv", "text/csv",
                use_container_width=True
            )

# ═══════════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ═══════════════════════════════════════════════════════
with tab4:
    st.subheader("ℹ️ How This App Works")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### 🔄 NLP Pipeline
        ```
        Raw Text Input
              ↓
        Lowercase + Remove URLs
              ↓
        Remove Punctuation & Numbers
              ↓
        Remove Stopwords (NLTK)
              ↓
        Lemmatization (WordNet)
              ↓
        TF-IDF Vectorization (8,000 features)
              ↓
        Logistic Regression Classifier
              ↓
        Prediction + Probability Scores
        ```

        ### 📊 Model Comparison
        | Model | Accuracy | AUC |
        |---|---|---|
        | **Logistic Regression** | **74.92%** | **0.921** |
        | Linear SVM | 74.70% | 0.918 |
        | Naive Bayes | 68.92% | 0.890 |
        | Random Forest | 65.43% | 0.872 |

        > Logistic Regression performs best on TF-IDF data
        because text in high-dimensional space is linearly separable.
        """)

    with col_b:
        st.markdown("""
        ### 🏷️ Mental Health Classes
        | Class | Risk | Description |
        |---|---|---|
        | 😊 Normal | Low | No concern |
        | 😔 Depression | Medium | Sadness, hopelessness |
        | 😰 Anxiety | Medium | Worry, panic |
        | 🚨 Suicidal | **HIGH** | Crisis indicators |
        | 😤 Stress | Low-Med | Overwhelmed |
        | 🔄 Bipolar | Medium | Mood swings |
        | 🌀 Personality | Medium | Unstable emotions |

        ### 📁 Dataset
        | Property | Value |
        |---|---|
        | Source | Kaggle — Reddit Posts |
        | Total Records | 53,043 |
        | Classes | 7 |
        | Balance | Roughly equal |
        | Missing Values | 0 |
        | Avg Post Length | ~74 words |

        ### 🧬 LIME Explainability
        LIME (Local Interpretable Model-agnostic Explanations)
        perturbs the input text and observes how predictions change
        to identify which words are most influential.
        """)

    st.markdown("---")
    st.markdown("""
    ---
    ### ⚠️ Disclaimer
    > This application is built for **research and awareness purposes only**.
    It is **NOT a clinical diagnostic tool** and must **NOT** be used as a
    substitute for professional mental health assessment or treatment.
    """)
