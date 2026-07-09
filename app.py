import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
 
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt_tab', quiet=True)
 
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide")
 
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 16px; padding: 2.5rem;
    text-align: center; margin-bottom: 2rem;
}
.hero h1 { color: #f8fafc; font-size: 2.5rem; margin: 0; }
.hero p  { color: #94a3b8; margin-top: 0.5rem; }
.accent  { color: #38bdf8; }
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #0f172a;
    border-left: 4px solid #38bdf8;
    padding-left: 0.75rem; margin: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class="hero">
    <h1>🔍 Fake News <span class="accent">Detector</span></h1>
    <p>Upload your datasets · Select models · Evaluate performance</p>
</div>
""", unsafe_allow_html=True)
 
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "📂 Upload Datasets (CSV)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files with columns: title, text, label"
    )
    st.markdown("### 🤖 Select Models")
    use_lr  = st.checkbox("Logistic Regression",    value=True)
    use_svm = st.checkbox("Support Vector Machine", value=True)
    use_rf  = st.checkbox("Random Forest",          value=True)
    test_size    = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    max_features = st.select_slider("TF-IDF Max Features",
                                     options=[5000, 10000, 20000], value=10000)
    run_btn = st.button("🚀 Run Analysis")
 
@st.cache_data
def preprocess(df):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens
                  if t not in stop_words and len(t) > 2]
        return ' '.join(tokens)
    df = df.copy()
    df['content']        = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['text_processed'] = df['content'].apply(clean)
    return df
 
def evaluate(name, y_test, y_pred, y_prob, color):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    with st.expander(f"📈 {name} — Accuracy: {acc*100:.2f}%", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",  f"{acc*100:.2f}%")
        c2.metric("Precision", f"{prec*100:.2f}%")
        c3.metric("Recall",    f"{rec*100:.2f}%")
        c4.metric("F1-Score",  f"{f1*100:.2f}%")
        col_l, col_r = st.columns(2)
        with col_l:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4,3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Real','Fake'],
                        yticklabels=['Real','Fake'], ax=ax)
            ax.set_title(f'Confusion Matrix — {name}', fontweight='bold')
            ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
            st.pyplot(fig); plt.close()
        with col_r:
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig, ax = plt.subplots(figsize=(4,3.5))
                ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.4f}")
                ax.plot([0,1],[0,1],'k--', alpha=0.4)
                ax.set_title(f'ROC Curve — {name}', fontweight='bold')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                st.pyplot(fig); plt.close()
    return dict(Model=name, Accuracy=acc, Precision=prec,
                Recall=rec, F1=f1, AUC=auc, y_prob=y_prob)
 
if not uploaded_files:
    st.info("👈 Upload one or more CSV files from the sidebar to get started.")
    st.markdown("""
    **Expected CSV format:**
    | title | text | label |
    |-------|------|-------|
    | News title... | Article text... | 0 (real) or 1 (fake) |
 
    💡 **You can upload multiple CSV files at once** — they will be automatically combined!
    """)
    st.stop()
 
dfs = []
file_info = []
for f in uploaded_files:
    try:
        df_temp = pd.read_csv(f, on_bad_lines='skip', engine='python')
        df_temp['label'] = pd.to_numeric(df_temp['label'], errors='coerce')
        df_temp = df_temp.dropna(subset=['label'])
        df_temp['label'] = df_temp['label'].astype(int)
        dfs.append(df_temp)
        file_info.append({'File': f.name, 'Articles': len(df_temp),
                          'Real': (df_temp['label']==0).sum(),
                          'Fake': (df_temp['label']==1).sum()})
    except Exception as e:
        st.warning(f"Could not read {f.name}: {e}")
 
if not dfs:
    st.error("No valid files could be loaded.")
    st.stop()
 
df_raw = pd.concat(dfs, ignore_index=True)
df_raw = df_raw.drop_duplicates(subset=['title']).reset_index(drop=True)
 
if len(uploaded_files) > 1:
    st.markdown('<div class="section-header">📁 Uploaded Files</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)
    st.success(f"✅ Combined {len(uploaded_files)} files → {len(df_raw):,} unique articles")
 
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
total = len(df_raw)
real  = (df_raw['label']==0).sum()
fake  = (df_raw['label']==1).sum()
 
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Articles", f"{total:,}")
c2.metric("Real (0)", f"{real:,}")
c3.metric("Fake (1)", f"{fake:,}")
c4.metric("Fake Ratio", f"{fake/total*100:.1f}%")
 
col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(5,3.5))
    ax.bar(['Real','Fake'], [real,fake], color=['#3b82f6','#ef4444'])
    ax.set_title('Class Distribution', fontweight='bold')
    ax.set_ylabel('Articles')
    st.pyplot(fig); plt.close()
with col_b:
    if 'source' in df_raw.columns:
        top = df_raw['source'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.barh(top.index[::-1], top.values[::-1], color='#0ea5e9')
        ax.set_title('Top Sources', fontweight='bold')
        st.pyplot(fig); plt.close()
 
if not run_btn:
    st.info("👈 Click **Run Analysis** to start training.")
    st.stop()
 
st.markdown('<div class="section-header">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)
with st.spinner("Tokenising · Removing stopwords · Lemmatising..."):
    df_proc = preprocess(df_raw)
st.success(f"✅ Preprocessed {len(df_proc):,} articles")
 
col1, col2 = st.columns(2)
with col1:
    real_text = ' '.join(df_proc[df_proc['label']==0]['text_processed'].tolist())
    if real_text.strip():
        wc = WordCloud(width=700,height=350,background_color='white',
                       colormap='Blues',max_words=80).generate(real_text)
        fig,ax = plt.subplots(figsize=(7,3.5))
        ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
        ax.set_title('Most Frequent Words — Real News', fontweight='bold')
        st.pyplot(fig); plt.close()
with col2:
    fake_text = ' '.join(df_proc[df_proc['label']==1]['text_processed'].tolist())
    if fake_text.strip():
        wc = WordCloud(width=700,height=350,background_color='white',
                       colormap='Reds',max_words=80).generate(fake_text)
        fig,ax = plt.subplots(figsize=(7,3.5))
        ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
        ax.set_title('Most Frequent Words — Fake News', fontweight='bold')
        st.pyplot(fig); plt.close()
 
st.markdown('<div class="section-header">🔢 Feature Extraction — TF-IDF</div>', unsafe_allow_html=True)
with st.spinner("Vectorising with TF-IDF..."):
    tfidf = TfidfVectorizer(max_features=max_features, max_df=0.95, min_df=2)
    X = tfidf.fit_transform(df_proc['text_processed'])
    y = df_proc['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
st.success(f"✅ Matrix: {X.shape[0]:,} × {X.shape[1]:,} | "
           f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
 
st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
models = []
if use_lr:  models.append(("Logistic Regression",
    LogisticRegression(max_iter=1000,C=1.0,random_state=42), '#3b82f6'))
if use_svm: models.append(("Support Vector Machine",
    CalibratedClassifierCV(LinearSVC(C=1.0,max_iter=2000,random_state=42)), '#ef4444'))
if use_rf:  models.append(("Random Forest",
    RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1), '#22c55e'))
 
if not models:
    st.warning("Please select at least one model.")
    st.stop()
 
all_results = []
for name, model, color in models:
    with st.spinner(f"Training {name}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else None
    all_results.append(evaluate(name, y_test, y_pred, y_prob, color))
 
st.markdown('<div class="section-header">🏆 Model Comparison</div>', unsafe_allow_html=True)
results_df = pd.DataFrame([{
    'Model':     r['Model'],
    'Accuracy':  f"{r['Accuracy']*100:.2f}%",
    'Precision': f"{r['Precision']*100:.2f}%",
    'Recall':    f"{r['Recall']*100:.2f}%",
    'F1-Score':  f"{r['F1']*100:.2f}%",
    'ROC-AUC':   f"{r['AUC']:.4f}" if r['AUC'] else 'N/A'
} for r in all_results])
st.dataframe(results_df, use_container_width=True, hide_index=True)
 
best = max(all_results, key=lambda x: x['F1'])
st.success(f"🥇 Best Model: **{best['Model']}** — "
           f"F1: {best['F1']*100:.2f}% | Accuracy: {best['Accuracy']*100:.2f}%")
 
colors = ['#3b82f6','#ef4444','#22c55e']
fig, ax = plt.subplots(figsize=(8,5))
for i, r in enumerate(all_results):
    if r['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, color=colors[i%len(colors)], lw=2,
                label=f"{r['Model']} (AUC={r['AUC']:.4f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
st.pyplot(fig); plt.close()
 
st.markdown("---")
st.markdown(
    "<center><small>Fake News Detector — LD7185 Programming for AI · Northumbria University</small></center>",
    unsafe_allow_html=True
)
 
