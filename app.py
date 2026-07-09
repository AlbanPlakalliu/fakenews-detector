
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
                              f1_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
 
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt_tab', quiet=True)
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 16px; padding: 3rem 2.5rem;
    margin-bottom: 2rem; text-align: center;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 700; color: #f8fafc; margin: 0;
}
.hero p { color: #94a3b8; font-size: 1.1rem; margin-top: 0.75rem; }
.accent { color: #38bdf8; }
 
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.25rem 1.5rem; text-align: center;
}
.metric-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: #0f172a;
}
.metric-card .label { font-size: 0.85rem; color: #64748b; margin-top: 0.25rem; }
 
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #0f172a;
    border-left: 4px solid #38bdf8;
    padding-left: 0.75rem; margin: 2rem 0 1rem 0;
}
 
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-weight: 600;
    font-size: 1rem; width: 100%;
}
</style>
""", unsafe_allow_html=True)
 
# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Fake News <span class="accent">Detector</span></h1>
    <p>Upload your dataset · Select models · Evaluate performance — end to end.</p>
</div>
""", unsafe_allow_html=True)
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
 
    uploaded_file = st.file_uploader(
        "📂 Upload Dataset (CSV)",
        type=["csv"],
        help="CSV must contain: title, text, label columns"
    )
 
    st.markdown("### 🤖 Select Models")
    use_lr   = st.checkbox("Logistic Regression",    value=True)
    use_svm  = st.checkbox("Support Vector Machine", value=True)
    use_rf   = st.checkbox("Random Forest",          value=True)
    use_xgb  = st.checkbox("XGBoost",                value=False)
    use_lstm = st.checkbox("Bi-LSTM",                value=False)
    use_bert = st.checkbox("DistilBERT",             value=False)
 
    st.markdown("---")
    test_size    = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    max_features = st.select_slider("TF-IDF Max Features",
                                     options=[5000, 10000, 20000], value=10000)
 
    run_btn = st.button("🚀 Run Analysis")
 
# ── Preprocessing ─────────────────────────────────────────────────────────────
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
 
# ── Evaluate helper ───────────────────────────────────────────────────────────
def evaluate(name, y_test, y_pred, y_prob=None):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    return dict(Model=name, Accuracy=acc, Precision=prec,
                Recall=rec, F1=f1, AUC=auc,
                y_pred=y_pred, y_prob=y_prob, y_test=y_test)
 
def show_model_results(res, color):
    m1, m2, m3, m4 = st.columns(4)
    for col, metric in zip([m1,m2,m3,m4], ['Accuracy','Precision','Recall','F1']):
        col.metric(metric, f"{res[metric]*100:.2f}%")
 
    col_l, col_r = st.columns(2)
    with col_l:
        cm = confusion_matrix(res['y_test'], res['y_pred'])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real','Fake'],
                    yticklabels=['Real','Fake'], ax=ax)
        ax.set_title(f"Confusion Matrix — {res['Model']}", fontweight='bold')
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
        st.pyplot(fig); plt.close()
 
    with col_r:
        if res['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"AUC = {res['AUC']:.4f}")
            ax.plot([0,1],[0,1],'k--', alpha=0.4)
            ax.set_title(f"ROC Curve — {res['Model']}", fontweight='bold')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
            st.pyplot(fig); plt.close()
 
# ── Main ──────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.markdown("""
    **Expected CSV format:**
    | title | text | label |
    |-------|------|-------|
    | News title... | Article text... | 0 (real) or 1 (fake) |
    """)
    st.stop()
 
df_raw = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')
df_raw['label'] = pd.to_numeric(df_raw['label'], errors='coerce')
df_raw = df_raw.dropna(subset=['label'])
df_raw['label'] = df_raw['label'].astype(int)
 
# ── Dataset Overview ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
 
total = len(df_raw)
real  = (df_raw['label']==0).sum()
fake  = (df_raw['label']==1).sum()
 
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip([c1,c2,c3,c4],
    [f"{total:,}", f"{real:,}", f"{fake:,}", f"{fake/total*100:.1f}%"],
    ["Total Articles","Real (0)","Fake (1)","Fake Ratio"]):
    col.markdown(f"""<div class="metric-card">
        <div class="value">{val}</div>
        <div class="label">{lbl}</div></div>""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
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
    else:
        st.info("No 'source' column found.")
 
if not run_btn:
    st.info("👈 Click **Run Analysis** in the sidebar to start.")
    st.stop()
 
# ── Preprocessing ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)
 
with st.spinner("Tokenising · Removing stopwords · Lemmatising..."):
    df_proc = preprocess(df_raw)
st.success(f"✅ Preprocessed {len(df_proc):,} articles")
 
col1, col2 = st.columns(2)
with col1:
    real_text = ' '.join(df_proc[df_proc['label']==0]['text_processed'])
    wc = WordCloud(width=700,height=350,background_color='white',
                   colormap='Blues',max_words=80).generate(real_text)
    fig,ax = plt.subplots(figsize=(7,3.5))
    ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
    ax.set_title('Most Frequent Words — Real News', fontweight='bold')
    st.pyplot(fig); plt.close()
 
with col2:
    fake_text = ' '.join(df_proc[df_proc['label']==1]['text_processed'])
    wc = WordCloud(width=700,height=350,background_color='white',
                   colormap='Reds',max_words=80).generate(fake_text)
    fig,ax = plt.subplots(figsize=(7,3.5))
    ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
    ax.set_title('Most Frequent Words — Fake News', fontweight='bold')
    st.pyplot(fig); plt.close()
 
# ── TF-IDF ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔢 Feature Extraction — TF-IDF</div>', unsafe_allow_html=True)
 
with st.spinner("Vectorising..."):
    tfidf = TfidfVectorizer(max_features=max_features, max_df=0.95, min_df=2)
    X = tfidf.fit_transform(df_proc['text_processed'])
    y = df_proc['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
 
st.success(f"✅ Matrix: {X.shape[0]:,} × {X.shape[1]:,} | "
           f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
 
# ── ML Models ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
 
all_results = []
colors = ['#3b82f6','#ef4444','#22c55e','#f59e0b','#8b5cf6','#ec4899']
color_idx = 0
 
# Classical models
classical = []
if use_lr:  classical.append(("Logistic Regression",
    LogisticRegression(max_iter=1000,C=1.0,random_state=42)))
if use_svm: classical.append(("Support Vector Machine",
    CalibratedClassifierCV(LinearSVC(C=1.0,max_iter=2000,random_state=42))))
if use_rf:  classical.append(("Random Forest",
    RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)))
 
if use_xgb:
    try:
        from xgboost import XGBClassifier
        classical.append(("XGBoost",
            XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
                          eval_metric='logloss',random_state=42,n_jobs=-1)))
    except ImportError:
        st.warning("XGBoost not installed. Skipping.")
 
for name, model in classical:
    with st.spinner(f"Training {name}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else None
 
    res = evaluate(name, y_test, y_pred, y_prob)
    all_results.append(res)
 
    with st.expander(f"📈 {name} — Accuracy: {res['Accuracy']*100:.2f}%", expanded=True):
        show_model_results(res, colors[color_idx % len(colors)])
    color_idx += 1
 
# Bi-LSTM
if use_lstm:
    st.markdown("#### 🧠 Bi-LSTM (Word2Vec embeddings)")
    with st.spinner("Training Word2Vec + Bi-LSTM (3 stages)..."):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
            from gensim.models import Word2Vec
 
            w2v = Word2Vec(sentences=df_proc['tokens_lemma'].tolist()
                           if 'tokens_lemma' in df_proc.columns
                           else [t.split() for t in df_proc['text_processed']],
                           vector_size=100, window=5, min_count=2,
                           workers=4, epochs=10)
 
            def get_w2v_vec(tokens, model):
                vecs = [model.wv[w] for w in tokens if w in model.wv]
                return np.mean(vecs, axis=0) if vecs else np.zeros(100)
 
            tokens_list = [t.split() for t in df_proc['text_processed']]
            X_w2v = np.array([get_w2v_vec(t, w2v) for t in tokens_list])
            Xtr_w, Xte_w, ytr_w, yte_w = train_test_split(
                X_w2v, y, test_size=test_size, random_state=42, stratify=y)
 
            Xtr_w = Xtr_w.reshape(Xtr_w.shape[0],1,100)
            Xte_w = Xte_w.reshape(Xte_w.shape[0],1,100)
 
            bilstm = Sequential([
                Input(shape=(1,100)),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(64)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            bilstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
 
            full_hist = {'accuracy':[],'val_accuracy':[],'loss':[],'val_loss':[]}
            lrs = [0.001, 0.0005, 0.0001]
 
            for stage, lr in enumerate(lrs, 1):
                st.write(f"Stage {stage}/3 — LR={lr}")
                bilstm.optimizer.learning_rate.assign(lr)
                h = bilstm.fit(Xtr_w, ytr_w, epochs=20, batch_size=256,
                               validation_split=0.1, verbose=0)
                for k in full_hist: full_hist[k].extend(h.history[k])
 
            y_prob_bl = bilstm.predict(Xte_w).flatten()
            y_pred_bl = (y_prob_bl > 0.5).astype(int)
            res = evaluate("Bi-LSTM", yte_w, y_pred_bl, y_prob_bl)
            all_results.append(res)
 
            # Training curves
            fig, axes = plt.subplots(1,2,figsize=(12,4))
            ep = range(1,61)
            for ax,key,title in zip(axes,
                [('accuracy','val_accuracy'),('loss','val_loss')],
                ['Accuracy','Loss']):
                ax.plot(ep, full_hist[key[0]], label='Train', color='steelblue', lw=2)
                ax.plot(ep, full_hist[key[1]], label='Validation', color='crimson', lw=2)
                for x in [20,40]: ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
                ax.set_title(f'Bi-LSTM — {title}', fontweight='bold')
                ax.set_xlabel('Epoch'); ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()
 
            with st.expander(f"📈 Bi-LSTM — Accuracy: {res['Accuracy']*100:.2f}%", expanded=True):
                show_model_results(res, colors[color_idx % len(colors)])
            color_idx += 1
 
        except Exception as e:
            st.error(f"Bi-LSTM failed: {e}")
 
# DistilBERT
if use_bert:
    st.markdown("#### 🤗 DistilBERT Fine-tuned")
    with st.spinner("Fine-tuning DistilBERT on 5,000 samples..."):
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            from torch.optim import AdamW
 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sample = df_proc.sample(min(5000, len(df_proc)), random_state=42)
            Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
                sample['text_processed'].tolist(), sample['label'].values,
                test_size=0.2, random_state=42, stratify=sample['label'].values)
 
            tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
 
            class FNDataset(Dataset):
                def __init__(self, texts, labels):
                    self.texts = texts; self.labels = labels
                def __len__(self): return len(self.texts)
                def __getitem__(self, i):
                    enc = tok(self.texts[i], max_length=128, padding='max_length',
                              truncation=True, return_tensors='pt')
                    return {'input_ids': enc['input_ids'].squeeze(),
                            'attention_mask': enc['attention_mask'].squeeze(),
                            'label': torch.tensor(self.labels[i], dtype=torch.long)}
 
            tr_dl = DataLoader(FNDataset(Xb_tr,yb_tr), batch_size=32, shuffle=True)
            te_dl = DataLoader(FNDataset(Xb_te,yb_te), batch_size=32)
 
            bm = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased', num_labels=2).to(device)
            opt = AdamW(bm.parameters(), lr=2e-5)
 
            bm.train()
            for batch in tr_dl:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbs  = batch['label'].to(device)
                opt.zero_grad()
                out = bm(input_ids=ids, attention_mask=mask, labels=lbs)
                out.loss.backward()
                opt.step()
 
            bm.eval()
            all_p, all_pr, all_l = [], [], []
            with torch.no_grad():
                for batch in te_dl:
                    ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    out = bm(input_ids=ids, attention_mask=mask)
                    pr = torch.softmax(out.logits,1)[:,1].cpu().numpy()
                    p  = out.logits.argmax(1).cpu().numpy()
                    all_p.extend(p); all_pr.extend(pr)
                    all_l.extend(batch['label'].numpy())
 
            res = evaluate("DistilBERT", np.array(all_l),
                           np.array(all_p), np.array(all_pr))
            all_results.append(res)
 
            with st.expander(f"📈 DistilBERT — Accuracy: {res['Accuracy']*100:.2f}%", expanded=True):
                show_model_results(res, colors[color_idx % len(colors)])
            color_idx += 1
 
        except Exception as e:
            st.error(f"DistilBERT failed: {e}")
 
# ── Final Comparison ──────────────────────────────────────────────────────────
if all_results:
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
 
    # All ROC curves
    probs_exist = [r for r in all_results if r['y_prob'] is not None]
    if probs_exist:
        fig, ax = plt.subplots(figsize=(9,6))
        for i,r in enumerate(probs_exist):
            fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
            ax.plot(fpr, tpr, color=colors[i%len(colors)], lw=2,
                    label=f"{r['Model']} (AUC={r['AUC']:.4f})")
        ax.plot([0,1],[0,1],'k--',alpha=0.4)
        ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        st.pyplot(fig); plt.close()
 
st.markdown("---")
st.markdown("<center><small>Fake News Detector — LD7185 Programming for AI · Northumbria University</small></center>",
            unsafe_allow_html=True)
