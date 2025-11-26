import os
import time
import math
import joblib
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve,auc
import pyshark
from sklearn.model_selection import train_test_split


# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Constants
CSV_DIR = "converted_csvs"
OUTPUT_DIR = "comparison_outputs"

# Helper Functions
def gini(arr):
    if len(arr) == 0: return 0.0
    vals, cnts = np.unique(arr, return_counts=True)
    p = cnts / cnts.sum()
    return 1.0 - np.sum(p**2)

def best_split_for_feature(X_col, y_col, max_cands=25):
    vals = np.unique(X_col)
    if len(vals) > max_cands:
        thresholds = np.percentile(X_col, np.linspace(1,99, max_cands))
    else:
        thresholds = vals
    best_gain = 0.0
    best_thresh = None
    current_impurity = gini(y_col)
    n = len(y_col)
    for t in thresholds:
        left_mask = X_col <= t
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        left_imp = gini(y_col[left_mask])
        right_imp = gini(y_col[right_mask])
        weighted = (left_mask.sum()/n)*left_imp + (right_mask.sum()/n)*right_imp
        gain = current_impurity - weighted
        if gain > best_gain:
            best_gain = gain
            best_thresh = t
    return best_gain, best_thresh

def find_best_split(X_block, y_block, feature_indices):
    best_f = None; best_t = None; best_gain = 0.0
    for fi in feature_indices:
        gain, thresh = best_split_for_feature(X_block[:,fi], y_block)
        if gain > best_gain:
            best_gain = gain; best_f = fi; best_t = thresh
    return best_f, best_t, best_gain

def build_tree(X_block, y_block, depth=0, max_depth=6, min_samples_split=10, m_features=None):
    if len(np.unique(y_block)) == 1:
        return {'leaf': True, 'prediction': int(y_block[0])}
    if depth >= max_depth or len(y_block) < min_samples_split:
        vals, cnts = np.unique(y_block, return_counts=True)
        return {'leaf': True, 'prediction': int(vals[np.argmax(cnts)])}
    n_features = X_block.shape[1]
    m = max(1, int(np.sqrt(n_features))) if m_features is None else m_features
    feat_indices = np.random.choice(n_features, m, replace=False)
    f, t, gain = find_best_split(X_block, y_block, feat_indices)
    if f is None or gain == 0:
        vals, cnts = np.unique(y_block, return_counts=True)
        return {'leaf': True, 'prediction': int(vals[np.argmax(cnts)])}
    left_mask = X_block[:,f] <= t
    right_mask = ~left_mask
    left_node = build_tree(X_block[left_mask], y_block[left_mask], depth+1, max_depth, min_samples_split, m_features)
    right_node = build_tree(X_block[right_mask], y_block[right_mask], depth+1, max_depth, min_samples_split, m_features)
    return {'leaf': False, 'feature': int(f), 'threshold': float(t), 'left': left_node, 'right': right_node}

def predict_tree(tree, sample):
    node = tree
    while not node['leaf']:
        if sample[node['feature']] <= node['threshold']:
            node = node['left']
        else:
            node = node['right']
    return node['prediction']

def train_forest(Xtr, ytr, n_trees=5):
    forest = []
    n = Xtr.shape[0]
    for i in range(n_trees):
        idxs = np.random.choice(n, n, replace=True)
        Xs = Xtr[idxs]
        ys = ytr[idxs]
        tree = build_tree(Xs, ys)
        forest.append(tree)
    return forest

def predict_forest(forest, Xdata):
    preds = []
    for i in range(Xdata.shape[0]):
        votes = [predict_tree(tree, Xdata[i]) for tree in forest]
        vals, cnts = np.unique(votes, return_counts=True)
        preds.append(int(vals[np.argmax(cnts)]))
    return np.array(preds)

class ManualNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}
        for c in self.classes:
            Xc = X[y == c]
            self.mean[c] = np.mean(Xc, axis=0)
            self.var[c] = np.var(Xc, axis=0) + 1e-9
            self.prior[c] = Xc.shape[0] / X.shape[0]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = np.clip(self.var[class_idx], 1e-9, None)
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.prior[c] + 1e-9)
            pdf = self._pdf(c, x)
            pdf = np.clip(pdf, 1e-9, None)
            conditional = np.sum(np.log(pdf))
            posteriors.append(prior + conditional)
        return self.classes[np.argmax(posteriors)]

    def predict(self, Xdata):
        return np.array([self._predict_single(Xdata[i]) for i in range(Xdata.shape[0])])

def confusion(y_true, y_pred):
    tp = int(((y_true==1)&(y_pred==1)).sum())
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    return tp, tn, fp, fn

def precision_recall_f1(tp, tn, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def robust_preprocess(X_df, clip_low=1.0, clip_high=99.5):
    X = X_df.copy().astype(float)
    X_log = np.log1p(X)
    low_q = X_log.quantile(clip_low/100.0)
    high_q = X_log.quantile(clip_high/100.0)
    X_clipped = X_log.clip(lower=low_q, upper=high_q, axis=1)
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clipped), columns=X_clipped.columns)
    mins = X_df.min()
    maxs = X_df.max()
    range_ = (maxs - mins).replace(0, 1.0)
    return X_scaled, mins, range_, scaler, X_clipped

def explain_rf(sample, tree, feature_names, depth_limit=5):
    node = tree
    path = []
    depth = 0
    while not node['leaf'] and depth < depth_limit:
        feat = feature_names[node['feature']]
        thresh = node['threshold']
        if sample[node['feature']] <= thresh:
            decision = f"{feat} ≤ {thresh:.2f}"
            node = node['left']
        else:
            decision = f"{feat} > {thresh:.2f}"
            node = node['right']
        path.append(decision)
        depth += 1
    return path, node['prediction']

def explain_nb(sample, nb_model, feature_names):
    expl = []
    for i, feat in enumerate(feature_names):
        m0, m1 = nb_model.mean[0][i], nb_model.mean[1][i]
        v0, v1 = nb_model.var[0][i], nb_model.var[1][i]
        d0 = abs(sample[i] - m0) / math.sqrt(v0)
        d1 = abs(sample[i] - m1) / math.sqrt(v1)
        expl.append((feat, d0, d1))
    contribs = []
    for feat, d0, d1 in expl:
        if d1 < d0:
            contribs.append(f"{feat} → Attack pattern")
        else:
            contribs.append(f"{feat} → Normal pattern")
    return contribs

# Main App
st.markdown('<h1 class="main-header">🛡 Network Intrusion Detection System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙ Configuration")
    
    st.subheader("📁 Dataset")
    csv_dir_input = st.text_input("CSV Directory", value=CSV_DIR)
    
    st.subheader("🌲 Random Forest")
    n_trees = st.slider("Number of Trees", 1, 20, 5)
    max_depth = st.slider("Max Depth", 3, 15, 6)
    min_samples_split = st.slider("Min Samples Split", 2, 50, 10)
    
    st.subheader("📡 Live Capture")
    capture_interface = st.text_input("Interface", value="Wi-Fi")
    capture_duration = st.slider("Duration (seconds)", 10, 120, 30)
    live_show_first = st.slider("Show first N packets", 5, 50, 12)
    
    st.divider()
    st.info("📊 All outputs saved to comparison_outputs/")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & Training", 
    "📈 Evaluation", 
    "🔍 Explainability", 
    "📡 Live Detection",
    "💾 Models"
])

# Tab 1: Data & Training
with tab1:
    st.header("📊 Dataset Loading & Model Training")
    
    if st.button("🔍 Load Dataset & Train Models", type="primary", use_container_width=True):
        with st.spinner("Loading dataset..."):
            # Find CSV files
            csv_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(csv_dir_input)
                for f in files if f.lower().endswith(".csv")
            ]
            
            if not csv_files:
                st.error(f"❌ No CSV files found in '{csv_dir_input}'")
                st.stop()
            
            csv_files = sorted(csv_files, key=lambda p: os.path.getsize(p), reverse=True)
            chosen_csv = csv_files[0]
            st.success(f"✅ Using CSV: {chosen_csv}")
            
            # Load data
            df = pd.read_csv(chosen_csv)
            st.info(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Display sample
            with st.expander("📋 View Data Sample"):
                st.dataframe(df.head(10))
            
            # Label detection
            label_col = None
            for c in df.columns:
                if c.lower() in ['label','attack','class','category','result','type']:
                    label_col = c
                    break
            if not label_col:
                label_col = df.columns[-1]
            
            st.info(f"Detected label column: {label_col}")
            
            def map_label(v):
                if pd.isna(v): return 0
                s = str(v).lower()
                if 'benign' in s or 'normal' in s or s in ['0','none','no']:
                    return 0
                return 1
            
            df['label_bin'] = df[label_col].apply(map_label).astype(int)
            label_counts = df['label_bin'].value_counts().to_dict()
            
            col_a, col_b = st.columns(2)
            col_a.metric("Normal Traffic", label_counts.get(0, 0))
            col_b.metric("Attack Traffic", label_counts.get(1, 0))
            
            # Feature selection
            def find_col(kw_list):
                for c in df.columns:
                    for kw in kw_list:
                        if kw in c.lower():
                            return c
                return None
            
            protocol_col = find_col(['protocol','proto'])
            packet_len_col = find_col(['len','length','bytes','fwd packets length total','fwd packets length'])
            tcp_flags_col = find_col(['tcp.flags','tcpflag','psh','psh flags','fwd psh flags','fwd_psh_flags'])
            feat_cols = [c for c in [protocol_col, packet_len_col, tcp_flags_col] if c]
            
            if not feat_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in ['label_bin']]
                feat_cols = numeric_cols[:3]
            
            st.success(f"✅ Selected features: {', '.join(feat_cols)}")
            
            # Prepare X, y
            X_raw = df[feat_cols].copy()
            y = df['label_bin'].values.astype(int)
            
            if protocol_col and protocol_col in X_raw.columns and X_raw[protocol_col].dtype == object:
                def proto_to_num(v):
                    s = str(v).lower()
                    if 'tcp' in s: return 6
                    if 'udp' in s: return 17
                    if 'icmp' in s: return 1
                    try: return int(float(s))
                    except: return 0
                X_raw[protocol_col] = X_raw[protocol_col].apply(proto_to_num)
            
            X_raw = X_raw.fillna(0).astype(float)
            
            # Normalization
            st.info("🔄 Applying robust normalization...")
            X_norm, mins, range_, scaler, X_clipped = robust_preprocess(X_raw)
            joblib.dump(scaler, os.path.join(OUTPUT_DIR, "robust_scaler.pkl"))
            
            # Visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            sns.boxplot(data=X_raw, ax=axes[0])
            axes[0].set_title("Before Normalization")
            axes[0].tick_params(axis='x', rotation=45)
            
            sns.boxplot(data=X_clipped, ax=axes[1])
            axes[1].set_title("After Log+Clip")
            axes[1].tick_params(axis='x', rotation=45)
            
            sns.boxplot(data=X_norm, ax=axes[2])
            axes[2].set_title("After Robust Normalization")
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.savefig(os.path.join(OUTPUT_DIR, "normalization_process.png"))
            plt.close()
            
            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(
            X_norm.values,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=True
            )

            st.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
            
            # Train Random Forest
            st.subheader("🌲 Training Random Forest")
            progress_bar = st.progress(0)
            forest = []
            n_samples = X_train.shape[0]
            for i in range(n_trees):
                idxs = np.random.choice(n_samples, n_samples, replace=True)
                Xs = X_train[idxs]
                ys = y_train[idxs]
                tree = build_tree(Xs, ys, max_depth=max_depth, min_samples_split=min_samples_split)
                forest.append(tree)
                progress_bar.progress((i + 1) / n_trees)
                st.text(f"Built tree {i+1}/{n_trees}")
            st.success(f"✅ Random Forest trained with {n_trees} trees")
            
            # Train Naive Bayes
            st.subheader("🤖 Training Naive Bayes")
            nb = ManualNaiveBayes()
            nb.fit(X_train, y_train)
            st.success("✅ Naive Bayes trained")
            
            # Predictions
            st.info("Making predictions on test set...")
            y_rf_pred = predict_forest(forest, X_test)
            y_nb_pred = nb.predict(X_test)
            
            # Metrics
            rf_acc = (y_rf_pred == y_test).mean()
            nb_acc = (y_nb_pred == y_test).mean()
            
            rf_tp, rf_tn, rf_fp, rf_fn = confusion(y_test, y_rf_pred)
            nb_tp, nb_tn, nb_fp, nb_fn = confusion(y_test, y_nb_pred)
            
            rf_prec, rf_rec, rf_f1 = precision_recall_f1(rf_tp, rf_tn, rf_fp, rf_fn)
            nb_prec, nb_rec, nb_f1 = precision_recall_f1(nb_tp, nb_tn, nb_fp, nb_fn)
            
            # Display results
            st.success("🎯 Training Complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🌲 Random Forest Accuracy", f"{rf_acc*100:.2f}%")
            with col2:
                st.metric("🤖 Naive Bayes Accuracy", f"{nb_acc*100:.2f}%")
            
            # Save to session state
            st.session_state.models_trained = True
            st.session_state.forest = forest
            st.session_state.nb = nb
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_rf_pred = y_rf_pred
            st.session_state.y_nb_pred = y_nb_pred
            st.session_state.feat_cols = feat_cols
            st.session_state.mins = mins
            st.session_state.range_ = range_
            st.session_state.protocol_col = protocol_col
            st.session_state.label_col = label_col
            st.session_state.rf_metrics = (rf_acc, rf_prec, rf_rec, rf_f1, rf_tp, rf_tn, rf_fp, rf_fn)
            st.session_state.nb_metrics = (nb_acc, nb_prec, nb_rec, nb_f1, nb_tp, nb_tn, nb_fp, nb_fn)
            st.session_state.scaler = scaler
            
            # Save models
            joblib.dump({'forest': forest, 'feature_columns': feat_cols, 'label_column': label_col}, 
                       os.path.join(OUTPUT_DIR, "rf_manual_joblib.pkl"))
            joblib.dump({'nb': nb, 'feature_columns': feat_cols, 'label_column': label_col}, 
                       os.path.join(OUTPUT_DIR, "nb_manual_joblib.pkl"))
            
            st.balloons()
            st.success("🎉 Models trained and saved successfully!")
    
    if not st.session_state.models_trained:
        st.info("Instructions:\n\n1. Place CSV files in the directory\n2. Configure parameters in sidebar\n3. Click 'Load & Train'\n4. View results in other tabs")

# Tab 2: Evaluation
with tab2:
    st.header("📈 Model Evaluation & Comparison")
    
    if not st.session_state.models_trained:
        st.warning("⚠ Please train models first in the 'Data & Training' tab")
    else:
        rf_acc, rf_prec, rf_rec, rf_f1, rf_tp, rf_tn, rf_fp, rf_fn = st.session_state.rf_metrics
        nb_acc, nb_prec, nb_rec, nb_f1, nb_tp, nb_tn, nb_fp, nb_fn = st.session_state.nb_metrics
        
        # Metrics Overview
        st.subheader("📊 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌲 Random Forest")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Accuracy", f"{rf_acc*100:.2f}%")
            metric_col2.metric("F1-Score", f"{rf_f1:.3f}")
            metric_col1.metric("Precision", f"{rf_prec:.3f}")
            metric_col2.metric("Recall", f"{rf_rec:.3f}")
        
        with col2:
            st.markdown("### 🤖 Naive Bayes")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Accuracy", f"{nb_acc*100:.2f}%")
            metric_col2.metric("F1-Score", f"{nb_f1:.3f}")
            metric_col1.metric("Precision", f"{nb_prec:.3f}")
            metric_col2.metric("Recall", f"{nb_rec:.3f}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(['Random Forest','Naive Bayes'], [rf_acc*100, nb_acc*100], color=['#4c78a8','#f58518'])
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Model Accuracy Comparison")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            labels = ["Precision", "Recall", "F1-Score"]
            rf_values = [rf_prec, rf_rec, rf_f1]
            nb_values = [nb_prec, nb_rec, nb_f1]
            
            x = np.arange(len(labels))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width/2, rf_values, width, label="Random Forest", color="skyblue")
            ax.bar(x + width/2, nb_values, width, label="Naive Bayes", color="salmon")
            ax.set_ylabel("Score")
            ax.set_title("Performance Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.divider()
        
        # Confusion Matrices
        st.subheader("🔢 Confusion Matrices")
        
        rf_cm = np.array([[rf_tn, rf_fp],[rf_fn, rf_tp]])
        nb_cm = np.array([[nb_tn, nb_fp],[nb_fn, nb_tp]])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            ax.set_title("Random Forest")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                       xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            ax.set_title("Naive Bayes")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()
        
        st.divider()
        
        # ROC Curve
        st.subheader("📉 ROC Curve Comparison")
        
        try:
            fpr_rf, tpr_rf, _ = roc_curve(st.session_state.y_test, st.session_state.y_rf_pred)
            roc_auc_rf = auc(fpr_rf, tpr_rf)
            
            fpr_nb, tpr_nb, _ = roc_curve(st.session_state.y_test, st.session_state.y_nb_pred)
            roc_auc_nb = auc(fpr_nb, tpr_nb)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fpr_rf, tpr_rf, color="blue", lw=2, label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
            ax.plot(fpr_nb, tpr_nb, color="green", lw=2, label=f"Naive Bayes (AUC = {roc_auc_nb:.2f})")
            ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve Comparison")
            ax.legend(loc="lower right")
            ax.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Could not generate ROC curves: {e}")

# Tab 3: Explainability
with tab3:
    st.header("🔍 Model Explainability")
    
    if not st.session_state.models_trained:
        st.warning("⚠ Please train models first in the 'Data & Training' tab")
    else:
        st.subheader("🔬 Sample Predictions Analysis")
        
        num_samples = st.slider("Number of samples to explain", 1, 10, 3)
        
        for i in range(min(num_samples, len(st.session_state.X_test))):
            with st.expander(f"📌 Sample {i+1} Explanation", expanded=(i==0)):
                x = st.session_state.X_test[i]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🌲 Random Forest Decision Path")
                    rf_path, rf_pred = explain_rf(x, st.session_state.forest[0], st.session_state.feat_cols)
                    
                    if rf_pred == 1:
                        st.error("⚠ Prediction: ATTACK")
                    else:
                        st.success("✅ Prediction: NORMAL")
                    
                    st.write("Decision Path:")
                    for step in rf_path:
                        st.write(f"→ {step}")
                
                with col2:
                    st.markdown("#### 🤖 Naive Bayes Feature Analysis")
                    nb_contribs = explain_nb(x, st.session_state.nb, st.session_state.feat_cols)
                    
                    nb_pred = st.session_state.nb.predict(x.reshape(1, -1))[0]
                    if nb_pred == 1:
                        st.error("⚠ Prediction: ATTACK")
                    else:
                        st.success("✅ Prediction: NORMAL")
                    
                    st.write("Key Indicators:")
                    for contrib in nb_contribs:
                        if "Attack" in contrib:
                            st.write(f"🔴 {contrib}")
                        else:
                            st.write(f"🟢 {contrib}")
                
                # Show feature values
                st.markdown("#### 📊 Feature Values")
                feature_data = {
                    "Feature": st.session_state.feat_cols,
                    "Value": [f"{x[j]:.4f}" for j in range(len(st.session_state.feat_cols))]
                }
                st.table(pd.DataFrame(feature_data))
        
        st.divider()
        
        # Feature Importance (based on split frequency in RF)
        st.subheader("🎯 Feature Importance Analysis")
        
        try:
            feature_counts = {feat: 0 for feat in st.session_state.feat_cols}
            
            def count_feature_usage(tree):
                if not tree['leaf']:
                    feat_name = st.session_state.feat_cols[tree['feature']]
                    feature_counts[feat_name] += 1
                    count_feature_usage(tree['left'])
                    count_feature_usage(tree['right'])
            
            for tree in st.session_state.forest:
                count_feature_usage(tree)
            
            # Normalize
            total = sum(feature_counts.values())
            if total > 0:
                feature_importance = {k: (v/total)*100 for k, v in feature_counts.items()}
                
                fig, ax = plt.subplots(figsize=(10, 5))
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                ax.barh(features, importance, color='steelblue')
                ax.set_xlabel("Importance (%)")
                ax.set_title("Feature Importance (Random Forest)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.error(f"Could not generate feature importance: {e}")


with tab4:
    st.header("📡 Live Network Traffic Detection")

    if not st.session_state.models_trained:
        st.warning("⚠ Please train models first in the 'Data & Training' tab")
    else:
        st.info("Note: Live capture requires tshark installed, and may require administrator privileges.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Interface", capture_interface)
        with col2:
            st.metric("Duration", f"{capture_duration}s")
        with col3:
            st.metric("Show First", f"{live_show_first} packets")

        if st.button("🚀 Start Live Capture", type="primary", use_container_width=True):
            try:
                import subprocess
                import tempfile
                
                st.info("⏱ Capturing packets... Please wait...")
                progress_bar = st.progress(0)
                
                # Create temporary file for capture
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pcap', delete=False) as tmp_file:
                    pcap_file = tmp_file.name
                
                # Run tshark command directly
                tshark_cmd = [
                    'tshark',
                    '-i', capture_interface,
                    '-a', f'duration:{capture_duration}',
                    '-w', pcap_file,
                    '-q'  # quiet mode
                ]
                
                # Start capture process
                start_time = time.time()
                process = subprocess.Popen(tshark_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Show progress while capturing
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    if elapsed >= capture_duration:
                        break
                    progress = min(elapsed / capture_duration, 1.0)
                    progress_bar.progress(progress)
                    time.sleep(0.5)
                
                process.wait()
                progress_bar.progress(1.0)
                progress_bar.empty()
                
                # Now read the captured packets
                read_cmd = [
                    'tshark',
                    '-r', pcap_file,
                    '-T', 'fields',
                    '-e', 'frame.protocols',
                    '-e', 'frame.len',
                    '-e', 'tcp.flags',
                    '-E', 'separator=,',
                    '-E', 'quote=d'
                ]
                
                result = subprocess.run(read_cmd, capture_output=True, text=True)
                
                # Clean up temp file
                try:
                    os.remove(pcap_file)
                except:
                    pass
                
                if result.returncode != 0:
                    raise Exception(f"tshark read failed: {result.stderr}")
                
                # Parse captured data
                live_rows = []
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    parts = line.split(',')
                    
                    proto = 0
                    pkt_len = 0
                    tcp_flags = 0
                    
                    # Parse protocol
                    if len(parts) > 0:
                        protocols = parts[0].lower()
                        if 'tcp' in protocols:
                            proto = 6
                        elif 'udp' in protocols:
                            proto = 17
                        elif 'icmp' in protocols:
                            proto = 1
                    
                    # Parse packet length
                    if len(parts) > 1:
                        try:
                            pkt_len = int(parts[1])
                        except:
                            pkt_len = 0
                    
                    # Parse TCP flags
                    if len(parts) > 2 and parts[2].strip():
                        try:
                            flags_str = parts[2].strip().strip('"')
                            if flags_str.startswith('0x'):
                                tcp_flags = int(flags_str, 16)
                            else:
                                tcp_flags = int(flags_str)
                        except:
                            tcp_flags = 0
                    
                    live_rows.append([proto, pkt_len, tcp_flags])

                if not live_rows:
                    st.warning("⚠ No packets captured during the interval.")
                else:
                    st.success(f"✅ Captured {len(live_rows)} packets")

                    # Build dataframe
                    df_live = pd.DataFrame(live_rows, columns=['protocol', 'packet_len', 'tcp_flags']).fillna(0)

                    # Align with trained feature columns
                    df_features_live = pd.DataFrame()
                    for feat in st.session_state.feat_cols:
                        if 'protocol' in feat.lower() or 'proto' in feat.lower():
                            df_features_live[feat] = df_live['protocol']
                        elif 'len' in feat.lower() or 'length' in feat.lower() or 'bytes' in feat.lower():
                            df_features_live[feat] = df_live['packet_len']
                        elif 'flag' in feat.lower() or 'psh' in feat.lower():
                            df_features_live[feat] = df_live['tcp_flags']
                        else:
                            df_features_live[feat] = 0

                    df_features_live = df_features_live.astype(float)

                    # Normalize
                    try:
                        live_norm = (df_features_live - st.session_state.mins[st.session_state.feat_cols].values) / (
                            st.session_state.range_[st.session_state.feat_cols].values
                        )
                    except:
                        live_norm = df_features_live.copy()
                        for i, col in enumerate(df_features_live.columns):
                            m = st.session_state.mins.iloc[i] if i < len(st.session_state.mins) else 0
                            r = st.session_state.range_.iloc[i] if i < len(st.session_state.range_) else 1
                            live_norm.iloc[:, i] = (df_features_live.iloc[:, i] - m) / (r if r != 0 else 1)

                    X_live_arr = live_norm.values

                    # Predict with models
                    rf_live_preds = predict_forest(st.session_state.forest, X_live_arr)
                    nb_live_preds = st.session_state.nb.predict(X_live_arr)

                    # Detection summary
                    rf_attack_count = int((rf_live_preds == 1).sum())
                    nb_attack_count = int((nb_live_preds == 1).sum())
                    total = len(rf_live_preds)

                    st.subheader("📊 Detection Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Packets", total)
                    col2.metric("🌲 RF Attacks", f"{rf_attack_count} ({rf_attack_count/total*100:.1f}%)")
                    col3.metric("🤖 NB Attacks", f"{nb_attack_count} ({nb_attack_count/total*100:.1f}%)")

                    st.divider()

                    # Show first N packets
                    st.subheader(f"🔍 First {min(live_show_first, total)} Packet Classifications")
                    results_data = []
                    for i in range(min(live_show_first, total)):
                        rf_result = "⚠ ATTACK" if rf_live_preds[i] == 1 else "✅ Normal"
                        nb_result = "⚠ ATTACK" if nb_live_preds[i] == 1 else "✅ Normal"
                        results_data.append({
                            "Packet": i + 1,
                            "Protocol": int(df_live.iloc[i]['protocol']),
                            "Length": int(df_live.iloc[i]['packet_len']),
                            "RF Prediction": rf_result,
                            "NB Prediction": nb_result
                        })
                    st.dataframe(pd.DataFrame(results_data), use_container_width=True)

                    # Save results
                    out_live = df_live.copy()
                    out_live['rf_pred'] = rf_live_preds
                    out_live['nb_pred'] = nb_live_preds
                    out_live['rf_label'] = out_live['rf_pred'].apply(lambda x: 'Attack' if x == 1 else 'Normal')
                    out_live['nb_label'] = out_live['nb_pred'].apply(lambda x: 'Attack' if x == 1 else 'Normal')

                    csv_path = os.path.join(OUTPUT_DIR, "live_classification_results.csv")
                    out_live.to_csv(csv_path, index=False)
                    st.success(f"💾 Results saved to: {csv_path}")

                    # Visualization
                    st.divider()
                    st.subheader("📈 Detection Distribution")
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                    # RF
                    rf_counts = pd.Series(rf_live_preds).value_counts()
                    axes[0].pie([rf_counts.get(0, 0), rf_counts.get(1, 0)],
                                labels=['Normal', 'Attack'],
                                autopct='%1.1f%%',
                                colors=['#90EE90', '#FF6B6B'])
                    axes[0].set_title("Random Forest")

                    # NB
                    nb_counts = pd.Series(nb_live_preds).value_counts()
                    axes[1].pie([nb_counts.get(0, 0), nb_counts.get(1, 0)],
                                labels=['Normal', 'Attack'],
                                autopct='%1.1f%%',
                                colors=['#90EE90', '#FF6B6B'])
                    axes[1].set_title("Naive Bayes")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            except FileNotFoundError:
                st.error("❌ tshark not found. Please install Wireshark/tshark first.")
                st.info("Installation:\n- Windows: Install Wireshark from wireshark.org\n- Linux: sudo apt-get install tshark\n- Mac: brew install wireshark")
            except Exception as e:
                st.error(f"❌ Capture failed: {e}")
                st.info("Make sure:\n- tshark is installed\n- You have administrator privileges\n- Interface name is correct")

        st.divider()

   
# Tab 5: Models
with tab5:
    st.header("💾 Model Management")
    
    if not st.session_state.models_trained:
        st.warning("⚠ No models trained yet")
    else:
        st.success("✅ Models are loaded and ready")
        
        st.subheader("📋 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌲 Random Forest")
            st.write(f"Number of Trees: {len(st.session_state.forest)}")
            st.write(f"Max Depth: {max_depth}")
            st.write(f"Min Samples Split: {min_samples_split}")
            st.write(f"Features: {len(st.session_state.feat_cols)}")
        
        with col2:
            st.markdown("### 🤖 Naive Bayes")
            st.write(f"Type: Gaussian Naive Bayes")
            st.write(f"Classes: {st.session_state.nb.classes.tolist()}")
            st.write(f"Features: {len(st.session_state.feat_cols)}")
        
        st.divider()
        
        st.subheader("📊 Feature Configuration")
        st.write(f"Selected Features: {', '.join(st.session_state.feat_cols)}")
        st.write(f"Label Column: {st.session_state.label_col}")
        
        # Show feature statistics
        feature_stats = pd.DataFrame({
            "Feature": st.session_state.feat_cols,
            "Min": st.session_state.mins[st.session_state.feat_cols].values,
            "Max": st.session_state.mins[st.session_state.feat_cols].values + st.session_state.range_[st.session_state.feat_cols].values
        })
        st.dataframe(feature_stats, use_container_width=True)
        
        st.divider()
        
        st.subheader("💾 Saved Model Files")
        
        rf_path = os.path.join(OUTPUT_DIR, "rf_manual_joblib.pkl")
        nb_path = os.path.join(OUTPUT_DIR, "nb_manual_joblib.pkl")
        scaler_path = os.path.join(OUTPUT_DIR, "robust_scaler.pkl")
        
        if os.path.exists(rf_path):
            st.success(f"✅ Random Forest: {rf_path}")
        else:
            st.error("❌ Random Forest model not found")
        
        if os.path.exists(nb_path):
            st.success(f"✅ Naive Bayes: {nb_path}")
        else:
            st.error("❌ Naive Bayes model not found")
        
        if os.path.exists(scaler_path):
            st.success(f"✅ Scaler: {scaler_path}")
        else:
            st.error("❌ Scaler not found")
        
        st.divider()
        
        st.subheader("🔄 Load Existing Models")
        
        if st.button("📥 Load Models from Disk", use_container_width=True):
            try:
                if os.path.exists(rf_path) and os.path.exists(nb_path):
                    with st.spinner("Loading models..."):
                        rf_data = joblib.load(rf_path)
                        nb_data = joblib.load(nb_path)
                        
                        st.session_state.forest = rf_data['forest']
                        st.session_state.nb = nb_data['nb']
                        st.session_state.feat_cols = rf_data['feature_columns']
                        st.session_state.label_col = rf_data['label_column']
                        st.session_state.models_trained = True
                        
                        if os.path.exists(scaler_path):
                            st.session_state.scaler = joblib.load(scaler_path)
                        
                        st.success("✅ Models loaded successfully!")
                        st.balloons()
                else:
                    st.error("❌ Model files not found. Please train models first.")
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
        
        st.divider()
        
        st.subheader("📤 Export & Download")
        st.info("Model files are saved in the comparison_outputs/ directory and can be used for deployment")
        
        # Show all output files
        if os.path.exists(OUTPUT_DIR):
            output_files = os.listdir(OUTPUT_DIR)
            if output_files:
                st.write("Available Output Files:")
                for file in sorted(output_files):
                    file_path = os.path.join(OUTPUT_DIR, file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    st.write(f"📄 {file} ({file_size:.2f} KB)")
            else:
                st.write("No output files yet")