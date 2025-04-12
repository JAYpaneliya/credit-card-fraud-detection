import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; font-size: 36px;'>💻 ML Model Comparison Dashboard</h1>", unsafe_allow_html=True)

# Load data and models
metrics = pd.read_csv("metrics_df.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

model_map = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# Tabs layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "📈  Visual Comparison",
    "📉  ROC Curve",
    "📥  Download Models"
])

# ----------- Tab 1: Metrics Table -----------
with tab1:
    st.markdown("<h2 style='font-size:28px; padding-top:10px;'>📊 Overview</h2>", unsafe_allow_html=True)
    styled_metrics = metrics.style.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '18px')]},
        {'selector': 'td', 'props': [('font-size', '17px')]}
    ])
    st.table(styled_metrics)

# ----------- Tab 2: Metric & Confusion Matrix -----------
with tab2:
    st.markdown("<h2 style='font-size:28px; padding-top:10px;'>📈 Visual Comparison</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        selected_metric = st.selectbox("Choose a metric:", metrics.columns[1:], key="metric_plot")
        y_vals = metrics[selected_metric]
        y_min = max(0, y_vals.min() - 0.02)
        y_max = min(1.05, y_vals.max() + 0.02)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        bars = sns.barplot(data=metrics, x="Model", y=selected_metric, palette="Set2", ax=ax1)
        ax1.set_ylim(y_min, y_max)
        ax1.set_ylabel(selected_metric, fontsize=12)
        ax1.set_title(f"{selected_metric} Across Models", fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars.patches:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9, color='black', fontweight='bold')
        st.pyplot(fig1)

    with col2:
        cm_model_choice = st.selectbox("Choose model for confusion matrix:", list(model_map.keys()), key="cm_model")
        cm_model = joblib.load(model_map[cm_model_choice])
        y_pred = cm_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        disp.plot(ax=ax2, cmap='Blues', colorbar=False)
        ax2.set_title(f"{cm_model_choice} - Confusion Matrix", fontsize=14)
        st.pyplot(fig2)

# ----------- Tab 3: ROC Curve -----------
with tab3:
    st.markdown("<h2 style='font-size:28px; padding-top:10px;'>📉 ROC Curve (AUC)</h2>", unsafe_allow_html=True)
    roc_col1, _ = st.columns(2)

    with roc_col1:
        roc_model_choice = st.selectbox("Choose model for ROC curve:", list(model_map.keys()), key="roc_model")
        roc_model = joblib.load(model_map[roc_model_choice])
        try:
            y_prob = roc_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='darkorange', linewidth=2)
            ax3.plot([0, 1], [0, 1], linestyle='--', color='navy')
            ax3.set_xlabel('False Positive Rate', fontsize=12)
            ax3.set_ylabel('True Positive Rate', fontsize=12)
            ax3.set_title(f'ROC Curve - {roc_model_choice}', fontsize=14)
            ax3.legend()
            ax3.grid(True)
            st.pyplot(fig3)

        except:
            st.warning("⚠️ This model does not support probability predictions.")

# ----------- Tab 4: Download Models -----------
with tab4:
    st.markdown("<h2 style='font-size:28px; padding-top:10px;'>📥 Download Trained Models</h2>", unsafe_allow_html=True)
    st.markdown("Click below to download any model you want to share or reuse.")

    cols = st.columns(len(model_map))
    for idx, (model_name, file_path) in enumerate(model_map.items()):
        with open(file_path, "rb") as f:
            model_bytes = f.read()
        with cols[idx]:
            st.download_button(
                label=f"⬇️ {model_name}",
                data=model_bytes,
                file_name=file_path,
                mime="application/octet-stream"
            )

# ----------- Footer -----------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size:18px; color: #555; padding-top: 10px;">
        🚀 Made with ❤️ by <strong>Anushka</strong>, <strong>Manish</strong> & <strong>Jay</strong>
    </div>
    """,
    unsafe_allow_html=True
)
