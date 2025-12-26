import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ==================== 模型路径（在线部署用相对路径） ====================
model_dir = "stacking_models"

# 检查路径（云端自动存在）
if not os.path.exists(model_dir):
    st.error(f"模型文件夹不存在！请确保 stacking_models/ 在仓库中。")
    st.stop()

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="术后异质性衰弱预测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 术后异质性衰弱预测系统（Stacking 集成模型）")
st.markdown("""
**模型名称**：术后异质性衰弱预测系统  
**基学习器**：Random Forest + CatBoost + Extra Trees（使用十折CV频率最高最优参数训练）  
**元学习器**：Logistic Regression  
**模型解释**：以 Extra Trees 为主体进行 SHAP 可解释性分析（全局 + 局部）
""")

# ==================== 检查文件 ====================
required_files = ["RF.pkl", "CatBoost.pkl", "ET.pkl", "meta_learner.pkl", "feature_names.pkl", "shap_summary_et.png", "shap_bar_et.png"]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
if missing_files:
    st.error(f"缺失文件：{', '.join(missing_files)}。请上传完整 stacking_models/ 文件夹到仓库。")
    st.stop()

# ==================== 加载模型 ====================
@st.cache_resource
def load_models():
    base_models = {
        'RF': joblib.load(os.path.join(model_dir, 'RF.pkl')),
        'CatBoost': joblib.load(os.path.join(model_dir, 'CatBoost.pkl')),
        'ET': joblib.load(os.path.join(model_dir, 'ET.pkl'))
    }
    meta_learner = joblib.load(os.path.join(model_dir, 'meta_learner.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    return base_models, meta_learner, feature_names

base_models, meta_learner, feature_names = load_models()
st.success("✅ 模型加载成功！")

# ==================== 预测函数 ====================
def predict_stacking(X_input: pd.DataFrame):
    base_proba = np.column_stack([model.predict_proba(X_input)[:, 1] for model in base_models.values()])
    final_proba = meta_learner.predict_proba(base_proba)[:, 1]
    final_pred = (final_proba >= 0.5).astype(int)
    return final_proba, final_pred

# ==================== 侧边栏 ====================
st.sidebar.header("📊 数据输入方式")
input_mode = st.sidebar.radio("请选择", ["手动输入单个样本", "上传 Excel 批量预测"])

# ==================== 手动预测 ====================
if input_mode == "手动输入单个样本":
    st.header("手动输入特征值")
    input_data = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(feat, value=0.0, step=0.0001, format="%.6f", key=f"feat_{i}")
            input_data[feat] = val

    if st.button("🚀 开始预测", type="primary"):
        X_input = pd.DataFrame([input_data])[feature_names]
        proba, pred = predict_stacking(X_input)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("预测概率 (TrajStage = 1)", f"{proba[0]:.4f}")
        with col2:
            result = "阳性 (1)" if pred[0] == 1 else "阴性 (0)"
            st.metric("预测分类结果", result)

        # SHAP 局部解释
        st.subheader("🔍 SHAP 局部解释（基于 Extra Trees）")
        try:
            et_model = base_models['ET']
            explainer = shap.TreeExplainer(et_model)
            shap_values = explainer.shap_values(X_input)
            if isinstance(shap_values, list):
                shap_val = shap_values[1]
                expected_value = explainer.expected_value[1]
            elif shap_values.ndim == 3:
                shap_val = shap_values[0, :, 1]
                expected_value = explainer.expected_value[1]
            else:
                shap_val = shap_values[0] if shap_values.shape[0] == 1 else shap_values[:, 1]
                expected_value = explainer.expected_value
            shap_val = np.ravel(shap_val)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap.Explanation(values=shap_val, base_values=expected_value, data=X_input.iloc[0].values, feature_names=feature_names))
            st.pyplot(plt)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP 图生成失败：{str(e)}")

# ==================== 批量预测 ====================
else:
    st.header("批量预测（上传 Excel）")
    uploaded_file = st.file_uploader("上传 Excel 文件", type=['xlsx'])
    if uploaded_file:
        df_input = pd.read_excel(uploaded_file)
        if list(df_input.columns) != feature_names:
            st.error(f"列不匹配！期望：{feature_names}")
            st.stop()
        proba, pred = predict_stacking(df_input)
        result_df = df_input.copy()
        result_df['Predicted_Probability'] = np.round(proba, 4)
        result_df['Predicted_Class'] = pred
        result_df['Predicted_Label'] = result_df['Predicted_Class'].map({1: '阳性', 0: '阴性'})
        st.dataframe(result_df)
        csv = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下载结果", csv, "predictions.csv", "text/csv")

# ==================== 全局 SHAP ====================
st.sidebar.header("📈 模型解释")
if st.sidebar.button("查看全局 SHAP"):
    st.subheader("全局 SHAP（Extra Trees）")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(model_dir, 'shap_summary_et.png'), caption="Summary Plot")
    with col2:
        st.image(os.path.join(model_dir, 'shap_bar_et.png'), caption="Bar Plot")