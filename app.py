import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- 页面配置 ---
st.set_page_config(page_title="NSCLC AE Predictor", layout="centered")

st.title("NSCLC Adverse Event Risk Calculator")
st.markdown("This online tool predicts the risk of AEs based on 8 clinical features.")

# --- 加载模型 ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    # 确保 catboost_ae_model.cbm 文件在当前目录下
    model.load_model("catboost_ae_model.cbm")
    return model

model = load_model()

# --- 侧边栏参数输入 ---
st.sidebar.header("Patient Characteristics")

# 1. Dose_Int
dose_int = st.sidebar.selectbox("Dose Intensity", options=[0, 1], 
                                format_func=lambda x: "Standard (0)" if x==0 else "Reduced (1)")

# 2-5. Numerical features
anc = st.sidebar.number_input("ANC (10^9/L)", value=4.50, step=0.01)
rbc = st.sidebar.number_input("RBC (10^12/L)", value=4.20, step=0.01)
ca = st.sidebar.number_input("Calcium (mmol/L)", value=2.30, step=0.01)
k = st.sidebar.number_input("Potassium (mmol/L)", value=4.00, step=0.01)

# 6. cN
cn = st.sidebar.selectbox("Clinical N Stage (cN)", options=[0, 1, 2, 3])

# 7. Pathology
pathology = st.sidebar.selectbox("Histological Pathology", options=[0, 1, 2], 
                                 format_func=lambda x: ["LUAD (0)", "LUSC (1)", "Others (2)"][x])

# 8. Cycles (新增变量)
cycles = st.sidebar.selectbox("Treatment Cycles", options=[0, 1], 
                               format_func=lambda x: "≤ 2 cycles (0)" if x==0 else "> 2 cycles (1)")

# --- 预测逻辑 ---
if st.sidebar.button("Calculate Risk"):
    # 构造 DataFrame: 这里的列名和顺序必须与 CatBoost 模型训练时输入的 X 矩阵完全一致
    # 确保第8个变量 'cycles' 排在最后
    input_data = [[dose_int, anc, rbc, ca, k, cn, pathology, cycles]]
    cols = ["Dose_Int", "ANC", "RBC", "Ca", "K", "cN", "Pathology", "cycles"]
    
    input_df = pd.DataFrame(input_data, columns=cols)
    
    # 获取概率
    prob = model.predict_proba(input_df)[0][1]
    
    # --- 结果展示 ---
    st.subheader("Results")
    
    # 使用列布局美化结果
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write(f"The predicted probability of AE is:")
        st.title(f"{prob*100:.1f}%")
        
    with col2:
        if prob >= 0.48:
            st.error("Prediction: **HIGH RISK**")
        else:
            st.success("Prediction: **LOW RISK**")
            
    st.progress(prob)

st.markdown("---")
st.caption("Disclaimer: For research use only. The model predictions are based on statistical patterns and should not replace clinical judgment.")