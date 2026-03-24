import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- 页面配置 ---
st.set_page_config(page_title="NSCLC AE Predictor", layout="centered")

st.title("NSCLC Adverse Event Risk Calculator")
st.markdown("This online tool predicts the risk of ≥3 TRAEs based on 8  features.")

# --- 加载模型 ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
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

# 6. Pathology (注意：顺序调整到了 cN 之前)
pathology = st.sidebar.selectbox("Histological Pathology", options=[0, 1, 2], 
                                 format_func=lambda x: ["LUAD (0)", "LUSC (1)", "Others (2)"][x])

# 7. cN
cn = st.sidebar.selectbox("Clinical N Stage (cN)", options=[0, 1, 2, 3])

# 8. Cycles (注意：首字母大写)
cycles = st.sidebar.selectbox("Treatment Cycles", options=[0, 1], 
                               format_func=lambda x: "≤ 2 cycles (0)" if x==0 else "> 2 cycles (1)")

# --- 预测按钮 ---
if st.sidebar.button("Calculate Risk"):
    # 构建数据框，严格遵循模型要求的顺序：
    # ["Dose_Int", "ANC", "RBC", "Ca", "K", "Pathology", "cN", "Cycles"]
    input_data = [[dose_int, anc, rbc, ca, k, pathology, cn, cycles]]
    cols = ["Dose_Int", "ANC", "RBC", "Ca", "K", "Pathology", "cN", "Cycles"]
    
    input_df = pd.DataFrame(input_data, columns=cols)
    
    # 获取概率
    try:
        prob = model.predict_proba(input_df)[0][1]
        
        # --- 结果展示 ---
        st.subheader("Results")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write(f"Predicted Probability:")
            st.title(f"{prob*100:.2f}%")
            
        with col2:
            if prob >= 0.48:
                st.error("Prediction: **HIGH RISK**")
            else:
                st.success("Prediction: **LOW RISK**")
        
        st.progress(prob)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Please ensure the model file matches the 8 input features.")

st.markdown("---")
st.caption("Disclaimer: For research use only.")