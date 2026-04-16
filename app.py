import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- 1. 页面配置 (优化 UI 体验) ---
st.set_page_config(page_title="NSCLC AE Predictor", layout="centered")

st.title("NSCLC Adverse Event Risk Calculator")
st.markdown("### Clinical prediction tool for Grade ≥3 TRAEs")
st.markdown("This tool predicts the risk based on **7 key clinical features**.")

# --- 2. 加载模型 ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    # 请确保你的新模型文件是以这 7 个特征训练的
    model.load_model("catboost_ae_model.cbm") 
    return model

try:
    model = load_model()
except Exception as e:
    st.error("Model file not found. Please ensure 'catboost_ae_model_7.cbm' is in the directory.")

# --- 3. 侧边栏参数输入 (调整为 7 个特征) ---
st.sidebar.header("Patient Characteristics")

# 分类变量 1: Dose Intensity
dose_int = st.sidebar.selectbox("Dose Intensity", options=[0, 1], 
                               format_func=lambda x: "Standard (0)" if x==0 else "Reduced (1)")

# 数值变量 1-4: ANC, RBC, Ca, K
anc = st.sidebar.number_input("ANC (10^9/L)", value=4.50, step=0.01, min_value=0.0)
rbc = st.sidebar.number_input("RBC (10^12/L)", value=4.20, step=0.01, min_value=0.0)
ca = st.sidebar.number_input("Calcium (mmol/L)", value=2.30, step=0.01, min_value=0.0)
k = st.sidebar.number_input("Potassium (mmol/L)", value=4.00, step=0.01, min_value=0.0)

# 分类变量 2: cN Stage
cn = st.sidebar.selectbox("Clinical N Stage (cN)", options=[0, 1, 2, 3])

# 分类变量 3: Treatment Cycles
cycles = st.sidebar.selectbox("Treatment Cycles", options=[0, 1], 
                               format_func=lambda x: "≤ 2 cycles (0)" if x==0 else "> 2 cycles (1)")

# --- 4. 风险预测逻辑 ---
if st.sidebar.button("Calculate Risk"):
    # 构建数据框，严格遵循你最新的特征顺序：
    # ["Dose_Int", "ANC", "Ca", "RBC", "Cycles", "K", "cN"]
    cols = ["Dose_Int", "ANC", "Ca", "RBC", "Cycles", "K", "cN"]
    input_data = [[dose_int, anc, ca, rbc, cycles, k, cn]]
    
    input_df = pd.DataFrame(input_data, columns=cols)
    
    try:
        # 获取正类概率
        prob = model.predict_proba(input_df)[0][1]
        
        # --- 5. 结果展示 ---
        st.subheader("Risk Prediction Results")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric(label="Calculated Risk Score", value=f"{prob:.2%}")
            
        with col2:
            # 这里的阈值 0.48 你可以根据 7 因子模型的最佳 Cut-off 值进行调整
            if prob >= 0.4114:
                st.error("Prediction: **HIGH RISK**")
            else:
                st.success("Prediction: **LOW RISK**")
        
        # 视觉进度条
        st.progress(prob)
        st.write(f"The probability of the patient experiencing Grade ≥3 TRAEs is **{prob*100:.1f}%**.")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure the model input order matches: Dose_Int, ANC, Ca, RBC, Cycles, K, cN")

st.markdown("---")
st.caption("Note: This model is for clinical research reference only and should not be used as the sole basis for medical decisions.")
