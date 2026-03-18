import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# 设置页面风格
st.set_page_config(page_title="NSCLC AE Predictor", layout="centered")

st.title("NSCLC Grade ≥3 TRAE Risk Calculator")

st.markdown("""
This interactive tool is designed to predict the individual risk of 
**Grade 3 or higher Treatment-Related Adverse Events (TRAE)** in patients with Non-Small Cell Lung Cancer (NSCLC).
""")

# 加载模型
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_ae_model.cbm")
    return model

model = load_model()

# 侧边栏参数输入
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

# 预测按钮
if st.sidebar.button("Calculate Risk"):
    # 构造 DataFrame (列名顺序必须与训练时完全一致)
    input_df = pd.DataFrame([[dose_int, anc, rbc, ca, k, cn, pathology]], 
                            columns=["Dose_Int", "ANC", "RBC", "Ca", "K", "cN", "Pathology"])
    
    # 获取概率
    prob = model.predict_proba(input_df)[0][1]
    
    # 结果展示
    st.subheader("Results")
    if prob >= 0.5:
        st.error(f"Prediction: HIGH RISK")
    else:
        st.success(f"Prediction: LOW RISK")
    
    st.write(f"The predicted probability of AE is: **{prob*100:.2f}%**")
    st.progress(prob)

st.markdown("---")
st.caption("Disclaimer: For research use only.")
