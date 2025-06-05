import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="수학적 수열 예측기", layout="wide")
st.title("📐 수열 예측 시뮬레이터")

# --- 설명 영역 ---
st.markdown("""
### 📘 학습 목표
- ###### 수학적 모델과 인공지능(AI) 모델의 예측 성능을 비교할 수 있다. \n
    동일한 데이터를 기반으로 한 수학 함수 모델(회귀식)과 AI 모델(선형 회귀 또는 딥러닝)의 
    예측값($\hat{y} = ax + b$)과 오차($SSE = \sum (y_i - \hat{y}_i)^2$)를 비교 분석한다.
- ###### 학습된 모델을 이용해 새로운 입력에 대한 결과를 예측할 수 있다. \n
    함수나 AI 모델을 활용하여 새로운 입력값에 대한 출력을 예측하고, 그 결과를 해석하는 수리적 사고력을 기른다.
""")

# --- 수식 해석 함수 ---
def get_polynomial_equation(model, poly, feature_names):
    terms = poly.get_feature_names_out(feature_names)
    coefs = model.coef_
    intercept = model.intercept_
    expression = []
    for term, coef in zip(terms, coefs):
        if abs(coef) > 1e-6:
            expression.append(f"{coef:.2f}·{term}")
    expression.append(f"{intercept:.2f}")
    return " + ".join(expression)

# --- 1. 입력 방식 선택 ---
st.subheader("1️⃣ 입력 방식 선택")
input_mode = st.radio("입력 방식 선택", ["수열 입력", "학생 입력 데이터"])

if input_mode == "수열 입력":
    default_seq = "2, 5, 8, 11"
    seq_input = st.text_input("수열을 입력하세요 (쉼표로 구분):", default_seq)
    y = np.array(list(map(float, seq_input.split(","))))
    x = np.arange(1, len(y) + 1).reshape(-1, 1)
    input_dim = 1
else:
    st.markdown("#### 🎓 학생 데이터 입력 (X1 → Y)")
    x1_input = st.text_input("X1 값 (쉼표로 구분):", "1,2,3")
    y_input = st.text_input("Y 값 (쉼표로 구분):", "5,7,9")

    x1 = list(map(float, x1_input.strip().split(",")))
    y = list(map(float, y_input.strip().split(",")))

    if len(x1) != len(y):
        st.error("❌ X1과 Y의 길이가 같아야 합니다.")
        st.stop()

    x = np.array(x1).reshape(-1, 1)
    y = np.array(y)
    input_dim = 1

# --- 2. 수동 회귀 vs AI 모델 ---
st.subheader("2️⃣ 수동 회귀 vs AI 모델")
manual_col, ai_col = st.columns(2)

# 수동 회귀
with manual_col:
    st.markdown("#### ✍️ 수동 회귀(Manual Regression)")
    degree_manual = st.selectbox("차수 선택 (최대 3차)", options=[1, 2, 3], index=0)
    coeffs = []
    y_pred_manual = np.zeros_like(y, dtype=float)

    for deg in range(degree_manual, 0, -1):
        coef = st.slider(f"x^{deg} 계수", -10.0, 10.0, 1.0 if deg == 1 else 0.0, 0.1,
                         key=f"manual_coef_deg{deg}")
        coeffs.append((0, deg, coef))  # 단일 x만 사용
        y_pred_manual += coef * x[:, 0]**deg

    b = st.slider("상수항 b (절편)", -20.0, 20.0, 0.0, 0.1)
    y_pred_manual += b
    manual_sse = np.sum((y - y_pred_manual)**2)
    terms = [f"{coef:.2f}x^{deg}" for (_, deg, coef) in coeffs if coef != 0]
    if b != 0.0:
        terms.append(f"{b:.2f}")
    equation = "y = " + " + ".join(terms) if terms else f"y = {b:.2f}"

# AI 모델
with ai_col:
    st.markdown("#### 🤖 AI 모델(AI model)")
    model_type = st.radio("모델 선택", ["회귀 모델", "딥러닝 모델"])
    if model_type == "회귀 모델":
        degree = st.selectbox("차수 선택", options=[1, 2, 3], index=0)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train = poly.fit_transform(x)
        model = LinearRegression()
        model.fit(X_train, y)
        y_pred = model.predict(X_train)
        feature_names = [f"x1"]
        model_equation = get_polynomial_equation(model, poly, feature_names)
    else:
        hidden1 = st.slider("1층 뉴런 수", 4, 64, 16)
        hidden2 = st.slider("2층 뉴런 수", 4, 64, 8)
        epochs = st.slider("학습 횟수", 50, 100, 50)
        model = Sequential([
            Dense(hidden1, input_shape=(input_dim,), activation='relu'),
            Dense(hidden2, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.01), loss='mse')
        model.fit(x, y, epochs=epochs, verbose=0)
        y_pred = model.predict(x).flatten()
        model_equation = f"딥러닝 ({input_dim}-{hidden1}-{hidden2}-1)"
    sse = np.sum((y - y_pred)**2)

# --- 3. 비교표 출력 ---
st.subheader("📋 모델 비교")
st.dataframe(pd.DataFrame({
    "모델": ["수동 회귀", model_type],
    "함수식": [equation, model_equation],
    "SSE": [f"{manual_sse:.2f}", f"{sse:.2f}"]
}), use_container_width=True)

# --- 4. 예측값 비교 ---
st.subheader("🔍 예측값 비교")
next_input = st.number_input("다음 x 입력값", value=float(len(y)+1))
x_next = np.array([[next_input]])

pred_manual_next = sum([
    coef * (x_next[0][0] ** deg) for (_, deg, coef) in coeffs
]) + b

if model_type == "회귀 모델":
    X_next_trans = poly.transform(x_next)
    pred_ai_next = model.predict(X_next_trans)[0]
else:
    pred_ai_next = model.predict(x_next)[0][0]

st.dataframe(pd.DataFrame({
    "모델": ["수동 회귀", model_type],
    "다음 입력 예측값 y": [f"{pred_manual_next:.2f}", f"{pred_ai_next:.2f}"]
}), use_container_width=True)

# --- 5. 시각화 ---
st.subheader("📊 시각화")
fig, ax = plt.subplots()
ax.scatter(np.arange(len(y)), y, color='blue', label='Input Data')
ax.plot(np.arange(len(y)), y_pred_manual, color='orange', label='Manual Regression')
ax.plot(np.arange(len(y)), y_pred, color='green', label='AI model')

marker_size = 100
ax.scatter([x_next[0][0]], [pred_manual_next], color='red', edgecolors='black',
           label='Manual Prediction', s=marker_size, marker='o')
ax.scatter([x_next[0][0]], [pred_ai_next], color='red', edgecolors='black',
           label='AI Prediction', s=marker_size, marker='x')

ax.legend()
st.pyplot(fig)

# --- 6. 요약 ---
st.subheader("📌 요약")
st.markdown(f"""
- 입력 방식: **{input_mode}**
- 수동 회귀와 {model_type} 모델로 예측을 수행했습니다.
- 다음 입력에 대한 예측값:
  - 수동 회귀: **{pred_manual_next:.2f}**
  - {model_type}: **{pred_ai_next:.2f}**
""")
