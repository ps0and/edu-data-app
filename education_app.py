import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="고등학교 데이터 분석 실습", layout="wide")

st.title("📊 데이터 분석 실습 앱")
st.markdown("학생들이 CSV 파일을 업로드하고, 데이터를 분석하고, 시각화하며, 머신러닝 예측까지 체험할 수 있는 교육용 앱입니다.")

# 1. 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ 파일 업로드 성공!")

    # 2. 데이터 미리보기
    st.subheader("1️⃣ 데이터 미리보기")
    st.dataframe(df.head())

    # 3. 컬럼 선택
    st.subheader("2️⃣ 컬럼 선택")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_columns) >= 1:
        col = st.selectbox("분석할 숫자형 컬럼을 선택하세요", numeric_columns)

        # 4. 기술 통계 분석
        st.subheader("3️⃣ 기술 통계 분석")
        st.write(df[col].describe())

        # 5. 히스토그램
        st.subheader("4️⃣ 히스토그램")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, color="skyblue", ax=ax)
        st.pyplot(fig)

        # 6. 상관 관계 히트맵
        if len(numeric_columns) > 1:
            st.subheader("5️⃣ 상관 관계 히트맵")
            corr = df[numeric_columns].corr()
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)

        # 7. 머신러닝 예측
        st.subheader("6️⃣ 머신러닝 예측: 선형 회귀")
        target = st.selectbox("🎯 예측할 대상(종속변수)을 선택하세요", numeric_columns, index=numeric_columns.index(col))
        features = st.multiselect("📌 입력 변수(독립변수)를 선택하세요", [c for c in numeric_columns if c != target])

        if features:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("✅ 모델 훈련 완료!")
            st.write(f"📉 평균제곱오차 (MSE): {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"📈 결정계수 (R²): {r2_score(y_test, y_pred):.2f}")

            # 예측 결과 시각화
            fig3, ax3 = plt.subplots()
            ax3.scatter(y_test, y_pred, color='green')
            ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax3.set_xlabel("실제 값")
            ax3.set_ylabel("예측 값")
            ax3.set_title("실제 값 vs 예측 값")
            st.pyplot(fig3)

    else:
        st.warning("⚠️ 숫자형 컬럼이 포함된 CSV 파일을 업로드해 주세요.")
else:
    st.info("📂 왼쪽 사이드바에서 CSV 파일을 업로드하면 분석을 시작할 수 있어요!")
