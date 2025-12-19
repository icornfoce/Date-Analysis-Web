import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="EcoTrack Insights", page_icon="♻️", layout="wide")

@st.cache_data
def get_initial_data():
    df = pd.read_csv('sustainable_waste_management_dataset_2024.csv', parse_dates=['date'])
    df.columns = df.columns.str.strip()
    return df

if 'main_df' not in st.session_state:
    st.session_state.main_df = get_initial_data()

display_df = st.session_state.main_df

# --- ส่วนที่ 1: แบบฟอร์มรับข้อมูล (เพิ่มช่องอุณหภูมิ) ---
st.title("♻️ EcoTrack: Temperature & Waste Analysis")
with st.expander("➕ เพิ่มข้อมูลชุดใหม่ (รวมสภาพอากาศ)"):
    with st.form("weather_waste_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_date = st.date_input("วันที่")
            in_area = st.selectbox("พื้นที่", options=display_df['area'].unique())
        with c2:
            in_waste = st.number_input("ขยะที่เก็บได้ (kg)", min_value=0.0)
            in_temp = st.slider("อุณหภูมิขณะนั้น (°C)", 10.0, 45.0, 28.0) # รับข้อมูลอุณหภูมิ
        with c3:
            in_pop = st.number_input("ประชากร", value=int(display_df['population'].mean()))
            in_cap = st.number_input("ความสามารถการจัดเก็บ (kg)", value=20000)

        submit = st.form_submit_button("บันทึกข้อมูล")

        if submit:
            new_row = pd.DataFrame([{
                'date': pd.to_datetime(in_date),
                'area': in_area,
                'population': in_pop,
                'waste_kg': in_waste,
                'temp_c': in_temp, # บันทึกอุณหภูมิใหม่เข้าไป
                'recyclable_kg': in_waste * 0.2, # สมมติค่ารีไซเคิล 20%
                'collection_capacity_kg': in_cap,
                'rain_mm': 0.0,
                'overflow': 1 if in_waste > in_cap else 0
            }])
            st.session_state.main_df = pd.concat([st.session_state.main_df, new_row], ignore_index=True)
            st.rerun()

# --- ส่วนที่ 2: การวิเคราะห์อุณหภูมิ ---
st.divider()
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("🌡️ สถิติอุณหภูมิในพื้นที่")
    avg_temp = display_df['temp_c'].mean()
    st.metric("อุณหภูมิเฉลี่ยสะสม", f"{avg_temp:.2f} °C")
    
    # กราฟ Histogram ดูการกระจายตัวของอุณหภูมิ
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(display_df['temp_c'], bins=15, color='orange', edgecolor='white')
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Frequency (Days)")
    st.pyplot(fig1)

with col_b:
    st.subheader("🔍 อุณหภูมิมีผลต่อขยะหรือไม่?")
    # กราฟ Scatter Plot ดูความสัมพันธ์
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(display_df['temp_c'], display_df['waste_kg'], alpha=0.5, color='red')
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Waste Amount (kg)")
    # เพิ่มเส้นแนวโน้ม (Trendline)
    m, b = np.polyfit(display_df['temp_c'], display_df['waste_kg'], 1)
    ax2.plot(display_df['temp_c'], m*display_df['temp_c'] + b, color='black', linestyle='--')
    st.pyplot(fig2)

# --- ส่วนที่ 3: Machine Learning (พยากรณ์โดยใช้อุณหภูมิเป็นปัจจัย) ---
st.divider()
st.subheader("🤖 AI Forecast (Temperature Factor)")

ml_df = st.session_state.main_df.dropna()
# รอบนี้เราใช้ทั้ง 'ประชากร' และ 'อุณหภูมิ' มาช่วยทำนาย
X = ml_df[['population', 'temp_c']] 
y = ml_df['waste_kg']

model = LinearRegression().fit(X, y)

c1, c2 = st.columns(2)
with c1:
    user_pop = st.number_input("ระบุจำนวนประชากร:", value=int(ml_df['population'].mean()))
with c2:
    user_temp = st.slider("ระบุอุณหภูมิที่คาดไว้ (°C):", 10.0, 45.0, 30.0)

if st.button("ให้ AI ทำนายปริมาณขยะ"):
    prediction = model.predict([[user_pop, user_temp]])
    st.success(f"ถ้าอุณหภูมิ {user_temp} °C และประชากร {user_pop} คน คาดว่าจะมีขยะประมาณ {prediction[0]:,.2f} kg")
