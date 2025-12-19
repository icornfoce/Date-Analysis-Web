import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="EcoTrack Insights", page_icon="♻️", layout="wide")

# --- ฟังก์ชันโหลดข้อมูลเริ่มต้น ---
@st.cache_data
def get_initial_data():
    df = pd.read_csv('sustainable_waste_management_dataset_2024.csv', parse_dates=['date'])
    df.columns = df.columns.str.strip()
    return df

# --- ส่วนการจัดการหน่วยความจำ (Session State) ---
# ถ้ายังไม่มีข้อมูลใน "หน่วยความจำเว็บ" ให้เอาข้อมูลจาก CSV มาใส่ไว้เป็นค่าเริ่มต้น
if 'main_df' not in st.session_state:
    st.session_state.main_df = get_initial_data()

# ดึงข้อมูลจากหน่วยความจำมาใช้งาน
display_df = st.session_state.main_df
display_df['year'] = display_df['date'].dt.year

# --- Header ---
st.title("♻️ EcoTrack Insights")
st.markdown("ข้อมูลจะถูกอัปเดตและแสดงผลทันทีเมื่อมีการบันทึกข้อมูลใหม่")

# --- ส่วนที่ 1: แบบฟอร์มรับข้อมูล (Data Entry) ---
with st.expander("➕ เพิ่มข้อมูลใหม่ลงใน Dashboard", expanded=False):
    with st.form("new_entry_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_date = st.date_input("วันที่")
            in_area = st.selectbox("พื้นที่", options=display_df['area'].unique())
        with c2:
            in_waste = st.number_input("ขยะที่เก็บได้ (kg)", min_value=0.0)
            in_recycle = st.number_input("รีไซเคิลได้ (kg)", min_value=0.0)
        with c3:
            in_pop = st.number_input("ประชากร", value=int(display_df['population'].mean()))
            in_cap = st.number_input("ขีดความสามารถ (kg)", value=int(display_df['collection_capacity_kg'].mean()))

        submit = st.form_submit_button("บันทึกข้อมูลลงตาราง")

        if submit:
            # สร้างแถวข้อมูลใหม่ (จัดลำดับคอลัมน์ให้ตรงกับไฟล์เดิม)
            new_row = pd.DataFrame([{
                'date': pd.to_datetime(in_date),
                'day_name': pd.to_datetime(in_date).day_name(),
                'area': in_area,
                'population': in_pop,
                'waste_kg': in_waste,
                'recyclable_kg': in_recycle,
                'collection_capacity_kg': in_cap,
                'temp_c': display_df['temp_c'].mean(), # ใส่ค่าเฉลี่ยไว้ก่อน
                'rain_mm': 0.0,
                'overflow': 1 if in_waste > in_cap else 0
            }])
            
            # อัปเดตข้อมูลในหน่วยความจำ (ต่อท้ายข้อมูลเดิม)
            st.session_state.main_df = pd.concat([st.session_state.main_df, new_row], ignore_index=True)
            st.success("อัปเดตข้อมูลสำเร็จ! กราฟด้านล่างจะเปลี่ยนไปตามข้อมูลใหม่ที่คุณกรอก")
            st.rerun() # สั่งให้หน้าเว็บโหลดใหม่เพื่อแสดงผลทันที

# --- ส่วนที่ 2: Dashboard แสดงผล ---
st.divider()
st.sidebar.header("🔍 ตัวกรองแดชบอร์ด")
selected_areas = st.sidebar.multiselect("เลือกพื้นที่:", options=display_df['area'].unique(), default=display_df['area'].unique())
filtered_df = display_df[display_df['area'].isin(selected_areas)]

# แสดง Metrics
m1, m2, m3 = st.columns(3)
m1.metric("ปริมาณขยะสะสมทั้งหมด (kg)", f"{filtered_df['waste_kg'].sum():,.0f}")
m2.metric("จำนวนรายการทั้งหมด", f"{len(filtered_df)} รายการ")
m3.metric("ปริมาณรีไซเคิลรวม (kg)", f"{filtered_df['recyclable_kg'].sum():,.0f}")

# กราฟแสดงแนวโน้ม (จะเห็นจุดใหม่ที่เพิ่งกรอกเข้าไปด้วย)
st.subheader("📈 วิเคราะห์แนวโน้ม (รวมข้อมูลล่าสุด)")
st.line_chart(filtered_df.set_index('date')[['waste_kg']])

# แสดงตารางข้อมูล 5 แถวที่เพิ่งเพิ่มเข้าไปล่าสุด
st.write("### 📄 ตารางข้อมูลล่าสุด")
st.dataframe(filtered_df.tail(10).sort_values(by='date', ascending=False))

# --- ส่วนที่ 3: Machine Learning ---
st.divider()
st.subheader("🤖 AI Forecast")
# AI จะเรียนรู้จากข้อมูลที่รวม "ข้อมูลใหม่" เข้าไปด้วยแล้ว
ml_df = st.session_state.main_df.copy().dropna()
ml_df['date_ord'] = ml_df['date'].map(pd.Timestamp.toordinal)
model = LinearRegression().fit(ml_df[['date_ord', 'population']], ml_df['waste_kg'])

p_pop = st.number_input("ระบุจำนวนประชากรเพื่อพยากรณ์:", value=int(ml_df['population'].mean()))
if st.button("ทำนายปริมาณขยะ"):
    today = pd.Timestamp.now().toordinal()
    pred = model.predict([[today, p_pop]])
    st.success(f"AI คาดการณ์ว่าจะมีขยะ: {pred[0]:,.2f} kg")