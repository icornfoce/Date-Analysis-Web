import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="EcoTrack Insights", page_icon="♻️", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('sustainable_waste_management_dataset_2024.csv', parse_dates=['date'])
    df.columns = df.columns.str.strip()
    df['year'] = df['date'].dt.year
    return df

try:
    df = load_data()

    # --- Header ---
    st.title("♻️ EcoTrack Insights")
    st.markdown("ระบบวิเคราะห์และบันทึกข้อมูลการจัดการขยะอัจฉริยะ")
    st.divider()

    # --- ส่วนใหม่: แบบฟอร์มรับข้อมูล (Data Entry) ---
    with st.expander("➕ บันทึกข้อมูลปริมาณขยะรายวันใหม่"):
        with st.form("waste_entry_form", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                in_date = st.date_input("วันที่บันทึก")
                in_area = st.selectbox("พื้นที่ (Area)", options=df['area'].unique())
            with f_col2:
                in_waste = st.number_input("ปริมาณขยะที่เก็บได้ (kg)", min_value=0.0)
                in_recycle = st.number_input("รีไซเคิลได้ (kg)", min_value=0.0)
            with f_col3:
                in_pop = st.number_input("จำนวนประชากรในวันนั้น", value=int(df['population'].mean()))
                in_temp = st.slider("อุณหภูมิ (°C)", 10.0, 45.0, 25.0)

            submitted = st.form_submit_button("บันทึกและคำนวณ")
            
            if submitted:
                # คำนวณเบื้องต้น
                eff_rate = (in_recycle / in_waste * 100) if in_waste > 0 else 0
                st.success(f"บันทึกข้อมูลเรียบร้อย! อัตราการรีไซเคิลของข้อมูลนี้คือ {eff_rate:.2f}%")
                
                # สร้าง DataFrame จำลองสำหรับข้อมูลใหม่
                new_data = pd.DataFrame({
                    'date': [in_date], 'area': [in_area], 'waste_kg': [in_waste],
                    'recyclable_kg': [in_recycle], 'population': [in_pop], 'temp_c': [in_temp]
                })
                st.write("ข้อมูลที่กรอก:", new_data)

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 ตัวกรองแดชบอร์ด")
    all_areas = df['area'].unique()
    selected_areas = st.sidebar.multiselect("เลือกพื้นที่:", options=all_areas, default=all_areas)
    filtered_df = df[df['area'].isin(selected_areas)]

    # --- ส่วนที่ 1: Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ปริมาณขยะรวม (kg)", f"{filtered_df['waste_kg'].sum():,.0f}")
    with col2:
        recycle_rate = (filtered_df['recyclable_kg'].sum() / filtered_df['waste_kg'].sum()) * 100
        st.metric("อัตราการรีไซเคิล", f"{recycle_rate:.1f}%")
    with col3:
        st.metric("อุณหภูมิเฉลี่ย", f"{filtered_df['temp_c'].mean():.1f} °C")
    with col4:
        # ปุ่มโหลดข้อมูล CSV ที่กรองแล้ว
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report CSV", data=csv, file_name='waste_report.csv', mime='text/csv')

    # --- ส่วนที่ 2: Visualizations ---
    tab1, tab2 = st.tabs(["📊 แนวโน้มการจัดการ", "🌦️ ปัจจัยสภาพอากาศ"])
    with tab1:
        st.line_chart(filtered_df.set_index('date')[['waste_kg', 'collection_capacity_kg']])
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(filtered_df['temp_c'], filtered_df['waste_kg'], color='salmon', alpha=0.5)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Waste (kg)")
        st.pyplot(fig)

    # --- ส่วนที่ 3: Machine Learning ---
    st.divider()
    st.write("### 🤖 ระบบพยากรณ์ปริมาณขยะ (AI Prediction)")
    
    # เตรียมข้อมูล ML
    df_ml = df.dropna().copy()
    df_ml['date_ordinal'] = df_ml['date'].apply(lambda x: x.toordinal())
    X = df_ml[['date_ordinal', 'population']]
    y = df_ml['waste_kg']
    
    model = LinearRegression().fit(X, y)

    ml_col1, ml_col2 = st.columns([1, 2])
    with ml_col1:
        st.write("กรอกจำนวนประชากรเพื่อพยากรณ์ขยะวันนี้:")
        pop_input = st.number_input("จำนวนประชากร:", value=int(df['population'].mean()), key='ml_pop')
        if st.button("ทำนายผล"):
            current_date_ord = pd.Timestamp.now().toordinal()
            pred = model.predict([[current_date_ord, pop_input]])
            st.success(f"ปริมาณขยะที่คาดการณ์: {pred[0]:,.2f} kg")
    
    with ml_col2:
        st.info("โมเดลเรียนรู้จากสถิติประชากรและช่วงเวลา เพื่อช่วยวางแผนรถเก็บขยะให้เพียงพอต่อความต้องการ")

except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {e}")