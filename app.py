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
def load_initial_data():
    df = pd.read_csv('sustainable_waste_management_dataset_2024.csv', parse_dates=['date'])
    df.columns = df.columns.str.strip()
    df['year'] = df['date'].dt.year
    return df

# --- ระบบหน่วยความจำ (Session State) ---
if 'main_df' not in st.session_state:
    st.session_state.main_df = load_initial_data()

# เริ่มต้นบล็อก try เพื่อป้องกันแอปพังถ้าไฟล์หรือการคำนวณมีปัญหา
try:
    df_display = st.session_state.main_df

    # --- Header ---
    st.title("♻️ EcoTrack Insights")
    st.markdown("ระบบวิเคราะห์และบันทึกข้อมูลการจัดการขยะอัจฉริยะ")
    st.divider()

    # --- ส่วนที่ 1: แบบฟอร์มรับข้อมูล ---
    with st.expander("➕ บันทึกข้อมูลปริมาณขยะรายวันใหม่"):
        with st.form("waste_entry_form", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                in_date = st.date_input("วันที่บันทึก")
                in_area = st.selectbox("พื้นที่ (Area)", options=df_display['area'].unique())
            with f_col2:
                in_waste = st.number_input("ปริมาณขยะที่เก็บได้ (kg)", min_value=0.0)
                in_recycle = st.number_input("รีไซเคิลได้ (kg)", min_value=0.0)
            with f_col3:
                in_pop = st.number_input("จำนวนประชากรในวันนั้น", value=int(df_display['population'].mean()))
                in_temp = st.slider("อุณหภูมิ (°C)", 10.0, 45.0, 25.0)

            submitted = st.form_submit_button("บันทึกข้อมูลลงตาราง")
            
            if submitted:
                new_row = pd.DataFrame([{
                    'date': pd.to_datetime(in_date),
                    'area': in_area,
                    'waste_kg': in_waste,
                    'recyclable_kg': in_recycle,
                    'population': in_pop,
                    'temp_c': in_temp,
                    'collection_capacity_kg': df_display['collection_capacity_kg'].mean(),
                    'overflow': 1 if in_waste > df_display['collection_capacity_kg'].mean() else 0
                }])
                st.session_state.main_df = pd.concat([st.session_state.main_df, new_row], ignore_index=True)
                st.success("✅ บันทึกข้อมูลเรียบร้อย!")
                st.rerun()

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 ตัวกรองแดชบอร์ด")
    all_areas = df_display['area'].unique()
    selected_areas = st.sidebar.multiselect("เลือกพื้นที่:", options=all_areas, default=all_areas)
    filtered_df = df_display[df_display['area'].isin(selected_areas)]

    # --- ส่วนที่ 2: Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ปริมาณขยะรวม (kg)", f"{filtered_df['waste_kg'].sum():,.0f}")
    with col2:
        total_w = filtered_df['waste_kg'].sum()
        recycle_rate = (filtered_df['recyclable_kg'].sum() / total_w * 100) if total_w > 0 else 0
        st.metric("อัตราการรีไซเคิล", f"{recycle_rate:.1f}%")
    with col3:
        st.metric("อุณหภูมิเฉลี่ย", f"{filtered_df['temp_c'].mean():.1f} °C")
    with col4:
        st.metric("จำนวนรายการ", len(filtered_df))

    # --- ส่วนที่ 3: Visualizations & Table ---
    st.write("### 📈 ข้อมูลเชิงลึกและตารางข้อมูล")
    tab1, tab2, tab3 = st.tabs(["📊 แนวโน้มการจัดการ", "🌦️ ปัจจัยสภาพอากาศ", "📋 ตารางข้อมูล"])

    with tab1:
        st.write("**แนวโน้มปริมาณขยะรายวัน**")
        chart_data = filtered_df.groupby('date')[['waste_kg', 'collection_capacity_kg']].sum()
        st.line_chart(chart_data)

    with tab2:
        st.write("**ความสัมพันธ์ระหว่างอุณหภูมิและปริมาณขยะ**")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(filtered_df['temp_c'], filtered_df['waste_kg'], color='salmon', alpha=0.5)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Waste (kg)")
        st.pyplot(fig)

    with tab3:
        st.write("**ตารางข้อมูลทั้งหมด**")
        st.dataframe(filtered_df.sort_values(by='date', ascending=False), use_container_width=True)
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv_data, file_name='waste_report.csv', mime='text/csv')

    # --- ส่วนที่ 4: Machine Learning ---
    st.divider()
    st.write("### 🤖 ระบบพยากรณ์ปริมาณขยะ (AI Prediction)")
    
    ml_df = st.session_state.main_df.dropna().copy()
    ml_df['date_ordinal'] = ml_df['date'].apply(lambda x: x.toordinal())
    
    X = ml_df[['date_ordinal', 'population', 'temp_c']]
    y = ml_df['waste_kg']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ml_col1, ml_col2 = st.columns([1, 2])
    with ml_col1:
        st.write("**ทำนายผลขยะรายวัน**")
        pop_in = st.number_input("ระบุจำนวนประชากร:", value=int(ml_df['population'].mean()))
        temp_in = st.slider("ระบุอุณหภูมิ (°C):", 10.0, 45.0, 30.0)
        
        if st.button("คำนวณโดย AI"):
            curr_date = pd.Timestamp.now().toordinal()
            pred = model.predict([[curr_date, pop_in, temp_in]])
            st.success(f"ปริมาณขยะที่คาดการณ์: {pred[0]:,.2f} kg")
            st.info(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    with ml_col2:
        fig_ml, ax_ml = plt.subplots(figsize=(8, 5))
        ax_ml.scatter(y_test, y_pred, color='skyblue', alpha=0.6, label='Predicted vs Actual')
        ax_ml.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Line')
        ax_ml.set_xlabel("Actual Waste (kg)")
        ax_ml.set_ylabel("Predicted Waste (kg)")
        ax_ml.legend()
        st.pyplot(fig_ml)

# ปิดบล็อก try ด้วย except เพื่อแก้ Syntax Error
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {e}")
