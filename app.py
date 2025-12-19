import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Sustainable Waste Dashboard", page_icon="♻️", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('sustainable_waste_management_dataset_2024.csv', parse_dates=['date'])
    # ตัดช่องว่างชื่อคอลัมน์เผื่อไว้
    df.columns = df.columns.str.strip()
    df['year'] = df['date'].dt.year
    return df

try:
    df = load_data()

    # --- Header ---
    st.title("♻️ Dashboard วิเคราะห์ข้อมูลขยะยั่งยืน 2024")
    st.markdown("---")

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 ตัวกรองข้อมูล")
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
        st.metric("จำนวนวันที่เกิด Overflow", f"{filtered_df['overflow'].sum()} วัน")

    # --- ส่วนที่ 2: Visualizations ---
    st.write("### 📈 แนวโน้มและสถิติข้อมูล")
    tab1, tab2 = st.tabs(["แนวโน้มปริมาณขยะ", "ความสัมพันธ์ของสภาพอากาศ"])
    
    with tab1:
        # กราฟเส้นแสดงขยะเปรียบเทียบกับขีดความสามารถการเก็บ
        st.line_chart(filtered_df.set_index('date')[['waste_kg', 'collection_capacity_kg']])
    
    with tab2:
        # กราฟความสัมพันธ์ระหว่างฝนกับปริมาณขยะ
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(filtered_df['rain_mm'], filtered_df['waste_kg'], color='teal', alpha=0.5)
        ax.set_xlabel("Rainfall (mm)")
        ax.set_ylabel("Waste (kg)")
        ax.set_title("Relationship between Rainfall and Waste Amount")
        st.pyplot(fig)

    # --- ส่วนที่ 3: Machine Learning ---
    st.divider()
    st.write("### 🤖 ระบบพยากรณ์ปริมาณขยะ (AI Prediction)")
    
    # เตรียมข้อมูล ML
    df_ml = df.dropna().copy()
    # แก้ไขจุดที่ Error: ใช้ toordinal()
    df_ml['date_ordinal'] = df_ml['date'].apply(lambda x: x.toordinal())
    
    X = df_ml[['date_ordinal', 'population']]
    y = df_ml['waste_kg']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ml_col1, ml_col2 = st.columns([1, 2])
    
    with ml_col1:
        st.info(f"**ความแม่นยำของโมเดล:** {r2_score(y_test, y_pred):.4f}")
        st.write("ระบุจำนวนประชากรเพื่อพยากรณ์ขยะที่จะเกิดขึ้นในวันนี้:")
        pop_input = st.number_input("จำนวนประชากร (คน):", value=int(df['population'].mean()))
        
        if st.button("ทำนายผล"):
            # แก้ไขจุดที่ Error: ใช้ toordinal() สำหรับวันที่ปัจจุบันด้วย
            current_date_ord = pd.Timestamp.now().toordinal()
            pred = model.predict([[current_date_ord, pop_input]])
            st.success(f"ปริมาณขยะที่คาดการณ์: {pred[0]:,.2f} kg")

    with ml_col2:
        # กราฟเปรียบเทียบค่าจริงกับค่าทำนาย
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test, y_pred, color='darkorange', alpha=0.4)
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax2.set_xlabel("Actual (kg)")
        ax2.set_ylabel("Predicted (kg)")
        st.pyplot(fig2)

except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")