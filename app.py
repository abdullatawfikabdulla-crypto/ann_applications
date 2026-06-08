import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="نظام توقع نجاح الطالب", page_icon="🎓", layout="wide")
st.title("📚 نظام توقع نجاح الطالب")
st.markdown("---")

# تحميل النموذج باستخدام pickle (مضمنة ولا تحتاج تثبيت)
@st.cache_resource
def load_models():
    with open('my_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('my_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_models()

# بقية الدوال كما هي (شرح القرار، قاعدة البيانات، إلخ)
def explain_prediction(study_hours, attendance, exam_score):
    reasons = []
    if study_hours >= 5: reasons.append("✅ ساعات الدراسة كافية (≥5)")
    else: reasons.append(f"❌ ساعات الدراسة غير كافية ({study_hours} < 5)")
    if attendance >= 70: reasons.append("✅ نسبة الحضور كافية (≥70%)")
    else: reasons.append(f"❌ نسبة الحضور غير كافية ({attendance}% < 70%)")
    if exam_score >= 65: reasons.append("✅ درجة الامتحان كافية (≥65)")
    else: reasons.append(f"❌ درجة الامتحان غير كافية ({exam_score} < 65)")
    return reasons

DB_NAME = "predictions.db"
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, study_hours REAL, attendance REAL, age INTEGER,
                  exam_score REAL, prediction INTEGER, probability REAL, result_text TEXT)''')
    conn.commit()
    conn.close()

def save_prediction(study_hours, attendance, age, exam_score, pred, prob, result_text):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''INSERT INTO predictions (timestamp, study_hours, attendance, age, exam_score, prediction, probability, result_text)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().isoformat(), study_hours, attendance, age, exam_score, int(pred), float(prob), result_text))
    conn.commit()
    conn.close()

def load_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

init_db()

# التبويبات
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 توقع فردي", "📊 شرح التنبؤ", "📁 رفع ملف CSV", "📈 تحليل الحساسية", "📊 لوحة التحكم"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.number_input("ساعات الدراسة", 0.0, 24.0, 5.0, 0.5)
        attendance = st.number_input("نسبة الحضور (%)", 0.0, 100.0, 75.0, 1.0)
    with col2:
        exam_score = st.number_input("درجة الامتحان", 0.0, 100.0, 65.0, 1.0)
        age = st.number_input("العمر", 15, 100, 20, 1)
    if st.button("توقع النتيجة", type="primary"):
        input_data = np.array([[study_hours, attendance, age, exam_score]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        result_text = "ناجح" if pred == 1 else "راسب"
        save_prediction(study_hours, attendance, age, exam_score, pred, prob, result_text)
        st.subheader("النتيجة:")
        if pred == 1:
            st.success(f"✅ **ناجح** (احتمال النجاح: {prob:.1%})")
        else:
            st.error(f"❌ **راسب** (احتمال النجاح: {prob:.1%})")
        with st.expander("🔍 لماذا هذه النتيجة؟"):
            for r in explain_prediction(study_hours, attendance, exam_score):
                st.write(r)

with tab2:
    st.subheader("شرح مفصل لقرار النموذج")
    sh = st.slider("ساعات الدراسة", 0.0, 12.0, 5.0)
    att = st.slider("نسبة الحضور (%)", 40.0, 100.0, 70.0)
    ex = st.slider("درجة الامتحان", 30.0, 100.0, 65.0)
    ag = st.number_input("العمر", 18, 30, 20)
    if st.button("شرح القرار"):
        input_data = np.array([[sh, att, ag, ex]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        st.markdown(f"### النتيجة المتوقعة: **{'ناجح' if pred==1 else 'راسب'}** (احتمال النجاح {prob:.1%})")
        st.markdown("#### أسباب القرار:")
        for r in explain_prediction(sh, att, ex):
            st.write(r)

with tab3:
    st.subheader("رفع ملف CSV")
    uploaded = st.file_uploader("اختر ملف CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        required = ['StudyHours', 'Attendance', 'Age', 'ExamScore']
        if all(col in df.columns for col in required):
            if st.button("تنفيذ التنبؤات"):
                X_batch = scaler.transform(df[required].values)
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)[:, 1]
                df['Prediction'] = ['ناجح' if p==1 else 'راسب' for p in preds]
                df['Probability'] = probs
                st.dataframe(df)
                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button("تحميل النتائج CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")
        else:
            st.error(f"الأعمدة المطلوبة: {required}")

with tab4:
    st.subheader("تحليل الحساسية")
    base_sh = st.slider("ساعات الأساس", 0.0, 12.0, 5.0)
    base_att = st.slider("حضور الأساس (%)", 40.0, 100.0, 70.0)
    base_ex = st.slider("درجة الأساس", 30.0, 100.0, 65.0)
    age_fixed = 20
    study_range = np.linspace(0,12,50)
    att_range = np.linspace(40,100,50)
    probs_sh = [model.predict_proba(scaler.transform([[s, base_att, age_fixed, base_ex]]))[0][1] for s in study_range]
    probs_att = [model.predict_proba(scaler.transform([[base_sh, a, age_fixed, base_ex]]))[0][1] for a in att_range]
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.plot(study_range, probs_sh); ax1.axvline(x=5, color='r', linestyle='--'); ax1.set_xlabel('ساعات الدراسة'); ax1.set_ylabel('احتمال النجاح')
    ax2.plot(att_range, probs_att); ax2.axvline(x=70, color='r', linestyle='--'); ax2.set_xlabel('الحضور (%)')
    st.pyplot(fig)

with tab5:
    st.subheader("لوحة التحكم الإحصائية")
    df_db = load_all_predictions()
    if df_db.empty:
        st.info("لا توجد تنبؤات بعد، قم بالتوقع من التبويب الأول.")
    else:
        col1, col2, col3 = st.columns(3)
        total = len(df_db); success = df_db['prediction'].sum()
        col1.metric("إجمالي التنبؤات", total)
        col2.metric("نسبة النجاح", f"{success/total:.1%}")
        col3.metric("متوسط ساعات الدراسة", f"{df_db['study_hours'].mean():.1f}")
        st.dataframe(df_db[['timestamp', 'study_hours', 'attendance', 'exam_score', 'result_text', 'probability']].head(10))
        csv_all = df_db.to_csv(index=False).encode('utf-8')
        st.download_button("تحميل كل البيانات CSV", data=csv_all, file_name="all_predictions.csv")

st.markdown("---")
st.caption("تم التطوير بواسطة Streamlit + Pickle (بدون joblib)")
