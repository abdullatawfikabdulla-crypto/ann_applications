import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
from datetime import datetime
from io import BytesIO

# ---------------------- إعداد الصفحة ----------------------
st.set_page_config(page_title="نظام توقع نجاح الطالب - إحصائيات", page_icon="🎓", layout="wide")
st.title("📚 نظام توقع نجاح الطالب (نسخة إحصائية متقدمة)")
st.markdown("---")

# ---------------------- تحميل النموذج والمعاملات ----------------------
@st.cache_resource
def load_models():
    model = joblib.load('my_model.pkl')
    scaler = joblib.load('my_scaler.pkl')
    return model, scaler

model, scaler = load_models()

# ---------------------- دالة شرح القرار ----------------------
def explain_prediction(study_hours, attendance, exam_score):
    reasons = []
    if study_hours >= 5:
        reasons.append("✅ ساعات الدراسة كافية (≥5)")
    else:
        reasons.append(f"❌ ساعات الدراسة غير كافية ({study_hours} < 5)")
    if attendance >= 70:
        reasons.append("✅ نسبة الحضور كافية (≥70%)")
    else:
        reasons.append(f"❌ نسبة الحضور غير كافية ({attendance}% < 70%)")
    if exam_score >= 65:
        reasons.append("✅ درجة الامتحان كافية (≥65)")
    else:
        reasons.append(f"❌ درجة الامتحان غير كافية ({exam_score} < 65)")
    return reasons

# ---------------------- قاعدة البيانات (SQLite) ----------------------
DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  study_hours REAL,
                  attendance REAL,
                  age INTEGER,
                  exam_score REAL,
                  prediction INTEGER,
                  probability REAL,
                  result_text TEXT)''')
    conn.commit()
    conn.close()

def save_prediction(study_hours, attendance, age, exam_score, pred, prob, result_text):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''INSERT INTO predictions 
                 (timestamp, study_hours, attendance, age, exam_score, prediction, probability, result_text)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().isoformat(), study_hours, attendance, age, exam_score, int(pred), float(prob), result_text))
    conn.commit()
    conn.close()

def load_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# تهيئة قاعدة البيانات
init_db()

# ---------------------- إنشاء التبويبات ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 توقع فردي",
    "📊 شرح التنبؤ",
    "📁 رفع ملف CSV",
    "📈 تحليل الحساسية",
    "📊 لوحة التحكم الإحصائية"
])

# ===================== التبويب 1: توقع فردي (مع حفظ في قاعدة البيانات) =====================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.number_input("ساعات الدراسة", 0.0, 24.0, 5.0, 0.5, key="sh")
        attendance = st.number_input("نسبة الحضور (%)", 0.0, 100.0, 75.0, 1.0, key="att")
    with col2:
        exam_score = st.number_input("درجة الامتحان", 0.0, 100.0, 65.0, 1.0, key="ex")
        age = st.number_input("العمر", 15, 100, 20, 1, key="age")
    
    if st.button("توقع النتيجة", type="primary", key="predict1"):
        input_data = np.array([[study_hours, attendance, age, exam_score]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        result_text = "ناجح" if pred == 1 else "راسب"
        
        # حفظ في قاعدة البيانات
        save_prediction(study_hours, attendance, age, exam_score, pred, prob, result_text)
        
        st.subheader("النتيجة:")
        if pred == 1:
            st.success(f"✅ **ناجح** (احتمال النجاح: {prob:.1%})")
        else:
            st.error(f"❌ **راسب** (احتمال النجاح: {prob:.1%})")
        
        with st.expander("🔍 لماذا هذه النتيجة؟"):
            reasons = explain_prediction(study_hours, attendance, exam_score)
            for r in reasons:
                st.write(r)

# ===================== التبويب 2: شرح التنبؤ المتقدم =====================
with tab2:
    st.subheader("شرح مفصل لقرار النموذج")
    sh = st.slider("ساعات الدراسة", 0.0, 12.0, 5.0, 0.5, key="sh_exp")
    att = st.slider("نسبة الحضور (%)", 40.0, 100.0, 70.0, 1.0, key="att_exp")
    ex = st.slider("درجة الامتحان", 30.0, 100.0, 65.0, 1.0, key="ex_exp")
    ag = st.number_input("العمر", 18, 30, 20, key="age_exp")
    
    if st.button("شرح القرار", key="explain_btn"):
        input_data = np.array([[sh, att, ag, ex]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        st.markdown(f"### النتيجة المتوقعة: **{'ناجح' if pred==1 else 'راسب'}** (احتمال النجاح {prob:.1%})")
        st.markdown("#### أسباب القرار:")
        reasons = explain_prediction(sh, att, ex)
        for r in reasons:
            st.write(r)
        
        st.markdown("#### أهمية الميزات (تقديرية):")
        st.bar_chart(pd.DataFrame({"ساعات الدراسة": 0.45, "الحضور": 0.35, "درجة الامتحان": 0.20}, index=["تأثير"]))

# ===================== التبويب 3: رفع ملف CSV =====================
with tab3:
    st.subheader("رفع ملف CSV لعدة طلاب")
    st.markdown("يجب أن يحتوي الملف على الأعمدة: `StudyHours, Attendance, Age, ExamScore`")
    uploaded_file = st.file_uploader("اختر ملف CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("البيانات المرفوعة:")
        st.dataframe(df.head())
        
        required_cols = ['StudyHours', 'Attendance', 'Age', 'ExamScore']
        if all(col in df.columns for col in required_cols):
            if st.button("تنفيذ التنبؤات", key="batch_predict"):
                X_batch = df[required_cols].values
                X_scaled_batch = scaler.transform(X_batch)
                predictions = model.predict(X_scaled_batch)
                probabilities = model.predict_proba(X_scaled_batch)[:, 1]
                df['Prediction'] = ['ناجح' if p==1 else 'راسب' for p in predictions]
                df['Probability'] = probabilities
                st.success("تمت التنبؤات!")
                st.dataframe(df)
                
                output = BytesIO()
                df.to_csv(output, index=False)
                st.download_button("📥 تحميل النتائج CSV", data=output.getvalue(), file_name="predictions.csv", mime="text/csv")
                
                st.markdown("#### ملخص التنبؤات:")
                st.write(df['Prediction'].value_counts())
        else:
            st.error(f"الملف يجب أن يحتوي على الأعمدة: {required_cols}")

# ===================== التبويب 4: تحليل الحساسية =====================
with tab4:
    st.subheader("تحليل الحساسية: كيف يتغير التنبؤ بتغير ساعات الدراسة أو الحضور؟")
    base_sh = st.slider("القيمة الأساسية لساعات الدراسة", 0.0, 12.0, 5.0, 0.5, key="base_sh")
    base_att = st.slider("القيمة الأساسية للحضور (%)", 40.0, 100.0, 70.0, 1.0, key="base_att")
    base_ex = st.slider("القيمة الأساسية لدرجة الامتحان", 30.0, 100.0, 65.0, 1.0, key="base_ex")
    age_fixed = 20
    
    study_range = np.linspace(0, 12, 50)
    attendance_range = np.linspace(40, 100, 50)
    
    probs_study = []
    for sh in study_range:
        inp = np.array([[sh, base_att, age_fixed, base_ex]])
        inp_scaled = scaler.transform(inp)
        prob = model.predict_proba(inp_scaled)[0][1]
        probs_study.append(prob)
    
    probs_att = []
    for att in attendance_range:
        inp = np.array([[base_sh, att, age_fixed, base_ex]])
        inp_scaled = scaler.transform(inp)
        prob = model.predict_proba(inp_scaled)[0][1]
        probs_att.append(prob)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(study_range, probs_study, 'b-', linewidth=2)
    ax1.axvline(x=5, color='r', linestyle='--', label='الحد الأدنى (5 ساعات)')
    ax1.set_xlabel('ساعات الدراسة')
    ax1.set_ylabel('احتمال النجاح')
    ax1.set_title('تأثير ساعات الدراسة')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(attendance_range, probs_att, 'g-', linewidth=2)
    ax2.axvline(x=70, color='r', linestyle='--', label='الحد الأدنى (70%)')
    ax2.set_xlabel('نسبة الحضور (%)')
    ax2.set_ylabel('احتمال النجاح')
    ax2.set_title('تأثير الحضور')
    ax2.grid(True)
    ax2.legend()
    
    st.pyplot(fig)

# ===================== التبويب 5: لوحة التحكم الإحصائية مع قاعدة البيانات =====================
with tab5:
    st.subheader("📊 لوحة التحكم الإحصائية - سجل التنبؤات وتحليلاتها")
    
    df_db = load_all_predictions()
    
    if df_db.empty:
        st.info("📭 لا توجد تنبؤات مسجلة بعد. قم بالتوقع من التبويب 'توقع فردي' وستظهر البيانات هنا.")
    else:
        # مؤشرات الأداء الرئيسية (KPIs)
        col1, col2, col3, col4, col5 = st.columns(5)
        total = len(df_db)
        success_count = df_db[df_db['prediction'] == 1].shape[0]
        success_rate = success_count / total if total > 0 else 0
        avg_study = df_db['study_hours'].mean()
        avg_att = df_db['attendance'].mean()
        avg_exam = df_db['exam_score'].mean()
        
        col1.metric("📌 إجمالي التنبؤات", total)
        col2.metric("🏆 نسبة النجاح", f"{success_rate:.1%}")
        col3.metric("📖 متوسط ساعات الدراسة", f"{avg_study:.1f}")
        col4.metric("👥 متوسط الحضور", f"{avg_att:.1f}%")
        col5.metric("📝 متوسط درجة الامتحان", f"{avg_exam:.1f}")
        
        # رسمان بيانيان تفاعليان
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            st.markdown("#### توزيع النتائج")
            fig_pie = px.pie(names=['ناجح', 'راسب'], 
                             values=[success_count, total - success_count],
                             color_discrete_sequence=['green', 'red'],
                             hole=0.4)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_ch2:
            st.markdown("#### توزيع ساعات الدراسة (لجميع التنبؤات)")
            fig_hist = px.histogram(df_db, x='study_hours', nbins=15, 
                                    title="", labels={'study_hours': 'ساعات الدراسة'})
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # جدول بأحدث التنبؤات
        st.markdown("### 📋 أحدث التنبؤات المسجلة")
        display_df = df_db[['timestamp', 'study_hours', 'attendance', 'exam_score', 'result_text', 'probability']].head(10)
        st.dataframe(display_df.style.format({'probability': '{:.1%}'}))
        
        # أزرار التصدير والمسح
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_all = df_db.to_csv(index=False).encode('utf-8')
            st.download_button("📥 تحميل جميع البيانات (CSV)", data=csv_all, file_name="all_predictions.csv", mime="text/csv")
        with col_exp2:
            if st.button("🗑️ مسح قاعدة البيانات (حذف كل التنبؤات)"):
                conn = sqlite3.connect(DB_NAME)
                conn.execute("DELETE FROM predictions")
                conn.commit()
                conn.close()
                st.success("تم مسح قاعدة البيانات بنجاح")
                st.rerun()

st.markdown("---")
st.caption("نظام متقدم لتوقع نجاح الطلاب - مزود بلوحة تحكم وقاعدة بيانات محلية")