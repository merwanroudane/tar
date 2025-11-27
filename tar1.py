# -*- coding: utf-8 -*-
"""
تطبيق تفاعلي شامل لشرح نماذج العتبة (Threshold Models)
Comprehensive Interactive Application for Threshold Models Education
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="نماذج العتبة | Threshold Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== التنسيقات CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #a2d2ff;
        font-size: 1.2rem;
    }
    
    .concept-box {
        background: linear-gradient(145deg, #1e3a5f, #2d5a87);
        padding: 1.5rem;
        border-radius: 12px;
        border-right: 5px solid #e94560;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .formula-box {
        background: linear-gradient(145deg, #0d1b2a, #1b263b);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #00b4d8;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 5px 25px rgba(0,180,216,0.2);
    }
    
    .term-box {
        background: linear-gradient(145deg, #2d3436, #353b48);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #00cec9;
    }
    
    .term-ar {
        color: #ffeaa7;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .term-en {
        color: #81ecec;
        font-size: 0.95rem;
        font-style: italic;
    }
    
    .term-def {
        color: #dfe6e9;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .warning-box {
        background: linear-gradient(145deg, #6c3483, #8e44ad);
        padding: 1.5rem;
        border-radius: 12px;
        border-right: 5px solid #f39c12;
        margin: 1rem 0;
        color: white;
    }
    
    .success-box {
        background: linear-gradient(145deg, #1e8449, #27ae60);
        padding: 1.5rem;
        border-radius: 12px;
        border-right: 5px solid #2ecc71;
        margin: 1rem 0;
        color: white;
    }
    
    .info-box {
        background: linear-gradient(145deg, #2471a3, #3498db);
        padding: 1.5rem;
        border-radius: 12px;
        border-right: 5px solid #5dade2;
        margin: 1rem 0;
        color: white;
    }
    
    .step-box {
        background: linear-gradient(145deg, #34495e, #2c3e50);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid #e74c3c;
    }
    
    .step-number {
        background: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 50%;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    .highlight {
        background: linear-gradient(120deg, #f39c12 0%, #e74c3c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e3a5f;
        border-radius: 8px;
        color: white;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
    }
    
    div[data-testid="stExpander"] {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    
    .comparison-table {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .davies-box {
        background: linear-gradient(145deg, #641e16, #922b21);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #f39c12;
        margin: 1rem 0;
        color: white;
    }
    
    .hansen-box {
        background: linear-gradient(145deg, #145a32, #1e8449);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #58d68d;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== الشريط الجانبي ====================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #e94560;'>📚 فهرس المحتويات</h2>
        <p style='color: #a2d2ff;'>Table of Contents</p>
    </div>
    """, unsafe_allow_html=True)
    
    section = st.radio(
        "اختر القسم | Select Section",
        [
            "🏠 المقدمة | Introduction",
            "📖 المفاهيم الأساسية | Basic Concepts",
            "📐 الصيغ الرياضية | Mathematical Formulas",
            "🔍 أنواع نماذج العتبة | Types of Models",
            "⚠️ مشكلة Davies | Davies Problem",
            "✅ حل Hansen | Hansen's Solution",
            "🧮 إيجاد العتبات | Finding Thresholds",
            "📊 الاختبارات الإحصائية | Statistical Tests",
            "🎯 التطبيق العملي | Practical Application",
            "📈 محاكاة تفاعلية | Interactive Simulation",
            "📋 ملخص شامل | Comprehensive Summary"
        ],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class='info-box' style='font-size: 0.85rem;'>
        <b>💡 نصيحة:</b><br>
        ابدأ من المقدمة وتابع بالترتيب للحصول على فهم شامل
        <br><br>
        <b>💡 Tip:</b><br>
        Start from Introduction and follow in order for comprehensive understanding
    </div>
    """, unsafe_allow_html=True)

# ==================== العنوان الرئيسي ====================
st.markdown("""
<div class='main-header'>
    <h1>🎯 نماذج العتبة الشاملة</h1>
    <p>Comprehensive Threshold Models Guide</p>
    <p style='color: #ffeaa7; font-size: 0.9rem;'>دليلك الكامل من الصفر إلى الاحتراف</p>
</div>
""", unsafe_allow_html=True)

# ==================== دوال مساعدة ====================
def create_threshold_data(n=200, threshold=5, beta1=2, beta2=-1, noise=1, seed=42):
    """إنشاء بيانات نموذج عتبة"""
    np.random.seed(seed)
    q = np.linspace(0, 10, n)
    x = np.random.randn(n) * 2 + 5
    e = np.random.randn(n) * noise
    
    y = np.where(q <= threshold, 
                 beta1 * x + e,
                 beta2 * x + e)
    
    return pd.DataFrame({'y': y, 'x': x, 'q': q})

def estimate_ssr(data, threshold, x_col='x', y_col='y', q_col='q'):
    """حساب مجموع مربعات البواقي لعتبة معينة"""
    regime1 = data[data[q_col] <= threshold]
    regime2 = data[data[q_col] > threshold]
    
    if len(regime1) < 5 or len(regime2) < 5:
        return np.inf
    
    # تقدير النظام الأول
    X1 = np.column_stack([np.ones(len(regime1)), regime1[x_col]])
    y1 = regime1[y_col].values
    try:
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        ssr1 = np.sum((y1 - X1 @ beta1)**2)
    except:
        return np.inf
    
    # تقدير النظام الثاني
    X2 = np.column_stack([np.ones(len(regime2)), regime2[x_col]])
    y2 = regime2[y_col].values
    try:
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        ssr2 = np.sum((y2 - X2 @ beta2)**2)
    except:
        return np.inf
    
    return ssr1 + ssr2

def grid_search_threshold(data, q_col='q', trim=0.15):
    """البحث الشبكي عن العتبة المثلى"""
    q_sorted = np.sort(data[q_col].values)
    n = len(q_sorted)
    lower_idx = int(n * trim)
    upper_idx = int(n * (1 - trim))
    
    candidates = q_sorted[lower_idx:upper_idx]
    
    ssr_values = []
    for gamma in candidates:
        ssr = estimate_ssr(data, gamma, q_col=q_col)
        ssr_values.append(ssr)
    
    min_idx = np.argmin(ssr_values)
    optimal_threshold = candidates[min_idx]
    min_ssr = ssr_values[min_idx]
    
    return optimal_threshold, min_ssr, candidates, ssr_values

def bootstrap_p_value(data, n_bootstrap=500, trim=0.15, seed=42):
    """حساب قيمة p باستخدام Bootstrap (Hansen 1996)"""
    np.random.seed(seed)
    
    # تقدير النموذج الخطي
    X = np.column_stack([np.ones(len(data)), data['x']])
    y = data['y'].values
    beta_linear = np.linalg.lstsq(X, y, rcond=None)[0]
    ssr_linear = np.sum((y - X @ beta_linear)**2)
    
    # تقدير نموذج العتبة
    opt_threshold, ssr_threshold, _, _ = grid_search_threshold(data, trim=trim)
    
    # إحصائية الاختبار الأصلية
    F_stat = (ssr_linear - ssr_threshold) / (ssr_threshold / (len(data) - 4))
    
    # Bootstrap
    residuals = y - X @ beta_linear
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # إعادة عينة البواقي
        boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        boot_y = X @ beta_linear + boot_residuals
        boot_data = data.copy()
        boot_data['y'] = boot_y
        
        # SSR للنموذج الخطي على بيانات Bootstrap
        ssr_linear_boot = np.sum(boot_residuals**2)
        
        # SSR لنموذج العتبة على بيانات Bootstrap
        _, ssr_threshold_boot, _, _ = grid_search_threshold(boot_data, trim=trim)
        
        # إحصائية Bootstrap
        F_boot = (ssr_linear_boot - ssr_threshold_boot) / (ssr_threshold_boot / (len(data) - 4))
        bootstrap_stats.append(F_boot)
    
    # قيمة p
    p_value = np.mean(np.array(bootstrap_stats) >= F_stat)
    
    return F_stat, p_value, bootstrap_stats

# ==================== الأقسام ====================

if section == "🏠 المقدمة | Introduction":
    st.markdown("""
    <div class='concept-box'>
        <h2 style='color: #ffeaa7;'>🎯 ما هي نماذج العتبة؟ | What are Threshold Models?</h2>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        نماذج العتبة هي فئة من النماذج الإحصائية التي تسمح بتغير العلاقة بين المتغيرات 
        عند نقطة معينة تسمى <span style='color: #e94560; font-weight: bold;'>العتبة (Threshold)</span>.
        <br><br>
        بمعنى آخر، بدلاً من افتراض علاقة ثابتة واحدة بين المتغيرات، 
        نسمح للعلاقة بالتغير بناءً على قيمة متغير معين.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <h3 style='color: #ffeaa7;'>🤔 لماذا نحتاج نماذج العتبة؟</h3>
            <ul style='line-height: 2;'>
                <li>العلاقات الاقتصادية قد تتغير في ظروف مختلفة</li>
                <li>السياسات النقدية قد تعمل بشكل مختلف في فترات التضخم</li>
                <li>سلوك المستهلك يتغير عند مستويات دخل معينة</li>
                <li>الأسواق المالية تتصرف بشكل مختلف في فترات الأزمات</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='success-box'>
            <h3 style='color: #ffeaa7;'>✨ مميزات نماذج العتبة</h3>
            <ul style='line-height: 2;'>
                <li>المرونة في التقاط العلاقات غير الخطية</li>
                <li>سهولة التفسير مقارنة بالنماذج غير الخطية الأخرى</li>
                <li>إمكانية تحديد نقاط التحول في البيانات</li>
                <li>تطبيقات واسعة في الاقتصاد والمالية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # مثال بصري
    st.markdown("### 📊 مثال بصري | Visual Example")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### النموذج الخطي التقليدي")
        # بيانات خطية
        np.random.seed(42)
        x_linear = np.linspace(0, 10, 100)
        y_linear = 2 * x_linear + np.random.randn(100) * 2
        
        fig_linear = go.Figure()
        fig_linear.add_trace(go.Scatter(
            x=x_linear, y=y_linear,
            mode='markers',
            marker=dict(color='#3498db', size=8),
            name='البيانات'
        ))
        fig_linear.add_trace(go.Scatter(
            x=x_linear, y=2*x_linear,
            mode='lines',
            line=dict(color='#e74c3c', width=3),
            name='خط الانحدار'
        ))
        fig_linear.update_layout(
            template='plotly_dark',
            height=350,
            title='علاقة خطية واحدة',
            xaxis_title='X',
            yaxis_title='Y'
        )
        st.plotly_chart(fig_linear, use_container_width=True)
    
    with col2:
        st.markdown("#### نموذج العتبة")
        # بيانات عتبة
        np.random.seed(42)
        x_thresh = np.linspace(0, 10, 100)
        threshold = 5
        y_thresh = np.where(x_thresh <= threshold,
                          2 * x_thresh + np.random.randn(100) * 1.5,
                          -1 * x_thresh + 15 + np.random.randn(100) * 1.5)
        
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(
            x=x_thresh[x_thresh <= threshold], 
            y=y_thresh[x_thresh <= threshold],
            mode='markers',
            marker=dict(color='#3498db', size=8),
            name='النظام الأول'
        ))
        fig_thresh.add_trace(go.Scatter(
            x=x_thresh[x_thresh > threshold], 
            y=y_thresh[x_thresh > threshold],
            mode='markers',
            marker=dict(color='#e74c3c', size=8),
            name='النظام الثاني'
        ))
        fig_thresh.add_vline(x=threshold, line_dash="dash", line_color="#f39c12", line_width=2)
        fig_thresh.add_annotation(x=threshold, y=max(y_thresh), text="العتبة γ",
                                 showarrow=True, arrowhead=1)
        fig_thresh.update_layout(
            template='plotly_dark',
            height=350,
            title='علاقة تتغير عند العتبة',
            xaxis_title='X (متغير العتبة)',
            yaxis_title='Y'
        )
        st.plotly_chart(fig_thresh, use_container_width=True)
    
    st.markdown("""
    <div class='warning-box'>
        <h3>💡 الفكرة الأساسية</h3>
        <p style='font-size: 1.1rem;'>
        في النموذج الخطي التقليدي، نفترض أن العلاقة بين X و Y ثابتة دائماً.
        <br><br>
        في نموذج العتبة، نسمح للعلاقة بالتغير: 
        <span style='color: #ffeaa7;'>قبل العتبة</span> تكون العلاقة بطريقة معينة،
        و<span style='color: #81ecec;'>بعد العتبة</span> تتغير العلاقة!
        </p>
    </div>
    """, unsafe_allow_html=True)

elif section == "📖 المفاهيم الأساسية | Basic Concepts":
    st.markdown("## 📖 المفاهيم والمصطلحات الأساسية")
    st.markdown("### Basic Concepts and Terminology")
    
    # قاموس المصطلحات
    terms = [
        {
            "ar": "العتبة",
            "en": "Threshold (γ)",
            "def": "القيمة الحرجة التي تقسم البيانات إلى نظامين مختلفين. عندما يتجاوز متغير العتبة هذه القيمة، تتغير العلاقة بين المتغيرات."
        },
        {
            "ar": "متغير العتبة",
            "en": "Threshold Variable (q)",
            "def": "المتغير الذي يُستخدم لتحديد أي نظام ينطبق. يُقارن بقيمة العتبة لتحديد النظام المناسب."
        },
        {
            "ar": "النظام / الحالة",
            "en": "Regime",
            "def": "كل جزء من النموذج له معاملات خاصة. النظام الأول عندما q ≤ γ، والنظام الثاني عندما q > γ."
        },
        {
            "ar": "الدالة المؤشرة",
            "en": "Indicator Function I(·)",
            "def": "دالة تأخذ القيمة 1 إذا تحقق الشرط، و0 إذا لم يتحقق. تُستخدم لتحديد النظام الفعال."
        },
        {
            "ar": "معاملات النظام",
            "en": "Regime Coefficients (β₁, β₂)",
            "def": "المعاملات التي تصف العلاقة في كل نظام. β₁ للنظام الأول و β₂ للنظام الثاني."
        },
        {
            "ar": "نسبة القص",
            "en": "Trimming Ratio (π)",
            "def": "نسبة من البيانات تُستبعد من طرفي التوزيع عند البحث عن العتبة. عادة 10-15%."
        },
        {
            "ar": "مجموع مربعات البواقي",
            "en": "Sum of Squared Residuals (SSR)",
            "def": "مقياس لجودة ملاءمة النموذج. SSR = Σ(yᵢ - ŷᵢ)². نختار العتبة التي تقلل SSR."
        },
        {
            "ar": "اختبار الخطية",
            "en": "Linearity Test",
            "def": "اختبار فرضية العدم بأن النموذج خطي (لا توجد عتبة) مقابل الفرضية البديلة بوجود عتبة."
        }
    ]
    
    for term in terms:
        st.markdown(f"""
        <div class='term-box'>
            <span class='term-ar'>🔹 {term['ar']}</span>
            <br>
            <span class='term-en'>{term['en']}</span>
            <p class='term-def'>{term['def']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📐 التمثيل البياني للمفاهيم")
    
    # رسم توضيحي تفاعلي
    col1, col2 = st.columns([1, 2])
    
    with col1:
        threshold_val = st.slider("قيمة العتبة γ", 2.0, 8.0, 5.0, 0.1)
        beta1_val = st.slider("معامل النظام الأول β₁", -3.0, 3.0, 2.0, 0.1)
        beta2_val = st.slider("معامل النظام الثاني β₂", -3.0, 3.0, -1.0, 0.1)
    
    with col2:
        np.random.seed(42)
        q = np.linspace(0, 10, 200)
        x = np.random.randn(200) * 2 + 5
        
        y = np.where(q <= threshold_val,
                    beta1_val * (q - threshold_val/2),
                    beta2_val * (q - threshold_val) + beta1_val * threshold_val/2)
        
        fig = go.Figure()
        
        # النظام الأول
        mask1 = q <= threshold_val
        fig.add_trace(go.Scatter(
            x=q[mask1], y=y[mask1],
            mode='lines',
            line=dict(color='#3498db', width=4),
            name=f'النظام الأول (β₁={beta1_val})'
        ))
        
        # النظام الثاني
        mask2 = q > threshold_val
        fig.add_trace(go.Scatter(
            x=q[mask2], y=y[mask2],
            mode='lines',
            line=dict(color='#e74c3c', width=4),
            name=f'النظام الثاني (β₂={beta2_val})'
        ))
        
        # خط العتبة
        fig.add_vline(x=threshold_val, line_dash="dash", line_color="#f39c12", line_width=3)
        fig.add_annotation(
            x=threshold_val, y=max(y)*1.1,
            text=f"γ = {threshold_val}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#f39c12",
            font=dict(size=14, color="#f39c12")
        )
        
        # منطقة النظام الأول
        fig.add_vrect(x0=0, x1=threshold_val, fillcolor="blue", opacity=0.1,
                     annotation_text="النظام الأول", annotation_position="top left")
        
        # منطقة النظام الثاني
        fig.add_vrect(x0=threshold_val, x1=10, fillcolor="red", opacity=0.1,
                     annotation_text="النظام الثاني", annotation_position="top right")
        
        fig.update_layout(
            template='plotly_dark',
            height=450,
            title='التمثيل البياني لنموذج العتبة',
            xaxis_title='متغير العتبة (q)',
            yaxis_title='المتغير التابع (y)',
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class='concept-box'>
        <h3>🎯 فهم الرسم:</h3>
        <ul style='line-height: 2;'>
            <li><span style='color: #3498db;'>الخط الأزرق</span>: يمثل النظام الأول حيث q ≤ γ</li>
            <li><span style='color: #e74c3c;'>الخط الأحمر</span>: يمثل النظام الثاني حيث q > γ</li>
            <li><span style='color: #f39c12;'>الخط المتقطع</span>: يمثل نقطة العتبة γ</li>
            <li>جرب تغيير قيم β₁ و β₂ لترى كيف تتغير العلاقة في كل نظام!</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif section == "📐 الصيغ الرياضية | Mathematical Formulas":
    st.markdown("## 📐 الصيغ الرياضية لنماذج العتبة")
    st.markdown("### Mathematical Formulas for Threshold Models")
    
    st.markdown("""
    <div class='formula-box'>
        <h3 style='color: #ffeaa7;'>الصيغة العامة لنموذج العتبة البسيط</h3>
        <h4 style='color: #81ecec;'>Simple Threshold Regression Model</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    y_t = \begin{cases}
    \beta_1' x_t + e_t & \text{if } q_t \leq \gamma \\
    \beta_2' x_t + e_t & \text{if } q_t > \gamma
    \end{cases}
    ''')
    
    st.markdown("""
    <div class='term-box'>
        <p><b>حيث | where:</b></p>
        <ul>
            <li><b>y_t</b>: المتغير التابع (Dependent Variable)</li>
            <li><b>x_t</b>: متجه المتغيرات المستقلة (Vector of Independent Variables)</li>
            <li><b>q_t</b>: متغير العتبة (Threshold Variable)</li>
            <li><b>γ</b>: قيمة العتبة (Threshold Value)</li>
            <li><b>β₁, β₂</b>: متجهات المعاملات لكل نظام (Coefficient Vectors)</li>
            <li><b>e_t</b>: حد الخطأ (Error Term)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='formula-box'>
        <h3 style='color: #ffeaa7;'>الصيغة باستخدام الدالة المؤشرة</h3>
        <h4 style='color: #81ecec;'>Using Indicator Function</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    y_t = \beta_1' x_t \cdot I(q_t \leq \gamma) + \beta_2' x_t \cdot I(q_t > \gamma) + e_t
    ''')
    
    st.markdown("""
    <div class='term-box'>
        <p><b>الدالة المؤشرة | Indicator Function:</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    I(q_t \leq \gamma) = \begin{cases}
    1 & \text{if } q_t \leq \gamma \\
    0 & \text{if } q_t > \gamma
    \end{cases}
    ''')
    
    st.markdown("---")
    
    st.markdown("""
    <div class='formula-box'>
        <h3 style='color: #ffeaa7;'>الصيغة المدمجة (Compact Form)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    y_t = \theta' x_t(\gamma) + e_t
    ''')
    
    st.markdown("""
    <div class='term-box'>
        <p><b>حيث:</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    x_t(\gamma) = \begin{pmatrix} x_t \cdot I(q_t \leq \gamma) \\ x_t \cdot I(q_t > \gamma) \end{pmatrix}
    \quad \text{و} \quad
    \theta = \begin{pmatrix} \beta_1 \\ \beta_2 \end{pmatrix}
    ''')
    
    st.markdown("---")
    
    st.markdown("### 📊 مجموع مربعات البواقي | Sum of Squared Residuals")
    
    st.latex(r'''
    S_n(\gamma) = \sum_{t=1}^{n} (y_t - \hat{y}_t(\gamma))^2 = \sum_{t=1}^{n} \hat{e}_t^2(\gamma)
    ''')
    
    st.markdown("""
    <div class='concept-box'>
        <h4>طريقة تقدير العتبة:</h4>
        <p>نختار العتبة المثلى γ̂ التي تقلل مجموع مربعات البواقي:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    \hat{\gamma} = \arg\min_{\gamma \in \Gamma} S_n(\gamma)
    ''')
    
    st.markdown("""
    <div class='term-box'>
        <p><b>حيث Γ هي مجموعة القيم المحتملة للعتبة:</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    \Gamma = [\gamma_L, \gamma_U] = [Q_{\pi}(q), Q_{1-\pi}(q)]
    ''')
    
    st.markdown("""
    <div class='info-box'>
        <p><b>ملاحظة:</b> Q_π(q) هو المئين π لمتغير العتبة q، و π هي نسبة القص (trimming ratio) عادة 10-15%.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🧮 تقدير المعاملات | Coefficient Estimation")
    
    st.markdown("#### لقيمة عتبة معطاة γ:")
    
    st.latex(r'''
    \hat{\beta}_1(\gamma) = \left( \sum_{q_t \leq \gamma} x_t x_t' \right)^{-1} \sum_{q_t \leq \gamma} x_t y_t
    ''')
    
    st.latex(r'''
    \hat{\beta}_2(\gamma) = \left( \sum_{q_t > \gamma} x_t x_t' \right)^{-1} \sum_{q_t > \gamma} x_t y_t
    ''')
    
    st.markdown("""
    <div class='success-box'>
        <h4>💡 خلاصة التقدير:</h4>
        <ol style='line-height: 2;'>
            <li>نحدد مجموعة قيم العتبة المحتملة Γ</li>
            <li>لكل قيمة γ ∈ Γ، نقسم البيانات إلى نظامين</li>
            <li>نقدر β₁ و β₂ بطريقة المربعات الصغرى لكل نظام</li>
            <li>نحسب SSR لكل قيمة γ</li>
            <li>نختار γ̂ التي تعطي أقل SSR</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif section == "🔍 أنواع نماذج العتبة | Types of Models":
    st.markdown("## 🔍 أنواع نماذج العتبة")
    st.markdown("### Types of Threshold Models")
    
    tabs = st.tabs([
        "TAR", "SETAR", "STAR", "LSTAR", "ESTAR", "Panel TAR"
    ])
    
    with tabs[0]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>نموذج الانحدار الذاتي العتبي</h3>
            <h4 style='color: #81ecec;'>Threshold Autoregressive (TAR) Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        y_t = \begin{cases}
        \phi_{10} + \phi_{11} y_{t-1} + ... + \phi_{1p} y_{t-p} + e_t & \text{if } q_t \leq \gamma \\
        \phi_{20} + \phi_{21} y_{t-1} + ... + \phi_{2p} y_{t-p} + e_t & \text{if } q_t > \gamma
        \end{cases}
        ''')
        
        st.markdown("""
        <div class='info-box'>
            <h4>الخصائص:</h4>
            <ul>
                <li>متغير العتبة q_t يمكن أن يكون خارجي أو متأخر من y</li>
                <li>يسمح بديناميكيات مختلفة في كل نظام</li>
                <li>مفيد لنمذجة دورات الأعمال والأسواق المالية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # رسم توضيحي
        np.random.seed(42)
        n = 300
        y = np.zeros(n)
        e = np.random.randn(n) * 0.5
        
        for t in range(1, n):
            if y[t-1] <= 0:
                y[t] = 0.5 + 0.8 * y[t-1] + e[t]
            else:
                y[t] = -0.3 + 0.4 * y[t-1] + e[t]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y, mode='lines', name='y_t',
                                line=dict(color='#3498db', width=1.5)))
        fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c",
                     annotation_text="العتبة γ=0")
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title='مثال على نموذج TAR',
            xaxis_title='الزمن',
            yaxis_title='y_t'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>نموذج الانحدار الذاتي العتبي الذاتي</h3>
            <h4 style='color: #81ecec;'>Self-Exciting TAR (SETAR) Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        y_t = \begin{cases}
        \phi_{10} + \phi_{11} y_{t-1} + ... + \phi_{1p} y_{t-p} + e_t & \text{if } y_{t-d} \leq \gamma \\
        \phi_{20} + \phi_{21} y_{t-1} + ... + \phi_{2p} y_{t-p} + e_t & \text{if } y_{t-d} > \gamma
        \end{cases}
        ''')
        
        st.markdown("""
        <div class='info-box'>
            <h4>الفرق عن TAR:</h4>
            <ul>
                <li>متغير العتبة هو <b>قيمة متأخرة من المتغير نفسه</b> (y_{t-d})</li>
                <li>d يسمى "تأخر العتبة" (delay parameter)</li>
                <li>النموذج "يثير نفسه" - لذا سمي Self-Exciting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        \text{SETAR}(k; p_1, p_2) : k = \text{عدد الأنظمة}, \quad p_i = \text{رتبة AR في النظام } i
        ''')
    
    with tabs[2]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>نموذج الانحدار الذاتي الانتقالي السلس</h3>
            <h4 style='color: #81ecec;'>Smooth Transition AR (STAR) Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        y_t = (\phi_{10} + \phi_{11} y_{t-1} + ... + \phi_{1p} y_{t-p})(1 - G(s_t; \gamma, c))
        ''')
        st.latex(r'''
        + (\phi_{20} + \phi_{21} y_{t-1} + ... + \phi_{2p} y_{t-p})G(s_t; \gamma, c) + e_t
        ''')
        
        st.markdown("""
        <div class='warning-box'>
            <h4>الفرق الجوهري:</h4>
            <p>بدلاً من الانتقال المفاجئ (0 أو 1)، نستخدم <b>دالة انتقال سلسة</b> G(·) تتراوح بين 0 و 1</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        G(s_t; \gamma, c) \in [0, 1]
        ''')
        
        st.markdown("حيث:")
        st.markdown("- **s_t**: متغير الانتقال (transition variable)")
        st.markdown("- **γ**: معامل سرعة الانتقال (smoothness parameter)")
        st.markdown("- **c**: موقع الانتقال (location parameter)")
    
    with tabs[3]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>نموذج STAR اللوجستي</h3>
            <h4 style='color: #81ecec;'>Logistic STAR (LSTAR) Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        G(s_t; \gamma, c) = \frac{1}{1 + \exp(-\gamma(s_t - c))}
        ''')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            gamma_lstar = st.slider("سرعة الانتقال γ", 0.5, 20.0, 5.0, 0.5)
            c_lstar = st.slider("موقع الانتقال c", -2.0, 2.0, 0.0, 0.1)
        
        with col2:
            s = np.linspace(-5, 5, 200)
            G_lstar = 1 / (1 + np.exp(-gamma_lstar * (s - c_lstar)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=G_lstar, mode='lines',
                                    line=dict(color='#e74c3c', width=3),
                                    name='G(s)'))
            fig.add_hline(y=0.5, line_dash="dot", line_color="#f39c12")
            fig.add_vline(x=c_lstar, line_dash="dot", line_color="#f39c12")
            fig.update_layout(
                template='plotly_dark',
                height=350,
                title=f'دالة الانتقال اللوجستية (γ={gamma_lstar}, c={c_lstar})',
                xaxis_title='s_t',
                yaxis_title='G(s_t)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
            <h4>خصائص LSTAR:</h4>
            <ul>
                <li>الانتقال <b>غير متماثل</b> - مختلف فوق وتحت العتبة</li>
                <li>مناسب عندما يختلف سلوك التوسع عن الانكماش</li>
                <li>كلما زاد γ، أصبح الانتقال أكثر حدة (يقترب من TAR)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[4]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>نموذج STAR الأسي</h3>
            <h4 style='color: #81ecec;'>Exponential STAR (ESTAR) Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        G(s_t; \gamma, c) = 1 - \exp(-\gamma(s_t - c)^2)
        ''')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            gamma_estar = st.slider("معامل السرعة γ", 0.1, 5.0, 1.0, 0.1, key='estar_gamma')
            c_estar = st.slider("المركز c", -2.0, 2.0, 0.0, 0.1, key='estar_c')
        
        with col2:
            s = np.linspace(-5, 5, 200)
            G_estar = 1 - np.exp(-gamma_estar * (s - c_estar)**2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s, y=G_estar, mode='lines',
                                    line=dict(color='#9b59b6', width=3),
                                    name='G(s)'))
            fig.add_vline(x=c_estar, line_dash="dot", line_color="#f39c12")
            fig.update_layout(
                template='plotly_dark',
                height=350,
                title=f'دالة الانتقال الأسية (γ={gamma_estar}, c={c_estar})',
                xaxis_title='s_t',
                yaxis_title='G(s_t)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='success-box'>
            <h4>خصائص ESTAR:</h4>
            <ul>
                <li>الانتقال <b>متماثل</b> حول c</li>
                <li>G(s) = 0 عندما s = c</li>
                <li>G(s) → 1 كلما ابتعد s عن c</li>
                <li>مناسب لنمذجة أسعار الصرف والانحرافات عن التوازن</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[5]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>نموذج العتبة للبيانات الطولية</h3>
            <h4 style='color: #81ecec;'>Panel Threshold Regression Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        y_{it} = \mu_i + \beta_1' x_{it} \cdot I(q_{it} \leq \gamma) + \beta_2' x_{it} \cdot I(q_{it} > \gamma) + e_{it}
        ''')
        
        st.markdown("""
        <div class='info-box'>
            <h4>الخصائص:</h4>
            <ul>
                <li><b>μ_i</b>: التأثير الثابت الفردي (Individual Fixed Effect)</li>
                <li>يجمع بين مزايا البيانات الطولية ونماذج العتبة</li>
                <li>Hansen (1999) طور طرق التقدير والاستدلال</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### اختبار العتبة في بيانات Panel:")
        
        st.latex(r'''
        F_1 = \frac{S_0 - S_1(\hat{\gamma})}{S_1(\hat{\gamma}) / [n(T-1) - 1]}
        ''')
        
        st.markdown("""
        <div class='warning-box'>
            <p>حيث S_0 هو SSR للنموذج الخطي و S_1 هو SSR لنموذج العتبة</p>
        </div>
        """, unsafe_allow_html=True)

elif section == "⚠️ مشكلة Davies | Davies Problem":
    st.markdown("## ⚠️ مشكلة Davies")
    st.markdown("### The Davies Problem")
    
    st.markdown("""
    <div class='davies-box'>
        <h3 style='color: #ffeaa7;'>🚨 ما هي مشكلة Davies؟</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        عند اختبار وجود العتبة، نواجه مشكلة إحصائية أساسية:
        <br><br>
        <b style='color: #f39c12;'>تحت فرضية العدم (H₀: لا توجد عتبة)، معامل العتبة γ غير معرّف!</b>
        <br><br>
        هذا يعني أن توزيع إحصائية الاختبار ليس قياسياً.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 الصياغة الرياضية للمشكلة")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='formula-box'>
            <h4 style='color: #ffeaa7;'>فرضية العدم H₀</h4>
            <p>النموذج خطي (لا توجد عتبة)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        H_0: \beta_1 = \beta_2
        ''')
    
    with col2:
        st.markdown("""
        <div class='formula-box'>
            <h4 style='color: #81ecec;'>الفرضية البديلة H₁</h4>
            <p>توجد عتبة (النموذج غير خطي)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        H_1: \beta_1 \neq \beta_2 \text{ لقيمة ما } \gamma
        ''')
    
    st.markdown("---")
    
    st.markdown("### 🔍 لماذا هذه مشكلة؟")
    
    st.markdown("""
    <div class='concept-box'>
        <h4>المشكلة الجوهرية:</h4>
        <p>تحت H₀، النموذج يصبح:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    y_t = \beta' x_t + e_t \quad \text{(لا يوجد } \gamma \text{ في النموذج!)}
    ''')
    
    st.markdown("""
    <div class='warning-box'>
        <h4>النتائج:</h4>
        <ol style='line-height: 2;'>
            <li><b>معامل مزعج غير معرّف:</b> γ موجود فقط تحت H₁</li>
            <li><b>إحصائية Wald/LR/LM القياسية:</b> لا تتبع توزيع χ² المعتاد</li>
            <li><b>الجداول الإحصائية العادية:</b> لا يمكن استخدامها</li>
            <li><b>القيم الحرجة:</b> تعتمد على خصائص البيانات</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 توضيح بياني للمشكلة")
    
    st.markdown("""
    <div class='info-box'>
        <p>إحصائية الاختبار كدالة في γ تبدو عشوائية تحت H₀:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # محاكاة إحصائيات الاختبار
    np.random.seed(42)
    gamma_values = np.linspace(2, 8, 100)
    
    # تحت H₀ (لا توجد عتبة حقيقية)
    F_stats_H0 = np.abs(np.random.randn(100)) * 2 + np.sin(gamma_values) * 0.5
    
    # تحت H₁ (توجد عتبة عند γ=5)
    F_stats_H1 = np.abs(np.random.randn(100)) * 1.5 + 5 * np.exp(-0.5 * (gamma_values - 5)**2)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('تحت H₀: لا توجد عتبة', 'تحت H₁: عتبة عند γ=5'))
    
    fig.add_trace(go.Scatter(x=gamma_values, y=F_stats_H0, mode='lines',
                            line=dict(color='#e74c3c', width=2),
                            name='F(γ) تحت H₀'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=gamma_values, y=F_stats_H1, mode='lines',
                            line=dict(color='#2ecc71', width=2),
                            name='F(γ) تحت H₁'), row=1, col=2)
    
    fig.add_vline(x=5, line_dash="dash", line_color="#f39c12", row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        showlegend=True
    )
    fig.update_xaxes(title_text='γ', row=1, col=1)
    fig.update_xaxes(title_text='γ', row=1, col=2)
    fig.update_yaxes(title_text='F(γ)', row=1, col=1)
    fig.update_yaxes(title_text='F(γ)', row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class='term-box'>
        <p><b>الملاحظة:</b> تحت H₀، إحصائية F(γ) تتذبذب عشوائياً لأن γ ليس له معنى. 
        تحت H₁، هناك قمة واضحة عند العتبة الحقيقية.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 مساهمة Davies (1977, 1987)")
    
    st.markdown("""
    <div class='concept-box'>
        <h4>اقترح Davies حدوداً علوية لقيمة p:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    P\left(\sup_{\gamma \in \Gamma} F(\gamma) > c\right) \leq P(F > c) + V \cdot c^{1/2} \cdot \phi(c^{1/2})
    ''')
    
    st.markdown("""
    <div class='term-box'>
        <p><b>حيث:</b></p>
        <ul>
            <li>V: عدد القمم المتوقع في F(γ)</li>
            <li>φ: دالة الكثافة للتوزيع الطبيعي</li>
            <li>c: القيمة الحرجة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='warning-box'>
        <h4>⚠️ قيود طريقة Davies:</h4>
        <ul>
            <li>تعطي حداً علوياً فقط، وليس قيمة p دقيقة</li>
            <li>قد تكون محافظة جداً (تعطي قيم p كبيرة)</li>
            <li>تتطلب افتراضات قد لا تتحقق دائماً</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif section == "✅ حل Hansen | Hansen's Solution":
    st.markdown("## ✅ حل Hansen لمشكلة Davies")
    st.markdown("### Hansen's Bootstrap Solution")
    
    st.markdown("""
    <div class='hansen-box'>
        <h3 style='color: #ffeaa7;'>💡 الحل: Bootstrap Procedure</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
        طور Bruce Hansen (1996, 2000) طريقة Bootstrap لحساب القيم الحرجة وقيم p
        بشكل دقيق لاختبار وجود العتبة.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 إحصائية الاختبار")
    
    st.markdown("""
    <div class='formula-box'>
        <h4 style='color: #ffeaa7;'>Sup-Wald Statistic</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    F_n = \sup_{\gamma \in \Gamma} F_n(\gamma)
    ''')
    
    st.markdown("حيث:")
    
    st.latex(r'''
    F_n(\gamma) = n \cdot \frac{S_0 - S_n(\gamma)}{S_n(\gamma)}
    ''')
    
    st.markdown("""
    <div class='term-box'>
        <ul>
            <li><b>S₀</b>: مجموع مربعات البواقي للنموذج الخطي (تحت H₀)</li>
            <li><b>Sₙ(γ)</b>: مجموع مربعات البواقي لنموذج العتبة عند γ</li>
            <li><b>sup</b>: الحد الأعلى على جميع قيم γ الممكنة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔄 خوارزمية Bootstrap")
    
    steps = [
        ("تقدير النموذج الخطي", "قدّر النموذج تحت H₀ (بدون عتبة) واحصل على البواقي ê_t والتقديرات β̂"),
        ("حساب إحصائية الاختبار الأصلية", "احسب Fₙ باستخدام البيانات الفعلية"),
        ("توليد عينات Bootstrap", "لكل تكرار b = 1, ..., B:\n- أعد عينة البواقي: e*_t من ê_t\n- أنشئ بيانات جديدة: y*_t = x_t'β̂ + e*_t"),
        ("حساب إحصائيات Bootstrap", "لكل عينة Bootstrap، احسب F*ₙ,b"),
        ("حساب قيمة p", "p-value = (عدد F*ₙ,b ≥ Fₙ) / B")
    ]
    
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step-box'>
            <span class='step-number'>{i}</span>
            <b style='color: #ffeaa7;'>{title}</b>
            <p style='color: #dfe6e9; margin-top: 0.5rem; white-space: pre-line;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 الصيغ التفصيلية")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### تقدير النموذج الخطي:")
        st.latex(r'''
        \hat{\beta}_0 = (X'X)^{-1}X'y
        ''')
        st.latex(r'''
        \hat{e}_t = y_t - x_t'\hat{\beta}_0
        ''')
        st.latex(r'''
        S_0 = \sum_{t=1}^n \hat{e}_t^2
        ''')
    
    with col2:
        st.markdown("#### بيانات Bootstrap:")
        st.latex(r'''
        e_t^* \sim \text{Resample from } \{\hat{e}_1, ..., \hat{e}_n\}
        ''')
        st.latex(r'''
        y_t^* = x_t'\hat{\beta}_0 + e_t^*
        ''')
    
    st.markdown("---")
    
    st.markdown("### 🧮 محاكاة تفاعلية لـ Bootstrap")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_boot = st.slider("عدد تكرارات Bootstrap", 100, 1000, 300, 50)
        sample_size = st.slider("حجم العينة", 50, 300, 100, 10)
        true_threshold = st.checkbox("وجود عتبة حقيقية", value=False)
    
    with col2:
        if st.button("🚀 تشغيل Bootstrap"):
            with st.spinner("جاري حساب Bootstrap..."):
                # توليد البيانات
                np.random.seed(42)
                q = np.linspace(0, 10, sample_size)
                x = np.random.randn(sample_size) * 2 + 5
                
                if true_threshold:
                    y = np.where(q <= 5, 2 * x, -1 * x + 15) + np.random.randn(sample_size) * 2
                else:
                    y = 0.5 * x + np.random.randn(sample_size) * 2
                
                data = pd.DataFrame({'y': y, 'x': x, 'q': q})
                
                # تقدير النموذج الخطي
                X = np.column_stack([np.ones(sample_size), x])
                beta_linear = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta_linear
                S0 = np.sum(residuals**2)
                
                # إيجاد أفضل عتبة
                opt_gamma, S1, _, _ = grid_search_threshold(data)
                F_original = sample_size * (S0 - S1) / S1
                
                # Bootstrap
                F_bootstrap = []
                progress_bar = st.progress(0)
                
                for b in range(n_boot):
                    boot_residuals = np.random.choice(residuals, size=sample_size, replace=True)
                    boot_y = X @ beta_linear + boot_residuals
                    boot_data = data.copy()
                    boot_data['y'] = boot_y
                    
                    S0_boot = np.sum(boot_residuals**2)
                    _, S1_boot, _, _ = grid_search_threshold(boot_data)
                    F_boot = sample_size * (S0_boot - S1_boot) / S1_boot
                    F_bootstrap.append(F_boot)
                    
                    progress_bar.progress((b + 1) / n_boot)
                
                p_value = np.mean(np.array(F_bootstrap) >= F_original)
                
                # رسم النتائج
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=F_bootstrap,
                    nbinsx=30,
                    name='توزيع Bootstrap',
                    marker_color='#3498db',
                    opacity=0.7
                ))
                fig.add_vline(x=F_original, line_dash="dash", line_color="#e74c3c", line_width=3,
                             annotation_text=f"F الأصلية = {F_original:.2f}")
                
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    title=f'توزيع Bootstrap لإحصائية F (p-value = {p_value:.4f})',
                    xaxis_title='F statistic',
                    yaxis_title='التكرار'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # النتيجة
                if p_value < 0.05:
                    st.markdown(f"""
                    <div class='success-box'>
                        <h4>✅ النتيجة: رفض H₀</h4>
                        <p>p-value = {p_value:.4f} < 0.05</p>
                        <p>يوجد دليل إحصائي على وجود عتبة عند γ̂ = {opt_gamma:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='warning-box'>
                        <h4>⚠️ النتيجة: عدم رفض H₀</h4>
                        <p>p-value = {p_value:.4f} ≥ 0.05</p>
                        <p>لا يوجد دليل إحصائي كافٍ على وجود عتبة</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 فترة الثقة للعتبة (Hansen 2000)")
    
    st.markdown("""
    <div class='concept-box'>
        <h4>Likelihood Ratio Statistic للعتبة:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    LR_n(\gamma) = n \cdot \frac{S_n(\gamma) - S_n(\hat{\gamma})}{S_n(\hat{\gamma})}
    ''')
    
    st.markdown("فترة الثقة 95% للعتبة:")
    
    st.latex(r'''
    C_\alpha = \{\gamma : LR_n(\gamma) \leq c_\alpha\}
    ''')
    
    st.markdown("""
    <div class='info-box'>
        <p><b>القيم الحرجة (Hansen 2000):</b></p>
        <ul>
            <li>90%: c = 5.94</li>
            <li>95%: c = 7.35</li>
            <li>99%: c = 10.59</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif section == "🧮 إيجاد العتبات | Finding Thresholds":
    st.markdown("## 🧮 طرق إيجاد العتبات")
    st.markdown("### Methods for Finding Thresholds")
    
    st.markdown("""
    <div class='concept-box'>
        <h3 style='color: #ffeaa7;'>الطريقة الأساسية: البحث الشبكي</h3>
        <h4 style='color: #81ecec;'>Grid Search Method</h4>
        <p>نبحث عن العتبة التي تقلل مجموع مربعات البواقي</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📐 الخوارزمية")
    
    st.latex(r'''
    \hat{\gamma} = \arg\min_{\gamma \in \Gamma} S_n(\gamma)
    ''')
    
    steps = [
        ("تحديد نطاق البحث Γ", 
         "استبعد π% من كل طرف (عادة 10-15%)\nΓ = [q_(πn), q_((1-π)n)]"),
        ("إنشاء شبكة من القيم المحتملة",
         "قسّم Γ إلى نقاط شبكية\nأو استخدم القيم الفعلية لـ q"),
        ("لكل قيمة γ في الشبكة",
         "1. قسّم البيانات إلى نظامين\n2. قدّر معاملات كل نظام\n3. احسب SSR"),
        ("اختيار العتبة المثلى",
         "اختر γ التي تعطي أقل SSR"),
        ("التحقق من الأهمية",
         "استخدم Bootstrap لاختبار معنوية العتبة")
    ]
    
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step-box'>
            <span class='step-number'>{i}</span>
            <b style='color: #ffeaa7;'>{title}</b>
            <p style='color: #dfe6e9; margin-top: 0.5rem; white-space: pre-line;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 تطبيق تفاعلي")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### إعدادات البيانات:")
        n_samples = st.slider("حجم العينة", 50, 500, 200, 25)
        true_gamma = st.slider("العتبة الحقيقية", 2.0, 8.0, 5.0, 0.5)
        beta1_true = st.slider("β₁ (النظام الأول)", -3.0, 3.0, 2.0, 0.2)
        beta2_true = st.slider("β₂ (النظام الثاني)", -3.0, 3.0, -1.0, 0.2)
        noise_level = st.slider("مستوى الضوضاء", 0.5, 3.0, 1.0, 0.1)
        trim_pct = st.slider("نسبة القص %", 5, 20, 15, 1)
    
    with col2:
        # توليد البيانات
        data = create_threshold_data(
            n=n_samples,
            threshold=true_gamma,
            beta1=beta1_true,
            beta2=beta2_true,
            noise=noise_level
        )
        
        # البحث الشبكي
        opt_gamma, min_ssr, candidates, ssr_values = grid_search_threshold(
            data, trim=trim_pct/100
        )
        
        # رسم دالة SSR
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('دالة SSR كدالة في γ', 'البيانات مع العتبة المقدرة'),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6]
        )
        
        # SSR curve
        fig.add_trace(go.Scatter(
            x=candidates, y=ssr_values,
            mode='lines',
            line=dict(color='#3498db', width=2),
            name='SSR(γ)'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[opt_gamma], y=[min_ssr],
            mode='markers',
            marker=dict(color='#e74c3c', size=15, symbol='star'),
            name=f'γ̂ = {opt_gamma:.3f}'
        ), row=1, col=1)
        
        fig.add_vline(x=true_gamma, line_dash="dash", line_color="#2ecc71",
                     annotation_text=f"γ الحقيقية = {true_gamma}", row=1, col=1)
        
        # Data plot
        mask1 = data['q'] <= opt_gamma
        mask2 = data['q'] > opt_gamma
        
        fig.add_trace(go.Scatter(
            x=data.loc[mask1, 'q'], y=data.loc[mask1, 'y'],
            mode='markers',
            marker=dict(color='#3498db', size=6),
            name='النظام الأول'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.loc[mask2, 'q'], y=data.loc[mask2, 'y'],
            mode='markers',
            marker=dict(color='#e74c3c', size=6),
            name='النظام الثاني'
        ), row=2, col=1)
        
        fig.add_vline(x=opt_gamma, line_dash="dash", line_color="#f39c12",
                     annotation_text=f"γ̂ = {opt_gamma:.3f}", row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # نتائج التقدير
        st.markdown(f"""
        <div class='success-box'>
            <h4>📊 نتائج التقدير:</h4>
            <ul>
                <li><b>العتبة المقدرة:</b> γ̂ = {opt_gamma:.4f}</li>
                <li><b>العتبة الحقيقية:</b> γ = {true_gamma:.4f}</li>
                <li><b>الخطأ:</b> |γ̂ - γ| = {abs(opt_gamma - true_gamma):.4f}</li>
                <li><b>SSR عند γ̂:</b> {min_ssr:.4f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔄 العتبات المتعددة | Multiple Thresholds")
    
    st.markdown("""
    <div class='info-box'>
        <h4>البحث التتابعي عن عتبات متعددة:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    y_t = \begin{cases}
    \beta_1' x_t + e_t & \text{if } q_t \leq \gamma_1 \\
    \beta_2' x_t + e_t & \text{if } \gamma_1 < q_t \leq \gamma_2 \\
    \beta_3' x_t + e_t & \text{if } q_t > \gamma_2
    \end{cases}
    ''')
    
    st.markdown("""
    <div class='step-box'>
        <b style='color: #ffeaa7;'>خطوات البحث التتابعي:</b>
        <ol style='line-height: 2;'>
            <li>ابحث عن العتبة الأولى γ₁ وتحقق من أهميتها</li>
            <li>إذا كانت معنوية، ابحث عن γ₂ مع تثبيت γ₁</li>
            <li>أعد تقدير γ₁ مع تثبيت γ₂</li>
            <li>كرر حتى التقارب</li>
            <li>اختبر الحاجة لعتبة ثالثة، وهكذا</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif section == "📊 الاختبارات الإحصائية | Statistical Tests":
    st.markdown("## 📊 الاختبارات الإحصائية لنماذج العتبة")
    st.markdown("### Statistical Tests for Threshold Models")
    
    tabs = st.tabs([
        "اختبار الخطية",
        "اختبار عدد العتبات",
        "فترات الثقة",
        "اختبارات التشخيص"
    ])
    
    with tabs[0]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>اختبار الخطية مقابل العتبة</h3>
            <h4 style='color: #81ecec;'>Linearity Test</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### الفرضيات:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.latex(r'''H_0: \beta_1 = \beta_2''')
            st.markdown("**(لا توجد عتبة)**")
        
        with col2:
            st.latex(r'''H_1: \beta_1 \neq \beta_2''')
            st.markdown("**(توجد عتبة)**")
        
        st.markdown("#### إحصائيات الاختبار:")
        
        st.markdown("**1. Sup-Wald:**")
        st.latex(r'''F_1 = \sup_{\gamma \in \Gamma} F_n(\gamma)''')
        
        st.markdown("**2. Average-Wald:**")
        st.latex(r'''F_2 = \frac{1}{|\Gamma|} \sum_{\gamma \in \Gamma} F_n(\gamma)''')
        
        st.markdown("**3. Exp-Wald:**")
        st.latex(r'''F_3 = \log\left(\frac{1}{|\Gamma|} \sum_{\gamma \in \Gamma} \exp\left(\frac{F_n(\gamma)}{2}\right)\right)''')
        
        st.markdown("""
        <div class='info-box'>
            <h4>ملاحظات:</h4>
            <ul>
                <li><b>Sup-Wald:</b> الأكثر شيوعاً، قوي ضد بدائل محددة</li>
                <li><b>Average-Wald:</b> قوي ضد بدائل منتشرة</li>
                <li><b>Exp-Wald:</b> توازن بين الاثنين</li>
                <li>جميعها تتطلب Bootstrap لحساب القيم الحرجة</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>اختبار عدد العتبات</h3>
            <h4 style='color: #81ecec;'>Testing Number of Thresholds</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### اختبار عتبة واحدة مقابل عتبتين:")
        
        st.latex(r'''
        H_0: \text{عتبة واحدة} \quad vs \quad H_1: \text{عتبتان}
        ''')
        
        st.latex(r'''
        F_{12} = \frac{S_1(\hat{\gamma}_1) - S_2(\hat{\gamma}_1, \hat{\gamma}_2)}{S_2(\hat{\gamma}_1, \hat{\gamma}_2) / (n - 2k - 2)}
        ''')
        
        st.markdown("""
        <div class='warning-box'>
            <h4>الإجراء التتابعي:</h4>
            <ol style='line-height: 2;'>
                <li>اختبر H₀: لا عتبة vs H₁: عتبة واحدة</li>
                <li>إذا رُفضت H₀، اختبر H₀: عتبة واحدة vs H₁: عتبتان</li>
                <li>استمر حتى عدم رفض H₀</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>فترات الثقة للعتبة</h3>
            <h4 style='color: #81ecec;'>Confidence Intervals for Threshold</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### طريقة Hansen (2000):")
        
        st.latex(r'''
        LR_n(\gamma) = n \cdot \frac{S_n(\gamma) - S_n(\hat{\gamma})}{S_n(\hat{\gamma})}
        ''')
        
        st.latex(r'''
        C_{1-\alpha} = \{\gamma : LR_n(\gamma) \leq c(\alpha)\}
        ''')
        
        st.markdown("#### القيم الحرجة:")
        
        critical_values = pd.DataFrame({
            'مستوى الثقة': ['90%', '95%', '99%'],
            'α': [0.10, 0.05, 0.01],
            'c(α)': [5.94, 7.35, 10.59]
        })
        
        st.dataframe(critical_values, use_container_width=True)
        
        # رسم توضيحي
        np.random.seed(42)
        gamma_range = np.linspace(3, 7, 100)
        LR_values = 10 * (1 - np.exp(-0.5 * (gamma_range - 5)**2))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gamma_range, y=LR_values, mode='lines',
                                line=dict(color='#3498db', width=3),
                                name='LR(γ)'))
        
        fig.add_hline(y=7.35, line_dash="dash", line_color="#e74c3c",
                     annotation_text="c(0.05) = 7.35")
        
        # منطقة فترة الثقة
        ci_mask = LR_values <= 7.35
        fig.add_trace(go.Scatter(
            x=gamma_range[ci_mask],
            y=LR_values[ci_mask],
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name='فترة الثقة 95%'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title='فترة الثقة للعتبة',
            xaxis_title='γ',
            yaxis_title='LR(γ)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>اختبارات التشخيص</h3>
            <h4 style='color: #81ecec;'>Diagnostic Tests</h4>
        </div>
        """, unsafe_allow_html=True)
        
        tests = [
            ("اختبار عدم الارتباط الذاتي", "Ljung-Box Test", 
             r"Q = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k}",
             "للتحقق من عدم وجود ارتباط ذاتي في البواقي"),
            
            ("اختبار تجانس التباين", "ARCH-LM Test",
             r"LM = nR^2 \sim \chi^2(q)",
             "للتحقق من ثبات التباين عبر الزمن"),
            
            ("اختبار الطبيعية", "Jarque-Bera Test",
             r"JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)",
             "للتحقق من التوزيع الطبيعي للبواقي"),
            
            ("اختبار الاستقرار", "CUSUM Test",
             r"CUSUM_t = \frac{\sum_{j=k+1}^{t} w_j}{\hat{\sigma}_w}",
             "للتحقق من استقرار المعاملات")
        ]
        
        for ar_name, en_name, formula, desc in tests:
            st.markdown(f"""
            <div class='term-box'>
                <span class='term-ar'>🔹 {ar_name}</span><br>
                <span class='term-en'>{en_name}</span>
                <p class='term-def'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            st.latex(formula)

elif section == "🎯 التطبيق العملي | Practical Application":
    st.markdown("## 🎯 التطبيق العملي الكامل")
    st.markdown("### Complete Practical Application")
    
    st.markdown("""
    <div class='concept-box'>
        <h3>📋 سنقوم بتنفيذ التحليل الكامل خطوة بخطوة</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # توليد أو رفع البيانات
    st.markdown("### 1️⃣ البيانات")
    
    data_option = st.radio(
        "اختر مصدر البيانات:",
        ["توليد بيانات محاكاة", "استخدام بيانات جاهزة"]
    )
    
    if data_option == "توليد بيانات محاكاة":
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_n = st.number_input("حجم العينة", 50, 1000, 200)
            sim_gamma = st.number_input("العتبة الحقيقية", 1.0, 9.0, 5.0)
        with col2:
            sim_beta1 = st.number_input("β₁", -5.0, 5.0, 2.0)
            sim_beta2 = st.number_input("β₂", -5.0, 5.0, -1.5)
        with col3:
            sim_noise = st.number_input("الضوضاء", 0.1, 5.0, 1.0)
            sim_seed = st.number_input("Random Seed", 1, 1000, 42)
        
        data = create_threshold_data(
            n=sim_n, threshold=sim_gamma, 
            beta1=sim_beta1, beta2=sim_beta2,
            noise=sim_noise, seed=sim_seed
        )
    else:
        # بيانات مثال جاهزة
        np.random.seed(123)
        n = 200
        q = np.linspace(0, 10, n)
        x = np.random.randn(n) * 2 + 5
        y = np.where(q <= 5, 1.5 * x + np.random.randn(n), 
                    -0.8 * x + 12 + np.random.randn(n))
        data = pd.DataFrame({'y': y, 'x': x, 'q': q})
        sim_gamma = 5.0
    
    st.markdown("#### معاينة البيانات:")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # التحليل
    if st.button("🚀 تشغيل التحليل الكامل", type="primary"):
        
        st.markdown("### 2️⃣ الإحصاءات الوصفية")
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data.describe().round(4), use_container_width=True)
        
        with col2:
            fig = px.histogram(data, x='q', nbins=30, 
                              title='توزيع متغير العتبة',
                              color_discrete_sequence=['#3498db'])
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 3️⃣ تقدير النموذج الخطي (H₀)")
        
        X = np.column_stack([np.ones(len(data)), data['x']])
        y = data['y'].values
        beta_linear = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred_linear = X @ beta_linear
        ssr_linear = np.sum((y - y_pred_linear)**2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='info-box'>
                <h4>نتائج النموذج الخطي:</h4>
                <ul>
                    <li>β₀ (الثابت) = {beta_linear[0]:.4f}</li>
                    <li>β₁ (الميل) = {beta_linear[1]:.4f}</li>
                    <li>SSR = {ssr_linear:.4f}</li>
                    <li>R² = {1 - ssr_linear/np.sum((y - y.mean())**2):.4f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='markers',
                                    marker=dict(color='#3498db', size=6),
                                    name='البيانات'))
            x_line = np.linspace(data['x'].min(), data['x'].max(), 100)
            fig.add_trace(go.Scatter(x=x_line, y=beta_linear[0] + beta_linear[1]*x_line,
                                    mode='lines', line=dict(color='#e74c3c', width=2),
                                    name='خط الانحدار'))
            fig.update_layout(template='plotly_dark', height=300,
                            title='النموذج الخطي', xaxis_title='x', yaxis_title='y')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 4️⃣ البحث عن العتبة المثلى")
        
        with st.spinner("جاري البحث عن العتبة..."):
            opt_gamma, min_ssr, candidates, ssr_values = grid_search_threshold(data, trim=0.15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='success-box'>
                <h4>نتائج البحث الشبكي:</h4>
                <ul>
                    <li><b>العتبة المقدرة:</b> γ̂ = {opt_gamma:.4f}</li>
                    <li><b>SSR عند γ̂:</b> {min_ssr:.4f}</li>
                    <li><b>تحسن SSR:</b> {((ssr_linear - min_ssr)/ssr_linear*100):.2f}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=candidates, y=ssr_values, mode='lines',
                                    line=dict(color='#3498db', width=2), name='SSR(γ)'))
            fig.add_trace(go.Scatter(x=[opt_gamma], y=[min_ssr], mode='markers',
                                    marker=dict(color='#e74c3c', size=15, symbol='star'),
                                    name=f'γ̂={opt_gamma:.3f}'))
            fig.update_layout(template='plotly_dark', height=300,
                            title='دالة SSR', xaxis_title='γ', yaxis_title='SSR(γ)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 5️⃣ تقدير نموذج العتبة")
        
        # تقدير المعاملات لكل نظام
        regime1 = data[data['q'] <= opt_gamma]
        regime2 = data[data['q'] > opt_gamma]
        
        X1 = np.column_stack([np.ones(len(regime1)), regime1['x']])
        beta1 = np.linalg.lstsq(X1, regime1['y'].values, rcond=None)[0]
        
        X2 = np.column_stack([np.ones(len(regime2)), regime2['x']])
        beta2 = np.linalg.lstsq(X2, regime2['y'].values, rcond=None)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='info-box'>
                <h4>النظام الأول (q ≤ {opt_gamma:.3f}):</h4>
                <ul>
                    <li>عدد المشاهدات: {len(regime1)}</li>
                    <li>β₁₀ = {beta1[0]:.4f}</li>
                    <li>β₁₁ = {beta1[1]:.4f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='warning-box'>
                <h4>النظام الثاني (q > {opt_gamma:.3f}):</h4>
                <ul>
                    <li>عدد المشاهدات: {len(regime2)}</li>
                    <li>β₂₀ = {beta2[0]:.4f}</li>
                    <li>β₂₁ = {beta2[1]:.4f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # رسم النتيجة النهائية
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=regime1['q'], y=regime1['y'], mode='markers',
                                marker=dict(color='#3498db', size=8),
                                name='النظام الأول'))
        fig.add_trace(go.Scatter(x=regime2['q'], y=regime2['y'], mode='markers',
                                marker=dict(color='#e74c3c', size=8),
                                name='النظام الثاني'))
        
        fig.add_vline(x=opt_gamma, line_dash="dash", line_color="#f39c12", line_width=3)
        fig.add_annotation(x=opt_gamma, y=max(data['y']), text=f"γ̂ = {opt_gamma:.3f}",
                          showarrow=True, arrowhead=2, font=dict(size=14, color="#f39c12"))
        
        fig.update_layout(template='plotly_dark', height=450,
                        title='نموذج العتبة المقدر',
                        xaxis_title='متغير العتبة (q)',
                        yaxis_title='المتغير التابع (y)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 6️⃣ اختبار الخطية (Bootstrap)")
        
        with st.spinner("جاري تنفيذ Bootstrap (قد يستغرق بعض الوقت)..."):
            F_stat, p_value, boot_stats = bootstrap_p_value(data, n_bootstrap=299)
        
        col1, col2 = st.columns(2)
        
        with col1:
            significance = "✅ معنوي" if p_value < 0.05 else "❌ غير معنوي"
            conclusion = "يوجد عتبة" if p_value < 0.05 else "لا يوجد دليل على عتبة"
            
            st.markdown(f"""
            <div class='{"success" if p_value < 0.05 else "warning"}-box'>
                <h4>نتائج اختبار الخطية:</h4>
                <ul>
                    <li><b>إحصائية F:</b> {F_stat:.4f}</li>
                    <li><b>قيمة p:</b> {p_value:.4f}</li>
                    <li><b>النتيجة:</b> {significance}</li>
                    <li><b>القرار:</b> {conclusion}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=boot_stats, nbinsx=30,
                                      marker_color='#3498db', opacity=0.7,
                                      name='توزيع Bootstrap'))
            fig.add_vline(x=F_stat, line_dash="dash", line_color="#e74c3c", line_width=3)
            fig.add_annotation(x=F_stat, y=0, text=f"F = {F_stat:.2f}",
                              showarrow=True, arrowhead=2)
            fig.update_layout(template='plotly_dark', height=300,
                            title='توزيع Bootstrap لإحصائية F')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 7️⃣ التقرير النهائي")
        
        st.markdown(f"""
        <div class='concept-box'>
            <h3 style='color: #ffeaa7;'>📋 ملخص التحليل</h3>
            <hr style='border-color: #ffeaa7;'>
            
            <h4>1. البيانات:</h4>
            <p>• حجم العينة: {len(data)} مشاهدة</p>
            
            <h4>2. نتائج التقدير:</h4>
            <p>• العتبة المقدرة: γ̂ = {opt_gamma:.4f}</p>
            <p>• النظام الأول ({len(regime1)} مشاهدة): y = {beta1[0]:.3f} + {beta1[1]:.3f}x</p>
            <p>• النظام الثاني ({len(regime2)} مشاهدة): y = {beta2[0]:.3f} + {beta2[1]:.3f}x</p>
            
            <h4>3. اختبار الخطية:</h4>
            <p>• إحصائية F = {F_stat:.4f}</p>
            <p>• p-value = {p_value:.4f}</p>
            <p>• القرار: {'رفض H₀ - يوجد عتبة معنوية' if p_value < 0.05 else 'عدم رفض H₀ - لا يوجد دليل على عتبة'}</p>
            
            <h4>4. التفسير:</h4>
            <p>{'تشير النتائج إلى وجود تغير هيكلي في العلاقة عند النقطة γ̂ = ' + f'{opt_gamma:.3f}' if p_value < 0.05 else 'النموذج الخطي كافٍ لوصف البيانات'}</p>
        </div>
        """, unsafe_allow_html=True)

elif section == "📈 محاكاة تفاعلية | Interactive Simulation":
    st.markdown("## 📈 محاكاة تفاعلية")
    st.markdown("### Interactive Simulation")
    
    st.markdown("""
    <div class='concept-box'>
        <p>جرّب تغيير المعاملات وشاهد كيف يتغير شكل نموذج العتبة!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### ⚙️ الإعدادات:")
        
        n_points = st.slider("عدد النقاط", 50, 500, 200, 25)
        
        st.markdown("---")
        st.markdown("**العتبة:**")
        gamma_sim = st.slider("γ (العتبة)", 1.0, 9.0, 5.0, 0.1)
        
        st.markdown("---")
        st.markdown("**النظام الأول (q ≤ γ):**")
        alpha1 = st.slider("α₁ (الثابت)", -10.0, 10.0, 1.0, 0.5)
        beta1_sim = st.slider("β₁ (الميل)", -5.0, 5.0, 2.0, 0.1)
        
        st.markdown("---")
        st.markdown("**النظام الثاني (q > γ):**")
        alpha2 = st.slider("α₂ (الثابت)", -10.0, 10.0, 8.0, 0.5)
        beta2_sim = st.slider("β₂ (الميل)", -5.0, 5.0, -1.0, 0.1)
        
        st.markdown("---")
        noise_sim = st.slider("σ (الضوضاء)", 0.1, 5.0, 1.0, 0.1)
        
        show_regression = st.checkbox("إظهار خطوط الانحدار", value=True)
        show_ci = st.checkbox("إظهار فترة الثقة", value=False)
    
    with col2:
        # توليد البيانات
        np.random.seed(42)
        q = np.linspace(0, 10, n_points)
        e = np.random.randn(n_points) * noise_sim
        
        y = np.where(q <= gamma_sim,
                    alpha1 + beta1_sim * q + e,
                    alpha2 + beta2_sim * q + e)
        
        # إنشاء الرسم
        fig = go.Figure()
        
        # نقاط البيانات
        mask1 = q <= gamma_sim
        mask2 = q > gamma_sim
        
        fig.add_trace(go.Scatter(
            x=q[mask1], y=y[mask1],
            mode='markers',
            marker=dict(color='#3498db', size=8, opacity=0.7),
            name=f'النظام الأول (n={sum(mask1)})'
        ))
        
        fig.add_trace(go.Scatter(
            x=q[mask2], y=y[mask2],
            mode='markers',
            marker=dict(color='#e74c3c', size=8, opacity=0.7),
            name=f'النظام الثاني (n={sum(mask2)})'
        ))
        
        if show_regression:
            # خط النظام الأول
            q1_line = np.linspace(0, gamma_sim, 50)
            y1_line = alpha1 + beta1_sim * q1_line
            fig.add_trace(go.Scatter(
                x=q1_line, y=y1_line,
                mode='lines',
                line=dict(color='#2980b9', width=3),
                name=f'y = {alpha1:.1f} + {beta1_sim:.1f}q'
            ))
            
            # خط النظام الثاني
            q2_line = np.linspace(gamma_sim, 10, 50)
            y2_line = alpha2 + beta2_sim * q2_line
            fig.add_trace(go.Scatter(
                x=q2_line, y=y2_line,
                mode='lines',
                line=dict(color='#c0392b', width=3),
                name=f'y = {alpha2:.1f} + {beta2_sim:.1f}q'
            ))
        
        if show_ci:
            # فترة ثقة تقريبية
            ci_width = 1.96 * noise_sim
            
            y1_upper = alpha1 + beta1_sim * q1_line + ci_width
            y1_lower = alpha1 + beta1_sim * q1_line - ci_width
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([q1_line, q1_line[::-1]]),
                y=np.concatenate([y1_upper, y1_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(41, 128, 185, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='CI النظام الأول'
            ))
            
            y2_upper = alpha2 + beta2_sim * q2_line + ci_width
            y2_lower = alpha2 + beta2_sim * q2_line - ci_width
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([q2_line, q2_line[::-1]]),
                y=np.concatenate([y2_upper, y2_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(192, 57, 43, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='CI النظام الثاني'
            ))
        
        # خط العتبة
        fig.add_vline(x=gamma_sim, line_dash="dash", line_color="#f39c12", line_width=3)
        
        # منطقتي النظامين
        fig.add_vrect(x0=0, x1=gamma_sim, fillcolor="blue", opacity=0.05)
        fig.add_vrect(x0=gamma_sim, x1=10, fillcolor="red", opacity=0.05)
        
        fig.add_annotation(
            x=gamma_sim, y=max(y)*1.1,
            text=f"γ = {gamma_sim}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#f39c12",
            font=dict(size=16, color="#f39c12")
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            title=dict(
                text='نموذج العتبة التفاعلي',
                font=dict(size=20)
            ),
            xaxis_title='متغير العتبة (q)',
            yaxis_title='المتغير التابع (y)',
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # معادلات النموذج
        col_eq1, col_eq2 = st.columns(2)
        
        with col_eq1:
            st.markdown(f"""
            <div class='info-box'>
                <h4>النظام الأول (q ≤ {gamma_sim}):</h4>
                <p style='font-size: 1.2rem;'>y = {alpha1:.2f} + {beta1_sim:.2f}q + ε</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_eq2:
            st.markdown(f"""
            <div class='warning-box'>
                <h4>النظام الثاني (q > {gamma_sim}):</h4>
                <p style='font-size: 1.2rem;'>y = {alpha2:.2f} + {beta2_sim:.2f}q + ε</p>
            </div>
            """, unsafe_allow_html=True)

elif section == "📋 ملخص شامل | Comprehensive Summary":
    st.markdown("## 📋 ملخص شامل لنماذج العتبة")
    st.markdown("### Comprehensive Summary of Threshold Models")
    
    st.markdown("""
    <div class='main-header' style='padding: 1.5rem;'>
        <h2>🎓 ما تعلمناه</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # المفاهيم الأساسية
    with st.expander("📖 المفاهيم الأساسية", expanded=True):
        st.markdown("""
        | المصطلح العربي | English Term | الشرح |
        |---------------|--------------|-------|
        | العتبة | Threshold (γ) | نقطة التحول بين الأنظمة |
        | متغير العتبة | Threshold Variable (q) | المتغير المحدد للنظام |
        | النظام | Regime | كل جزء من النموذج بمعاملات مختلفة |
        | الدالة المؤشرة | Indicator Function | تحدد النظام الفعال |
        | نسبة القص | Trimming Ratio | نسبة الاستبعاد من الأطراف |
        """)
    
    # الصيغ الأساسية
    with st.expander("📐 الصيغ الرياضية الأساسية"):
        st.latex(r'''
        y_t = \beta_1' x_t \cdot I(q_t \leq \gamma) + \beta_2' x_t \cdot I(q_t > \gamma) + e_t
        ''')
        
        st.latex(r'''
        \hat{\gamma} = \arg\min_{\gamma \in \Gamma} S_n(\gamma)
        ''')
        
        st.latex(r'''
        F_n = \sup_{\gamma} n \cdot \frac{S_0 - S_n(\gamma)}{S_n(\gamma)}
        ''')
    
    # أنواع النماذج
    with st.expander("🔍 أنواع نماذج العتبة"):
        models_summary = pd.DataFrame({
            'النموذج': ['TAR', 'SETAR', 'LSTAR', 'ESTAR', 'Panel TAR'],
            'الخاصية الرئيسية': [
                'متغير عتبة خارجي',
                'متغير العتبة = قيمة متأخرة',
                'انتقال سلس لوجستي',
                'انتقال سلس أسي متماثل',
                'بيانات طولية'
            ],
            'الاستخدام الشائع': [
                'دورات الأعمال',
                'السلاسل الزمنية المالية',
                'النمو غير المتماثل',
                'أسعار الصرف',
                'بيانات الدول/الشركات'
            ]
        })
        st.dataframe(models_summary, use_container_width=True)
    
    # مشكلة Davies وحل Hansen
    with st.expander("⚠️ مشكلة Davies وحل Hansen"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **المشكلة:**
            - تحت H₀، γ غير معرّف
            - التوزيع غير قياسي
            - الجداول العادية لا تصلح
            """)
        
        with col2:
            st.markdown("""
            **الحل (Hansen Bootstrap):**
            1. قدّر النموذج الخطي
            2. أعد عينة البواقي
            3. أنشئ بيانات Bootstrap
            4. احسب p-value تجريبياً
            """)
    
    # خطوات التطبيق
    with st.expander("🎯 خطوات التطبيق العملي"):
        steps_df = pd.DataFrame({
            'الخطوة': range(1, 8),
            'الوصف': [
                'تحضير البيانات والتحقق من جودتها',
                'تقدير النموذج الخطي كمرجع',
                'تحديد نطاق البحث عن العتبة',
                'البحث الشبكي عن العتبة المثلى',
                'تقدير معاملات كل نظام',
                'اختبار معنوية العتبة (Bootstrap)',
                'بناء فترة الثقة للعتبة'
            ]
        })
        st.dataframe(steps_df, use_container_width=True)
    
    # نصائح مهمة
    st.markdown("""
    <div class='success-box'>
        <h3>💡 نصائح مهمة للتطبيق:</h3>
        <ol style='line-height: 2;'>
            <li><b>نسبة القص:</b> استخدم 10-15% لضمان عدد كافٍ من المشاهدات في كل نظام</li>
            <li><b>Bootstrap:</b> استخدم على الأقل 1000 تكرار للحصول على نتائج موثوقة</li>
            <li><b>التحقق:</b> دائماً تحقق من افتراضات النموذج (استقلالية البواقي، تجانس التباين)</li>
            <li><b>التفسير:</b> العتبة المقدرة لها معنى اقتصادي - ابحث عنه!</li>
            <li><b>المقارنة:</b> قارن نموذج العتبة مع النموذج الخطي باستخدام معايير المعلومات</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # المراجع
    st.markdown("---")
    st.markdown("### 📚 المراجع الأساسية | Key References")
    
    st.markdown("""
    <div class='concept-box'>
        <ul style='line-height: 2.5;'>
            <li><b>Hansen, B.E. (1996)</b>: "Inference When a Nuisance Parameter Is Not Identified Under the Null Hypothesis" - Econometrica</li>
            <li><b>Hansen, B.E. (1999)</b>: "Threshold Effects in Non-dynamic Panels" - Journal of Econometrics</li>
            <li><b>Hansen, B.E. (2000)</b>: "Sample Splitting and Threshold Estimation" - Econometrica</li>
            <li><b>Tong, H. (1990)</b>: "Non-linear Time Series: A Dynamical System Approach"</li>
            <li><b>Davies, R.B. (1977, 1987)</b>: "Hypothesis Testing When a Nuisance Parameter is Present Only Under the Alternative"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='main-header' style='padding: 1rem;'>
        <h3>🎉 تهانينا!</h3>
        <p>لقد أكملت دراسة نماذج العتبة من الصفر إلى الاحتراف</p>
        <p style='color: #81ecec;'>Congratulations! You've completed the comprehensive guide to Threshold Models</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; color: #a2d2ff;'>
    <p>📊 تطبيق نماذج العتبة الشامل | Comprehensive Threshold Models Application</p>
    <p style='font-size: 0.8rem;'>تم التطوير باستخدام Python, Streamlit, and Plotly</p>
</div>
""", unsafe_allow_html=True)
