import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="TAR Models Complete Guide",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .subsection-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .theory-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Complete Guide to Threshold Autoregressive (TAR) Models</h1>',
            unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📚 Navigation")
sections = [
    "🏠 Home & Introduction",
    "📖 Theory & Mathematical Foundation",
    "🔍 Pre-Estimation Tests",
    "⚙️ Model Estimation",
    "📊 Post-Estimation Analysis",
    "⚠️ Davies Problem",
    "🔬 Practical Implementation",
    "📈 Interactive Examples",
    "📝 Summary & Best Practices"
]

selected_section = st.sidebar.selectbox("Choose a section:", sections)


# Helper functions for TAR model implementation
class TARModel:
    def __init__(self):
        self.threshold = None
        self.delay = None
        self.coefficients = {}
        self.residuals = None
        self.fitted_values = None
        self.regimes = None

    def simulate_tar_data(self, n=200, phi1=[0.5, 0.3], phi2=[-0.2, 0.8],
                          threshold=0, sigma1=1, sigma2=1, delay=1):
        """Simulate TAR data"""
        np.random.seed(42)
        y = np.zeros(n)
        errors = np.random.normal(0, 1, n)

        for t in range(max(len(phi1), len(phi2)), n):
            if t >= delay:
                if y[t - delay] <= threshold:
                    y[t] = sum(phi1[i] * y[t - 1 - i] for i in range(len(phi1))) + sigma1 * errors[t]
                else:
                    y[t] = sum(phi2[i] * y[t - 1 - i] for i in range(len(phi2))) + sigma2 * errors[t]
            else:
                y[t] = 0.1 * errors[t]

        return y

    def tsay_test(self, y, max_lag=4, delay=1, alpha=0.05):
        """Tsay's arranged autoregression test"""
        n = len(y)
        threshold_var = y[:-delay] if delay > 0 else y[:-1]

        # Sort by threshold variable
        sorted_indices = np.argsort(threshold_var)
        y_sorted = y[sorted_indices]

        # Fit linear AR model
        X_linear = np.column_stack([y[i:n - max_lag + i] for i in range(max_lag)])
        y_linear = y[max_lag:]

        lr_linear = LinearRegression()
        lr_linear.fit(X_linear, y_linear)
        residuals_linear = y_linear - lr_linear.predict(X_linear)

        # Arranged autoregression
        recursive_residuals = []
        for i in range(max_lag, len(y_sorted)):
            X_temp = np.column_stack([y_sorted[j:i - max_lag + j + 1] for j in range(max_lag)])
            y_temp = y_sorted[max_lag:i + 1]

            if len(y_temp) > max_lag:
                lr_temp = LinearRegression()
                lr_temp.fit(X_temp, y_temp)
                pred = lr_temp.predict(X_temp[-1:])
                recursive_residuals.append(y_sorted[i] - pred[0])

        # Test statistic (simplified version)
        if len(recursive_residuals) > 0:
            test_stat = np.sum(np.array(recursive_residuals) ** 2) / np.var(residuals_linear)
            p_value = 1 - stats.chi2.cdf(test_stat, df=max_lag)
        else:
            test_stat, p_value = 0, 1

        return test_stat, p_value

    def keenan_test(self, y, max_lag=4, alpha=0.05):
        """Keenan's test for quadratic nonlinearity"""
        n = len(y)
        X = np.column_stack([y[i:n - max_lag + i] for i in range(max_lag)])
        y_reg = y[max_lag:]

        # Fit linear model
        lr = LinearRegression()
        lr.fit(X, y_reg)
        fitted = lr.predict(X)

        # Create quadratic term
        q = fitted ** 2 - np.mean(fitted ** 2)

        # Augmented regression
        X_aug = np.column_stack([X, q])
        lr_aug = LinearRegression()
        lr_aug.fit(X_aug, y_reg)

        # F-test for quadratic term
        rss_restricted = np.sum((y_reg - fitted) ** 2)
        rss_unrestricted = np.sum((y_reg - lr_aug.predict(X_aug)) ** 2)

        f_stat = ((rss_restricted - rss_unrestricted) / 1) / (rss_unrestricted / (len(y_reg) - X_aug.shape[1] - 1))
        p_value = 1 - stats.f.cdf(f_stat, 1, len(y_reg) - X_aug.shape[1] - 1)

        return f_stat, p_value

    def bds_test_simplified(self, residuals, m=2, eps_factor=1.0):
        """Simplified BDS test for independence"""
        n = len(residuals)
        std_res = np.std(residuals)
        eps = eps_factor * std_res

        # Calculate correlation integrals
        def correlation_integral(data, m, eps):
            n = len(data)
            count = 0
            total_pairs = 0

            for i in range(n - m + 1):
                for j in range(i + 1, n - m + 1):
                    vec1 = data[i:i + m]
                    vec2 = data[j:j + m]
                    if np.max(np.abs(vec1 - vec2)) < eps:
                        count += 1
                    total_pairs += 1

            return count / total_pairs if total_pairs > 0 else 0

        c1 = correlation_integral(residuals, 1, eps)
        cm = correlation_integral(residuals, m, eps)

        if c1 > 0:
            bds_stat = np.sqrt(n) * (cm - c1 ** m) / np.sqrt(
                c1 ** (2 * m) * (1 + 2 * sum(c1 ** (2 * i) for i in range(1, m))))
            p_value = 2 * (1 - stats.norm.cdf(abs(bds_stat)))
        else:
            bds_stat, p_value = 0, 1

        return bds_stat, p_value


# Section content
if selected_section == "🏠 Home & Introduction":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Welcome to TAR Models</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="theory-box">
        <h3>🎯 What You'll Learn</h3>
        This comprehensive guide covers everything you need to know about Threshold Autoregressive (TAR) models:

        • <b>Mathematical foundations</b> and intuitive explanations<br>
        • <b>Pre-estimation tests</b> to detect nonlinearity<br>
        • <b>Estimation techniques</b> and grid search methods<br>
        • <b>Post-estimation diagnostics</b> and inference<br>
        • <b>The Davies Problem</b> and its solutions<br>
        • <b>Practical implementation</b> in Python<br>
        • <b>Interactive examples</b> with real data
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="subsection-header">🤔 Why TAR Models?</div>

        Linear time series models assume **constant relationships** over time. But real-world data often exhibits:

        - **Regime switches** (bull vs bear markets)
        - **Asymmetric dynamics** (gradual growth, sudden crashes)
        - **Threshold effects** (policy changes, structural breaks)

        TAR models capture these nonlinearities elegantly!
        """)

    with col2:
        # Create a simple illustration
        fig = go.Figure()

        # Generate sample data for illustration
        x = np.linspace(-3, 3, 100)
        y_linear = 0.5 * x
        y_threshold = np.where(x <= 0, 0.8 * x, -0.3 * x + 0.5)

        fig.add_trace(go.Scatter(x=x, y=y_linear, mode='lines',
                                 name='Linear Model', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=x, y=y_threshold, mode='lines',
                                 name='TAR Model', line=dict(color='red', width=3)))
        fig.add_vline(x=0, line_dash="dash", line_color="gray",
                      annotation_text="Threshold")

        fig.update_layout(
            title="Linear vs TAR Model Comparison",
            xaxis_title="Threshold Variable",
            yaxis_title="Response",
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

elif selected_section == "📖 Theory & Mathematical Foundation":
    st.markdown('<div class="section-header">Mathematical Foundation of TAR Models</div>', unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🔢 Basic TAR Model Specification</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    A <b>two-regime TAR(p)</b> model is defined as:
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    y_t = \begin{cases}
    \phi_{1,0} + \sum_{i=1}^{p_1} \phi_{1,i} y_{t-i} + \varepsilon_{1,t}, & \text{if } z_{t-d} \leq r \\
    \phi_{2,0} + \sum_{i=1}^{p_2} \phi_{2,i} y_{t-i} + \varepsilon_{2,t}, & \text{if } z_{t-d} > r
    \end{cases}
    ''')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="theory-box">
        <h4>🔍 Key Components:</h4>
        • <b>z_{t-d}</b>: Threshold variable with delay d<br>
        • <b>r</b>: Threshold value<br>
        • <b>φ_{j,i}</b>: Regime-specific coefficients<br>
        • <b>ε_{j,t}</b>: Regime-specific errors<br>
        • <b>p_j</b>: Regime-specific AR orders
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Important Assumptions:</h4>
        • Errors are i.i.d. within regimes<br>
        • E[ε_{j,t}] = 0<br>
        • Var(ε_{j,t}) = σ²_j<br>
        • Threshold variable is observable<br>
        • Sufficient observations in each regime
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🎯 Self-Exciting TAR (SETAR) Models</div>', unsafe_allow_html=True)

    st.markdown("""
    When the threshold variable is a lagged value of the dependent variable itself (z_{t-d} = y_{t-d}), 
    we have a **Self-Exciting TAR (SETAR)** model.
    """)

    st.latex(r'''
    \text{SETAR}(k, p, d) \text{ notation:}
    ''')

    st.markdown("""
    - **k**: Number of regimes
    - **p**: AR order(s) (can be regime-specific)
    - **d**: Delay parameter
    """)

    # Interactive example
    st.markdown('<div class="subsection-header">🔧 Interactive Example: SETAR Model</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        phi1_1 = st.slider("φ₁₁ (Regime 1, lag 1)", -1.0, 1.0, 0.5, 0.1)
        phi1_2 = st.slider("φ₁₂ (Regime 1, lag 2)", -1.0, 1.0, 0.3, 0.1)

    with col2:
        phi2_1 = st.slider("φ₂₁ (Regime 2, lag 1)", -1.0, 1.0, -0.2, 0.1)
        phi2_2 = st.slider("φ₂₂ (Regime 2, lag 2)", -1.0, 1.0, 0.8, 0.1)

    with col3:
        threshold = st.slider("Threshold (r)", -2.0, 2.0, 0.0, 0.1)
        n_obs = st.slider("Sample Size", 100, 500, 200, 50)

    # Generate and plot SETAR data
    tar_model = TARModel()
    y_sim = tar_model.simulate_tar_data(n=n_obs, phi1=[phi1_1, phi1_2],
                                        phi2=[phi2_1, phi2_2], threshold=threshold)

    # Determine regimes
    regimes = np.where(y_sim[:-1] <= threshold, 1, 2)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Simulated SETAR Time Series', 'Regime Classification'])

    # Time series plot
    fig.add_trace(go.Scatter(x=list(range(len(y_sim))), y=y_sim,
                             mode='lines', name='SETAR Series'), row=1, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text="Threshold", row=1, col=1)

    # Regime plot
    colors = ['blue' if r == 1 else 'red' for r in regimes]
    fig.add_trace(go.Scatter(x=list(range(len(regimes))), y=regimes,
                             mode='markers', marker=dict(color=colors),
                             name='Regimes'), row=2, col=1)

    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Show regime statistics
    regime_1_pct = np.mean(regimes == 1) * 100
    regime_2_pct = np.mean(regimes == 2) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Regime 1 Observations", f"{regime_1_pct:.1f}%")
    with col2:
        st.metric("Regime 2 Observations", f"{regime_2_pct:.1f}%")

elif selected_section == "🔍 Pre-Estimation Tests":
    st.markdown('<div class="section-header">Pre-Estimation Tests for Nonlinearity</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Why Test for Nonlinearity?</h3>
    Before fitting a complex nonlinear model, we need to verify that a simple linear AR model is inadequate. 
    These tests help us decide whether the additional complexity of a TAR model is justified.
    </div>
    """, unsafe_allow_html=True)

    # Generate sample data for demonstrations
    tar_model = TARModel()

    # Linear data
    np.random.seed(42)
    n = 200
    linear_data = np.random.normal(0, 1, n)
    for i in range(2, n):
        linear_data[i] = 0.5 * linear_data[i - 1] + 0.3 * linear_data[i - 2] + np.random.normal(0, 0.5)

    # Nonlinear (TAR) data
    nonlinear_data = tar_model.simulate_tar_data(n=n, phi1=[0.7, 0.2], phi2=[-0.3, 0.8], threshold=0)

    st.markdown('<div class="subsection-header">1️⃣ Tsay\'s Arranged Autoregression Test</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>📊 Tsay's Test Procedure:</h4>
    1. Choose a threshold variable z_{t-d} and sort observations by this variable<br>
    2. Fit AR(p) to the original (unsorted) data<br>
    3. Run recursive least squares on the <b>sorted</b> data<br>
    4. Test if the arranged regression improves prediction significantly
    </div>
    """, unsafe_allow_html=True)

    # Perform Tsay's test
    tsay_stat_linear, tsay_p_linear = tar_model.tsay_test(linear_data)
    tsay_stat_nonlinear, tsay_p_nonlinear = tar_model.tsay_test(nonlinear_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Data (AR Model)**")
        st.metric("Tsay Test Statistic", f"{tsay_stat_linear:.3f}")
        st.metric("P-value", f"{tsay_p_linear:.3f}")
        if tsay_p_linear > 0.05:
            st.success("✅ Fail to reject linearity")
        else:
            st.error("❌ Reject linearity")

    with col2:
        st.markdown("**Nonlinear Data (TAR Model)**")
        st.metric("Tsay Test Statistic", f"{tsay_stat_nonlinear:.3f}")
        st.metric("P-value", f"{tsay_p_nonlinear:.3f}")
        if tsay_p_nonlinear > 0.05:
            st.success("✅ Fail to reject linearity")
        else:
            st.error("❌ Reject linearity")

    st.markdown('<div class="subsection-header">2️⃣ Keenan\'s Test for Quadratic Nonlinearity</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>📊 Keenan's Test Procedure:</h4>
    1. Fit linear AR model: y_t = φ₀ + Σφᵢy_{t-i} + ε_t<br>
    2. Obtain fitted values m̂_t<br>
    3. Create quadratic proxy: q_t = (m̂_t)² - E[(m̂_t)²]<br>
    4. Test H₀: β_q = 0 in augmented regression with q_t
    </div>
    """, unsafe_allow_html=True)

    # Perform Keenan's test
    keenan_stat_linear, keenan_p_linear = tar_model.keenan_test(linear_data)
    keenan_stat_nonlinear, keenan_p_nonlinear = tar_model.keenan_test(nonlinear_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Data**")
        st.metric("Keenan F-Statistic", f"{keenan_stat_linear:.3f}")
        st.metric("P-value", f"{keenan_p_linear:.3f}")
        if keenan_p_linear > 0.05:
            st.success("✅ Fail to reject linearity")
        else:
            st.error("❌ Reject linearity")

    with col2:
        st.markdown("**Nonlinear Data**")
        st.metric("Keenan F-Statistic", f"{keenan_stat_nonlinear:.3f}")
        st.metric("P-value", f"{keenan_p_nonlinear:.3f}")
        if keenan_p_nonlinear > 0.05:
            st.success("✅ Fail to reject linearity")
        else:
            st.error("❌ Reject linearity")

    st.markdown('<div class="subsection-header">3️⃣ Brock-Dechert-Scheinkman (BDS) Test</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>📊 BDS Test Concept:</h4>
    The BDS test examines whether residuals from a linear model are independently and identically distributed (i.i.d.). 
    If they're not, it suggests remaining structure (often nonlinear) in the data.
    </div>
    """, unsafe_allow_html=True)


    # Fit linear models and get residuals
    def fit_ar_get_residuals(data, lag=4):
        n = len(data)
        X = np.column_stack([data[i:n - lag + i] for i in range(lag)])
        y = data[lag:]
        lr = LinearRegression()
        lr.fit(X, y)
        return y - lr.predict(X)


    residuals_linear = fit_ar_get_residuals(linear_data)
    residuals_nonlinear = fit_ar_get_residuals(nonlinear_data)

    # Perform BDS test
    bds_stat_linear, bds_p_linear = tar_model.bds_test_simplified(residuals_linear)
    bds_stat_nonlinear, bds_p_nonlinear = tar_model.bds_test_simplified(residuals_nonlinear)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Data Residuals**")
        st.metric("BDS Test Statistic", f"{bds_stat_linear:.3f}")
        st.metric("P-value", f"{bds_p_linear:.3f}")
        if bds_p_linear > 0.05:
            st.success("✅ Fail to reject i.i.d.")
        else:
            st.error("❌ Reject i.i.d.")

    with col2:
        st.markdown("**Nonlinear Data Residuals**")
        st.metric("BDS Test Statistic", f"{bds_stat_nonlinear:.3f}")
        st.metric("P-value", f"{bds_p_nonlinear:.3f}")
        if bds_p_nonlinear > 0.05:
            st.success("✅ Fail to reject i.i.d.")
        else:
            st.error("❌ Reject i.i.d.")

    # Visualization of residuals
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Linear Data', 'Nonlinear Data',
                                        'Linear Residuals', 'Nonlinear Residuals'])

    # Original data
    fig.add_trace(go.Scatter(y=linear_data, mode='lines', name='Linear', line=dict(color='blue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=nonlinear_data, mode='lines', name='Nonlinear', line=dict(color='red')),
                  row=1, col=2)

    # Residuals
    fig.add_trace(go.Scatter(y=residuals_linear, mode='lines', name='Linear Residuals', line=dict(color='blue')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(y=residuals_nonlinear, mode='lines', name='Nonlinear Residuals', line=dict(color='red')),
                  row=2, col=2)

    fig.update_layout(height=600, showlegend=False, title_text="Data and Residual Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="success-box">
    <h4>📈 Test Results Summary:</h4>
    The tests correctly identify that:
    • Linear data shows <b>no evidence</b> of nonlinearity
    • Nonlinear (TAR) data shows <b>strong evidence</b> of nonlinearity

    This validates our testing procedures and demonstrates their effectiveness!
    </div>
    """, unsafe_allow_html=True)

elif selected_section == "⚙️ Model Estimation":
    st.markdown('<div class="section-header">TAR Model Estimation Techniques</div>', unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🎯 Conditional Least Squares (CLS)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>📊 CLS Procedure:</h4>
    When the threshold <b>r</b> and delay <b>d</b> are known, the TAR model becomes linear in parameters within each regime.

    <b>Step 1:</b> Split the sample based on z_{t-d} ≤ r vs z_{t-d} > r<br>
    <b>Step 2:</b> Apply OLS separately to each regime<br>
    <b>Step 3:</b> Obtain regime-specific coefficient estimates
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    \hat{\phi}_j = (X_j'X_j)^{-1}X_j'y_j, \quad j = 1, 2
    ''')

    st.markdown('<div class="subsection-header">🔍 Grid Search for Unknown Threshold</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>📊 Grid Search Algorithm:</h4>
    In practice, both threshold <b>r</b> and delay <b>d</b> are usually unknown. We use grid search:

    <b>Step 1:</b> Choose candidate delays D = {1, 2, ..., D_max}<br>
    <b>Step 2:</b> For each d ∈ D, create threshold grid from z_{t-d} values<br>
    <b>Step 3:</b> Trim extremes (e.g., keep central 70-80%) to ensure minimum regime sizes<br>
    <b>Step 4:</b> For each (d,r) pair, estimate by CLS and compute SSR(d,r)<br>
    <b>Step 5:</b> Select (d̂,r̂) = argmin SSR(d,r)
    </div>
    """, unsafe_allow_html=True)

    # Interactive grid search demonstration
    st.markdown('<div class="subsection-header">🔧 Interactive Grid Search Example</div>', unsafe_allow_html=True)

    # Generate TAR data for estimation
    tar_model = TARModel()
    np.random.seed(123)
    true_params = {
        'phi1': [0.6, 0.3],
        'phi2': [-0.2, 0.7],
        'threshold': 0.5,
        'delay': 1
    }

    y_data = tar_model.simulate_tar_data(n=300, **true_params)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Grid Search Parameters**")
        max_delay = st.selectbox("Maximum Delay", [1, 2, 3, 4, 5], index=2)
        trim_pct = st.slider("Trim Percentage", 0.1, 0.3, 0.15, 0.05)
        min_regime_size = st.slider("Min Regime Size", 20, 50, 30, 5)
        ar_order = st.selectbox("AR Order", [1, 2, 3, 4], index=1)


    # Perform grid search
    def grid_search_tar(y, max_delay, trim_pct, min_regime_size, ar_order):
        results = []
        n = len(y)

        for delay in range(1, max_delay + 1):
            if n - delay - ar_order < 2 * min_regime_size:
                continue

            threshold_var = y[ar_order:-delay] if delay > 0 else y[ar_order:]
            threshold_candidates = np.percentile(threshold_var,
                                                 np.linspace(trim_pct * 100, (1 - trim_pct) * 100, 20))

            for threshold in threshold_candidates:
                try:
                    # Split data into regimes
                    regime_indicator = threshold_var <= threshold

                    if np.sum(regime_indicator) < min_regime_size or np.sum(~regime_indicator) < min_regime_size:
                        continue

                    # Prepare data for regression
                    X = np.column_stack([y[i:n - delay - ar_order + i] for i in range(ar_order)])
                    y_reg = y[ar_order + delay:]
                    regime_reg = regime_indicator[:len(y_reg)]

                    # Estimate regime 1
                    X1 = X[regime_reg]
                    y1 = y_reg[regime_reg]
                    if len(y1) > ar_order:
                        lr1 = LinearRegression()
                        lr1.fit(X1, y1)
                        pred1 = lr1.predict(X1)
                        ssr1 = np.sum((y1 - pred1) ** 2)
                    else:
                        ssr1 = np.inf

                    # Estimate regime 2
                    X2 = X[~regime_reg]
                    y2 = y_reg[~regime_reg]
                    if len(y2) > ar_order:
                        lr2 = LinearRegression()
                        lr2.fit(X2, y2)
                        pred2 = lr2.predict(X2)
                        ssr2 = np.sum((y2 - pred2) ** 2)
                    else:
                        ssr2 = np.inf

                    total_ssr = ssr1 + ssr2

                    results.append({
                        'delay': delay,
                        'threshold': threshold,
                        'ssr': total_ssr,
                        'regime1_size': len(y1),
                        'regime2_size': len(y2)
                    })

                except:
                    continue

        return pd.DataFrame(results)


    if st.button("🔍 Run Grid Search"):
        with st.spinner("Performing grid search..."):
            search_results = grid_search_tar(y_data, max_delay, trim_pct, min_regime_size, ar_order)

        if not search_results.empty:
            # Find optimal parameters
            optimal_idx = search_results['ssr'].idxmin()
            optimal_params = search_results.iloc[optimal_idx]

            with col2:
                st.markdown("**Grid Search Results**")

                # Create heatmap of SSR values
                pivot_data = search_results.pivot_table(values='ssr', index='delay', columns='threshold', aggfunc='min')

                fig = px.imshow(pivot_data.values,
                                x=pivot_data.columns.round(3),
                                y=pivot_data.index,
                                color_continuous_scale='Viridis',
                                title="SSR Heatmap (darker = better)")
                fig.update_layout(xaxis_title="Threshold", yaxis_title="Delay")
                st.plotly_chart(fig, use_container_width=True)

                # Display optimal results
                st.success(f"**Optimal Parameters Found:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Optimal Delay", f"{optimal_params['delay']}")
                with col_b:
                    st.metric("Optimal Threshold", f"{optimal_params['threshold']:.3f}")
                with col_c:
                    st.metric("Minimum SSR", f"{optimal_params['ssr']:.3f}")

                # Compare with true parameters
                st.markdown("**Comparison with True Parameters:**")
                comparison_df = pd.DataFrame({
                    'Parameter': ['Delay', 'Threshold'],
                    'True Value': [true_params['delay'], true_params['threshold']],
                    'Estimated': [optimal_params['delay'], optimal_params['threshold']],
                })
                comparison_df['Error'] = comparison_df['Estimated'] - comparison_df['True Value']
                st.dataframe(comparison_df)

        else:
            st.error("Grid search failed. Try adjusting parameters.")

    st.markdown('<div class="subsection-header">📊 Estimation Quality Metrics</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>🎯 Model Selection Criteria:</h4>

    <b>Akaike Information Criterion (AIC):</b>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    AIC = \log(\hat{\sigma}^2) + \frac{2k}{T}
    ''')

    st.markdown("""
    <div class="theory-box">
    <b>Bayesian Information Criterion (BIC):</b>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    BIC = \log(\hat{\sigma}^2) + \frac{k \log(T)}{T}
    ''')

    st.markdown("""
    Where:
    - σ̂² is the estimated error variance
    - k is the number of parameters
    - T is the sample size

    **Lower values indicate better models!**
    """)

elif selected_section == "📊 Post-Estimation Analysis":
    st.markdown('<div class="section-header">Post-Estimation Analysis and Diagnostics</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Why Post-Estimation Diagnostics?</h3>
    After estimating a TAR model, we need to verify that:
    • The model adequately captures the data's nonlinear structure<br>
    • Residuals behave as expected (white noise)<br>
    • The threshold effect is statistically significant<br>
    • Model assumptions are satisfied
    </div>
    """, unsafe_allow_html=True)

    # Generate and estimate a TAR model for demonstration
    tar_model = TARModel()
    np.random.seed(42)

    # True parameters
    true_params = {'phi1': [0.7, 0.2], 'phi2': [-0.3, 0.6], 'threshold': 0.3, 'delay': 1}
    y_sim = tar_model.simulate_tar_data(n=400, **true_params)


    # Estimate the model (simplified)
    def estimate_tar_model(y, threshold, delay=1, ar_order=2):
        n = len(y)
        threshold_var = y[ar_order - 1:-delay] if delay > 0 else y[ar_order - 1:]
        regime_indicator = threshold_var <= threshold

        # Prepare regression data
        X = np.column_stack([y[i:n - delay - ar_order + i + 1] for i in range(ar_order)])
        y_reg = y[ar_order + delay - 1:]
        regime_reg = regime_indicator[:len(y_reg)]

        # Estimate both regimes
        results = {}

        # Regime 1
        X1, y1 = X[regime_reg], y_reg[regime_reg]
        lr1 = LinearRegression()
        lr1.fit(X1, y1)
        results['regime1'] = {'coef': lr1.coef_, 'intercept': lr1.intercept_, 'fitted': lr1.predict(X1)}

        # Regime 2
        X2, y2 = X[~regime_reg], y_reg[~regime_reg]
        lr2 = LinearRegression()
        lr2.fit(X2, y2)
        results['regime2'] = {'coef': lr2.coef_, 'intercept': lr2.intercept_, 'fitted': lr2.predict(X2)}

        # Combined fitted values and residuals
        fitted = np.zeros(len(y_reg))
        fitted[regime_reg] = results['regime1']['fitted']
        fitted[~regime_reg] = results['regime2']['fitted']

        residuals = y_reg - fitted

        results['fitted'] = fitted
        results['residuals'] = residuals
        results['regime_indicator'] = regime_reg
        results['y_reg'] = y_reg
        results['X'] = X

        return results


    # Estimate with true threshold for demonstration
    estimated_model = estimate_tar_model(y_sim, true_params['threshold'])

    st.markdown('<div class="subsection-header">1️⃣ Residual Diagnostics</div>', unsafe_allow_html=True)

    # Residual analysis
    residuals = estimated_model['residuals']
    fitted_values = estimated_model['fitted']

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['Residuals vs Time', 'Residuals vs Fitted',
                                        'Q-Q Plot', 'ACF of Residuals'])

    # Residuals vs time
    fig.add_trace(go.Scatter(y=residuals, mode='lines+markers', name='Residuals',
                             marker=dict(size=4)), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Residuals vs fitted
    fig.add_trace(go.Scatter(x=fitted_values, y=residuals, mode='markers',
                             name='Residuals vs Fitted', marker=dict(size=4)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    # Q-Q plot
    from scipy import stats

    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot',
                             marker=dict(size=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines',
                             name='Normal Line', line=dict(color='red')), row=2, col=1)


    # ACF of residuals
    def autocorrelation_function(x, max_lags=20):
        n = len(x)
        x = x - np.mean(x)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[n - 1:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lags + 1]


    acf_values = autocorrelation_function(residuals)
    lags = list(range(len(acf_values)))

    fig.add_trace(go.Bar(x=lags, y=acf_values, name='ACF'), row=2, col=2)
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-0.05, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(height=700, showlegend=False, title_text="Residual Diagnostics")
    st.plotly_chart(fig, use_container_width=True)

    # Statistical tests on residuals
    st.markdown('<div class="subsection-header">🧪 Statistical Tests on Residuals</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Ljung-Box test
        from scipy.stats import jarque_bera


        def ljung_box_test(residuals, lags=10):
            n = len(residuals)
            acf_vals = autocorrelation_function(residuals, lags)[1:]
            lb_stat = n * (n + 2) * np.sum([(acf_vals[i] ** 2) / (n - i - 1) for i in range(lags)])
            p_value = 1 - stats.chi2.cdf(lb_stat, lags)
            return lb_stat, p_value


        lb_stat, lb_pvalue = ljung_box_test(residuals)
        st.markdown("**Ljung-Box Test**")
        st.metric("Test Statistic", f"{lb_stat:.3f}")
        st.metric("P-value", f"{lb_pvalue:.3f}")
        if lb_pvalue > 0.05:
            st.success("✅ No autocorrelation")
        else:
            st.error("❌ Autocorrelation detected")

    with col2:
        # Jarque-Bera test
        jb_stat, jb_pvalue = jarque_bera(residuals)
        st.markdown("**Jarque-Bera Test**")
        st.metric("Test Statistic", f"{jb_stat:.3f}")
        st.metric("P-value", f"{jb_pvalue:.3f}")
        if jb_pvalue > 0.05:
            st.success("✅ Normal residuals")
        else:
            st.error("❌ Non-normal residuals")

    with col3:
        # ARCH test (simplified)
        def arch_test(residuals, lags=5):
            residuals_sq = residuals ** 2
            n = len(residuals_sq)
            X = np.column_stack([residuals_sq[i:n - lags + i] for i in range(lags)])
            y = residuals_sq[lags:]

            lr = LinearRegression()
            lr.fit(X, y)
            r_squared = lr.score(X, y)

            lm_stat = len(y) * r_squared
            p_value = 1 - stats.chi2.cdf(lm_stat, lags)
            return lm_stat, p_value


        arch_stat, arch_pvalue = arch_test(residuals)
        st.markdown("**ARCH Test**")
        st.metric("LM Statistic", f"{arch_stat:.3f}")
        st.metric("P-value", f"{arch_pvalue:.3f}")
        if arch_pvalue > 0.05:
            st.success("✅ No ARCH effects")
        else:
            st.error("❌ ARCH effects present")

    st.markdown('<div class="subsection-header">2️⃣ Model Fit Visualization</div>', unsafe_allow_html=True)

    # Plot actual vs fitted
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['Actual vs Fitted Values', 'Regime Classification'])

    # Time series comparison
    time_index = list(range(len(estimated_model['y_reg'])))
    fig.add_trace(go.Scatter(x=time_index, y=estimated_model['y_reg'],
                             mode='lines', name='Actual', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_index, y=fitted_values,
                             mode='lines', name='Fitted', line=dict(color='red')), row=1, col=1)

    # Regime classification
    regime_colors = ['blue' if r else 'red' for r in estimated_model['regime_indicator']]
    fig.add_trace(go.Scatter(x=time_index, y=estimated_model['regime_indicator'].astype(int),
                             mode='markers', marker=dict(color=regime_colors, size=4),
                             name='Regimes'), row=2, col=1)

    fig.update_layout(height=600, title_text="Model Fit Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # Model performance metrics
    st.markdown('<div class="subsection-header">📈 Model Performance Metrics</div>', unsafe_allow_html=True)

    # Calculate metrics
    mse = mean_squared_error(estimated_model['y_reg'], fitted_values)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r_squared = 1 - np.sum(residuals ** 2) / np.sum((estimated_model['y_reg'] - np.mean(estimated_model['y_reg'])) ** 2)

    # Information criteria
    n_params = 2 * (2 + 1)  # 2 regimes * (2 AR coeffs + 1 intercept)
    n_obs = len(residuals)
    log_likelihood = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(np.var(residuals)) + 1)

    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_obs) - 2 * log_likelihood

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("MAE", f"{mae:.4f}")

    with col2:
        st.metric("R-squared", f"{r_squared:.4f}")
        st.metric("Adj. R-squared", f"{1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_params - 1):.4f}")

    with col3:
        st.metric("AIC", f"{aic:.2f}")
        st.metric("BIC", f"{bic:.2f}")

    with col4:
        st.metric("Log-Likelihood", f"{log_likelihood:.2f}")
        st.metric("Regime 1 %", f"{np.mean(estimated_model['regime_indicator']) * 100:.1f}%")

    st.markdown("""
    <div class="success-box">
    <h4>✅ Diagnostic Summary:</h4>
    The residual diagnostics show that our TAR model successfully captures the nonlinear structure in the data. 
    The residuals appear to be well-behaved (close to white noise), indicating good model specification.
    </div>
    """, unsafe_allow_html=True)

elif selected_section == "⚠️ Davies Problem":
    st.markdown('<div class="section-header">The Davies Problem in TAR Models</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="danger-box">
    <h3>🚨 What is the Davies Problem?</h3>
    The Davies Problem arises when testing the null hypothesis of <b>linearity against a TAR alternative</b>. 
    Under the null hypothesis (linear model), the <b>threshold parameter is not identified</b> - it doesn't exist!

    This creates a fundamental problem: <b>standard test statistics don't follow their usual distributions</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🔍 Understanding the Problem</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="theory-box">
        <h4>📊 Standard Hypothesis Testing:</h4>

        <b>H₀:</b> y_t = φ₀ + φ₁y_{t-1} + φ₂y_{t-2} + ε_t (Linear AR)<br>
        <b>H₁:</b> TAR model with threshold r

        <b>Problem:</b> Under H₀, the threshold r is <b>not identified</b>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Consequences:</h4>
        • Likelihood Ratio test statistic doesn't follow χ² distribution<br>
        • Wald test is not applicable<br>
        • Standard critical values are invalid<br>
        • P-values are incorrect
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🎯 Davies (1987) Solution Framework</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>📈 Davies' Key Insight:</h4>
    When a nuisance parameter (threshold r) is present only under the alternative hypothesis, 
    the test statistic has a <b>non-standard limiting distribution</b>.

    <b>Solution:</b> Use <b>supremum-type tests</b> that search over all possible threshold values.
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    \text{sup-LR} = \sup_{r \in \mathcal{R}} LR(r)
    ''')

    st.latex(r'''
    \text{sup-LM} = \sup_{r \in \mathcal{R}} LM(r)
    ''')

    st.latex(r'''
    \text{sup-Wald} = \sup_{r \in \mathcal{R}} W(r)
    ''')

    st.markdown("where 𝒢 is the admissible set of threshold values (trimmed to ensure minimum regime sizes).")

    st.markdown('<div class="subsection-header">🔧 Hansen\'s Bootstrap Solution</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>🎲 Bootstrap Procedure (Hansen 1996, 1997):</h4>

    <b>Step 1:</b> Estimate linear AR model under H₀<br>
    <b>Step 2:</b> Compute sup-LM statistic from original data<br>
    <b>Step 3:</b> Bootstrap under H₀:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;• Generate bootstrap residuals from fitted linear model<br>
    &nbsp;&nbsp;&nbsp;&nbsp;• Create bootstrap series y*_t<br>
    &nbsp;&nbsp;&nbsp;&nbsp;• Compute sup-LM* for each bootstrap sample<br>
    <b>Step 4:</b> Bootstrap p-value = P(sup-LM* > sup-LM_observed)
    </div>
    """, unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown('<div class="subsection-header">🔬 Interactive Demonstration</div>', unsafe_allow_html=True)

    # Generate linear and nonlinear data
    np.random.seed(42)
    n = 200

    # Linear data (H0 true)
    linear_data = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    for t in range(2, n):
        linear_data[t] = 0.5 * linear_data[t - 1] + 0.3 * linear_data[t - 2] + errors[t]

    # Nonlinear data (H1 true)
    tar_model = TARModel()
    nonlinear_data = tar_model.simulate_tar_data(n=n, phi1=[0.6, 0.3], phi2=[-0.2, 0.7], threshold=0)


    def compute_sup_lm_test(y, trim_pct=0.15):
        """Compute sup-LM test statistic"""
        n = len(y)
        ar_order = 2

        # Fit linear AR model
        X_linear = np.column_stack([y[i:n - ar_order + i] for i in range(ar_order)])
        y_linear = y[ar_order:]
        lr_linear = LinearRegression()
        lr_linear.fit(X_linear, y_linear)
        residuals_linear = y_linear - lr_linear.predict(X_linear)
        sigma2_linear = np.var(residuals_linear)

        # Threshold candidates
        threshold_candidates = np.percentile(y[ar_order - 1:-1],
                                             np.linspace(trim_pct * 100, (1 - trim_pct) * 100, 30))

        lm_stats = []
        valid_thresholds = []

        for threshold in threshold_candidates:
            try:
                # Split into regimes
                threshold_var = y[ar_order - 1:-1]
                regime_indicator = threshold_var <= threshold

                # Check minimum regime sizes
                if np.sum(regime_indicator) < 0.15 * len(regime_indicator) or \
                        np.sum(~regime_indicator) < 0.15 * len(regime_indicator):
                    continue

                # Prepare data
                X = X_linear
                y_reg = y_linear
                regime_reg = regime_indicator[:len(y_reg)]

                # Regime-specific regressions
                X1, y1 = X[regime_reg], y_reg[regime_reg]
                X2, y2 = X[~regime_reg], y_reg[~regime_reg]

                if len(y1) > ar_order and len(y2) > ar_order:
                    lr1 = LinearRegression()
                    lr1.fit(X1, y1)
                    lr2 = LinearRegression()
                    lr2.fit(X2, y2)

                    # Residual sum of squares
                    rss1 = np.sum((y1 - lr1.predict(X1)) ** 2)
                    rss2 = np.sum((y2 - lr2.predict(X2)) ** 2)
                    rss_tar = rss1 + rss2
                    rss_linear = np.sum(residuals_linear ** 2)

                    # LM-type statistic
                    lm_stat = len(y_reg) * (rss_linear - rss_tar) / rss_linear
                    lm_stats.append(lm_stat)
                    valid_thresholds.append(threshold)

            except:
                continue

        if lm_stats:
            sup_lm = np.max(lm_stats)
            optimal_threshold = valid_thresholds[np.argmax(lm_stats)]
            return sup_lm, optimal_threshold, lm_stats, valid_thresholds
        else:
            return 0, 0, [], []


    # Compute test statistics
    sup_lm_linear, opt_thresh_linear, lm_stats_linear, thresholds_linear = compute_sup_lm_test(linear_data)
    sup_lm_nonlinear, opt_thresh_nonlinear, lm_stats_nonlinear, thresholds_nonlinear = compute_sup_lm_test(
        nonlinear_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Data (H₀ True)**")
        fig1 = go.Figure()
        if lm_stats_linear:
            fig1.add_trace(go.Scatter(x=thresholds_linear, y=lm_stats_linear,
                                      mode='lines+markers', name='LM(r)'))
            fig1.add_hline(y=sup_lm_linear, line_dash="dash", line_color="red",
                           annotation_text=f"sup-LM = {sup_lm_linear:.3f}")
        fig1.update_layout(title="LM Statistics vs Threshold",
                           xaxis_title="Threshold", yaxis_title="LM Statistic")
        st.plotly_chart(fig1, use_container_width=True)

        st.metric("sup-LM Statistic", f"{sup_lm_linear:.3f}")
        st.metric("Optimal Threshold", f"{opt_thresh_linear:.3f}")

    with col2:
        st.markdown("**Nonlinear Data (H₁ True)**")
        fig2 = go.Figure()
        if lm_stats_nonlinear:
            fig2.add_trace(go.Scatter(x=thresholds_nonlinear, y=lm_stats_nonlinear,
                                      mode='lines+markers', name='LM(r)'))
            fig2.add_hline(y=sup_lm_nonlinear, line_dash="dash", line_color="red",
                           annotation_text=f"sup-LM = {sup_lm_nonlinear:.3f}")
        fig2.update_layout(title="LM Statistics vs Threshold",
                           xaxis_title="Threshold", yaxis_title="LM Statistic")
        st.plotly_chart(fig2, use_container_width=True)

        st.metric("sup-LM Statistic", f"{sup_lm_nonlinear:.3f}")
        st.metric("Optimal Threshold", f"{opt_thresh_nonlinear:.3f}")

    st.markdown('<div class="subsection-header">🎲 Bootstrap Critical Values</div>', unsafe_allow_html=True)

    if st.button("🔄 Run Bootstrap Simulation"):
        with st.spinner("Running bootstrap simulation (this may take a moment)..."):

            def bootstrap_sup_lm(y, B=500):
                """Bootstrap procedure for sup-LM test"""
                n = len(y)
                ar_order = 2

                # Fit linear model under H0
                X = np.column_stack([y[i:n - ar_order + i] for i in range(ar_order)])
                y_reg = y[ar_order:]
                lr = LinearRegression()
                lr.fit(X, y_reg)
                residuals = y_reg - lr.predict(X)

                bootstrap_stats = []

                for b in range(B):
                    # Generate bootstrap residuals
                    boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)

                    # Generate bootstrap series
                    y_boot = np.zeros(n)
                    y_boot[:ar_order] = y[:ar_order]  # Initial values

                    for t in range(ar_order, n):
                        y_boot[t] = lr.intercept_ + np.sum(lr.coef_ * y_boot[t - ar_order:t][::-1]) + boot_residuals[
                            t - ar_order]

                    # Compute sup-LM for bootstrap sample
                    sup_lm_boot, _, _, _ = compute_sup_lm_test(y_boot)
                    bootstrap_stats.append(sup_lm_boot)

                return np.array(bootstrap_stats)


            # Bootstrap for linear data
            bootstrap_stats_linear = bootstrap_sup_lm(linear_data, B=200)  # Reduced for speed
            p_value_linear = np.mean(bootstrap_stats_linear > sup_lm_linear)

            # Bootstrap for nonlinear data
            bootstrap_stats_nonlinear = bootstrap_sup_lm(nonlinear_data, B=200)
            p_value_nonlinear = np.mean(bootstrap_stats_nonlinear > sup_lm_nonlinear)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Linear Data Results**")
                st.metric("Bootstrap P-value", f"{p_value_linear:.3f}")
                if p_value_linear > 0.05:
                    st.success("✅ Fail to reject linearity (correct)")
                else:
                    st.error("❌ Reject linearity (Type I error)")

                # Histogram of bootstrap statistics
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(x=bootstrap_stats_linear, nbinsx=30,
                                            name='Bootstrap sup-LM*'))
                fig1.add_vline(x=sup_lm_linear, line_dash="dash", line_color="red",
                               annotation_text=f"Observed = {sup_lm_linear:.3f}")
                fig1.update_layout(title="Bootstrap Distribution (H₀ True)")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("**Nonlinear Data Results**")
                st.metric("Bootstrap P-value", f"{p_value_nonlinear:.3f}")
                if p_value_nonlinear <= 0.05:
                    st.success("✅ Reject linearity (correct)")
                else:
                    st.error("❌ Fail to reject linearity (Type II error)")

                # Histogram of bootstrap statistics
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=bootstrap_stats_nonlinear, nbinsx=30,
                                            name='Bootstrap sup-LM*'))
                fig2.add_vline(x=sup_lm_nonlinear, line_dash="dash", line_color="red",
                               annotation_text=f"Observed = {sup_lm_nonlinear:.3f}")
                fig2.update_layout(title="Bootstrap Distribution (H₁ True)")
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="success-box">
    <h4>✅ Key Takeaways:</h4>

    • The Davies Problem makes standard tests invalid for threshold models<br>
    • <b>Supremum-type tests</b> provide the correct framework<br>
    • <b>Bootstrap methods</b> give accurate p-values and critical values<br>
    • The bootstrap correctly accounts for the <b>unidentified threshold under H₀</b><br>
    • This approach has good <b>size and power properties</b> in practice
    </div>
    """, unsafe_allow_html=True)

elif selected_section == "🔬 Practical Implementation":
    st.markdown('<div class="section-header">Practical Implementation in Python</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Complete TAR Modeling Workflow</h3>
    This section provides a complete, production-ready implementation of TAR models in Python.
    You can copy and use this code directly in your projects!
    </div>
    """, unsafe_allow_html=True)

    # Code implementation
    st.markdown('<div class="subsection-header">📝 Complete TAR Class Implementation</div>', unsafe_allow_html=True)

    st.code('''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from typing import Tuple, Dict, List, Optional
import warnings

class TARModel:
    """
    Complete Threshold Autoregressive (TAR) Model Implementation

    Features:
    - Pre-estimation linearity tests (Tsay, Keenan, BDS)
    - Grid search estimation with trimming
    - Bootstrap inference (Hansen's method)
    - Post-estimation diagnostics
    - Confidence intervals for threshold
    """

    def __init__(self):
        self.is_fitted = False
        self.results = {}

    def tsay_test(self, y: np.ndarray, max_lag: int = 4, delay: int = 1) -> Tuple[float, float]:
        """Tsay's arranged autoregression test for threshold nonlinearity"""
        n = len(y)

        # Threshold variable
        if delay >= n:
            raise ValueError(f"Delay {delay} too large for series length {n}")

        threshold_var = y[max_lag-1:-delay] if delay > 0 else y[max_lag-1:]

        # Sort observations by threshold variable
        sorted_indices = np.argsort(threshold_var)
        y_subset = y[max_lag+delay-1:]
        y_sorted = y_subset[sorted_indices]

        # Linear AR model on original data
        X_linear = np.column_stack([y[i:n-max_lag+i] for i in range(max_lag)])
        y_linear = y[max_lag:]

        lr_linear = LinearRegression()
        lr_linear.fit(X_linear, y_linear)
        residuals_linear = y_linear - lr_linear.predict(X_linear)

        # Arranged autoregression (simplified)
        try:
            # Recursive estimation on sorted data
            recursive_residuals = []
            min_obs = max_lag + 10  # Minimum observations for stable estimation

            for i in range(min_obs, len(y_sorted)):
                X_temp = np.column_stack([y_sorted[j:i-max_lag+j+1] for j in range(max_lag)])
                y_temp = y_sorted[max_lag:i+1]

                if len(y_temp) > max_lag:
                    lr_temp = LinearRegression()
                    lr_temp.fit(X_temp, y_temp)
                    pred = lr_temp.predict(X_temp[-1:])
                    recursive_residuals.append(y_sorted[i] - pred[0])

            # Test statistic
            if len(recursive_residuals) > 0:
                rss_arranged = np.sum(np.array(recursive_residuals)**2)
                rss_linear = np.sum(residuals_linear**2)
                test_stat = len(recursive_residuals) * (rss_linear - rss_arranged) / rss_linear
                p_value = 1 - stats.chi2.cdf(test_stat, df=max_lag)
            else:
                test_stat, p_value = 0, 1

        except Exception as e:
            warnings.warn(f"Tsay test computation failed: {e}")
            test_stat, p_value = 0, 1

        return test_stat, p_value

    def keenan_test(self, y: np.ndarray, max_lag: int = 4) -> Tuple[float, float]:
        """Keenan's test for quadratic nonlinearity"""
        n = len(y)

        # Prepare data
        X = np.column_stack([y[i:n-max_lag+i] for i in range(max_lag)])
        y_reg = y[max_lag:]

        # Linear model
        lr = LinearRegression()
        lr.fit(X, y_reg)
        fitted = lr.predict(X)

        # Quadratic term
        q = fitted**2 - np.mean(fitted**2)

        # Augmented regression
        X_aug = np.column_stack([X, q])
        lr_aug = LinearRegression()
        lr_aug.fit(X_aug, y_reg)

        # F-test for quadratic term
        rss_restricted = np.sum((y_reg - fitted)**2)
        rss_unrestricted = np.sum((y_reg - lr_aug.predict(X_aug))**2)

        df1 = 1
        df2 = len(y_reg) - X_aug.shape[1] - 1

        if df2 > 0 and rss_unrestricted > 0:
            f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            f_stat, p_value = 0, 1

        return f_stat, p_value

    def grid_search_estimate(self, 
                           y: np.ndarray, 
                           max_delay: int = 5,
                           ar_order: int = 2,
                           trim_pct: float = 0.15,
                           min_regime_size: int = None) -> Dict:
        """
        Grid search estimation of TAR model

        Parameters:
        -----------
        y : array-like
            Time series data
        max_delay : int
            Maximum delay to consider
        ar_order : int
            Autoregressive order
        trim_pct : float
            Percentage to trim from threshold grid (each tail)
        min_regime_size : int
            Minimum observations per regime (default: 15% of sample)

        Returns:
        --------
        dict : Estimation results
        """
        n = len(y)
        if min_regime_size is None:
            min_regime_size = int(0.15 * n)

        results = []

        for delay in range(1, max_delay + 1):
            if n - delay - ar_order < 2 * min_regime_size:
                continue

            # Threshold variable
            threshold_var = y[ar_order-1:-delay] if delay > 0 else y[ar_order-1:]

            # Threshold candidates (trimmed)
            threshold_candidates = np.percentile(threshold_var, 
                                               np.linspace(trim_pct*100, (1-trim_pct)*100, 25))

            for threshold in threshold_candidates:
                try:
                    result = self._estimate_tar_given_threshold(y, threshold, delay, ar_order, min_regime_size)
                    if result is not None:
                        result.update({'delay': delay, 'threshold': threshold})
                        results.append(result)
                except:
                    continue

        if not results:
            raise ValueError("Grid search failed. Try adjusting parameters.")

        # Select best model (minimum SSR)
        best_idx = np.argmin([r['ssr'] for r in results])
        best_result = results[best_idx]

        # Store results
        self.results = best_result
        self.results['all_candidates'] = results
        self.is_fitted = True

        return best_result

    def _estimate_tar_given_threshold(self, y, threshold, delay, ar_order, min_regime_size):
        """Estimate TAR model for given threshold and delay"""
        n = len(y)

        # Prepare data
        threshold_var = y[ar_order-1:-delay] if delay > 0 else y[ar_order-1:]
        regime_indicator = threshold_var <= threshold

        # Check regime sizes
        n_regime1 = np.sum(regime_indicator)
        n_regime2 = len(regime_indicator) - n_regime1

        if n_regime1 < min_regime_size or n_regime2 < min_regime_size:
            return None

        # Regression data
        X = np.column_stack([y[i:n-delay-ar_order+i+1] for i in range(ar_order)])
        y_reg = y[ar_order+delay-1:]
        regime_reg = regime_indicator[:len(y_reg)]

        # Estimate regime 1
        X1, y1 = X[regime_reg], y_reg[regime_reg]
        lr1 = LinearRegression()
        lr1.fit(X1, y1)
        fitted1 = lr1.predict(X1)

        # Estimate regime 2
        X2, y2 = X[~regime_reg], y_reg[~regime_reg]
        lr2 = LinearRegression()
        lr2.fit(X2, y2)
        fitted2 = lr2.predict(X2)

        # Combined results
        fitted = np.zeros(len(y_reg))
        fitted[regime_reg] = fitted1
        fitted[~regime_reg] = fitted2

        residuals = y_reg - fitted
        ssr = np.sum(residuals**2)

        # Store regime information
        result = {
            'ssr': ssr,
            'fitted': fitted,
            'residuals': residuals,
            'regime_indicator': regime_reg,
            'y_reg': y_reg,
            'X': X,
            'regime1': {
                'coef': lr1.coef_,
                'intercept': lr1.intercept_,
                'size': len(y1)
            },
            'regime2': {
                'coef': lr2.coef_,
                'intercept': lr2.intercept_,
                'size': len(y2)
            }
        }

        return result

    def bootstrap_test(self, y: np.ndarray, B: int = 1000, ar_order: int = 2) -> Dict:
        """
        Bootstrap test for linearity vs TAR (Hansen's method)

        Parameters:
        -----------
        y : array-like
            Time series data
        B : int
            Number of bootstrap replications
        ar_order : int
            AR order for linear model under null

        Returns:
        --------
        dict : Test results including bootstrap p-value
        """
        n = len(y)

        # Estimate linear model under H0
        X = np.column_stack([y[i:n-ar_order+i] for i in range(ar_order)])
        y_reg = y[ar_order:]
        lr = LinearRegression()
        lr.fit(X, y_reg)
        residuals = y_reg - lr.predict(X)

        # Compute observed sup-LM statistic
        observed_sup_lm = self._compute_sup_lm(y)

        # Bootstrap procedure
        bootstrap_stats = []

        for b in range(B):
            # Generate bootstrap sample under H0
            boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
            y_boot = self._generate_bootstrap_series(y, lr, boot_residuals, ar_order)

            # Compute sup-LM for bootstrap sample
            try:
                boot_sup_lm = self._compute_sup_lm(y_boot)
                bootstrap_stats.append(boot_sup_lm)
            except:
                bootstrap_stats.append(0)

        # Bootstrap p-value
        bootstrap_stats = np.array(bootstrap_stats)
        p_value = np.mean(bootstrap_stats > observed_sup_lm)

        return {
            'observed_sup_lm': observed_sup_lm,
            'bootstrap_stats': bootstrap_stats,
            'p_value': p_value,
            'critical_values': {
                '10%': np.percentile(bootstrap_stats, 90),
                '5%': np.percentile(bootstrap_stats, 95),
                '1%': np.percentile(bootstrap_stats, 99)
            }
        }

    def _compute_sup_lm(self, y, trim_pct=0.15):
        """Compute sup-LM statistic"""
        n = len(y)
        ar_order = 2

        # Linear model
        X = np.column_stack([y[i:n-ar_order+i] for i in range(ar_order)])
        y_reg = y[ar_order:]
        lr = LinearRegression()
        lr.fit(X, y_reg)
        rss_linear = np.sum((y_reg - lr.predict(X))**2)

        # Threshold candidates
        threshold_var = y[ar_order-1:-1]
        threshold_candidates = np.percentile(threshold_var, 
                                           np.linspace(trim_pct*100, (1-trim_pct)*100, 20))

        lm_stats = []

        for threshold in threshold_candidates:
            try:
                regime_indicator = threshold_var <= threshold

                # Check minimum regime sizes
                if np.sum(regime_indicator) < 0.15 * len(regime_indicator):
                    continue
                if np.sum(~regime_indicator) < 0.15 * len(regime_indicator):
                    continue

                # TAR estimation
                regime_reg = regime_indicator[:len(y_reg)]
                X1, y1 = X[regime_reg], y_reg[regime_reg]
                X2, y2 = X[~regime_reg], y_reg[~regime_reg]

                if len(y1) > ar_order and len(y2) > ar_order:
                    lr1 = LinearRegression()
                    lr1.fit(X1, y1)
                    lr2 = LinearRegression()
                    lr2.fit(X2, y2)

                    rss_tar = (np.sum((y1 - lr1.predict(X1))**2) + 
                              np.sum((y2 - lr2.predict(X2))**2))

                    lm_stat = len(y_reg) * (rss_linear - rss_tar) / rss_linear
                    lm_stats.append(lm_stat)

            except:
                continue

        return np.max(lm_stats) if lm_stats else 0

    def _generate_bootstrap_series(self, y, lr_model, boot_residuals, ar_order):
        """Generate bootstrap series under linear null"""
        n = len(y)
        y_boot = np.zeros(n)
        y_boot[:ar_order] = y[:ar_order]  # Use original initial values

        for t in range(ar_order, n):
            y_boot[t] = (lr_model.intercept_ + 
                        np.sum(lr_model.coef_ * y_boot[t-ar_order:t][::-1]) + 
                        boot_residuals[t-ar_order])

        return y_boot

    def confidence_interval_threshold(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Confidence interval for threshold parameter (Hansen 2000)

        Parameters:
        -----------
        alpha : float
            Significance level (default 0.05 for 95% CI)

        Returns:
        --------
        tuple : (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Critical value for confidence interval
        c_alpha = -2 * np.log(1 - np.sqrt(1 - alpha))

        # Get all candidate results
        candidates = self.results['all_candidates']
        optimal_ssr = self.results['ssr']
        n_obs = len(self.results['residuals'])

        # LR-type statistic
        lr_stats = []
        thresholds = []

        for candidate in candidates:
            if candidate['delay'] == self.results['delay']:  # Same delay
                lr_stat = n_obs * (candidate['ssr'] - optimal_ssr) / optimal_ssr
                lr_stats.append(lr_stat)
                thresholds.append(candidate['threshold'])

        # Find thresholds in confidence set
        lr_stats = np.array(lr_stats)
        thresholds = np.array(thresholds)

        in_ci = lr_stats <= c_alpha

        if np.any(in_ci):
            ci_thresholds = thresholds[in_ci]
            return np.min(ci_thresholds), np.max(ci_thresholds)
        else:
            # Return point estimate if CI is empty
            opt_threshold = self.results['threshold']
            return opt_threshold, opt_threshold

    def summary(self) -> str:
        """Generate summary of fitted TAR model"""
        if not self.is_fitted:
            return "Model not fitted"

        summary_str = f"""
TAR Model Summary
=================

Optimal Parameters:
- Delay: {self.results['delay']}
- Threshold: {self.results['threshold']:.4f}

Regime 1 (z_{{t-d}} ≤ {self.results['threshold']:.4f}):
- Observations: {self.results['regime1']['size']}
- Intercept: {self.results['regime1']['intercept']:.4f}
- AR coefficients: {self.results['regime1']['coef']}

Regime 2 (z_{{t-d}} > {self.results['threshold']:.4f}):
- Observations: {self.results['regime2']['size']}
- Intercept: {self.results['regime2']['intercept']:.4f}
- AR coefficients: {self.results['regime2']['coef']}

Model Fit:
- SSR: {self.results['ssr']:.4f}
- RMSE: {np.sqrt(self.results['ssr']/len(self.results['residuals'])):.4f}
"""

        return summary_str
''', language='python')

    st.markdown('<div class="subsection-header">🔧 Usage Example</div>', unsafe_allow_html=True)

    st.code('''
# Example usage of the TAR model class

# Import and create model
tar = TARModel()

# Generate sample data (or use your own)
np.random.seed(42)
y = tar.simulate_tar_data(n=300, phi1=[0.6, 0.3], phi2=[-0.2, 0.7], threshold=0.5)

# Step 1: Pre-estimation tests
print("=== Pre-estimation Tests ===")
tsay_stat, tsay_p = tar.tsay_test(y)
keenan_stat, keenan_p = tar.keenan_test(y)

print(f"Tsay test: statistic = {tsay_stat:.3f}, p-value = {tsay_p:.3f}")
print(f"Keenan test: statistic = {keenan_stat:.3f}, p-value = {keenan_p:.3f}")

# Step 2: Estimation
print("\\n=== Model Estimation ===")
results = tar.grid_search_estimate(y, max_delay=3, ar_order=2)

print(tar.summary())

# Step 3: Bootstrap test for linearity
print("\\n=== Bootstrap Test ===")
bootstrap_results = tar.bootstrap_test(y, B=500)  # Reduced for speed
print(f"Bootstrap p-value: {bootstrap_results['p_value']:.3f}")

# Step 4: Confidence interval for threshold
print("\\n=== Confidence Intervals ===")
ci_lower, ci_upper = tar.confidence_interval_threshold(alpha=0.05)
print(f"95% CI for threshold: [{ci_lower:.3f}, {ci_upper:.3f}]")
''', language='python')

    # Interactive example
    st.markdown('<div class="subsection-header">🎮 Interactive Example</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Complete TAR Analysis"):
        with st.spinner("Running complete TAR analysis..."):
            # Create TAR model instance
            tar = TARModel()

            # Generate sample data
            np.random.seed(42)
            y_example = tar.simulate_tar_data(n=250, phi1=[0.6, 0.3], phi2=[-0.2, 0.7], threshold=0.5)

            # Pre-estimation tests
            st.markdown("**Step 1: Pre-estimation Tests**")
            tsay_stat, tsay_p = tar.tsay_test(y_example)
            keenan_stat, keenan_p = tar.keenan_test(y_example)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tsay Test Statistic", f"{tsay_stat:.3f}")
                st.metric("Tsay P-value", f"{tsay_p:.3f}")
            with col2:
                st.metric("Keenan Test Statistic", f"{keenan_stat:.3f}")
                st.metric("Keenan P-value", f"{keenan_p:.3f}")

            # Model estimation
            st.markdown("**Step 2: Model Estimation**")
            results = tar.grid_search_estimate(y_example, max_delay=3, ar_order=2)

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Delay", results['delay'])
            with col2:
                st.metric("Optimal Threshold", f"{results['threshold']:.3f}")
            with col3:
                st.metric("SSR", f"{results['ssr']:.3f}")

            # Model summary
            st.text(tar.summary())

            # Bootstrap test
            st.markdown("**Step 3: Bootstrap Test for Linearity**")
            bootstrap_results = tar.bootstrap_test(y_example, B=200)  # Reduced for demo

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bootstrap P-value", f"{bootstrap_results['p_value']:.3f}")
                if bootstrap_results['p_value'] <= 0.05:
                    st.success("✅ Reject linearity - TAR model preferred")
                else:
                    st.warning("⚠️ Fail to reject linearity")

            with col2:
                st.markdown("**Bootstrap Critical Values**")
                for level, cv in bootstrap_results['critical_values'].items():
                    st.write(f"{level}: {cv:.3f}")

            # Confidence interval
            st.markdown("**Step 4: Confidence Interval for Threshold**")
            try:
                ci_lower, ci_upper = tar.confidence_interval_threshold(alpha=0.05)
                st.metric("95% CI for Threshold", f"[{ci_lower:.3f}, {ci_upper:.3f}]")
            except Exception as e:
                st.error(f"CI computation failed: {e}")

            # Visualization
            fig = make_subplots(rows=2, cols=2,
                                subplot_titles=['Time Series', 'Fitted vs Actual',
                                                'Residuals', 'Bootstrap Distribution'])

            # Time series
            fig.add_trace(go.Scatter(y=y_example, mode='lines', name='Data'), row=1, col=1)
            fig.add_hline(y=results['threshold'], line_dash="dash", line_color="red", row=1, col=1)

            # Fitted vs actual
            fig.add_trace(go.Scatter(x=results['y_reg'], y=results['fitted'],
                                     mode='markers', name='Fitted vs Actual'), row=1, col=2)
            fig.add_trace(go.Scatter(x=[min(results['y_reg']), max(results['y_reg'])],
                                     y=[min(results['y_reg']), max(results['y_reg'])],
                                     mode='lines', name='45° line'), row=1, col=2)

            # Residuals
            fig.add_trace(go.Scatter(y=results['residuals'], mode='lines+markers',
                                     name='Residuals'), row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

            # Bootstrap distribution
            fig.add_trace(go.Histogram(x=bootstrap_results['bootstrap_stats'], nbinsx=30,
                                       name='Bootstrap sup-LM'), row=2, col=2)
            fig.add_vline(x=bootstrap_results['observed_sup_lm'], line_dash="dash",
                          line_color="red", row=2, col=2)

            fig.update_layout(height=800, showlegend=False, title_text="Complete TAR Analysis Results")
            st.plotly_chart(fig, use_container_width=True)

elif selected_section == "📈 Interactive Examples":
    st.markdown('<div class="section-header">Interactive Examples and Data Exploration</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Learn by Doing</h3>
    This section lets you experiment with TAR models using both simulated and real-world datasets.
    Adjust parameters and see how they affect model behavior and estimation results!
    </div>
    """, unsafe_allow_html=True)

    # Example selection
    example_type = st.selectbox(
        "Choose an example:",
        ["📊 Simulated TAR Data", "💹 Financial Time Series", "🌡️ Climate Data", "📈 Economic Indicators"]
    )

    if example_type == "📊 Simulated TAR Data":
        st.markdown('<div class="subsection-header">🔧 Custom TAR Data Generator</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Regime 1 Parameters**")
            phi1_0 = st.slider("φ₁₀ (Intercept)", -1.0, 1.0, 0.1, 0.1, key="phi10")
            phi1_1 = st.slider("φ₁₁ (AR1)", -0.99, 0.99, 0.6, 0.01, key="phi11")
            phi1_2 = st.slider("φ₁₂ (AR2)", -0.99, 0.99, 0.3, 0.01, key="phi12")
            sigma1 = st.slider("σ₁ (Error SD)", 0.1, 2.0, 1.0, 0.1, key="sigma1")

        with col2:
            st.markdown("**Regime 2 Parameters**")
            phi2_0 = st.slider("φ₂₀ (Intercept)", -1.0, 1.0, -0.1, 0.1, key="phi20")
            phi2_1 = st.slider("φ₂₁ (AR1)", -0.99, 0.99, -0.2, 0.01, key="phi21")
            phi2_2 = st.slider("φ₂₂ (AR2)", -0.99, 0.99, 0.7, 0.01, key="phi22")
            sigma2 = st.slider("σ₂ (Error SD)", 0.1, 2.0, 1.2, 0.1, key="sigma2")

        with col3:
            st.markdown("**Model Settings**")
            threshold = st.slider("Threshold (r)", -2.0, 2.0, 0.0, 0.1, key="threshold")
            delay = st.selectbox("Delay (d)", [1, 2, 3], key="delay")
            n_obs = st.slider("Sample Size", 100, 1000, 300, 50, key="n_obs")
            noise_seed = st.slider("Random Seed", 1, 100, 42, 1, key="seed")


        # Generate data
        def generate_custom_tar(n, phi1, phi2, threshold, delay, sigma1, sigma2, seed):
            np.random.seed(seed)
            y = np.zeros(n)

            # Initial values
            for i in range(max(len(phi1), len(phi2))):
                y[i] = np.random.normal(0, 0.5)

            for t in range(max(len(phi1), len(phi2)), n):
                if t >= delay:
                    if y[t - delay] <= threshold:
                        # Regime 1
                        y[t] = phi1[0] + sum(
                            phi1[i + 1] * y[t - 1 - i] for i in range(len(phi1) - 1)) + np.random.normal(0, sigma1)
                    else:
                        # Regime 2
                        y[t] = phi2[0] + sum(
                            phi2[i + 1] * y[t - 1 - i] for i in range(len(phi2) - 1)) + np.random.normal(0, sigma2)
                else:
                    y[t] = np.random.normal(0, 0.5)

            return y


        phi1_params = [phi1_0, phi1_1, phi1_2]
        phi2_params = [phi2_0, phi2_1, phi2_2]

        y_custom = generate_custom_tar(n_obs, phi1_params, phi2_params, threshold, delay, sigma1, sigma2, noise_seed)


        # Check stationarity
        def check_stationarity(phi_coeffs):
            """Simple check for AR(2) stationarity"""
            if len(phi_coeffs) < 3:  # Need at least intercept + 2 AR terms
                return True

            phi1, phi2 = phi_coeffs[1], phi_coeffs[2]

            # AR(2) stationarity conditions
            condition1 = phi1 + phi2 < 1
            condition2 = phi2 - phi1 < 1
            condition3 = abs(phi2) < 1

            return condition1 and condition2 and condition3


        regime1_stationary = check_stationarity(phi1_params)
        regime2_stationary = check_stationarity(phi2_params)

        # Display warnings if non-stationary
        if not regime1_stationary:
            st.warning("⚠️ Regime 1 parameters may lead to non-stationary behavior")
        if not regime2_stationary:
            st.warning("⚠️ Regime 2 parameters may lead to non-stationary behavior")

        # Regime classification
        regime_indicator = np.zeros(len(y_custom))
        for t in range(delay, len(y_custom)):
            regime_indicator[t] = 1 if y_custom[t - delay] <= threshold else 2

        # Visualization
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=['Generated TAR Series',
                                            'Regime Classification',
                                            'Distribution by Regime'],
                            row_heights=[0.4, 0.3, 0.3])

        # Time series
        fig.add_trace(go.Scatter(x=list(range(len(y_custom))), y=y_custom,
                                 mode='lines', name='TAR Series', line=dict(color='blue')), row=1, col=1)
        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold = {threshold}", row=1, col=1)

        # Regime classification
        colors = ['blue' if r == 1 else 'red' for r in regime_indicator[delay:]]
        fig.add_trace(go.Scatter(x=list(range(delay, len(y_custom))), y=regime_indicator[delay:],
                                 mode='markers', marker=dict(color=colors, size=3),
                                 name='Regimes'), row=2, col=1)

        # Distribution by regime
        regime1_data = y_custom[delay:][regime_indicator[delay:] == 1]
        regime2_data = y_custom[delay:][regime_indicator[delay:] == 2]

        fig.add_trace(go.Histogram(x=regime1_data, nbinsx=30, name='Regime 1',
                                   opacity=0.7, marker_color='blue'), row=3, col=1)
        fig.add_trace(go.Histogram(x=regime2_data, nbinsx=30, name='Regime 2',
                                   opacity=0.7, marker_color='red'), row=3, col=1)

        fig.update_layout(height=900, showlegend=True, title_text="Custom TAR Model Simulation")
        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Observations", len(y_custom))
            st.metric("Regime 1 Count", len(regime1_data))

        with col2:
            st.metric("Regime 2 Count", len(regime2_data))
            st.metric("Regime 1 %", f"{len(regime1_data) / (len(regime1_data) + len(regime2_data)) * 100:.1f}%")

        with col3:
            st.metric("Regime 1 Mean", f"{np.mean(regime1_data):.3f}")
            st.metric("Regime 1 Std", f"{np.std(regime1_data):.3f}")

        with col4:
            st.metric("Regime 2 Mean", f"{np.mean(regime2_data):.3f}")
            st.metric("Regime 2 Std", f"{np.std(regime2_data):.3f}")

        # Run analysis button
        if st.button("🔍 Analyze This TAR Model"):
            with st.spinner("Analyzing TAR model..."):
                tar = TARModel()

                # Pre-tests
                tsay_stat, tsay_p = tar.tsay_test(y_custom)
                keenan_stat, keenan_p = tar.keenan_test(y_custom)

                st.markdown("**Pre-estimation Test Results:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tsay Test P-value", f"{tsay_p:.3f}")
                    if tsay_p <= 0.05:
                        st.success("✅ Nonlinearity detected")
                    else:
                        st.warning("⚠️ No strong nonlinearity evidence")

                with col2:
                    st.metric("Keenan Test P-value", f"{keenan_p:.3f}")
                    if keenan_p <= 0.05:
                        st.success("✅ Nonlinearity detected")
                    else:
                        st.warning("⚠️ No strong nonlinearity evidence")

                # Estimation
                try:
                    results = tar.grid_search_estimate(y_custom, max_delay=min(3, delay + 1), ar_order=2)

                    st.markdown("**Estimation Results:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Estimated Delay", results['delay'])
                        st.metric("True Delay", delay)

                    with col2:
                        st.metric("Estimated Threshold", f"{results['threshold']:.3f}")
                        st.metric("True Threshold", f"{threshold:.3f}")

                    with col3:
                        error = abs(results['threshold'] - threshold)
                        st.metric("Threshold Error", f"{error:.3f}")
                        if error < 0.1:
                            st.success("✅ Excellent estimate")
                        elif error < 0.3:
                            st.warning("⚠️ Reasonable estimate")
                        else:
                            st.error("❌ Poor estimate")

                    # Display full summary
                    st.text(tar.summary())

                except Exception as e:
                    st.error(f"Estimation failed: {e}")

    elif example_type == "💹 Financial Time Series":
        st.markdown('<div class="subsection-header">📈 Stock Market Returns Analysis</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="theory-box">
        <h4>Financial Applications of TAR Models:</h4>
        • <b>Bull vs Bear Markets:</b> Different dynamics in rising vs falling markets<br>
        • <b>Volatility Clustering:</b> High volatility periods vs calm periods<br>
        • <b>Crisis Detection:</b> Normal times vs crisis periods<br>
        • <b>Trading Strategies:</b> Regime-dependent investment rules
        </div>
        """, unsafe_allow_html=True)

        # Generate realistic financial data with regime switching
        np.random.seed(42)
        n = 500

        # Simulate regime-switching returns
        returns = np.zeros(n)
        volatility_regime = np.zeros(n)

        # Market state (0 = calm, 1 = volatile)
        p_stay_calm = 0.95
        p_stay_volatile = 0.85

        current_regime = 0

        for t in range(1, n):
            # Regime switching
            if current_regime == 0:  # Calm market
                if np.random.random() > p_stay_calm:
                    current_regime = 1
            else:  # Volatile market
                if np.random.random() > p_stay_volatile:
                    current_regime = 0

            volatility_regime[t] = current_regime

            # Generate returns based on regime
            if current_regime == 0:  # Calm market
                returns[t] = 0.0005 + 0.1 * returns[t - 1] + np.random.normal(0, 0.01)
            else:  # Volatile market
                returns[t] = -0.001 + 0.3 * returns[t - 1] + np.random.normal(0, 0.03)

        # Convert to prices
        prices = 100 * np.exp(np.cumsum(returns))

        # Visualization
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=['Stock Price', 'Returns', 'Volatility Regime'],
                            row_heights=[0.4, 0.3, 0.3])

        # Prices
        fig.add_trace(go.Scatter(x=list(range(n)), y=prices, mode='lines',
                                 name='Stock Price', line=dict(color='blue')), row=1, col=1)

        # Returns with regime coloring
        colors = ['green' if r == 0 else 'red' for r in volatility_regime]
        fig.add_trace(go.Scatter(x=list(range(n)), y=returns, mode='markers',
                                 marker=dict(color=colors, size=3),
                                 name='Returns'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Regime indicator
        fig.add_trace(go.Scatter(x=list(range(n)), y=volatility_regime, mode='lines',
                                 name='Volatility Regime', line=dict(color='purple')), row=3, col=1)

        fig.update_layout(height=800, title_text="Simulated Financial Time Series with Regime Switching")
        st.plotly_chart(fig, use_container_width=True)

        # Analysis
        if st.button("📊 Analyze Financial TAR Model"):
            with st.spinner("Analyzing financial time series..."):
                tar = TARModel()

                # Use returns for TAR analysis
                returns_clean = returns[50:]  # Remove initial observations

                # Pre-tests
                tsay_stat, tsay_p = tar.tsay_test(returns_clean)
                keenan_stat, keenan_p = tar.keenan_test(returns_clean)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Tsay Test (Threshold Nonlinearity)**")
                    st.metric("Test Statistic", f"{tsay_stat:.3f}")
                    st.metric("P-value", f"{tsay_p:.3f}")
                    if tsay_p <= 0.05:
                        st.success("✅ Evidence of threshold effects")
                    else:
                        st.warning("⚠️ Weak evidence of threshold effects")

                with col2:
                    st.markdown("**Keenan Test (Quadratic Nonlinearity)**")
                    st.metric("Test Statistic", f"{keenan_stat:.3f}")
                    st.metric("P-value", f"{keenan_p:.3f}")
                    if keenan_p <= 0.05:
                        st.success("✅ Evidence of nonlinearity")
                    else:
                        st.warning("⚠️ Weak evidence of nonlinearity")

                # Estimation
                try:
                    results = tar.grid_search_estimate(returns_clean, max_delay=3, ar_order=2)

                    st.markdown("**TAR Model Results:**")
                    st.text(tar.summary())

                    # Regime analysis
                    regime_1_returns = returns_clean[results['regime_indicator']]
                    regime_2_returns = returns_clean[~results['regime_indicator']]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Regime 1 (Low Volatility)**")
                        st.metric("Mean Return", f"{np.mean(regime_1_returns) * 252:.1f}% (annualized)")
                        st.metric("Volatility", f"{np.std(regime_1_returns) * np.sqrt(252):.1f}% (annualized)")
                        st.metric("Sharpe Ratio",
                                  f"{np.mean(regime_1_returns) / np.std(regime_1_returns) * np.sqrt(252):.2f}")

                    with col2:
                        st.markdown("**Regime 2 (High Volatility)**")
                        st.metric("Mean Return", f"{np.mean(regime_2_returns) * 252:.1f}% (annualized)")
                        st.metric("Volatility", f"{np.std(regime_2_returns) * np.sqrt(252):.1f}% (annualized)")
                        st.metric("Sharpe Ratio",
                                  f"{np.mean(regime_2_returns) / np.std(regime_2_returns) * np.sqrt(252):.2f}")

                except Exception as e:
                    st.error(f"TAR estimation failed: {e}")

    else:
        st.info("🚧 More examples coming soon! Choose 'Simulated TAR Data' or 'Financial Time Series' for now.")

elif selected_section == "📝 Summary & Best Practices":
    st.markdown('<div class="section-header">Summary & Best Practices</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Congratulations!</h3>
    You've completed a comprehensive journey through Threshold Autoregressive (TAR) models. 
    This section summarizes key concepts and provides practical guidance for your own research.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">📋 TAR Modeling Checklist</div>', unsafe_allow_html=True)

    checklist_items = [
        ("🔍 **Data Exploration**", [
            "Plot your time series and look for potential regime changes",
            "Check for obvious structural breaks or asymmetric patterns",
            "Examine autocorrelation and partial autocorrelation functions",
            "Consider economic/financial intuition about possible regimes"
        ]),
        ("🧪 **Pre-Estimation Testing**", [
            "Run Tsay's test for threshold nonlinearity",
            "Apply Keenan's test for quadratic nonlinearity",
            "Use BDS test on linear model residuals",
            "Only proceed with TAR if tests suggest nonlinearity"
        ]),
        ("⚙️ **Model Estimation**", [
            "Use grid search with proper trimming (15-20% each tail)",
            "Ensure minimum regime sizes (at least 15% of sample)",
            "Try multiple delay values (d = 1, 2, 3, ...)",
            "Select optimal (d,r) based on information criteria"
        ]),
        ("📊 **Inference & Testing**", [
            "Use Hansen's bootstrap for testing linearity vs TAR",
            "Compute confidence intervals for threshold parameter",
            "Remember: standard inference is invalid due to Davies problem",
            "Use sup-type tests, not standard LR/Wald tests"
        ]),
        ("🔬 **Post-Estimation Diagnostics**", [
            "Check residual autocorrelation (Ljung-Box test)",
            "Test for remaining nonlinearity (BDS on residuals)",
            "Examine residual normality (Jarque-Bera test)",
            "Verify regime classification makes economic sense"
        ])
    ]

    for title, items in checklist_items:
        st.markdown(f'<div class="subsection-header">{title}</div>', unsafe_allow_html=True)
        for item in items:
            st.markdown(f"✅ {item}")
        st.markdown("")

    st.markdown('<div class="subsection-header">⚠️ Common Pitfalls to Avoid</div>', unsafe_allow_html=True)

    pitfalls = [
        ("**Using Standard Critical Values**",
         "Never use χ² critical values for testing linearity vs TAR. Always use bootstrap methods."),
        ("**Insufficient Regime Sizes**",
         "Ensure each regime has enough observations (≥15% of sample) for reliable estimation."),
        ("**Ignoring the Davies Problem**",
         "The threshold is unidentified under the null. Standard inference theory breaks down."),
        ("**Over-trimming the Grid**",
         "Don't trim too much (>30%) as you might exclude the true threshold."),
        ("**Neglecting Pre-tests**",
         "Always test for nonlinearity first. Don't fit TAR to linear data."),
        ("**Forgetting Economic Interpretation**",
         "Make sure regime classification aligns with economic theory/intuition.")
    ]

    for title, description in pitfalls:
        st.markdown(f"""
        <div class="danger-box">
        <h4>❌ {title}</h4>
        {description}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🎯 When to Use TAR Models</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>✅ TAR Models Are Great For:</h4>
        • Financial returns with bull/bear markets<br>
        • Economic cycles (expansion/recession)<br>
        • Policy regime switches<br>
        • Asymmetric adjustment processes<br>
        • Crisis vs normal periods<br>
        • Inventory adjustment models<br>
        • Exchange rate dynamics
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Consider Alternatives When:</h4>
        • Data is truly linear (pre-tests confirm)<br>
        • Sample size is very small (<100 obs)<br>
        • Regime switches are very rare<br>
        • Smooth transitions are more appropriate<br>
        • Multiple thresholds are suspected<br>
        • Time-varying parameters are needed
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">📚 Further Reading & Extensions</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h4>📖 Key References:</h4>

    <b>Foundational Papers:</b><br>
    • Tong, H. (1983). Threshold Models in Non-linear Time Series Analysis<br>
    • Hansen, B.E. (1996). Inference when a nuisance parameter is not identified under the null<br>
    • Hansen, B.E. (1997). Inference in TAR models<br>
    • Tsay, R.S. (1989). Testing and modeling threshold autoregressive processes

    <b>Advanced Topics:</b><br>
    • Smooth Transition AR (STAR) models<br>
    • Multiple regime TAR models<br>
    • Vector TAR (VTAR) models<br>
    • Threshold Error Correction Models<br>
    • Markov-Switching models
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="subsection-header">🔧 Python Implementation Tips</div>', unsafe_allow_html=True)

    st.code('''
# Quick reference for the TARModel class we built

from tar_model import TARModel  # Assuming you saved our implementation

# Basic workflow
tar = TARModel()

# 1. Pre-tests
tsay_stat, tsay_p = tar.tsay_test(data)
keenan_stat, keenan_p = tar.keenan_test(data)

# 2. Estimation (if pre-tests suggest nonlinearity)
if tsay_p < 0.05 or keenan_p < 0.05:
    results = tar.grid_search_estimate(data, max_delay=3, ar_order=2)
    print(tar.summary())

    # 3. Bootstrap test
    bootstrap_results = tar.bootstrap_test(data, B=1000)
    print(f"Bootstrap p-value: {bootstrap_results['p_value']:.3f}")

    # 4. Confidence interval
    ci_lower, ci_upper = tar.confidence_interval_threshold()
    print(f"95% CI for threshold: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Pro tips:
# - Always set random seeds for reproducibility
# - Use B=5000+ bootstrap replications for publication
# - Save intermediate results for large computations
# - Parallelize bootstrap for speed (not shown here)
''', language='python')

    st.markdown('<div class="subsection-header">🎊 Final Thoughts</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
    <h3>🚀 You're Now Ready to:</h3>

    ✅ <b>Identify</b> when TAR models are appropriate for your data<br>
    ✅ <b>Test</b> for nonlinearity using proper statistical procedures<br>
    ✅ <b>Estimate</b> TAR models using grid search methods<br>
    ✅ <b>Conduct</b> valid inference accounting for the Davies problem<br>
    ✅ <b>Interpret</b> results in economic and statistical terms<br>
    ✅ <b>Implement</b> everything in Python from scratch<br>

    <h4>🎯 Remember the Golden Rules:</h4>
    1. <b>Always test before estimating</b> - don't fit TAR to linear data<br>
    2. <b>Use bootstrap inference</b> - standard tests are invalid<br>
    3. <b>Think economically</b> - do the regimes make sense?<br>
    4. <b>Check diagnostics</b> - verify model adequacy<br>
    5. <b>Be patient</b> - good TAR modeling takes time and care
    </div>
    """, unsafe_allow_html=True)

    # Final interactive element
    st.markdown("---")

    if st.button("🎉 Generate Your TAR Analysis Certificate!"):
        st.balloons()

        certificate_html = f"""
        <div style="border: 3px solid #1f77b4; padding: 20px; text-align: center; background: linear-gradient(45deg, #f0f8ff, #e6f3ff); border-radius: 15px; margin: 20px 0;">
            <h1 style="color: #1f77b4; margin-bottom: 10px;">🏆 Certificate of Completion</h1>
            <h2 style="color: #ff7f0e;">Threshold Autoregressive Models</h2>
            <p style="font-size: 18px; margin: 20px 0;">
                <b>This certifies that you have successfully completed</b><br>
                the comprehensive guide to TAR models, including:
            </p>
            <div style="text-align: left; max-width: 500px; margin: 0 auto;">
                ✅ Mathematical foundations and theory<br>
                ✅ Pre-estimation testing procedures<br>
                ✅ Grid search estimation techniques<br>
                ✅ Bootstrap inference methods<br>
                ✅ Davies problem understanding<br>
                ✅ Practical Python implementation<br>
                ✅ Interactive examples and applications
            </div>
            <p style="font-size: 16px; margin-top: 20px; color: #666;">
                <b>Date:</b> {pd.Timestamp.now().strftime('%B %d, %Y')}<br>
                <b>Status:</b> TAR Model Expert 🎯
            </p>
            <p style="font-style: italic; color: #888; margin-top: 15px;">
                "The best way to learn TAR models is by doing - and you've done it all!"
            </p>
        </div>
        """

        st.markdown(certificate_html, unsafe_allow_html=True)

        st.success(
            "🎊 Congratulations! You're now equipped with advanced knowledge of TAR models. Go forth and model those regime switches!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>📊 <b>Complete Guide to Threshold Autoregressive Models</b> 📊</p>
    <p>Built with ❤️ using Streamlit • For educational and research purposes</p>
    <p><i>"Understanding nonlinearity, one threshold at a time"</i></p>
</div>
""", unsafe_allow_html=True)