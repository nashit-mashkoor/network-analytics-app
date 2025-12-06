"""
Shared UI styling utilities for the Network Analytics App.
Provides consistent, beautiful formatting across all analysis pages.
"""
import streamlit as st

# Custom CSS for analysis pages
ANALYSIS_CSS = """
<style>
/* Section headers */
.section-header {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #e94560;
    margin: 1.5rem 0 1rem 0;
}

.section-header h3 {
    color: #e94560;
    margin: 0;
    font-size: 1.4rem;
}

/* Metric cards */
.metric-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border: 1px solid rgba(233, 69, 96, 0.3);
    min-width: 150px;
    flex: 1;
}

.metric-label {
    color: #a2d2ff;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}

.metric-value {
    color: #e94560;
    font-size: 1.8rem;
    font-weight: 700;
}

.metric-value.good {
    color: #4ade80;
}

.metric-value.warning {
    color: #fbbf24;
}

.metric-value.bad {
    color: #f87171;
}

/* Info boxes */
.info-box {
    background: rgba(162, 210, 255, 0.1);
    border: 1px solid rgba(162, 210, 255, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.info-box-title {
    color: #a2d2ff;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.info-box-content {
    color: #ccd6f6;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-success {
    background: rgba(74, 222, 128, 0.2);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.4);
}

.status-processing {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.4);
}

.status-info {
    background: rgba(162, 210, 255, 0.2);
    color: #a2d2ff;
    border: 1px solid rgba(162, 210, 255, 0.4);
}

/* Data summary cards */
.summary-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    padding: 1.2rem;
    border-radius: 10px;
    border: 1px solid rgba(233, 69, 96, 0.2);
    margin: 0.5rem 0;
}

.summary-title {
    color: #e94560;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.summary-content {
    color: #ccd6f6;
}

/* Divider */
.styled-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(233, 69, 96, 0.5), transparent);
    margin: 1.5rem 0;
}

/* Progress indicator */
.progress-step {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(233, 69, 96, 0.1);
    border-radius: 20px;
    color: #e94560;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* Outlier count display */
.outlier-display {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    border-left: 3px solid #e94560;
    margin: 0.3rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.outlier-label {
    color: #ccd6f6;
}

.outlier-count {
    color: #e94560;
    font-weight: 700;
    font-size: 1.2rem;
}

/* Comparison table styling */
.comparison-header {
    background: linear-gradient(90deg, #e94560, #0f3460);
    padding: 0.8rem;
    border-radius: 8px 8px 0 0;
    text-align: center;
    color: white;
    font-weight: 600;
}
</style>
"""

def inject_analysis_css():
    """Inject the analysis page CSS styles."""
    st.markdown(ANALYSIS_CSS, unsafe_allow_html=True)

def section_header(title, icon="📊"):
    """Create a styled section header."""
    st.markdown(f"""
    <div class="section-header">
        <h3>{icon} {title}</h3>
    </div>
    """, unsafe_allow_html=True)

def styled_divider():
    """Create a styled divider."""
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

def display_metrics_row(metrics_dict):
    """
    Display metrics in a styled row.
    metrics_dict: dict with format {"label": value, ...}
    """
    cols = st.columns(len(metrics_dict))
    for col, (label, value) in zip(cols, metrics_dict.items()):
        with col:
            # Format value based on type
            if isinstance(value, float):
                if value < 0.01:
                    formatted = f"{value:.6f}"
                elif value < 1:
                    formatted = f"{value:.4f}"
                else:
                    formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{formatted}</div>
            </div>
            """, unsafe_allow_html=True)

def display_metric_card(label, value, color_class=""):
    """Display a single metric card."""
    if isinstance(value, float):
        if value < 0.01:
            formatted = f"{value:.6f}"
        elif value < 1:
            formatted = f"{value:.4f}"
        else:
            formatted = f"{value:.4f}"
    else:
        formatted = str(value)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{formatted}</div>
    </div>
    """, unsafe_allow_html=True)

def info_box(title, content):
    """Display an info box."""
    st.markdown(f"""
    <div class="info-box">
        <div class="info-box-title">{title}</div>
        <div class="info-box-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def status_badge(text, status="info"):
    """Display a status badge. status: 'success', 'processing', or 'info'"""
    return f'<span class="status-badge status-{status}">{text}</span>'

def summary_card(title, content, icon="📋"):
    """Display a summary card."""
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">{icon} {title}</div>
        <div class="summary-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def progress_step(step_name):
    """Display a progress step indicator."""
    st.markdown(f"""
    <div class="progress-step">
        ⏳ {step_name}
    </div>
    """, unsafe_allow_html=True)

def outlier_display(label, count):
    """Display outlier count in styled format."""
    st.markdown(f"""
    <div class="outlier-display">
        <span class="outlier-label">{label}</span>
        <span class="outlier-count">{count}</span>
    </div>
    """, unsafe_allow_html=True)

def comparison_header(title):
    """Display a comparison section header."""
    st.markdown(f"""
    <div class="comparison-header">
        {title}
    </div>
    """, unsafe_allow_html=True)

def display_data_shape(shape, description=""):
    """Display data shape in a styled format."""
    if len(shape) == 2:
        rows, cols = shape
        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">📐 Data Dimensions</div>
            <div class="info-box-content">
                <strong>{rows:,}</strong> rows × <strong>{cols:,}</strong> columns
                {f'<br><em>{description}</em>' if description else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">📐 Data Shape</div>
            <div class="info-box-content">{shape}</div>
        </div>
        """, unsafe_allow_html=True)

def format_model_performance(r2=None, mae=None, mse=None, rmse=None, mape=None, mase=None):
    """Display model performance metrics in a beautiful grid."""
    metrics = {}
    if r2 is not None:
        metrics["R² Score"] = r2
    if mae is not None:
        metrics["MAE"] = mae
    if mape is not None:
        metrics["MAPE"] = mape
    if mse is not None:
        metrics["MSE"] = mse
    if rmse is not None:
        metrics["RMSE"] = rmse
    if mase is not None:
        metrics["MASE"] = mase
    
    if metrics:
        display_metrics_row(metrics)

