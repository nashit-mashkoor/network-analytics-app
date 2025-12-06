import streamlit as st
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

from tqdm import notebook
from prettytable import PrettyTable 
from astropy.table import Table, Column
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from src.TimeSeries import msc_traffic_process_render, apn_utilisation_process_render
from src.Prediction import eCell_Accessibility_process_render, eCell_Retainability_process_render
from src.Anomaly import jitter_15min_15days_30nodes_process_render, total_traffic_rate_5min_7days_20nodes

# App setting
st.set_page_config(
    page_title="Network Analytics App", layout="wide", initial_sidebar_state="collapsed",
    page_icon='🕸'
)

# Custom CSS for beautiful styling
CUSTOM_CSS = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

#network-analytics-app {
    text-align: center;
}

.hero-section {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #e94560;
    box-shadow: 0 4px 20px rgba(233, 69, 96, 0.2);
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #e94560;
    text-align: center;
    margin-bottom: 0.5rem;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: #a2d2ff;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 300;
}

.feature-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #e94560;
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}

.feature-card:hover {
    transform: translateX(5px);
}

.feature-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e94560;
    margin-bottom: 0.5rem;
}

.feature-desc {
    color: #ccd6f6;
    font-size: 0.95rem;
    line-height: 1.6;
}

.model-tag {
    display: inline-block;
    background: rgba(233, 69, 96, 0.15);
    color: #e94560;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 0.2rem;
    border: 1px solid rgba(233, 69, 96, 0.3);
}

.dataset-tag {
    display: inline-block;
    background: rgba(162, 210, 255, 0.15);
    color: #a2d2ff;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 0.2rem;
    border: 1px solid rgba(162, 210, 255, 0.3);
}

.getting-started {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-top: 2rem;
    border: 1px solid #a2d2ff;
    text-align: center;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: #e94560;
    color: white;
    border-radius: 50%;
    font-weight: bold;
    margin-right: 0.5rem;
}

.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #e94560, transparent);
    margin: 2rem 0;
}

.stats-container {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}

.stat-item {
    text-align: center;
    padding: 1rem;
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: #e94560;
}

.stat-label {
    color: #a2d2ff;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.model-detail-card {
    background: linear-gradient(145deg, #1a1a2e, #0f3460);
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border: 1px solid rgba(233, 69, 96, 0.2);
}

.model-name {
    color: #e94560;
    font-weight: 600;
    font-size: 1rem;
}

.model-desc {
    color: #ccd6f6;
    font-size: 0.85rem;
    margin-top: 0.3rem;
}
            </style>
            """
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.container():
    st.title('🕸 Network Analytics App')
    TITLE_ALIGNMENT="""
                    <style>
                    #network-analytics-app {
                    text-align: center
                    }
                    </style>
                    """
    st.markdown(TITLE_ALIGNMENT, unsafe_allow_html=True)

# Setting Sidebar for configuration
sidebar_col1, sidebar_col2 = st.sidebar.columns([1, 5])
with sidebar_col1:
    home_button = st.button('🏠')
with sidebar_col2:
    st.markdown("### ⚙️ Configuration")

st.sidebar.markdown("---")
selected_usecase = st.sidebar.radio('🎯 Select Use Case', ('Time_Series_Forecasting','Prediction', 'Anomaly_Detection'), index=0)

if selected_usecase == 'Time_Series_Forecasting':
    dataset_name = st.sidebar.selectbox('📊 Select Dataset',
                                ('MSC_Traffic', 'APN_Utilization'))
    if  dataset_name == 'MSC_Traffic':       
        model_name = st.sidebar.selectbox('🤖 Select Model',
                                    ('HWES', 'SARIMA', 'XGBoost', 'Prophet', 'LSTM'))
    elif dataset_name == 'APN_Utilization': 
        model_name = st.sidebar.selectbox('🤖 Select Model',
                                    ('LSTM_GRU', 'NBEATS', 'CNN_Wavenets'))
elif selected_usecase == 'Prediction':
    dataset_name = st.sidebar.selectbox('📊 Select Dataset',
                                ('eCell_Accessibility', 'eCell_Retainability'))
    model_name = st.sidebar.selectbox('🤖 Select Model',
                            ('Linear_Regression', 'Decision_Tree_Regression', 'Gradient_Boosting_Regression', 'AdaBoost_Regression', 'Support_Vector_Machine',
                            'Regression_Using_Neural_Networks', 'SGD_Neural_Network'))    
elif selected_usecase == 'Anomaly_Detection':
    dataset_name = st.sidebar.selectbox('📊 Select Dataset',
                                ('IP_Link_Jitter', 'IP_Router_Port_Total_Traffic_Rate'))
    model_name = st.sidebar.selectbox('🤖 Select Model',
                            ('Isolation_Forest', 'Autoencoder', 'Local_Outlier_Factor', 'One_Class_SVM', 'DBSCAN'))
    compare = st.sidebar.checkbox('📈 Compare with other models')

st.sidebar.markdown("---")
process = st.sidebar.button('🚀 Analyze')

# Handle home button - redirect to landing page
if home_button:
    st.experimental_rerun()

# Main content area
analysis_placeholder = st.empty()

def render_landing_page():
    """Render the beautiful landing page with app information"""
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📡 Network Analytics Platform</div>
        <div class="hero-subtitle">Explore, Analyze & Predict Network Performance with Machine Learning</div>
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-number">3</div>
                <div class="stat-label">Use Cases</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">15+</div>
                <div class="stat-label">ML Models</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">6</div>
                <div class="stat-label">Datasets</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Cases Section
    st.markdown("### 🎯 Available Use Cases")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📈 Time Series Forecasting</div>
            <div class="feature-desc">
                Predict future network traffic patterns and resource utilization using advanced forecasting models. Essential for capacity planning, resource allocation, and proactive network management.
            </div>
            <div style="margin-top: 1rem;">
                <strong style="color: #a2d2ff;">Datasets:</strong><br>
                <span class="dataset-tag">MSC Traffic</span>
                <span class="dataset-tag">APN Utilization</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Prediction</div>
            <div class="feature-desc">
                Predict eCell accessibility and retainability metrics to ensure optimal network coverage and quality of service. Critical for maintaining high customer satisfaction and SLA compliance.
            </div>
            <div style="margin-top: 1rem;">
                <strong style="color: #a2d2ff;">Datasets:</strong><br>
                <span class="dataset-tag">eCell Accessibility</span>
                <span class="dataset-tag">eCell Retainability</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔍 Anomaly Detection</div>
            <div class="feature-desc">
                Identify unusual patterns and potential issues in network performance before they impact users. Enable proactive maintenance and reduce mean time to resolution (MTTR).
            </div>
            <div style="margin-top: 1rem;">
                <strong style="color: #a2d2ff;">Datasets:</strong><br>
                <span class="dataset-tag">IP Link Jitter</span>
                <span class="dataset-tag">Traffic Rate</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Dataset Details Section
    st.markdown("### 📊 Datasets In Detail")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📶 MSC Traffic Data", expanded=False):
            st.markdown("""
            **Mobile Switching Center (MSC) Traffic Dataset**
            
            | Attribute | Value |
            |-----------|-------|
            | **Granularity** | Daily measurements |
            | **Duration** | 1 year of historical data |
            | **Nodes** | 10 network nodes |
            | **Features** | Call volume, data throughput, signaling load |
            
            **Use Cases:**
            - 📈 Long-term traffic trend analysis
            - 🎯 Capacity planning and forecasting
            - 📊 Seasonal pattern identification
            - ⚡ Peak load prediction
            
            **Ideal Models:** HWES, SARIMA, Prophet, XGBoost, LSTM
            """)
        
        with st.expander("📡 APN Utilization Data", expanded=False):
            st.markdown("""
            **Access Point Name (APN) Utilization Dataset**
            
            | Attribute | Value |
            |-----------|-------|
            | **Granularity** | Hourly measurements |
            | **Duration** | 3 months of data |
            | **Nodes** | 50 access points |
            | **Features** | Bandwidth usage, active sessions, latency |
            
            **Use Cases:**
            - 📈 Hourly resource optimization
            - 🔄 Load balancing decisions
            - 📊 Usage pattern analysis
            - 🎯 QoS prediction
            
            **Ideal Models:** LSTM-GRU, N-BEATS, CNN-WaveNet
            """)
        
        with st.expander("📱 eCell Accessibility Data", expanded=False):
            st.markdown("""
            **Enhanced Cell Accessibility Dataset**
            
            | Attribute | Value |
            |-----------|-------|
            | **Metrics** | RRC success rate, RACH success rate |
            | **Features** | Signal strength, interference levels, load |
            | **Coverage** | Multiple cell towers and sectors |
            
            **Use Cases:**
            - 📱 Coverage optimization
            - 🎯 Connection success prediction
            - 📊 Network planning
            - ⚠️ Early warning for accessibility issues
            
            **Ideal Models:** Linear/Decision Tree/Gradient Boosting Regression
            """)
    
    with col2:
        with st.expander("📞 eCell Retainability Data", expanded=False):
            st.markdown("""
            **Enhanced Cell Retainability Dataset**
            
            | Attribute | Value |
            |-----------|-------|
            | **Metrics** | Call drop rate, handover success rate |
            | **Features** | Session duration, mobility patterns |
            | **Coverage** | Multiple cell towers and handover zones |
            
            **Use Cases:**
            - 📞 Call drop prediction
            - 🔄 Handover optimization
            - 📊 Quality of experience improvement
            - 🎯 Customer churn prevention
            
            **Ideal Models:** SVM, Neural Networks, AdaBoost Regression
            """)
        
        with st.expander("⏱️ IP Link Jitter Data", expanded=False):
            st.markdown("""
            **IP Link Jitter Measurement Dataset**
            
            | Attribute | Value |
            |-----------|-------|
            | **Granularity** | 15-minute intervals |
            | **Duration** | 15 days of continuous monitoring |
            | **Nodes** | 30 IP links |
            | **Metrics** | Jitter (ms), packet delay variation |
            
            **Use Cases:**
            - ⏱️ Latency anomaly detection
            - 🎮 VoIP/Video quality assurance
            - 📊 SLA compliance monitoring
            - ⚠️ Proactive issue identification
            
            **Ideal Models:** Isolation Forest, Autoencoder, DBSCAN
            """)
        
        with st.expander("🌐 Total Traffic Rate Data", expanded=False):
            st.markdown("""
            **Router Port Traffic Throughput Dataset**
            
            | Attribute | Value |
            |-----------|-------|
            | **Granularity** | 5-minute intervals |
            | **Duration** | 7 days of measurements |
            | **Nodes** | 20 router ports |
            | **Metrics** | Throughput (Mbps), packet count, error rate |
            
            **Use Cases:**
            - 🌐 Traffic spike detection
            - 🔍 DDoS attack identification
            - 📊 Bandwidth anomaly detection
            - ⚡ Congestion prediction
            
            **Ideal Models:** One-Class SVM, LOF, Isolation Forest
            """)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Models Section
    st.markdown("### 🤖 Machine Learning Models")
    
    with st.expander("📈 Forecasting Models", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">HWES (Holt-Winters Exponential Smoothing)</div>
                <div class="model-desc">Triple exponential smoothing that captures level, trend, and seasonality. Best for data with clear seasonal patterns.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">SARIMA (Seasonal ARIMA)</div>
                <div class="model-desc">Extends ARIMA with seasonal components. Excellent for stationary time series with seasonal cycles.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">XGBoost</div>
                <div class="model-desc">Gradient boosting algorithm that handles non-linear patterns. Robust to outliers and missing data.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Prophet</div>
                <div class="model-desc">Facebook's forecasting tool. Handles holidays, missing data, and trend changes automatically.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">LSTM (Long Short-Term Memory)</div>
                <div class="model-desc">Deep learning model that captures long-term dependencies. Ideal for complex sequential patterns.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">LSTM-GRU Hybrid</div>
                <div class="model-desc">Combines LSTM with Gated Recurrent Units for faster training while maintaining accuracy.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">N-BEATS</div>
                <div class="model-desc">Neural Basis Expansion Analysis. State-of-the-art deep learning for univariate time series.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">CNN-WaveNet</div>
                <div class="model-desc">Dilated causal convolutions for capturing multi-scale temporal patterns efficiently.</div>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("🎯 Prediction Models", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Linear Regression</div>
                <div class="model-desc">Simple yet powerful baseline. Assumes linear relationship between features and target. Fast and interpretable.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Decision Tree Regression</div>
                <div class="model-desc">Non-parametric model using tree structure. Captures non-linear patterns and feature interactions.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Gradient Boosting Regression</div>
                <div class="model-desc">Ensemble method that builds trees sequentially. High accuracy with proper tuning.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">AdaBoost Regression</div>
                <div class="model-desc">Adaptive boosting that focuses on hard examples. Reduces bias and variance effectively.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Support Vector Machine (SVM)</div>
                <div class="model-desc">Kernel-based method for regression. Works well with high-dimensional data and clear margins.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Neural Network Regression</div>
                <div class="model-desc">Multi-layer perceptron for complex patterns. Universal function approximator with sufficient depth.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">SGD Neural Network</div>
                <div class="model-desc">Stochastic Gradient Descent optimized network. Efficient for large datasets with online learning capability.</div>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("🔍 Anomaly Detection Models", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Isolation Forest</div>
                <div class="model-desc">Tree-based anomaly detection. Isolates anomalies instead of profiling normal data. Fast and scalable.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Autoencoder</div>
                <div class="model-desc">Neural network that learns to reconstruct normal data. Anomalies have high reconstruction error.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">Local Outlier Factor (LOF)</div>
                <div class="model-desc">Density-based method comparing local density to neighbors. Detects local anomalies effectively.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">One-Class SVM</div>
                <div class="model-desc">SVM trained on normal data only. Creates decision boundary around normal samples.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="model-detail-card">
                <div class="model-name">DBSCAN</div>
                <div class="model-desc">Density-Based Spatial Clustering. Points not belonging to any cluster are anomalies. No predefined cluster count needed.</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("""
    <div class="getting-started">
        <h3 style="color: #e94560; margin-bottom: 1.5rem;">🚀 Getting Started</h3>
        <div style="display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; text-align: left;">
            <div>
                <span class="step-number">1</span>
                <span style="color: #ccd6f6;">Open the <strong>Sidebar</strong> (arrow on top-left)</span>
            </div>
            <div>
                <span class="step-number">2</span>
                <span style="color: #ccd6f6;">Select a <strong>Use Case</strong></span>
            </div>
            <div>
                <span class="step-number">3</span>
                <span style="color: #ccd6f6;">Choose <strong>Dataset</strong> & <strong>Model</strong></span>
            </div>
            <div>
                <span class="step-number">4</span>
                <span style="color: #ccd6f6;">Click <strong>🚀 Analyze</strong></span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.info("👈 **Click the arrow on the top-left** to open the sidebar and start your analysis!")

# Creating main skeleton of each page
if home_button:
    render_landing_page()
elif selected_usecase == 'Time_Series_Forecasting':
    if process:
        with st.container():
            st.header(f'📈 {selected_usecase.replace("_", " ")} Analysis')
        if dataset_name == 'MSC_Traffic':
            msc_traffic_process_render(dataset_name, model_name)
        elif dataset_name == 'APN_Utilization':
            apn_utilisation_process_render(dataset_name, model_name)
    else:
        render_landing_page()

elif selected_usecase == 'Prediction':
    if process:
        with st.container():
            st.header(f'🎯 {selected_usecase} Analysis')
        if dataset_name == 'eCell_Accessibility':
            eCell_Accessibility_process_render(dataset_name, model_name)
        elif dataset_name == 'eCell_Retainability':
            eCell_Retainability_process_render(dataset_name, model_name)
    else:
        render_landing_page()

elif selected_usecase == 'Anomaly_Detection':
    if process:
        with st.container():
            st.header(f'🔍 {selected_usecase.replace("_", " ")} Analysis')
        if dataset_name == 'IP_Link_Jitter':
            jitter_15min_15days_30nodes_process_render(dataset_name, model_name, compare)
        elif dataset_name == 'IP_Router_Port_Total_Traffic_Rate':
            total_traffic_rate_5min_7days_20nodes(dataset_name, model_name, compare)
    else:
        render_landing_page()

else:
    render_landing_page()
