"""
============================================================================
INTERACTIVE CUSTOMER SEGMENTATION DASHBOARD
Streamlit Application
============================================================================
Run with: streamlit run streamlit_app.py
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load the clustered data"""
    try:
        df = pd.read_csv('customer_segments_final.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Could not find 'customer_segments_final.csv'. Run main.py first!")
        st.stop()
        return None

def create_3d_scatter(df):
    """Create interactive 3D scatter plot"""
    fig = px.scatter_3d(
        df, 
        x='Age', 
        y='Income', 
        z='Total_Spending',
        color='Cluster',
        title='3D Customer Segmentation (Age, Income, Spending)',
        labels={'Age': 'Age (years)', 'Income': 'Income ($)', 'Total_Spending': 'Total Spending ($)'},
        color_continuous_scale='viridis',
        height=600
    )
    fig.update_traces(marker=dict(size=5))
    return fig

def create_2d_scatter(df, x_col, y_col):
    """Create interactive 2D scatter plot"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color='Cluster',
        title=f'{y_col} vs {x_col} by Cluster',
        labels={x_col: x_col, y_col: y_col},
        color_continuous_scale='viridis',
        height=500
    )
    return fig

def create_cluster_comparison(df):
    """Create cluster comparison charts"""
    features = ['Age', 'Income', 'Total_Spending', 'Total_Purchases', 'Recency']
    cluster_means = df.groupby('Cluster')[features].mean().reset_index()
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=features + [''],
        specs=[[{'type': 'bar'}]*3, [{'type': 'bar'}]*3]
    )
    
    colors = px.colors.qualitative.Plotly
    
    for idx, feature in enumerate(features):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        fig.add_trace(
            go.Bar(
                x=cluster_means['Cluster'],
                y=cluster_means[feature],
                name=feature,
                marker_color=colors[idx % len(colors)],
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Cluster", row=row, col=col)
        fig.update_yaxes(title_text=f"Avg {feature}", row=row, col=col)
    
    fig.update_layout(height=600, title_text="Average Feature Values by Cluster")
    return fig

def display_cluster_insights(df, cluster_id):
    """Display insights for a specific cluster"""
    cluster_data = df[df['Cluster'] == cluster_id]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Size", f"{len(cluster_data):,}")
        st.metric("📊 Percentage", f"{len(cluster_data)/len(df)*100:.1f}%")
    
    with col2:
        st.metric("🎂 Avg Age", f"{cluster_data['Age'].mean():.1f} years")
        st.metric("💰 Avg Income", f"${cluster_data['Income'].mean():,.0f}")
    
    with col3:
        st.metric("🛍️ Avg Spending", f"${cluster_data['Total_Spending'].mean():,.0f}")
        st.metric("📦 Avg Purchases", f"{cluster_data['Total_Purchases'].mean():.1f}")
    
    with col4:
        st.metric("📅 Avg Recency", f"{cluster_data['Recency'].mean():.0f} days")
        st.metric("👶 Avg Children", f"{cluster_data['Total_Children'].mean():.1f}")
    
    # Business segment classification
    avg_income = cluster_data['Income'].mean()
    avg_spending = cluster_data['Total_Spending'].mean()
    avg_purchases = cluster_data['Total_Purchases'].mean()
    
    st.markdown("---")
    
    if avg_income > 60000 and avg_spending > 1000:
        st.success("💎 **HIGH-VALUE CUSTOMERS**")
        st.info("**Strategy:** Premium products, VIP treatment, exclusive offers")
    elif avg_spending < 200:
        st.warning("💰 **BUDGET SHOPPERS**")
        st.info("**Strategy:** Discounts, value bundles, promotions")
    elif avg_purchases > 15:
        st.success("🛍️ **FREQUENT BUYERS**")
        st.info("**Strategy:** Loyalty programs, subscription services")
    elif avg_income > 50000 and avg_spending < 500:
        st.info("🎯 **POTENTIAL CUSTOMERS**")
        st.info("**Strategy:** Engagement campaigns, personalized recommendations")
    else:
        st.info("👥 **STANDARD CUSTOMERS**")
        st.info("**Strategy:** General marketing, seasonal campaigns")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">📊 Customer Segmentation Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("### K-Means Clustering Analysis & Business Insights")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigate to:",
        ["🏠 Overview", "📈 Cluster Analysis", "🔍 Individual Clusters", 
         "📊 Visualizations", "💡 Business Insights"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Total Customers:** {len(df):,}\n\n**Clusters:** {df['Cluster'].nunique()}")
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "🏠 Overview":
        st.markdown('<div class="sub-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Customers", f"{len(df):,}")
        with col2:
            st.metric("🎯 Clusters", df['Cluster'].nunique())
        with col3:
            st.metric("💵 Avg Income", f"${df['Income'].mean():,.0f}")
        with col4:
            st.metric("🛒 Avg Spending", f"${df['Total_Spending'].mean():,.0f}")
        
        st.markdown("---")
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cluster Size Distribution")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Customers'},
                color=cluster_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Cluster Distribution (Pie)")
            fig = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index,
                title='Customer Distribution',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Data preview
        st.markdown("#### 📋 Data Preview")
        st.dataframe(df.head(100), height=300)
        
        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset",
            data=csv,
            file_name='customer_segments.csv',
            mime='text/csv',
        )
    
    # ========================================================================
    # PAGE 2: CLUSTER ANALYSIS
    # ========================================================================
    elif page == "📈 Cluster Analysis":
        st.markdown('<div class="sub-header">📈 Cluster Comparison</div>', unsafe_allow_html=True)
        
        # Cluster comparison
        st.plotly_chart(create_cluster_comparison(df), use_container_width=True)
        
        st.markdown("---")
        
        # Statistical summary
        st.markdown("#### 📊 Statistical Summary by Cluster")
        features = ['Age', 'Income', 'Total_Spending', 'Total_Purchases', 'Recency', 'Total_Children']
        summary = df.groupby('Cluster')[features].mean().round(2)
        st.dataframe(summary, use_container_width=True)
        
        # Box plots
        st.markdown("---")
        st.markdown("#### 📦 Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", features, key='box1')
            fig = px.box(df, x='Cluster', y=feature, color='Cluster',
                        title=f'{feature} Distribution by Cluster')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            feature2 = st.selectbox("Select Feature", features, index=2, key='box2')
            fig = px.box(df, x='Cluster', y=feature2, color='Cluster',
                        title=f'{feature2} Distribution by Cluster')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 3: INDIVIDUAL CLUSTERS
    # ========================================================================
    elif page == "🔍 Individual Clusters":
        st.markdown('<div class="sub-header">🔍 Detailed Cluster Analysis</div>', unsafe_allow_html=True)
        
        cluster_id = st.selectbox(
            "Select Cluster:",
            sorted(df['Cluster'].unique()),
            format_func=lambda x: f"Cluster {x}"
        )
        
        st.markdown(f"### Cluster {cluster_id} - Detailed Insights")
        
        display_cluster_insights(df, cluster_id)
        
        st.markdown("---")
        
        cluster_data = df[df['Cluster'] == cluster_id]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Age Distribution")
            fig = px.histogram(cluster_data, x='Age', nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Income Distribution")
            fig = px.histogram(cluster_data, x='Income', nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(cluster_data, x='Income', y='Total_Spending',
                           title='Income vs Spending')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(cluster_data, x='Age', y='Total_Spending',
                           title='Age vs Spending')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 4: VISUALIZATIONS
    # ========================================================================
    elif page == "📊 Visualizations":
        st.markdown('<div class="sub-header">📊 Interactive Visualizations</div>', unsafe_allow_html=True)
        
        # 3D Plot
        st.markdown("#### 🎨 3D Cluster Visualization")
        st.plotly_chart(create_3d_scatter(df), use_container_width=True)
        
        st.markdown("---")
        
        # Custom 2D plots
        st.markdown("#### 📉 Custom Scatter Plots")
        
        col1, col2 = st.columns(2)
        
        numeric_cols = ['Age', 'Income', 'Total_Spending', 'Total_Purchases', 
                       'Recency', 'NumWebVisitsMonth', 'Total_Children']
        
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols, index=1)
        
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, index=2)
        
        fig = create_2d_scatter(df, x_axis, y_axis)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Correlation heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        corr_features = ['Age', 'Income', 'Total_Spending', 'Total_Purchases', 'Recency']
        correlation = df[corr_features].corr()
        
        fig = px.imshow(correlation, 
                       text_auto=True,
                       aspect='auto',
                       color_continuous_scale='RdBu',
                       title='Feature Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 5: BUSINESS INSIGHTS
    # ========================================================================
    elif page == "💡 Business Insights":
        st.markdown('<div class="sub-header">💡 Strategic Recommendations</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("### 🎯 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**✅ Segmentation Success**")
            st.write(f"Successfully segmented {len(df):,} customers into {df['Cluster'].nunique()} groups")
            
            st.info("**📊 Data Quality**")
            st.write("Comprehensive features provide rich insights")
        
        with col2:
            st.warning("**🎯 Actionable Insights**")
            st.write("Each cluster shows unique patterns")
            
            st.success("**💰 Revenue Opportunities**")
            st.write("High-value segments identified")
        
        st.markdown("---")
        
        # Cluster strategies
        st.markdown("### 📈 Cluster-Specific Strategies")
        
        for cluster_id in sorted(df['Cluster'].unique()):
            with st.expander(f"🔹 Cluster {cluster_id} Strategy", expanded=True):
                cluster_data = df[df['Cluster'] == cluster_id]
                
                avg_income = cluster_data['Income'].mean()
                avg_spending = cluster_data['Total_Spending'].mean()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Size", f"{len(cluster_data):,}")
                    st.metric("Avg Income", f"${avg_income:,.0f}")
                    st.metric("Avg Spending", f"${avg_spending:,.0f}")
                
                with col2:
                    if avg_income > 60000 and avg_spending > 1000:
                        st.markdown("**💎 HIGH-VALUE CUSTOMERS**")
                        st.markdown("""
                        - Premium product lines
                        - VIP loyalty program
                        - Dedicated account management
                        - Early product access
                        """)
                    elif avg_spending < 200:
                        st.markdown("**💰 BUDGET SHOPPERS**")
                        st.markdown("""
                        - Value bundles and discounts
                        - Budget-friendly products
                        - Promotional campaigns
                        - Volume discounts
                        """)
                    else:
                        st.markdown("**👥 STANDARD CUSTOMERS**")
                        st.markdown("""
                        - Seasonal campaigns
                        - Email marketing
                        - Social media engagement
                        - Customer surveys
                        """)
        
        st.markdown("---")
        
        # Implementation roadmap
        st.markdown("### 🗺️ Implementation Roadmap")
        
        st.markdown("""
        #### Phase 1: Immediate Actions (Week 1-2)
        - ✅ Segment customer database
        - ✅ Create segment-specific templates
        - ✅ Brief teams on characteristics
        - ✅ Set up tracking
        
        #### Phase 2: Campaign Launch (Week 3-4)
        - 📧 Launch personalized campaigns
        - 🎯 Create targeted ads
        - 💰 Implement pricing strategies
        - 📱 Customize user experience
        
        #### Phase 3: Optimization (Month 2-3)
        - 📊 Monitor performance
        - 🔄 A/B test strategies
        - 💡 Refine segments
        - 📈 Scale initiatives
        
        #### Phase 4: Long-term (Month 4+)
        - 🔮 Predictive modeling
        - 🎓 Lifetime value analysis
        - 🤝 CRM integration
        - 📊 Regular updates
        """)

# Run the app
if __name__ == "__main__":
    main()