# Customer Segmentation Using K-Means Clustering

## 📋 Project Overview

A comprehensive machine learning project that segments customers using K-Means clustering algorithm. The project includes data preprocessing, model training, visualization, and an interactive Streamlit dashboard for business insights.

---

## 🏗️ Project Structure

```
customer-segmentation/
│
├── preprocessing.py          # Data preprocessing module
├── model_training.py         # K-Means training module
├── visualization.py          # Visualization module
├── main.py                   # Main execution script
├── streamlit_app.py         # Interactive dashboard
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
│
├── marketing_campaign.csv   # Input dataset (required)
│
└── outputs/                # Generated files
    ├── processed_data.csv
    ├── customer_segments_final.csv
    ├── optimal_clusters.png
    ├── cluster_analysis.png
    ├── 3d_clusters.png
    └── kmeans_model.pkl
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure `marketing_campaign.csv` is in the project directory.

### 3. Run Complete Pipeline

```bash
python main.py
```

This will:
- Preprocess the data
- Find optimal number of clusters
- Train K-Means model
- Generate visualizations
- Save results

### 4. Launch Interactive Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📦 Module Descriptions

### `preprocessing.py`
**Purpose:** Handle all data preprocessing tasks

**Features:**
- Load dataset from CSV
- Handle missing values
- Feature engineering (Age, Total_Spending, Total_Purchases, etc.)
- Encode categorical variables
- Feature scaling using StandardScaler

**Usage:**
```python
from preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('marketing_campaign.csv')
df_processed, X_scaled = preprocessor.preprocess_pipeline()
```

---

### `model_training.py`
**Purpose:** Train and evaluate K-Means clustering model

**Features:**
- Find optimal K using Elbow and Silhouette methods
- Train K-Means model
- Analyze cluster characteristics
- Assign business labels to clusters
- Save trained model

**Usage:**
```python
from model_training import KMeansTrainer

trainer = KMeansTrainer(X_scaled, df_processed)
optimal_k = trainer.find_optimal_clusters(range(2, 11))
clusters = trainer.train_model(n_clusters=4)
```

---

### `visualization.py`
**Purpose:** Create static visualizations for analysis

**Features:**
- Comprehensive cluster analysis plots
- 3D scatter plots
- Silhouette analysis
- Correlation heatmaps

**Usage:**
```python
from visualization import ClusterVisualizer

visualizer = ClusterVisualizer(df, X_scaled, clusters)
visualizer.create_comprehensive_analysis('cluster_analysis.png')
visualizer.create_3d_plot('3d_clusters.png')
```

---

### `main.py`
**Purpose:** Main execution script that runs the complete pipeline

**Workflow:**
1. Data Preprocessing
2. Model Training
3. Visualization Generation
4. Results Saving

**Usage:**
```bash
python main.py
```

---

### `streamlit_app.py`
**Purpose:** Interactive dashboard for exploring results

**Features:**
- 🏠 **Overview:** Dataset summary and cluster distribution
- 📈 **Cluster Analysis:** Compare clusters statistically
- 🔍 **Individual Clusters:** Deep dive into each segment
- 📊 **Visualizations:** Interactive 3D and 2D plots
- 💡 **Business Insights:** Strategic recommendations

**Usage:**
```bash
streamlit run streamlit_app.py
```

---

## 📊 Features Used for Clustering

The model uses 8 key features:

1. **Age** - Customer age (derived from Year_Birth)
2. **Income** - Annual household income
3. **Total_Spending** - Sum of all product spending
4. **Total_Purchases** - Total number of purchases
5. **Total_Children** - Number of children at home
6. **Recency** - Days since last purchase
7. **NumWebVisitsMonth** - Monthly website visits
8. **Education_Encoded** - Encoded education level

---

## 📈 Output Files

### Data Files
- `processed_data.csv` - Cleaned and engineered features
- `customer_segments_final.csv` - Dataset with cluster labels
- `kmeans_model.pkl` - Trained K-Means model

### Visualizations
- `optimal_clusters.png` - Elbow and Silhouette analysis
- `cluster_analysis.png` - 8-panel comprehensive analysis
- `3d_clusters.png` - 3D visualization of clusters

---

## 💡 Business Insights

The model identifies distinct customer segments:

### 💎 High-Value Customers
- High income (>$60k) and spending (>$1000)
- **Strategy:** Premium products, VIP treatment, exclusive offers

### 💰 Budget Shoppers
- Low spending (<$200)
- **Strategy:** Discounts, value bundles, promotions

### 🛍️ Frequent Buyers
- High purchase frequency (>15 purchases)
- **Strategy:** Loyalty programs, subscription services

### 🎯 Potential Customers
- High income but low spending
- **Strategy:** Engagement campaigns, personalized recommendations

### 👥 Standard Customers
- Average metrics across the board
- **Strategy:** General marketing, seasonal campaigns

---

## 🎨 Streamlit Dashboard Pages

### 1. Overview
- Dataset metrics
- Cluster distribution charts
- Data preview and download

### 2. Cluster Analysis
- Feature comparison across clusters
- Statistical summaries
- Distribution box plots

### 3. Individual Clusters
- Detailed metrics per cluster
- Business segment classification
- Cluster-specific visualizations

### 4. Visualizations
- Interactive 3D scatter plot
- Custom 2D scatter plots
- Correlation heatmap

### 5. Business Insights
- Strategic recommendations
- Cluster-specific strategies
- Implementation roadmap

---

## 🔧 Customization

### Change Number of Clusters

In `main.py`, modify:
```python
optimal_k = 4  # Change this value
```

### Add/Remove Features

In `preprocessing.py`, modify:
```python
self.features_for_clustering = [
    'Age',
    'Income',
    # Add or remove features here
]
```

### Adjust Visualizations

In `visualization.py`, customize:
- Plot colors
- Figure sizes
- Chart types

---

## 📋 Requirements

- Python 3.8+
- pandas 2.0.3
- numpy 1.24.3
- matplotlib 3.7.2
- seaborn 0.12.2
- scikit-learn 1.3.0
- streamlit 1.28.0
- plotly 5.17.0

---

## 🐛 Troubleshooting

### Issue: "Could not find marketing_campaign.csv"
**Solution:** Ensure the CSV file is in the same directory as the scripts.

### Issue: Module import errors
**Solution:** Install all requirements:
```bash
pip install -r requirements.txt
```

### Issue: Streamlit not opening
**Solution:** Check if port 8501 is available, or specify a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## 📝 Project Workflow

1. **Data Loading** → Load CSV data
2. **Preprocessing** → Clean, engineer features, scale
3. **Optimal K** → Find best number of clusters
4. **Training** → Train K-Means model
5. **Analysis** → Analyze cluster characteristics
6. **Visualization** → Create plots and charts
7. **Dashboard** → Interactive exploration

---

## 🎓 Academic Usage

This project is designed for:
- Machine Learning lab projects
- Data Science coursework
- Customer analytics case studies
- K-Means clustering demonstrations

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review module documentation
3. Examine error messages in console output

---

## 📄 License

This project is for educational purposes. Feel free to modify and adapt for your needs.

---

## 🙏 Acknowledgments

- Dataset: Marketing Campaign Customer Data
- Algorithm: K-Means Clustering (scikit-learn)
- Visualization: Matplotlib, Seaborn, Plotly
- Dashboard: Streamlit

---

**Happy Clustering! 📊🎯**
