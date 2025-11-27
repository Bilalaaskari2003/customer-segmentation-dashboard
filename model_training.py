# ============================================================================
# FILE 2: model_training.py
# ============================================================================

"""
Model Training Module - K-Means Clustering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import pickle
import warnings
warnings.filterwarnings('ignore')


class KMeansTrainer:
    """Handle K-Means clustering model training and evaluation"""
    
    def __init__(self, X_scaled, df, random_state=42):
        self.X_scaled = X_scaled
        self.df = df
        self.random_state = random_state
        self.kmeans = None
        self.optimal_k = None
        self.inertias = []
        self.silhouette_scores = []
        
    def find_optimal_clusters(self, k_range=range(2, 11)):
        """Find optimal number of clusters"""
        print("\n[TRAINING] Finding Optimal Clusters...")
        print("="*70)
        self.inertias = []
        self.silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', 
                          random_state=self.random_state, n_init=10)
            kmeans.fit(self.X_scaled)
            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.X_scaled, kmeans.labels_)
            self.inertias.append(inertia)
            self.silhouette_scores.append(silhouette)
            print(f"K={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.3f}")
        best_k_idx = self.silhouette_scores.index(max(self.silhouette_scores))
        suggested_k = list(k_range)[best_k_idx]
        print(f"\n✓ Suggested K: {suggested_k}")
        print("="*70)
        return suggested_k
    
    def plot_optimal_k(self, k_range=range(2, 11), save_path='optimal_clusters.png'):
        """Plot Elbow and Silhouette analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(list(k_range), self.inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax2.plot(list(k_range), self.silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
        plt.close()
    
    def train_model(self, n_clusters=4):
        """Train final K-Means model"""
        print(f"\n[TRAINING] Training K-Means with K={n_clusters}...")
        print("="*70)
        self.optimal_k = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                           random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(self.X_scaled)
        self.df['Cluster'] = clusters
        inertia = self.kmeans.inertia_
        silhouette = silhouette_score(self.X_scaled, clusters)
        print(f"✓ Training completed!")
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"\n{'Cluster':<10} {'Count':<10} {'Percentage':<10}")
        print("-" * 30)
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"{cluster_id:<10} {count:<10} {percentage:.1f}%")
        print("="*70)
        return clusters
    
    def analyze_clusters(self, feature_names):
        """Analyze cluster characteristics"""
        print("\n[ANALYSIS] Cluster Characteristics...")
        print("="*70)
        cluster_features = ['Age', 'Income', 'Total_Spending', 'Total_Purchases',
                          'Recency', 'Total_Children', 'NumWebVisitsMonth']
        available_features = [f for f in cluster_features if f in self.df.columns]
        cluster_summary = self.df.groupby('Cluster')[available_features].mean()
        print("\nAverage Values by Cluster:")
        print(cluster_summary.round(2))
        print("\n" + "="*70)
        print("DETAILED CLUSTER INTERPRETATION")
        print("="*70)
        for i in range(self.optimal_k):
            cluster_data = self.df[self.df['Cluster'] == i]
            print(f"\n🔹 CLUSTER {i} (n={len(cluster_data)}, {len(cluster_data)/len(self.df)*100:.1f}%)")
            print("-" * 70)
            if 'Age' in cluster_data.columns:
                print(f"   Age: {cluster_data['Age'].mean():.1f} years")
            if 'Income' in cluster_data.columns:
                print(f"   Income: ${cluster_data['Income'].mean():.0f}")
            if 'Total_Spending' in cluster_data.columns:
                print(f"   Total Spending: ${cluster_data['Total_Spending'].mean():.0f}")
            avg_income = cluster_data['Income'].mean() if 'Income' in cluster_data.columns else 0
            avg_spending = cluster_data['Total_Spending'].mean() if 'Total_Spending' in cluster_data.columns else 0
            if avg_income > 60000 and avg_spending > 1000:
                label = "💎 HIGH-VALUE CUSTOMERS"
                strategy = "Premium products, VIP treatment"
            elif avg_spending < 200:
                label = "💰 BUDGET SHOPPERS"
                strategy = "Discounts, value bundles"
            else:
                label = "👥 STANDARD CUSTOMERS"
                strategy = "General marketing"
            print(f"\n   📊 SEGMENT: {label}")
            print(f"   📈 STRATEGY: {strategy}")
        return cluster_summary
    
    def save_model(self, model_path='kmeans_model.pkl'):
        """Save trained model"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.kmeans, f)
        print(f"\n✓ Model saved: {model_path}")
    
    def save_clustered_data(self, output_path='customer_segments_final.csv'):
        """Save dataframe with cluster labels"""
        self.df.to_csv(output_path, index=False)
        print(f"✓ Clustered data saved: {output_path}")


