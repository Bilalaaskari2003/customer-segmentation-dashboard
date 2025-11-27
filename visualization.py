# ============================================================================
# FILE 3: visualization.py
# ============================================================================

"""
Visualization Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ClusterVisualizer:
    """Create visualizations for cluster analysis"""
    
    def __init__(self, df, X_scaled, clusters):
        self.df = df.copy()
        self.df['Cluster'] = clusters
        self.X_scaled = X_scaled
        self.clusters = clusters
        self.n_clusters = len(np.unique(clusters))
        self.colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
        
    def create_comprehensive_analysis(self, save_path='cluster_analysis.png'):
        """Create comprehensive cluster analysis"""
        print("\n[VISUALIZATION] Creating comprehensive analysis...")
        fig = plt.figure(figsize=(20, 12))
        
        # Income vs Spending
        ax1 = fig.add_subplot(2, 4, 1)
        for i in range(self.n_clusters):
            cluster_data = self.df[self.df['Cluster'] == i]
            ax1.scatter(cluster_data['Income'], cluster_data['Total_Spending'],
                       label=f'Cluster {i}', s=50, alpha=0.6)
        ax1.set_xlabel('Income ($)')
        ax1.set_ylabel('Total Spending ($)')
        ax1.set_title('Income vs Spending', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Age vs Spending
        ax2 = fig.add_subplot(2, 4, 2)
        for i in range(self.n_clusters):
            cluster_data = self.df[self.df['Cluster'] == i]
            ax2.scatter(cluster_data['Age'], cluster_data['Total_Spending'],
                       label=f'Cluster {i}', s=50, alpha=0.6)
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Total Spending ($)')
        ax2.set_title('Age vs Spending', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Purchases vs Recency
        ax3 = fig.add_subplot(2, 4, 3)
        for i in range(self.n_clusters):
            cluster_data = self.df[self.df['Cluster'] == i]
            ax3.scatter(cluster_data['Total_Purchases'], cluster_data['Recency'],
                       label=f'Cluster {i}', s=50, alpha=0.6)
        ax3.set_xlabel('Total Purchases')
        ax3.set_ylabel('Recency (days)')
        ax3.set_title('Purchases vs Recency', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cluster Size
        ax4 = fig.add_subplot(2, 4, 4)
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        bars = ax4.bar(cluster_counts.index, cluster_counts.values, 
                      color=self.colors, edgecolor='black')
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Count')
        ax4.set_title('Cluster Size', fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Average Spending
        ax5 = fig.add_subplot(2, 4, 5)
        avg_spending = self.df.groupby('Cluster')['Total_Spending'].mean()
        bars = ax5.bar(avg_spending.index, avg_spending.values, 
                      color=self.colors, edgecolor='black')
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Avg Spending ($)')
        ax5.set_title('Average Spending', fontweight='bold')
        
        # Average Income
        ax6 = fig.add_subplot(2, 4, 6)
        avg_income = self.df.groupby('Cluster')['Income'].mean()
        bars = ax6.bar(avg_income.index, avg_income.values, 
                      color=self.colors, edgecolor='black')
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Avg Income ($)')
        ax6.set_title('Average Income', fontweight='bold')
        
        # Age Distribution
        ax7 = fig.add_subplot(2, 4, 7)
        for i in range(self.n_clusters):
            cluster_data = self.df[self.df['Cluster'] == i]['Age']
            ax7.hist(cluster_data, bins=20, alpha=0.5, label=f'Cluster {i}')
        ax7.set_xlabel('Age')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Age Distribution', fontweight='bold')
        ax7.legend()
        
        # Silhouette Analysis
        ax8 = fig.add_subplot(2, 4, 8)
        silhouette_vals = silhouette_samples(self.X_scaled, self.clusters)
        y_lower = 10
        for i in range(self.n_clusters):
            cluster_silhouette_vals = silhouette_vals[self.clusters == i]
            cluster_silhouette_vals.sort()
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            ax8.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                             facecolor=self.colors[i], alpha=0.7)
            ax8.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax8.set_xlabel('Silhouette Coefficient')
        ax8.set_ylabel('Cluster')
        ax8.set_title('Silhouette Analysis', fontweight='bold')
        avg_score = silhouette_score(self.X_scaled, self.clusters)
        ax8.axvline(x=avg_score, color='red', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Analysis saved: {save_path}")
        plt.close()
    
    def create_3d_plot(self, save_path='3d_clusters.png'):
        """Create 3D scatter plot"""
        print("\n[VISUALIZATION] Creating 3D visualization...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.n_clusters):
            cluster_data = self.df[self.df['Cluster'] == i]
            ax.scatter(cluster_data['Age'], cluster_data['Income'], 
                      cluster_data['Total_Spending'],
                      label=f'Cluster {i}', s=50, alpha=0.6, c=[self.colors[i]])
        ax.set_xlabel('Age')
        ax.set_ylabel('Income ($)')
        ax.set_zlabel('Total Spending ($)')
        ax.set_title('3D Cluster Visualization', fontsize=14, fontweight='bold')
        ax.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 3D plot saved: {save_path}")
        plt.close()


