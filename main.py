# ============================================================================
# FILE 4: main.py
# ============================================================================

"""
Main Execution Script
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from preprocessing import DataPreprocessor
from model_training import KMeansTrainer
from visualization import ClusterVisualizer
from datetime import datetime

def print_header():
    print("\n" + "="*70)
    print(" " * 15 + "CUSTOMER SEGMENTATION PROJECT")
    print(" " * 20 + "K-Means Clustering")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

def print_footer():
    print("\n" + "="*70)
    print(" " * 20 + "PROJECT COMPLETED!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 Generated Files:")
    print("   • processed_data.csv")
    print("   • customer_segments_final.csv")
    print("   • optimal_clusters.png")
    print("   • cluster_analysis.png")
    print("   • 3d_clusters.png")
    print("   • kmeans_model.pkl")
    print("\n🚀 Run Streamlit: streamlit run streamlit_app.py")
    print("="*70)

def main():
    print_header()
    try:
        # PREPROCESSING
        print("\n" + "="*70)
        print("STEP 1: DATA PREPROCESSING")
        print("="*70)
        preprocessor = DataPreprocessor('marketing_campaign.csv')
        df_processed, X_scaled = preprocessor.preprocess_pipeline()
        preprocessor.save_processed_data('processed_data.csv')
        
        # MODEL TRAINING
        print("\n" + "="*70)
        print("STEP 2: MODEL TRAINING")
        print("="*70)
        trainer = KMeansTrainer(X_scaled, df_processed)
        k_range = range(2, 11)
        suggested_k = trainer.find_optimal_clusters(k_range)
        trainer.plot_optimal_k(k_range, 'optimal_clusters.png')
        optimal_k = 4
        print(f"\nUsing K = {optimal_k}")
        clusters = trainer.train_model(n_clusters=optimal_k)
        feature_names = preprocessor.get_feature_names()
        cluster_summary = trainer.analyze_clusters(feature_names)
        trainer.save_model('kmeans_model.pkl')
        trainer.save_clustered_data('customer_segments_final.csv')
        
        # VISUALIZATION
        print("\n" + "="*70)
        print("STEP 3: CREATING VISUALIZATIONS")
        print("="*70)
        visualizer = ClusterVisualizer(df_processed, X_scaled, clusters)
        visualizer.create_comprehensive_analysis('cluster_analysis.png')
        visualizer.create_3d_plot('3d_clusters.png')
        
        print_footer()
        return 0
    except FileNotFoundError:
        print("\n❌ ERROR: Could not find 'marketing_campaign.csv'")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)