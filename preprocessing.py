# ============================================================================
# FILE 1: preprocessing.py
# ============================================================================

"""
Data Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handle all data preprocessing tasks"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()
        self.features_for_clustering = [
            'Age', 'Income', 'Total_Spending', 'Total_Purchases',
            'Total_Children', 'Recency', 'NumWebVisitsMonth', 'Education_Encoded'
        ]
        
    def load_data(self):
        """Load dataset from CSV file"""
        print("\n[PREPROCESSING] Loading Dataset...")
        self.df = pd.read_csv(self.filepath, sep='\t')
        print(f"✓ Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("\n[PREPROCESSING] Handling Missing Values...")
        if self.df['Income'].isnull().sum() > 0:
            missing_count = self.df['Income'].isnull().sum()
            print(f"⚠ Found {missing_count} missing values in Income")
            self.df['Income'].fillna(self.df['Income'].median(), inplace=True)
            print("✓ Filled with median")
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        dropped = initial_rows - len(self.df)
        if dropped > 0:
            print(f"✓ Dropped {dropped} rows with missing values")
        print(f"✓ Clean dataset: {len(self.df)} rows")
        return self
    
    def engineer_features(self):
        """Create new features from existing ones"""
        print("\n[PREPROCESSING] Engineering Features...")
        current_year = 2024
        self.df['Age'] = current_year - self.df['Year_Birth']
        print("✓ Age calculated")
        self.df['Total_Spending'] = (self.df['MntWines'] + self.df['MntFruits'] + 
                                     self.df['MntMeatProducts'] + self.df['MntFishProducts'] + 
                                     self.df['MntSweetProducts'] + self.df['MntGoldProds'])
        print("✓ Total_Spending calculated")
        self.df['Total_Purchases'] = (self.df['NumWebPurchases'] + 
                                      self.df['NumCatalogPurchases'] + 
                                      self.df['NumStorePurchases'])
        print("✓ Total_Purchases calculated")
        self.df['Total_Children'] = self.df['Kidhome'] + self.df['Teenhome']
        print("✓ Total_Children calculated")
        self.df['Dt_Customer'] = pd.to_datetime(self.df['Dt_Customer'], format='%d-%m-%Y')
        self.df['Customer_Days'] = (pd.to_datetime('2024-12-01') - self.df['Dt_Customer']).dt.days
        print("✓ Customer_Days calculated")
        return self
    
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("\n[PREPROCESSING] Encoding Categorical Variables...")
        education_mapping = {'Basic': 0, '2n Cycle': 1, 'Graduation': 2, 'Master': 3, 'PhD': 4}
        self.df['Education_Encoded'] = self.df['Education'].map(education_mapping)
        print("✓ Education encoded")
        self.df['Is_Single'] = self.df['Marital_Status'].apply(
            lambda x: 1 if x in ['Single', 'Alone', 'Absurd', 'YOLO'] else 0)
        print("✓ Marital_Status encoded")
        return self
    
    def scale_features(self):
        """Scale features for clustering"""
        print("\n[PREPROCESSING] Scaling Features...")
        X = self.df[self.features_for_clustering].values
        X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Features scaled - Shape: {X_scaled.shape}")
        return X_scaled
    
    def get_processed_data(self):
        """Return processed dataframe and scaled features"""
        return self.df, self.scale_features()
    
    def preprocess_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*70)
        self.load_data()
        self.handle_missing_values()
        self.engineer_features()
        self.encode_categorical_features()
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETED")
        print("="*70)
        return self.get_processed_data()
    
    def get_feature_names(self):
        """Return list of features used for clustering"""
        return self.features_for_clustering
    
    def save_processed_data(self, output_path='processed_data.csv'):
        """Save processed dataframe"""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Processed data saved to: {output_path}")


