# src/modules/crop_recommendation/eda/analyze_crop_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CropDataAnalyzer:
    def __init__(self):
        # Use current project root instead of adding an extra "CropCareAI"
        self.base_dir = Path(__file__).resolve().parents[4]  # goes up to project root
        self.data_dir = self.base_dir / "data" / "raw"
        self.results_dir = self.base_dir / "src" / "modules" / "crop_recommendation" / "eda" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_datasets(self):
        """Load both crop recommendation datasets"""
        try:
            # Dataset 1: Comprehensive crop data
            self.df1 = pd.read_csv(self.data_dir / "D1_crop_recommendation.csv")
            print(f"✅ D1 loaded: {self.df1.shape}")
            
            # Dataset 4: Simplified crop data  
            self.df4 = pd.read_csv(self.data_dir / "D4_crop_recommendation.csv")
            print(f"✅ D4 loaded: {self.df4.shape}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def basic_info(self):
        """Display basic dataset information"""
        print("=" * 60)
        print("📊 DATASET 1 (D1) - COMPREHENSIVE CROP DATA")
        print("=" * 60)
        print(f"Shape: {self.df1.shape}")
        print("\nColumns:")
        print(self.df1.columns.tolist())
        print("\nData Types:")
        print(self.df1.dtypes)
        print("\nMissing Values:")
        print(self.df1.isnull().sum())
        
        print("\n" + "=" * 60)
        print("📊 DATASET 4 (D4) - SIMPLIFIED CROP DATA")
        print("=" * 60)
        print(f"Shape: {self.df4.shape}")
        print("\nColumns:")
        print(self.df4.columns.tolist())
        print("\nMissing Values:")
        print(self.df4.isnull().sum())
    
    def analyze_numerical_features(self):
        """Analyze numerical features distribution"""
        # Select numerical columns (assuming common structure)
        numerical_cols = self.df1.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols[:9]):
            self.df1[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical summary
        print("\n📈 Numerical Features Summary:")
        print(self.df1[numerical_cols].describe())
    
    def analyze_categorical_features(self):
        """Analyze categorical features (especially crop types)"""
        if 'label' in self.df1.columns:  # Assuming 'label' is the crop type
            crop_distribution = self.df1['label'].value_counts()
            
            plt.figure(figsize=(12, 6))
            crop_distribution.plot(kind='bar')
            plt.title('Distribution of Crop Types')
            plt.xlabel('Crop Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'crop_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\n🌱 Crop Type Distribution:")
            print(crop_distribution)
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        numerical_cols = self.df1.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = self.df1[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return correlation_matrix
        return None
    
    def outlier_analysis(self):
        """Check for outliers in numerical features"""
        numerical_cols = self.df1.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols[:9]):
            self.df1.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def dataset_comparison(self):
        """Compare D1 and D4 datasets"""
        print("\n" + "=" * 60)
        print("🔍 DATASET COMPARISON: D1 vs D4")
        print("=" * 60)
        
        print(f"D1 Shape: {self.df1.shape}")
        print(f"D4 Shape: {self.df4.shape}")
        
        # Check common columns
        common_cols = set(self.df1.columns) & set(self.df4.columns)
        print(f"\nCommon columns: {len(common_cols)}")
        print(common_cols)
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        if not self.load_datasets():
            return
        
        print("🚀 STARTING COMPREHENSIVE EDA FOR CROP RECOMMENDATION DATA")
        print("=" * 70)
        
        self.basic_info()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        correlation_matrix = self.correlation_analysis()
        self.outlier_analysis()
        self.dataset_comparison()
        
        print("\n✅ EDA COMPLETED! Check the results/ folder for visualizations.")
        
        # Return the dataframes for enhanced analysis
        return self.df1, self.df4


class EnhancedCropAnalyzer:
    def __init__(self, df1, df4):
        self.df1 = df1
        self.df4 = df4
        self.base_dir = Path(__file__).resolve().parents[4]
        self.results_dir = self.base_dir / "src" / "modules" / "crop_recommendation" / "eda" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def create_comprehensive_visualizations(self):
        """Create enhanced visualizations based on the data structure"""
        self.plot_crop_distribution()
        self.plot_seasonal_comparison()
        self.plot_feature_correlations()
        self.plot_soil_nutrient_analysis()
        self.dataset_feature_comparison()
        
    def plot_crop_distribution(self):
        """Enhanced crop distribution plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # D1 Crop distribution
        crop_counts_d1 = self.df1['label'].value_counts()
        ax1.pie(crop_counts_d1.values, labels=crop_counts_d1.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('D1: Crop Distribution (Comprehensive Dataset)')
        
        # D4 Crop distribution
        crop_counts_d4 = self.df4['label'].value_counts()
        ax2.pie(crop_counts_d4.values, labels=crop_counts_d4.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('D4: Crop Distribution (Simplified Dataset)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'crop_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_seasonal_comparison(self):
        """Compare seasonal weather patterns"""
        seasonal_cols = [col for col in self.df1.columns if any(season in col for season in ['W', 'Sp', 'Su', 'Au'])]
        
        # Temperature comparison
        temp_cols = [col for col in seasonal_cols if 'T2M' in col]
        if len(temp_cols) >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
            for i, season in enumerate(['W', 'Sp', 'Su', 'Au']):
                max_temp_col = f'T2M_MAX-{season}'
                min_temp_col = f'T2M_MIN-{season}'
                
                axes[i].hist(self.df1[max_temp_col], alpha=0.7, label=f'Max Temp', bins=30)
                axes[i].hist(self.df1[min_temp_col], alpha=0.7, label=f'Min Temp', bins=30)
                axes[i].set_title(f'{seasons[i]} Temperature Distribution')
                axes[i].set_xlabel('Temperature (°C)')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'seasonal_temperature_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_feature_correlations(self):
        """Enhanced correlation analysis focusing on key nutrients"""
        key_nutrients = ['N', 'P', 'K', 'Ph', 'Zn', 'S']
        available_nutrients = [col for col in key_nutrients if col in self.df1.columns]
        
        if len(available_nutrients) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.df1[available_nutrients].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
            plt.title('Nutrient Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'nutrient_correlations.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_soil_nutrient_analysis(self):
        """Analyze soil nutrient relationships with crops"""
        if 'Soilcolor' in self.df1.columns and 'Ph' in self.df1.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.df1, x='Soilcolor', y='Ph', hue='label')
            plt.title('Soil pH Distribution by Crop Type and Soil Color')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'soil_ph_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def dataset_feature_comparison(self):
        """Compare common features between D1 and D4"""
        common_features = ['N', 'P', 'K']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, feature in enumerate(common_features):
            if feature in self.df1.columns and feature in self.df4.columns:
                axes[i].hist(self.df1[feature], alpha=0.7, label='D1', bins=30, color='blue')
                axes[i].hist(self.df4[feature], alpha=0.7, label='D4', bins=30, color='red')
                axes[i].set_title(f'{feature} Distribution Comparison')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


# Run the analysis
if __name__ == "__main__":
    # Run basic EDA
    analyzer = CropDataAnalyzer()
    df1, df4 = analyzer.generate_report()
    
    # Run enhanced analysis with the returned dataframes
    enhanced_analyzer = EnhancedCropAnalyzer(df1, df4)
    enhanced_analyzer.create_comprehensive_visualizations()