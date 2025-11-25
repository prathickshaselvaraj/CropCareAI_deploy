# src/modules/yield_analysis/eda/analyze_yield_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class YieldDataAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[4]
        self.data_dir = self.base_dir / "data" / "raw"
        self.results_dir = self.base_dir / "src" / "modules" / "yield_analysis" / "eda" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self):
        """Load the yield analysis dataset"""
        try:
            self.df = pd.read_csv(self.data_dir / "D3_complete_dataset.csv")
            print(f"✅ D3 loaded: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def basic_info(self):
        """Display basic dataset information"""
        print("=" * 60)
        print("📊 DATASET 3 (D3) - YIELD ANALYSIS DATA")
        print("=" * 60)
        print(f"Shape: {self.df.shape}")
        print("\nColumns:")
        print(self.df.columns.tolist())
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print(f"\nTotal missing values: {self.df.isnull().sum().sum()}")
    
    def analyze_temporal_features(self):
        """Analyze time-series aspects of the data"""
        print("\n📅 TEMPORAL ANALYSIS")
        print("=" * 60)
        
        # Check for date/time columns
        date_columns = [col for col in self.df.columns if any(keyword in col.lower() 
                      for keyword in ['year', 'date', 'time', 'season'])]
        
        if date_columns:
            print(f"📅 Date-related columns: {date_columns}")
            
            # Analyze year distribution if present
            if 'Year' in self.df.columns:
                year_counts = self.df['Year'].value_counts().sort_index()
                print(f"📊 Year distribution: {year_counts.to_dict()}")
                
                plt.figure(figsize=(12, 6))
                year_counts.plot(kind='bar', color='skyblue')
                plt.title('Data Distribution by Year')
                plt.xlabel('Year')
                plt.ylabel('Number of Records')
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.results_dir / 'year_distribution.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return date_columns
    
    def analyze_geographical_features(self):
        """Analyze geographical aspects of the data"""
        print("\n🌍 GEOGRAPHICAL ANALYSIS")
        print("=" * 60)
        
        # Check for geographical columns
        geo_columns = [col for col in self.df.columns if any(keyword in col.lower() 
                     for keyword in ['state', 'district', 'region', 'zone', 'area'])]
        
        if geo_columns:
            print(f"🌍 Geographical columns: {geo_columns}")
            
            # Analyze state/district distribution
            for geo_col in geo_columns[:2]:  # Analyze first 2 geographical columns
                if geo_col in self.df.columns:
                    geo_distribution = self.df[geo_col].value_counts().head(15)
                    print(f"\n📊 Top 15 {geo_col}s by record count:")
                    for location, count in geo_distribution.items():
                        print(f"   {location}: {count} records")
                    
                    # Plot geographical distribution
                    plt.figure(figsize=(12, 6))
                    geo_distribution.plot(kind='bar', color='lightgreen')
                    plt.title(f'Records Distribution by {geo_col}')
                    plt.xlabel(geo_col)
                    plt.ylabel('Number of Records')
                    plt.xticks(rotation=45)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.results_dir / f'{geo_col}_distribution.png', dpi=300, bbox_inches='tight')
                    plt.show()
        
        return geo_columns
    
    def analyze_crop_features(self):
        """Analyze crop-related features"""
        print("\n🌱 CROP ANALYSIS")
        print("=" * 60)
        
        # Check for crop-related columns
        crop_columns = [col for col in self.df.columns if any(keyword in col.lower() 
                       for keyword in ['crop', 'production', 'yield', 'area'])]
        
        if crop_columns:
            print(f"🌱 Crop-related columns: {crop_columns}")
            
            # Analyze crop type distribution
            if 'Crop' in self.df.columns:
                crop_distribution = self.df['Crop'].value_counts().head(20)
                print(f"\n📊 Top 20 crops by record count:")
                for crop, count in crop_distribution.items():
                    print(f"   {crop}: {count} records")
                
                plt.figure(figsize=(12, 6))
                crop_distribution.plot(kind='bar', color='gold')
                plt.title('Top 20 Crops by Record Count')
                plt.xlabel('Crop Type')
                plt.ylabel('Number of Records')
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.results_dir / 'crop_distribution.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return crop_columns
    
    def analyze_production_metrics(self):
        """Analyze production and yield metrics"""
        print("\n📈 PRODUCTION METRICS ANALYSIS")
        print("=" * 60)
        
        # Identify numerical production metrics
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        production_metrics = [col for col in numerical_cols if any(keyword in col.lower() 
                             for keyword in ['production', 'yield', 'area', 'quantity'])]
        
        if production_metrics:
            print(f"📈 Production metrics: {production_metrics}")
            
            # Statistical summary
            print("\n📊 Production metrics summary:")
            print(self.df[production_metrics].describe())
            
            # Distribution plots
            n_metrics = len(production_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows > 1:
                axes = axes.ravel()
            
            for i, metric in enumerate(production_metrics[:n_rows*n_cols]):
                if n_rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[i]
                
                self.df[metric].hist(bins=30, ax=ax, alpha=0.7, color='lightcoral')
                ax.set_title(f'Distribution of {metric}')
                ax.set_xlabel(metric)
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(production_metrics), n_rows*n_cols):
                if n_rows == 1:
                    axes[i].axis('off')
                else:
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'production_metrics_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Correlation analysis
            if len(production_metrics) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = self.df[production_metrics].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
                plt.title('Production Metrics Correlation Matrix')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'production_correlations.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return production_metrics
    
    def analyze_outliers(self):
        """Detect and analyze outliers in numerical features"""
        print("\n📊 OUTLIER ANALYSIS")
        print("=" * 60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            # Select first 9 numerical columns for visualization
            cols_to_plot = numerical_cols[:9]
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if len(cols_to_plot) == 1:
                axes = [axes]
            elif n_rows > 1:
                axes = axes.ravel()
            
            for i, col in enumerate(cols_to_plot):
                if n_rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[i]
                
                self.df.boxplot(column=col, ax=ax)
                ax.set_title(f'Boxplot of {col}')
                ax.grid(alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(cols_to_plot), n_rows*n_cols):
                if n_rows == 1:
                    axes[i].axis('off')
                else:
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        if not self.load_dataset():
            return
        
        print("🚀 STARTING COMPREHENSIVE EDA FOR YIELD ANALYSIS DATA")
        print("=" * 70)
        
        self.basic_info()
        date_cols = self.analyze_temporal_features()
        geo_cols = self.analyze_geographical_features()
        crop_cols = self.analyze_crop_features()
        production_metrics = self.analyze_production_metrics()
        self.analyze_outliers()
        
        print("\n✅ YIELD ANALYSIS EDA COMPLETED!")
        print(f"📅 Temporal features: {len(date_cols) if date_cols else 0}")
        print(f"🌍 Geographical features: {len(geo_cols) if geo_cols else 0}")
        print(f"🌱 Crop features: {len(crop_cols) if crop_cols else 0}")
        print(f"📈 Production metrics: {len(production_metrics) if production_metrics else 0}")
        print(f"💾 Results saved to: {self.results_dir}")

# Run the analysis
if __name__ == "__main__":
    analyzer = YieldDataAnalyzer()
    analyzer.generate_report()