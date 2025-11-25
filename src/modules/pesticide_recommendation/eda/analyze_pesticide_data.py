# src/modules/pesticide_recommendation/eda/analyze_pesticide_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PesticideDataAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[4]
        self.data_dir = self.base_dir / "data" / "raw"
        self.results_dir = self.base_dir / "src" / "modules" / "pesticide_recommendation" / "eda" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_datasets(self):
        """Load both pesticide recommendation datasets"""
        try:
            # Dataset 5: Indoor plants data
            self.df5 = pd.read_csv(self.data_dir / "D5_indoor_plants.csv")
            print(f"✅ D5 loaded: {self.df5.shape}")
            
            # Dataset 6: Farmer advisor data
            self.df6 = pd.read_csv(self.data_dir / "D6_farmer_advisor.csv")
            print(f"✅ D6 loaded: {self.df6.shape}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def basic_info(self):
        """Display basic dataset information"""
        print("=" * 60)
        print("📊 DATASET 5 (D5) - INDOOR PLANTS DATA")
        print("=" * 60)
        print(f"Shape: {self.df5.shape}")
        print("\nColumns:")
        print(self.df5.columns.tolist())
        print("\nData Types:")
        print(self.df5.dtypes)
        print("\nMissing Values:")
        print(self.df5.isnull().sum())
        
        print("\n" + "=" * 60)
        print("📊 DATASET 6 (D6) - FARMER ADVISOR DATA")
        print("=" * 60)
        print(f"Shape: {self.df6.shape}")
        print("\nColumns:")
        print(self.df6.columns.tolist())
        print("\nData Types:")
        print(self.df6.dtypes)
        print("\nMissing Values:")
        print(self.df6.isnull().sum())
    
    def analyze_indoor_plants(self):
        """Analyze indoor plants dataset (D5)"""
        print("\n🏠 INDOOR PLANTS ANALYSIS (D5)")
        print("=" * 60)
        
        # Analyze plant types
        if 'plant_type' in self.df5.columns or 'Plant_Type' in self.df5.columns:
            plant_col = 'plant_type' if 'plant_type' in self.df5.columns else 'Plant_Type'
            plant_distribution = self.df5[plant_col].value_counts()
            
            print(f"🌿 Plant type distribution:")
            for plant, count in plant_distribution.items():
                print(f"   {plant}: {count} records")
            
            plt.figure(figsize=(10, 6))
            plant_distribution.plot(kind='bar', color='lightgreen')
            plt.title('Indoor Plant Types Distribution')
            plt.xlabel('Plant Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'indoor_plant_types.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Analyze health metrics
        health_metrics = [col for col in self.df5.columns if any(keyword in col.lower() 
                         for keyword in ['health', 'condition', 'status', 'score'])]
        
        if health_metrics:
            print(f"❤️  Health metrics: {health_metrics}")
            
            # Plot health metric distributions
            n_metrics = len(health_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows > 1:
                axes = axes.ravel()
            
            for i, metric in enumerate(health_metrics[:n_rows*n_cols]):
                ax = axes[i] if n_rows > 1 else axes[i]
                self.df5[metric].hist(bins=30, ax=ax, alpha=0.7, color='lightcoral')
                ax.set_title(f'Distribution of {metric}')
                ax.set_xlabel(metric)
                ax.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'health_metrics_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Analyze pest-related columns
        pest_columns = [col for col in self.df5.columns if any(keyword in col.lower() 
                       for keyword in ['pest', 'disease', 'infestation', 'damage'])]
        
        if pest_columns:
            print(f"🐛 Pest-related columns: {pest_columns}")
            
            for col in pest_columns:
                if self.df5[col].dtype == 'object' or self.df5[col].nunique() < 20:
                    value_counts = self.df5[col].value_counts()
                    print(f"\n📊 {col} distribution:")
                    for value, count in value_counts.items():
                        print(f"   {value}: {count} records")
    
    def analyze_farmer_advisor(self):
        """Analyze farmer advisor dataset (D6)"""
        print("\n👨‍🌾 FARMER ADVISOR ANALYSIS (D6)")
        print("=" * 60)
        
        # Analyze crop types
        crop_columns = [col for col in self.df6.columns if any(keyword in col.lower() 
                       for keyword in ['crop', 'plant', 'variety'])]
        
        if crop_columns:
            crop_col = crop_columns[0]  # Use first crop-related column
            crop_distribution = self.df6[crop_col].value_counts().head(15)
            
            print(f"🌱 Top 15 crops:")
            for crop, count in crop_distribution.items():
                print(f"   {crop}: {count} records")
            
            plt.figure(figsize=(12, 6))
            crop_distribution.plot(kind='bar', color='gold')
            plt.title('Crop Distribution in Farmer Advisor Data')
            plt.xlabel('Crop Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'farmer_crop_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Analyze pesticide recommendations
        pesticide_columns = [col for col in self.df6.columns if any(keyword in col.lower() 
                           for keyword in ['pesticide', 'chemical', 'treatment', 'spray'])]
        
        if pesticide_columns:
            print(f"💊 Pesticide-related columns: {pesticide_columns}")
            
            for col in pesticide_columns:
                if self.df6[col].dtype == 'object' or self.df6[col].nunique() < 20:
                    value_counts = self.df6[col].value_counts().head(10)
                    print(f"\n📊 Top 10 {col} values:")
                    for value, count in value_counts.items():
                        print(f"   {value}: {count} records")
        
        # Analyze farming practices
        practice_columns = [col for col in self.df6.columns if any(keyword in col.lower() 
                          for keyword in ['practice', 'method', 'technique', 'system'])]
        
        if practice_columns:
            print(f"🌾 Farming practice columns: {practice_columns}")
    
    def analyze_environmental_factors(self):
        """Analyze environmental factors affecting pesticide recommendations"""
        print("\n🌡️ ENVIRONMENTAL FACTORS ANALYSIS")
        print("=" * 60)
        
        # Common environmental factors
        env_factors = ['temperature', 'humidity', 'rainfall', 'soil', 'ph', 'season']
        
        # Check D5 for environmental factors
        d5_env = [col for col in self.df5.columns if any(factor in col.lower() for factor in env_factors)]
        if d5_env:
            print(f"🏠 D5 Environmental factors: {d5_env}")
            
            # Correlation analysis for numerical environmental factors
            num_env = self.df5[d5_env].select_dtypes(include=[np.number]).columns
            if len(num_env) > 1:
                plt.figure(figsize=(8, 6))
                correlation_matrix = self.df5[num_env].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('D5: Environmental Factors Correlation')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'd5_environmental_correlations.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Check D6 for environmental factors
        d6_env = [col for col in self.df6.columns if any(factor in col.lower() for factor in env_factors)]
        if d6_env:
            print(f"👨‍🌾 D6 Environmental factors: {d6_env}")
    
    def dataset_comparison(self):
        """Compare D5 and D6 datasets"""
        print("\n🔍 DATASET COMPARISON: D5 vs D6")
        print("=" * 60)
        
        print(f"D5 Shape: {self.df5.shape} (Indoor Plants)")
        print(f"D6 Shape: {self.df6.shape} (Farmer Advisor)")
        
        # Check common columns
        common_cols = set(self.df5.columns) & set(self.df6.columns)
        print(f"\nCommon columns: {len(common_cols)}")
        if common_cols:
            print(common_cols)
        
        # Analyze complementary features
        d5_unique = set(self.df5.columns) - set(self.df6.columns)
        d6_unique = set(self.df6.columns) - set(self.df5.columns)
        
        print(f"\nD5 unique columns: {len(d5_unique)}")
        print(f"D6 unique columns: {len(d6_unique)}")
        
        # Potential integration points
        integration_points = []
        if any('pest' in col.lower() for col in d5_unique) and any('pest' in col.lower() for col in d6_unique):
            integration_points.append("Pest identification and treatment")
        if any('health' in col.lower() for col in d5_unique) and any('condition' in col.lower() for col in d6_unique):
            integration_points.append("Plant health assessment")
        
        if integration_points:
            print(f"\n💡 Potential integration points:")
            for point in integration_points:
                print(f"   • {point}")
    
    def generate_insights(self):
        """Generate insights for pesticide recommendation system"""
        print("\n💡 PESTICIDE RECOMMENDATION INSIGHTS")
        print("=" * 60)
        
        insights = []
        
        # Insight 1: Data coverage
        total_records = len(self.df5) + len(self.df6)
        insights.append(f"Total records available: {total_records:,}")
        
        # Insight 2: Scope analysis
        if 'indoor' in str(self.df5.columns).lower() or 'house' in str(self.df5.columns).lower():
            insights.append("D5 focuses on indoor/domestic plant care")
        if any(keyword in str(self.df6.columns).lower() for keyword in ['farm', 'field', 'crop', 'agriculture']):
            insights.append("D6 focuses on agricultural/farming scenarios")
        
        # Insight 3: Recommendation factors
        factors = []
        if any('temperature' in col.lower() for col in self.df5.columns) or any('temperature' in col.lower() for col in self.df6.columns):
            factors.append("temperature")
        if any('humidity' in col.lower() for col in self.df5.columns) or any('humidity' in col.lower() for col in self.df6.columns):
            factors.append("humidity")
        if any('soil' in col.lower() for col in self.df5.columns) or any('soil' in col.lower() for col in self.df6.columns):
            factors.append("soil conditions")
        
        if factors:
            insights.append(f"Key recommendation factors: {', '.join(factors)}")
        
        print("Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        return insights
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        if not self.load_datasets():
            return
        
        print("🚀 STARTING COMPREHENSIVE EDA FOR PESTICIDE RECOMMENDATION DATA")
        print("=" * 70)
        
        self.basic_info()
        self.analyze_indoor_plants()
        self.analyze_farmer_advisor()
        self.analyze_environmental_factors()
        self.dataset_comparison()
        insights = self.generate_insights()
        
        print("\n✅ PESTICIDE RECOMMENDATION EDA COMPLETED!")
        print(f"💾 Results saved to: {self.results_dir}")

# Run the analysis
if __name__ == "__main__":
    analyzer = PesticideDataAnalyzer()
    analyzer.generate_report()