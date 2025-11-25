# src/modules/market_analysis/eda/analyze_market_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MarketDataAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[4]
        self.data_dir = self.base_dir / "data" / "raw"
        self.results_dir = self.base_dir / "src" / "modules" / "market_analysis" / "eda" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self):
        """Load the market analysis dataset"""
        try:
            self.df = pd.read_csv(self.data_dir / "D7_market_researcher.csv")
            print(f"✅ D7 loaded: {self.df.shape}")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def basic_info(self):
        """Display basic dataset information"""
        print("=" * 60)
        print("📊 DATASET 7 (D7) - MARKET ANALYSIS DATA")
        print("=" * 60)
        print(f"Shape: {self.df.shape}")
        print("\nColumns:")
        print(self.df.columns.tolist())
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print(f"\nTotal missing values: {self.df.isnull().sum().sum()}")
        
        # Basic statistics for numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n📈 Numerical columns: {len(numerical_cols)}")
    
    def analyze_temporal_features(self):
        """Analyze time-series and seasonal patterns"""
        print("\n📅 TEMPORAL & SEASONAL ANALYSIS")
        print("=" * 60)
        
        # Identify temporal columns
        temporal_keywords = ['year', 'month', 'date', 'season', 'quarter', 'week']
        temporal_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in temporal_keywords)]
        
        if temporal_cols:
            print(f"📅 Temporal columns: {temporal_cols}")
            
            # Analyze each temporal column
            for col in temporal_cols:
                if col in self.df.columns:
                    unique_values = self.df[col].nunique()
                    print(f"\n📊 {col}:")
                    print(f"   Unique values: {unique_values}")
                    print(f"   Value range: {self.df[col].min()} to {self.df[col].max()}")
                    
                    if unique_values < 50:  # Plot if not too many unique values
                        value_counts = self.df[col].value_counts().sort_index()
                        plt.figure(figsize=(10, 6))
                        value_counts.plot(kind='bar', color='lightblue')
                        plt.title(f'Distribution of {col}')
                        plt.xlabel(col)
                        plt.ylabel('Frequency')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(self.results_dir / f'{col}_distribution.png', dpi=300, bbox_inches='tight')
                        plt.show()
        
        return temporal_cols
    
    def analyze_price_data(self):
        """Analyze price-related columns"""
        print("\n💰 PRICE ANALYSIS")
        print("=" * 60)
        
        # Identify price-related columns
        price_keywords = ['price', 'cost', 'value', 'rate', 'amount']
        price_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in price_keywords)]
        
        if price_cols:
            print(f"💰 Price-related columns: {price_cols}")
            
            # Statistical summary
            numerical_price_cols = self.df[price_cols].select_dtypes(include=[np.number]).columns
            if len(numerical_price_cols) > 0:
                print("\n📊 Price statistics:")
                print(self.df[numerical_price_cols].describe())
                
                # Distribution plots
                n_cols = min(3, len(numerical_price_cols))
                n_rows = (len(numerical_price_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                if len(numerical_price_cols) == 1:
                    axes = [axes]
                elif n_rows > 1:
                    axes = axes.ravel()
                
                for i, col in enumerate(numerical_price_cols[:n_rows*n_cols]):
                    ax = axes[i] if n_rows > 1 else axes[i]
                    self.df[col].hist(bins=30, ax=ax, alpha=0.7, color='gold')
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel('Price')
                    ax.set_ylabel('Frequency')
                    ax.grid(alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.results_dir / 'price_distributions.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # Price trends over time (if temporal data available)
                temporal_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in ['year', 'date'])]
                if temporal_cols and len(numerical_price_cols) > 0:
                    time_col = temporal_cols[0]
                    price_col = numerical_price_cols[0]
                    
                    if self.df[time_col].dtype in [np.int64, np.float64] or self.df[time_col].nunique() < 100:
                        plt.figure(figsize=(12, 6))
                        time_price = self.df.groupby(time_col)[price_col].mean()
                        time_price.plot(kind='line', marker='o', color='green')
                        plt.title(f'Average {price_col} Over Time')
                        plt.xlabel(time_col)
                        plt.ylabel(f'Average {price_col}')
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.results_dir / 'price_trends.png', dpi=300, bbox_inches='tight')
                        plt.show()
        
        return price_cols
    
    def analyze_crop_market(self):
        """Analyze crop-specific market data"""
        print("\n🌾 CROP MARKET ANALYSIS")
        print("=" * 60)
        
        # Identify crop-related columns
        crop_keywords = ['crop', 'commodity', 'product', 'item']
        crop_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in crop_keywords)]
        
        if crop_cols:
            print(f"🌾 Crop-related columns: {crop_cols}")
            
            crop_col = crop_cols[0]  # Use first crop-related column
            crop_distribution = self.df[crop_col].value_counts().head(20)
            
            print(f"\n📊 Top 20 crops/commodities:")
            for crop, count in crop_distribution.items():
                percentage = (count / len(self.df)) * 100
                print(f"   {crop}: {count} records ({percentage:.1f}%)")
            
            # Plot crop distribution
            plt.figure(figsize=(12, 6))
            crop_distribution.plot(kind='bar', color='lightgreen')
            plt.title('Top 20 Crops/Commodities in Market Data')
            plt.xlabel('Crop/Commodity')
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'crop_market_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Analyze price by crop (if price data available)
            price_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in ['price', 'cost'])]
            if price_cols and len(price_cols) > 0:
                price_col = price_cols[0]
                crop_price_stats = self.df.groupby(crop_col)[price_col].agg(['mean', 'std', 'min', 'max']).sort_values('mean', ascending=False)
                
                print(f"\n💰 Price statistics by crop:")
                print(crop_price_stats.head(10))
                
                # Plot top crops by average price
                plt.figure(figsize=(12, 6))
                crop_price_stats.head(15)['mean'].plot(kind='bar', color='orange')
                plt.title('Top 15 Crops by Average Price')
                plt.xlabel('Crop')
                plt.ylabel('Average Price')
                plt.xticks(rotation=45)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.results_dir / 'crop_price_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return crop_cols
    
    def analyze_geographical_markets(self):
        """Analyze geographical market patterns"""
        print("\n🌍 GEOGRAPHICAL MARKET ANALYSIS")
        print("=" * 60)
        
        # Identify geographical columns
        geo_keywords = ['state', 'region', 'district', 'market', 'location', 'zone']
        geo_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in geo_keywords)]
        
        if geo_cols:
            print(f"🌍 Geographical columns: {geo_cols}")
            
            geo_col = geo_cols[0]  # Use first geographical column
            geo_distribution = self.df[geo_col].value_counts().head(15)
            
            print(f"\n📊 Top 15 geographical regions:")
            for region, count in geo_distribution.items():
                percentage = (count / len(self.df)) * 100
                print(f"   {region}: {count} records ({percentage:.1f}%)")
            
            # Plot geographical distribution
            plt.figure(figsize=(12, 6))
            geo_distribution.plot(kind='bar', color='lightcoral')
            plt.title('Market Data Distribution by Region')
            plt.xlabel('Geographical Region')
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'geographical_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Analyze regional price variations (if price data available)
            price_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in ['price', 'cost'])]
            if price_cols and len(price_cols) > 0:
                price_col = price_cols[0]
                region_price_stats = self.df.groupby(geo_col)[price_col].agg(['mean', 'std']).sort_values('mean', ascending=False)
                
                print(f"\n💰 Regional price variations:")
                print(region_price_stats.head(10))
        
        return geo_cols
    
    def analyze_correlations(self):
        """Analyze correlations between market factors"""
        print("\n📈 MARKET FACTORS CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Select numerical columns for correlation analysis
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            print(f"📊 Numerical columns for correlation: {len(numerical_cols)}")
            
            # Correlation matrix
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', linewidths=0.5)
            plt.title('Market Factors Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'market_correlations.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Identify strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                #Strong correlation threshold    
                    if corr_value > 0.7:  
                        strong_correlations.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j]
                        ))
            
            if strong_correlations:
                print("\n🔗 Strong correlations (|r| > 0.7):")
                for var1, var2, corr in strong_correlations:
                    print(f"   {var1} ↔ {var2}: {corr:.3f}")
        
        return len(numerical_cols)
    
    def generate_market_insights(self):
        """Generate actionable market insights"""
        print("\n💡 MARKET ANALYSIS INSIGHTS")
        print("=" * 60)
        
        insights = []
        
        # Insight 1: Data scope
        insights.append(f"Dataset contains {len(self.df):,} market records")
        
        # Insight 2: Temporal coverage
        temporal_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in ['year', 'date'])]
        if temporal_cols:
            time_col = temporal_cols[0]
            if self.df[time_col].dtype in [np.int64, np.float64]:
                year_range = f"{self.df[time_col].min()} - {self.df[time_col].max()}"
                insights.append(f"Temporal coverage: {year_range}")
        
        # Insight 3: Market diversity
        crop_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in ['crop', 'commodity'])]
        if crop_cols:
            crop_col = crop_cols[0]
            unique_crops = self.df[crop_col].nunique()
            insights.append(f"Market covers {unique_crops} different crops/commodities")
        
        # Insight 4: Geographical coverage
        geo_cols = [col for col in self.df.columns if any(keyword in col.lower() for keyword in ['state', 'region'])]
        if geo_cols:
            geo_col = geo_cols[0]
            unique_regions = self.df[geo_col].nunique()
            insights.append(f"Geographical coverage: {unique_regions} regions")
        
        print("Key Market Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        return insights
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        if not self.load_dataset():
            return
        
        print("🚀 STARTING COMPREHENSIVE EDA FOR MARKET ANALYSIS DATA")
        print("=" * 70)
        
        self.basic_info()
        temporal_cols = self.analyze_temporal_features()
        price_cols = self.analyze_price_data()
        crop_cols = self.analyze_crop_market()
        geo_cols = self.analyze_geographical_markets()
        num_correlations = self.analyze_correlations()
        insights = self.generate_market_insights()
        
        print("\n✅ MARKET ANALYSIS EDA COMPLETED!")
        print(f"📅 Temporal features: {len(temporal_cols)}")
        print(f"💰 Price metrics: {len(price_cols)}")
        print(f"🌾 Crop commodities: {len(crop_cols)}")
        print(f"🌍 Geographical regions: {len(geo_cols)}")
        print(f"📈 Correlation analysis: {num_correlations} numerical features")
        print(f"💾 Results saved to: {self.results_dir}")

# Run the analysis
if __name__ == "__main__":
    analyzer = MarketDataAnalyzer()
    analyzer.generate_report()