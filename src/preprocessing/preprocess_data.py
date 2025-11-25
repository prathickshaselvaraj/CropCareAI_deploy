# src/preprocessing/preprocess_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

class CropCareDataPreprocessor:
    def __init__(self, data_path="data/raw/"):
        # Use absolute path to avoid directory issues
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / data_path
        self.processed_path = self.project_root / "data/processed/"
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Data path: {self.data_path}")
        print(f"📁 Processed path: {self.processed_path}")
    
    def load_all_datasets(self):
        """Load datasets with correct filenames"""
        datasets = {}
        
        # Map with actual filenames
        file_mapping = {
            'D1': 'D1_crop_recommendation.csv',
            'D2': 'D2_plantvillage',  # This is a directory, not CSV
            'D3': 'D3_complete_dataset.csv', 
            'D4': 'D4_crop_recommendation.csv',
            'D5': 'D5_indoor_plants.csv',
            'D6': 'D6_farmer_advisor.csv',
            'D7': 'D7_market_researcher.csv'
        }
        
        for dataset_name, filename in file_mapping.items():
            file_path = self.data_path / filename
            
            if dataset_name == 'D2':
                # Handle D2 as directory (image dataset)
                if file_path.exists() and file_path.is_dir():
                    print(f"✅ Found D2 image directory: {filename}")
                    datasets['D2'] = {'path': file_path, 'type': 'images'}
                else:
                    print(f"⚠️  D2 image directory not found: {filename}")
            else:
                # Handle CSV files
                if file_path.exists():
                    try:
                        datasets[dataset_name] = pd.read_csv(file_path)
                        print(f"✅ Loaded {dataset_name}: {filename} - Shape: {datasets[dataset_name].shape}")
                    except Exception as e:
                        print(f"❌ Error loading {filename}: {e}")
                else:
                    print(f"⚠️  File not found: {filename}")
        
        return datasets if datasets else None

    def fix_d3_yield_data(self, df):
        """Fix D3 production column (object to numeric)"""
        print("🔄 Fixing D3 yield data...")
        
        if 'production' not in df.columns:
            # Check for alternative column names
            production_cols = [col for col in df.columns if 'production' in col.lower() or 'yield' in col.lower()]
            if production_cols:
                print(f"⚠️  Using alternative column: {production_cols[0]}")
                df['production'] = df[production_cols[0]]
            else:
                print("⚠️  No production-related column found")
                return df
        
        # Check current dtype and sample values
        print(f"Current production dtype: {df['production'].dtype}")
        print(f"Sample values: {df['production'].head(3).tolist()}")
        
        try:
            # Clean and convert production column
            df['production_cleaned'] = (
                df['production']
                .astype(str)
                .str.replace(',', '')
                .str.extract(r'(\d+\.?\d*)')[0]  # Fixed regex
            )
            df['production_cleaned'] = pd.to_numeric(df['production_cleaned'], errors='coerce')
            
            # Handle missing values
            initial_missing = df['production_cleaned'].isna().sum()
            if initial_missing > 0:
                print(f"⚠️  {initial_missing} missing values in production, filling with median...")
                median_production = df['production_cleaned'].median()
                df['production_cleaned'].fillna(median_production, inplace=True)
            
            print(f"✅ Production column fixed. Range: {df['production_cleaned'].min():.2f} to {df['production_cleaned'].max():.2f}")
            return df
            
        except Exception as e:
            print(f"❌ Error processing production column: {e}")
            return df

    def fix_d5_pesticide_data(self, df):
        """Fix D5 missing values in key columns"""
        print("🔄 Fixing D5 pesticide data...")
        
        # Analyze missing values
        missing_report = df.isnull().sum()
        missing_cols = missing_report[missing_report > 0]
        
        if len(missing_cols) == 0:
            print("✅ No missing values found in D5")
            return df
        
        print("Missing values per column:")
        for col, missing in missing_cols.items():
            print(f"  {col}: {missing} missing ({missing/len(df)*100:.1f}%)")
        
        # Fix specific columns if they exist
        if 'Pest_Presence' in df.columns:
            if df['Pest_Presence'].nunique() <= 2:
                mode_val = df['Pest_Presence'].mode()[0] if not df['Pest_Presence'].mode().empty else 'No'
                df['Pest_Presence'].fillna(mode_val, inplace=True)
            else:
                df['Pest_Presence'].fillna('Unknown', inplace=True)
        
        if 'Fertilizer_Type' in df.columns:
            mode_fert = df['Fertilizer_Type'].mode()[0] if not df['Fertilizer_Type'].mode().empty else 'Organic'
            df['Fertilizer_Type'].fillna(mode_fert, inplace=True)
        
        # Fix remaining categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        # Fix numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        print("✅ D5 missing values handled")
        return df

    def analyze_d2_image_dataset(self, d2_info):
        """Analyze D2 PlantVillage image dataset structure"""
        print("🔄 Analyzing D2 PlantVillage image dataset...")
        
        image_path = d2_info['path']
        print(f"📁 Image directory: {image_path}")
        
        # Count images and analyze structure
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(image_path.rglob(ext)))
        
        print(f"📸 Total image files found: {len(image_files)}")
        
        # Analyze directory structure (should be: disease_type/plant_type/images)
        subdirs = [d for d in image_path.iterdir() if d.is_dir()]
        print(f"📂 Subdirectories (disease types): {len(subdirs)}")
        
        disease_stats = {}
        for subdir in subdirs:
            disease_name = subdir.name
            disease_images = []
            for ext in image_extensions:
                disease_images.extend(list(subdir.glob(ext)))
            disease_stats[disease_name] = len(disease_images)
            if len(disease_images) > 0:  # Only show directories with images
                print(f"  {disease_name}: {len(disease_images)} images")
        
        # Create a metadata dataframe for analysis
        metadata = []
        for disease, count in disease_stats.items():
            if count > 0:  # Only include directories with images
                metadata.append({
                    'disease_type': disease,
                    'image_count': count,
                    'label': disease.replace('___', ' ').replace('_', ' ').title()  # Clean label names
                })
        
        return pd.DataFrame(metadata), disease_stats

    def create_processed_datasets(self):
        """Main preprocessing pipeline"""
        print("🚀 Starting CropCareAI Data Preprocessing Pipeline...")
        print("=" * 60)
        
        # Load all datasets
        datasets = self.load_all_datasets()
        if datasets is None:
            print("❌ No datasets could be loaded")
            return False
        
        processed_datasets = {}
        
        # Process each dataset
        for name, data in datasets.items():
            print(f"\n📊 Processing {name}...")
            
            if name == 'D2':
                # Handle image dataset
                df_processed, stats = self.analyze_d2_image_dataset(data)
                print(f"   Image analysis complete: {len(stats)} disease types")
            else:
                # Handle CSV datasets
                print(f"   Original shape: {data.shape}")
                
                # Apply specific fixes
                if name == 'D3':
                    df_processed = self.fix_d3_yield_data(data.copy())
                elif name == 'D5':
                    df_processed = self.fix_d5_pesticide_data(data.copy())
                else:
                    df_processed = data.copy()
                
                # Basic cleaning for all datasets
                df_processed = df_processed.drop_duplicates()
                print(f"   Processed shape: {df_processed.shape}")
            
            # Save processed dataset
            if name == 'D2':
                save_path = self.processed_path / f"{name}_metadata.csv"
                if hasattr(df_processed, 'to_csv'):
                    df_processed.to_csv(save_path, index=False)
                    processed_datasets[name] = df_processed
                    print(f"   ✅ Saved to: {save_path}")
                else:
                    processed_datasets[name] = data  # Keep original for D2
                    print(f"   ✅ Preserved original {name} dataset")
            else:
                save_path = self.processed_path / f"{name}_processed.csv"
                df_processed.to_csv(save_path, index=False)
                processed_datasets[name] = df_processed
                print(f"   ✅ Saved to: {save_path}")
        
        # Generate preprocessing report
        self.generate_preprocessing_report(datasets, processed_datasets)
        
        return processed_datasets

    def generate_preprocessing_report(self, original, processed):
        """Generate comprehensive preprocessing report"""
        report_path = self.processed_path / "preprocessing_report.md"
        
        report_content = "# CropCareAI Data Preprocessing Report\n\n"
        report_content += "## Dataset Summary\n\n"
        report_content += "| Dataset | Original | Processed | Changes |\n"
        report_content += "|---------|----------|-----------|---------|\n"
        
        for name, data in original.items():
            if name == 'D2':
                orig_info = "Image Directory"
                if name in processed and hasattr(processed[name], 'shape'):
                    proc_info = f"Metadata: {processed[name].shape[0]} classes"
                else:
                    proc_info = "Preserved"
                changes = "Image analysis completed"
            else:
                orig_info = f"{data.shape[0]} rows, {data.shape[1]} cols"
                if name in processed and hasattr(processed[name], 'shape'):
                    proc_info = f"{processed[name].shape[0]} rows, {processed[name].shape[1]} cols"
                else:
                    proc_info = "Preserved"
                
                changes = []
                if name == 'D3':
                    changes.append("Production column cleaned")
                if name == 'D5':
                    changes.append("Missing values handled")
                if not changes:
                    changes = ["Basic cleaning"]
                changes = ", ".join(changes)
            
            report_content += f"| {name} | {orig_info} | {proc_info} | {changes} |\n"
        
        # Fix: Use UTF-8 encoding
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📋 Preprocessing report saved to: {report_path}")

# Execute the preprocessing pipeline
if __name__ == "__main__":
    preprocessor = CropCareDataPreprocessor()
    processed_data = preprocessor.create_processed_datasets()
    
    if processed_data:
        print("\n🎉 DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("🔧 Next step: Model Development")
        
        # Show summary
        print("\n📊 PROCESSED DATASETS SUMMARY:")
        for name, data in processed_data.items():
            if name == 'D2':
                if hasattr(data, 'shape'):
                    print(f"  {name}: {data.shape[0]} disease classes analyzed")
                else:
                    print(f"  {name}: Image dataset preserved")
            else:
                print(f"  {name}: {data.shape[0]} rows, {data.shape[1]} columns")
    else:
        print("\n❌ Preprocessing failed. Check data files.")