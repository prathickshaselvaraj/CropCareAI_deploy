# src/modules/disease_detection/eda/analyze_plantvillage.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class PlantVillageAnalyzer:
    def __init__(self):
        # Use the same path resolution as the crop analyzer
        self.base_dir = Path(__file__).resolve().parents[4]  # goes up to project root
        self.data_dir = self.base_dir / "data" / "raw" / "D2_plantvillage"
        self.results_dir = self.base_dir / "src" / "modules" / "disease_detection" / "eda" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_dataset_structure(self):
        """Analyze the folder structure and class distribution"""
        print("🌿 ANALYZING PLANTVILLAGE DATASET STRUCTURE")
        print("=" * 60)
        
        if not self.data_dir.exists():
            print(f"❌ Dataset path not found: {self.data_dir}")
            return None, None
        
        # Get all subdirectories (each represents a class)
        classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        print(f"📁 Number of classes: {len(classes)}")
        print(f"🌱 Classes: {classes[:10]}...")  # Show first 10 classes
        
        # Count images per class
        class_stats = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        total_images = 0
        
        for class_name in classes:
            class_path = self.data_dir / class_name
            images = [f for f in class_path.iterdir() if f.suffix in image_extensions]
            image_count = len(images)
            class_stats[class_name] = image_count
            total_images += image_count
        
        # Create summary dataframe
        stats_df = pd.DataFrame({
            'Class': list(class_stats.keys()),
            'Image_Count': list(class_stats.values())
        })
        stats_df = stats_df.sort_values('Image_Count', ascending=False)
        
        print(f"🖼️  Total images in dataset: {total_images}")
        print(f"📊 Class distribution summary:")
        print(f"   Max images per class: {stats_df['Image_Count'].max()}")
        print(f"   Min images per class: {stats_df['Image_Count'].min()}")
        print(f"   Average images per class: {stats_df['Image_Count'].mean():.1f}")
        
        return stats_df, class_stats
    
    def plot_class_distribution(self, stats_df):
        """Plot the distribution of images across classes"""
        plt.figure(figsize=(15, 8))
        
        # Plot top 30 classes for readability
        top_classes = stats_df.head(30)
        bars = plt.bar(range(len(top_classes)), top_classes['Image_Count'], color='lightgreen')
        
        plt.title('PlantVillage Dataset: Top 30 Classes by Image Count', fontsize=16, fontweight='bold')
        plt.xlabel('Plant Disease Classes', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(range(len(top_classes)), top_classes['Class'], rotation=90, fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'class_distribution_top30.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot overall distribution (all classes)
        plt.figure(figsize=(12, 6))
        plt.hist(stats_df['Image_Count'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Images per Class (All Classes)')
        plt.xlabel('Number of Images per Class')
        plt.ylabel('Frequency (Number of Classes)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'images_per_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
    
    def analyze_image_properties(self, class_stats, sample_size=5):
        """Analyze image dimensions and properties"""
        print("\n📊 ANALYZING IMAGE PROPERTIES")
        print("=" * 60)
        
        image_properties = []
        sampled_classes = list(class_stats.keys())[:sample_size]  # Sample first 5 classes
        
        for class_name in sampled_classes:
            class_path = self.data_dir / class_name
            image_files = [f for f in class_path.iterdir() if f.suffix in {'.jpg', '.jpeg', '.png'}]
            
            # Sample a few images from each class
            for img_path in image_files[:5]:  # First 5 images per class
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        format_type = img.format
                        
                        image_properties.append({
                            'Class': class_name,
                            'Filename': img_path.name,
                            'Width': width,
                            'Height': height,
                            'Aspect_Ratio': width/height,
                            'Mode': mode,
                            'Format': format_type,
                            'File_Size_KB': img_path.stat().st_size / 1024
                        })
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {e}")
        
        props_df = pd.DataFrame(image_properties)
        
        if len(props_df) > 0:
            print(f"📐 Analyzed {len(props_df)} images from {len(sampled_classes)} classes")
            print(f"📏 Image dimensions summary:")
            print(f"   Average size: {props_df['Width'].mean():.0f} x {props_df['Height'].mean():.0f}")
            print(f"   Min size: {props_df['Width'].min()} x {props_df['Height'].min()}")
            print(f"   Max size: {props_df['Width'].max()} x {props_df['Height'].max()}")
            print(f"   Common aspect ratios: {props_df['Aspect_Ratio'].value_counts().head(3)}")
            print(f"   Color modes: {props_df['Mode'].value_counts().to_dict()}")
            print(f"   File formats: {props_df['Format'].value_counts().to_dict()}")
        else:
            print("❌ No images were successfully processed")
        
        return props_df
    
    def plot_image_size_distribution(self, props_df):
        """Plot distribution of image sizes"""
        if len(props_df) == 0:
            print("❌ No image data to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Width distribution
        ax1.hist(props_df['Width'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Image Width (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Image Widths')
        ax1.grid(alpha=0.3)
        
        # Height distribution
        ax2.hist(props_df['Height'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Image Height (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Image Heights')
        ax2.grid(alpha=0.3)
        
        # Aspect ratio distribution
        ax3.hist(props_df['Aspect_Ratio'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Aspect Ratio (Width/Height)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Aspect Ratios')
        ax3.grid(alpha=0.3)
        
        # File size distribution
        ax4.hist(props_df['File_Size_KB'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_xlabel('File Size (KB)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of File Sizes')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'image_properties_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_disease_insights(self, stats_df):
        """Generate insights about disease types and frequencies"""
        print("\n🔍 DISEASE CATEGORY INSIGHTS")
        print("=" * 60)
        
        # Extract disease information from class names (PlantVillage format: Plant___Disease)
        stats_df['Plant_Type'] = stats_df['Class'].apply(
            lambda x: x.split('___')[0] if '___' in x else 'Unknown'
        )
        stats_df['Disease_Status'] = stats_df['Class'].apply(
            lambda x: x.split('___')[1] if '___' in x else 'Unknown'
        )
        
        plant_summary = stats_df.groupby('Plant_Type')['Image_Count'].sum().sort_values(ascending=False)
        disease_summary = stats_df.groupby('Disease_Status')['Image_Count'].sum().sort_values(ascending=False)
        
        print("🌿 Top 10 Plants by image count:")
        for plant, count in plant_summary.head(10).items():
            print(f"   {plant}: {count} images")
        
        print("\n🦠 Disease status distribution:")
        for disease, count in disease_summary.items():
            percentage = (count / disease_summary.sum()) * 100
            print(f"   {disease}: {count} images ({percentage:.1f}%)")
        
        # Plot plant type distribution
        plt.figure(figsize=(12, 6))
        plant_summary.head(10).plot(kind='bar', color='lightgreen')
        plt.title('Top 10 Plant Types by Image Count')
        plt.xlabel('Plant Type')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'plant_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot disease status distribution
        plt.figure(figsize=(10, 6))
        disease_summary.plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title('Disease Status Distribution')
        plt.ylabel('')  # Hide y-label for pie chart
        plt.tight_layout()
        plt.savefig(self.results_dir / 'disease_status_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
    
    def sample_and_display_images(self, class_stats, num_classes=5, images_per_class=3):
        """Display sample images from different classes"""
        print("\n🖼️  SAMPLE IMAGES FROM DATASET")
        print("=" * 60)
        
        sampled_classes = list(class_stats.keys())[:num_classes]
        
        fig, axes = plt.subplots(num_classes, images_per_class, figsize=(15, 3*num_classes))
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, class_name in enumerate(sampled_classes):
            class_path = self.data_dir / class_name
            image_files = [f for f in class_path.iterdir() if f.suffix in {'.jpg', '.jpeg', '.png'}]
            
            for j in range(images_per_class):
                if j < len(image_files):
                    try:
                        img = Image.open(image_files[j])
                        ax = axes[i, j] if num_classes > 1 else axes[j]
                        ax.imshow(img)
                        ax.set_title(f'{class_name}\n{img.size[0]}x{img.size[1]}', fontsize=8)
                        ax.axis('off')
                    except Exception as e:
                        print(f"❌ Error displaying {image_files[j]}: {e}")
                else:
                    # Hide axes if no image
                    ax = axes[i, j] if num_classes > 1 else axes[j]
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive EDA report"""
        print("🚀 STARTING COMPREHENSIVE EDA FOR PLANTVILLAGE DATASET")
        print("=" * 70)
        
        stats_df, class_stats = self.analyze_dataset_structure()
        if stats_df is None:
            return
        
        # Generate analyses
        stats_df = self.plot_class_distribution(stats_df)
        props_df = self.analyze_image_properties(class_stats)
        
        if len(props_df) > 0:
            self.plot_image_size_distribution(props_df)
        
        enhanced_stats_df = self.generate_disease_insights(stats_df)
        self.sample_and_display_images(class_stats)
        
        # Save summary statistics
        summary_path = self.results_dir / 'dataset_summary.csv'
        enhanced_stats_df.to_csv(summary_path, index=False)
        
        print(f"\n✅ PLANTVILLAGE EDA COMPLETED!")
        print(f"📊 Total classes: {len(class_stats)}")
        print(f"🖼️  Total images: {sum(class_stats.values())}")
        print(f"💾 Results saved to: {self.results_dir}")

# Run the analysis
if __name__ == "__main__":
    analyzer = PlantVillageAnalyzer()
    analyzer.generate_report()