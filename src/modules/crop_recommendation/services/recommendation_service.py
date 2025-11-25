import pandas as pd
import joblib
import os
import sys
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Remove the problematic import and create a local version
def load_crop_data(file_path):
    """Local version of data loading function"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✅ Loaded dataset from: {file_path}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

class CropRecommendationService:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # ✅ CONFIGURED PATHS - Using your D1 dataset
        self.default_dataset_path = "data/raw/D1_crop_recommendation.csv"
        self.default_model_path = "src/models/crop_model.pkl"
        
        self._load_existing_model()
    
    def _load_existing_model(self):
        """Load existing trained model"""
        try:
            if os.path.exists(self.default_model_path):
                self.model = joblib.load(self.default_model_path)
                self.is_trained = True
                logger.info(f"✅ Loaded existing model from: {self.default_model_path}")
            else:
                logger.warning("⚠️ No existing model found. Train model first.")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
    
    def _create_sample_data(self):
        """Create sample data for demonstration (fallback)"""
        sample_data = {
            'N': [90, 85, 80, 70, 60, 50, 40, 30, 20, 10],
            'P': [40, 45, 50, 35, 30, 25, 20, 15, 10, 5],
            'K': [40, 45, 50, 35, 30, 25, 20, 15, 10, 5],
            'temperature': [25, 22, 28, 20, 26, 23, 29, 19, 27, 18],
            'humidity': [80, 70, 85, 65, 75, 72, 82, 68, 78, 60],
            'ph': [6.0, 6.5, 5.8, 7.0, 6.2, 7.2, 5.5, 7.5, 5.2, 7.8],
            'rainfall': [200, 150, 250, 100, 180, 120, 220, 90, 240, 80],
            'label': ['rice', 'wheat', 'maize', 'cotton', 'jute', 'tea', 'coffee', 'sugarcane', 'apple', 'banana']
        }
        return pd.DataFrame(sample_data)
    
    def load_dataset(self, data_path=None):
        """Load dataset from path or use sample data as fallback"""
        if data_path is None:
            data_path = self.default_dataset_path
            
        try:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                logger.info(f"✅ Loaded dataset from: {data_path}")
                logger.info(f"📊 Dataset shape: {df.shape}")
                logger.info(f"📋 Columns: {df.columns.tolist()}")
                return df
            else:
                logger.warning(f"⚠️ Dataset not found at: {data_path}")
                logger.info("📊 Using sample data for training/demo")
                return self._create_sample_data()
        except Exception as e:
            logger.error(f"❌ Error loading data from {data_path}: {e}")
            logger.info("📊 Falling back to sample data")
            return self._create_sample_data()
    
    def train_model(self, data_path=None, save_path=None):
        """Train the crop recommendation model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # ✅ USE CONFIGURED PATHS
        if data_path is None:
            data_path = self.default_dataset_path
        
        if save_path is None:
            save_path = self.default_model_path
        
        try:
            # Load data
            logger.info(f"🔍 Loading dataset from: {data_path}")
            df = self.load_dataset(data_path)
            
            # Check if dataset has required features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                logger.error(f"❌ Missing features in dataset: {missing_features}")
                return {
                    'success': False, 
                    'error': f'Dataset missing required features: {missing_features}'
                }
            
            if 'label' not in df.columns:
                logger.error("❌ Dataset missing 'label' column")
                return {'success': False, 'error': "Dataset missing 'label' column"}
            
            # Prepare features and target
            X = df[self.features]
            y = df['label']
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            
            logger.info(f"🎯 Model trained successfully!")
            logger.info(f"📈 Accuracy: {accuracy:.4f}")
            logger.info(f"📁 Model saved to: {save_path}")
            logger.info(f"📊 Training samples: {len(X_train)}")
            logger.info(f"📊 Test samples: {len(X_test)}")
            logger.info(f"🌱 Unique crops: {len(y.unique())}")
            
            return {
                'success': True,
                'accuracy': float(accuracy),
                'model_path': save_path,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(X),
                'unique_crops': len(y.unique()),
                'features': self.features,
                'dataset_used': data_path
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_recommendation(self, input_data):
        """Get crop recommendation"""
        if not self.is_trained or self.model is None:
            return {
                'success': False,
                'error': 'Model not trained. Please train the model first using /api/crop/train'
            }
        
        try:
            # Validate input
            if len(input_data) != len(self.features):
                return {
                    'success': False,
                    'error': f'Expected {len(self.features)} parameters: {self.features}',
                    'received': len(input_data)
                }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data], columns=self.features)
            
            # Predict
            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            confidence = np.max(probabilities)
            
            # Get top 3 recommendations
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            recommendations = []
            
            for i, idx in enumerate(top_3_indices):
                recommendations.append({
                    'crop': self.model.classes_[idx],
                    'confidence': float(probabilities[idx]),
                    'rank': i + 1
                })
            
            return {
                'success': True,
                'top_recommendation': recommendations[0],
                'alternative_recommendations': recommendations[1:],
                'all_recommendations': recommendations,
                'input_parameters': dict(zip(self.features, input_data)),
                'model_confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_service_info(self):
        """Get information about this service"""
        return {
            'module': 'crop_recommendation',
            'description': 'Recommends optimal crops based on soil and weather conditions',
            'model_trained': self.is_trained,
            'features_used': self.features,
            'model_path': self.default_model_path,
            'dataset_path': self.default_dataset_path,
            'dataset_exists': os.path.exists(self.default_dataset_path)
        }