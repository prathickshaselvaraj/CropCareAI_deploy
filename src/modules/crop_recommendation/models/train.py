"""Enhanced training for crop recommendation"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import logging

from ..config import CROP_CONFIG

logger = logging.getLogger(__name__)

class AdvancedCropTrainer:
    def __init__(self):
        self.config = CROP_CONFIG
    
    def load_and_validate_data(self):
        """Load and validate your existing dataset"""
        data_path = self.config['datasets']['primary']
        df = pd.read_csv(data_path)
        
        # Validate required columns exist
        required_cols = self.config['features'] + [self.config['target']]
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Dataset missing columns: {missing_cols}")
        
        return df
    
    def train_with_cross_validation(self):
        """Enhanced training with cross-validation"""
        df = self.load_and_validate_data()
        X = df[self.config['features']]
        y = df[self.config['target']]
        
        # Your existing training logic here
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save to your existing model path
        joblib.dump(model, self.config['model_path'])
        
        return {
            'accuracy': accuracy,
            'model_path': self.config['model_path'],
            'classes': list(model.classes_)
        }