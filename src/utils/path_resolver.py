"""Utility to handle paths in your existing structure"""
import os
from pathlib import Path

def get_data_path(filename):
    """Get path to your existing data files"""
    base_dir = Path(__file__).parent.parent.parent  # Project root
    return base_dir / 'data' / 'raw' / filename

def get_model_path(model_name):
    """Get path to your existing model files"""
    base_dir = Path(__file__).parent.parent.parent  # Project root
    return base_dir / 'src' / 'models' / model_name

# Example usage:
# dataset_path = get_data_path('Dataset1.csv')
# model_path = get_model_path('crop_model.pkl')