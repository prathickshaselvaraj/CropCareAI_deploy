"""Crop Recommendation Module Configuration"""
CROP_CONFIG = {
    'module_name': 'crop_recommendation',
    'description': 'Recommends optimal crops based on soil and weather conditions',
    'datasets': {
        'primary': '../data/raw/Dataset1.csv',  # Your existing dataset path
        'secondary': '../data/raw/Dataset4.csv' # Your existing dataset path
    },
    'features': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
    'target': 'label',
    'model_path': '../models/crop_model.pkl'  # Your existing model path
}