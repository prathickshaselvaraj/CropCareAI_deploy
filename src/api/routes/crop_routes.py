"""API routes for crop recommendation"""
from flask import Blueprint, request, jsonify
import sys
import os
from pathlib import Path

# Add the project root to Python path
BASE_DIR = Path(__file__).parent.parent.parent.parent  # Goes up to CropCareAI/
sys.path.insert(0, str(BASE_DIR))

try:
    # Try absolute import
    from src.modules.crop_recommendation.services.recommendation_service import CropRecommendationService
    print("✅ Import successful using absolute path")
except ImportError:
    try:
        # Fallback: relative import
        from src.modules.crop_recommendation.services.recommendation_service import CropRecommendationService
        print("✅ Import successful using relative path")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        # Create a simple fallback class
        class CropRecommendationService:
            def __init__(self):
                self.is_trained = False
            
            def get_recommendation(self, input_data):
                return {'success': False, 'error': 'Service not properly imported'}
            
            def train_model(self):
                return {'success': False, 'error': 'Service not properly imported'}
            
            def get_service_info(self):
                return {'module': 'crop_recommendation', 'status': 'import_error'}

crop_bp = Blueprint('crop', __name__)
service = CropRecommendationService()

@crop_bp.route('/recommend', methods=['POST'])
def recommend_crop():
    """Crop recommendation endpoint"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        # Define expected features
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = []
        
        for feature in features:
            if feature not in data:
                return jsonify({
                    'success': False, 
                    'error': f'Missing parameter: {feature}',
                    'required_parameters': features
                }), 400
            input_data.append(data[feature])
        
        result = service.get_recommendation(input_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@crop_bp.route('/train', methods=['POST'])
def train_model():
    """Train the crop recommendation model"""
    try:
        data = request.json or {}
        data_path = data.get('data_path')
        
        result = service.train_model(data_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@crop_bp.route('/info', methods=['GET'])
def get_info():
    """Get service information"""
    return jsonify(service.get_service_info())

@crop_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for crop module"""
    return jsonify({
        'status': 'healthy',
        'module': 'crop_recommendation',
        'model_loaded': service.is_trained
    })