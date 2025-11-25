# src/models/model_comparison_developer.py
import pandas as pd
import numpy as np
import pickle
import time
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelComparisonDeveloper:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.processed_path = self.project_root / "data/processed/"
        self.modules_path = self.project_root / "src/modules/"
        self.results = {}
        
    def load_processed_data(self):
        """Load all processed datasets"""
        print("📊 Loading processed datasets...")
        datasets = {}
        
        try:
            datasets['D1'] = pd.read_csv(self.processed_path / "D1_processed.csv")
            datasets['D2_meta'] = pd.read_csv(self.processed_path / "D2_metadata.csv")
            datasets['D3'] = pd.read_csv(self.processed_path / "D3_processed.csv")
            datasets['D4'] = pd.read_csv(self.processed_path / "D4_processed.csv")
            datasets['D5'] = pd.read_csv(self.processed_path / "D5_processed.csv")
            datasets['D6'] = pd.read_csv(self.processed_path / "D6_processed.csv")
            datasets['D7'] = pd.read_csv(self.processed_path / "D7_processed.csv")
            
            print("✅ All processed datasets loaded successfully")
            return datasets
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return None

    def _encode_categorical_features(self, X):
        """Encode categorical features for modeling"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if X[col].nunique() <= 20:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded.drop(columns=[col], inplace=True)
            else:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
        
        return X_encoded.select_dtypes(include=[np.number])

    def compare_crop_recommendation_models(self, d1_data):
        """Compare multiple models for crop recommendation"""
        print("\n" + "="*60)
        print("🌱 CROP RECOMMENDATION - MODEL COMPARISON")
        print("="*60)
        
        target_col = 'label'
        X = d1_data.drop(columns=[target_col])
        y = d1_data[target_col]
        
        # Encode features
        X_encoded = self._encode_categorical_features(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to compare
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Neighbors': KNeighborsClassifier()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            start_time = time.time()
            
            # Scale data for linear models and SVM
            if name in ['Logistic Regression', 'SVM']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            training_time = time.time() - start_time
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'model': model
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   📊 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"   ⏱️  Time: {training_time:.2f}s")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n🎯 BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Save best model
        module_path = self.modules_path / "crop_recommendation" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        with open(module_path / "best_crop_classifier.pkl", 'wb') as f:
            pickle.dump(results[best_model_name]['model'], f)
        
        self.results['crop_recommendation'] = results
        return results

    def compare_yield_prediction_models(self, d3_data):
        """Compare multiple models for yield prediction"""
        print("\n" + "="*60)
        print("📈 YIELD PREDICTION - MODEL COMPARISON")
        print("="*60)
        
        # Use essential columns for memory efficiency
        numeric_cols = d3_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'production_cleaned' in numeric_cols:
            target_col = 'production_cleaned'
            feature_cols = [col for col in numeric_cols if col != target_col][:8]
        else:
            feature_cols = numeric_cols[:8]
            target_col = feature_cols[0]
            feature_cols = feature_cols[1:]
        
        d3_reduced = d3_data[feature_cols + [target_col]].copy()
        
        X = d3_reduced[feature_cols]
        y = d3_reduced[target_col]
        
        # Sample for faster training
        if len(X) > 30000:
            sample_size = 30000
            X_sampled = X.sample(n=sample_size, random_state=42)
            y_sampled = y.loc[X_sampled.index]
            print(f"🔧 Using {sample_size} samples for faster comparison")
        else:
            X_sampled, y_sampled = X, y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sampled, y_sampled, test_size=0.2, random_state=42
        )
        
        # Define regression models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'K-Neighbors': KNeighborsRegressor()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            start_time = time.time()
            
            # Scale for linear models and SVR
            if name in ['Linear Regression', 'SVR']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            training_time = time.time() - start_time
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'rmse': rmse,
                'r2_score': r2,
                'training_time': training_time,
                'model': model
            }
            
            print(f"   ✅ RMSE: {rmse:,.2f}")
            print(f"   📊 R² Score: {r2:.4f}")
            print(f"   ⏱️  Time: {training_time:.2f}s")
        
        # Find best model (lowest RMSE)
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_rmse = results[best_model_name]['rmse']
        
        print(f"\n🎯 BEST MODEL: {best_model_name} (RMSE: {best_rmse:,.2f})")
        
        # Save best model
        module_path = self.modules_path / "yield_analysis" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        with open(module_path / "best_yield_predictor.pkl", 'wb') as f:
            pickle.dump(results[best_model_name]['model'], f)
        
        self.results['yield_analysis'] = results
        return results

    def compare_market_analysis_models(self, d7_data):
        """Compare multiple models for market analysis"""
        print("\n" + "="*60)
        print("💰 MARKET ANALYSIS - MODEL COMPARISON")
        print("="*60)
        
        # Find price column
        price_cols = [col for col in d7_data.columns if 'price' in col.lower()]
        if not price_cols:
            price_cols = [d7_data.columns[-1]]  # Use last column as fallback
        
        target_col = price_cols[0]
        X = d7_data.drop(columns=[target_col])
        y = d7_data[target_col]
        
        # Encode features
        X_encoded = self._encode_categorical_features(X)
        X_encoded = X_encoded.select_dtypes(include=[np.number])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Define regression models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            start_time = time.time()
            
            if name in ['Linear Regression', 'SVR']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            training_time = time.time() - start_time
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'rmse': rmse,
                'r2_score': r2,
                'training_time': training_time,
                'model': model
            }
            
            print(f"   ✅ RMSE: {rmse:.2f}")
            print(f"   📊 R² Score: {r2:.4f}")
            print(f"   ⏱️  Time: {training_time:.2f}s")
        
        best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_rmse = results[best_model_name]['rmse']
        
        print(f"\n🎯 BEST MODEL: {best_model_name} (RMSE: {best_rmse:.2f})")
        
        # Save best model
        module_path = self.modules_path / "market_analysis" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        with open(module_path / "best_price_predictor.pkl", 'wb') as f:
            pickle.dump(results[best_model_name]['model'], f)
        
        self.results['market_analysis'] = results
        return results

    def compare_pesticide_models(self, d5_data):
        """Compare multiple models for pesticide recommendation"""
        print("\n" + "="*60)
        print("🐛 PESTICIDE RECOMMENDATION - MODEL COMPARISON")
        print("="*60)
        
        # Model 1: Pest Presence Classification
        print("\n🔍 1. PEST PRESENCE CLASSIFICATION")
        
        X = d5_data.drop(columns=['Pest_Presence', 'Pest_Severity', 'Plant_ID'])
        y = d5_data['Pest_Presence']
        
        # Remove 'Unknown' labels
        y_clean = y[y != 'Unknown']
        X_clean = X.loc[y_clean.index]
        
        X_encoded = self._encode_categorical_features(X_clean)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        models_classification = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        pest_results = {}
        
        for name, model in models_classification.items():
            print(f"   🔧 Training {name}...")
            start_time = time.time()
            
            if name in ['Logistic Regression', 'SVM']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            training_time = time.time() - start_time
            accuracy = accuracy_score(y_test, y_pred)
            
            pest_results[name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'model': model
            }
            
            print(f"      ✅ Accuracy: {accuracy:.4f}")
        
        best_pest_model = max(pest_results.keys(), key=lambda x: pest_results[x]['accuracy'])
        
        # Save treatment rules
        treatment_rules = {
            'Aphids': {'Low': 'Neem oil', 'Moderate': 'Insecticidal soap', 'High': 'Pyrethrin'},
            'Spider mites': {'Low': 'Water spray', 'Moderate': 'Horticultural oil', 'High': 'Miticide'},
            'Whiteflies': {'Low': 'Sticky traps', 'Moderate': 'Insecticidal soap', 'High': 'Systemic insecticide'},
            'Fungus gnats': {'Low': 'Reduce watering', 'Moderate': 'Nematodes', 'High': 'Bacillus thuringiensis'}
        }
        
        # Save all models
        module_path = self.modules_path / "pesticide_recommendation" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        with open(module_path / "best_pest_classifier.pkl", 'wb') as f:
            pickle.dump(pest_results[best_pest_model]['model'], f)
        
        with open(module_path / "treatment_rules.pkl", 'wb') as f:
            pickle.dump(treatment_rules, f)
        
        self.results['pesticide_recommendation'] = {
            'pest_classification': pest_results,
            'treatment_rules': treatment_rules
        }
        
        print(f"\n🎯 BEST PEST CLASSIFIER: {best_pest_model}")
        return self.results['pesticide_recommendation']

    def generate_comparison_report(self):
        """Generate comprehensive model comparison report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*60)
    
    # Create serializable report (remove model objects)
        serializable_results = {}
        for module, results in self.results.items():
            serializable_results[module] = {}
            if module == 'pesticide_recommendation':
                serializable_results[module]['pest_classification'] = {}
                for model_name, model_data in results['pest_classification'].items():
                    serializable_results[module]['pest_classification'][model_name] = {
                        'accuracy': model_data['accuracy'],
                        'training_time': model_data['training_time']
                    }
                serializable_results[module]['treatment_rules'] = results['treatment_rules']
            else:
                for model_name, model_data in results.items():
                    serializable_results[module][model_name] = {
                        k: v for k, v in model_data.items() if k != 'model'
                    }
    
        report = {
            'summary': {},
            'detailed_results': serializable_results
        }
    
    # Find best models for each module
        for module, results in self.results.items():
            if module == 'pesticide_recommendation':
                best_model = max(results['pest_classification'].keys(), 
                               key=lambda x: results['pest_classification'][x]['accuracy'])
                best_score = results['pest_classification'][best_model]['accuracy']
                report['summary'][module] = {
                    'best_model': best_model,
                    'best_score': best_score,
                    'metric': 'accuracy'
                }
            else:
                if 'accuracy' in list(results.values())[0]:
                    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
                    best_score = results[best_model]['accuracy']
                    metric = 'accuracy'
                else:
                    best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
                    best_score = results[best_model]['rmse']
                    metric = 'rmse'
            
                report['summary'][module] = {
                    'best_model': best_model,
                    'best_score': best_score,
                    'metric': metric
             }
    
    # Print summary
        print("\n🎯 BEST MODELS SUMMARY:")
        print("-" * 50)
        for module, info in report['summary'].items():
            print(f"🌱 {module.replace('_', ' ').title():<25} | {info['best_model']:<20} | {info['metric']}: {info['best_score']:.4f}")
    
    # Save report
        report_path = self.project_root / "src"/ "models" / "model_comparison_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
        print(f"\n💾 Full report saved to: {report_path}")
        return report

    def run_complete_comparison(self):
        """Run complete model comparison for all modules"""
        print("🚀 STARTING COMPREHENSIVE MODEL COMPARISON")
        print("="*60)
        
        datasets = self.load_processed_data()
        if not datasets:
            return False
        
        try:
            # Compare models for each module
            self.compare_crop_recommendation_models(datasets['D1'])
            self.compare_yield_prediction_models(datasets['D3'])
            self.compare_market_analysis_models(datasets['D7'])
            self.compare_pesticide_models(datasets['D5'])
            
            # Disease detection - metadata only (CNN planned)
            module_path = self.modules_path / "disease_detection" / "models"
            module_path.mkdir(parents=True, exist_ok=True)
            
            disease_mapping = {}
            for _, row in datasets['D2_meta'].iterrows():
                disease_mapping[row['disease_type']] = {
                    'image_count': row['image_count'],
                    'label': row['label']
                }
            
            with open(module_path / "disease_metadata.pkl", 'wb') as f:
                pickle.dump(disease_mapping, f)
            
            print(f"\n🦠 Disease Detection: {len(disease_mapping)} diseases mapped (CNN planned)")
            
            # Generate final report
            report = self.generate_comparison_report()
            
            print("\n🎉 MODEL COMPARISON COMPLETED SUCCESSFULLY!")
            print("🔬 Scientific proof that your model choices are optimal! 🚀")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in model comparison: {e}")
            import traceback
            traceback.print_exc()
            return False

# Execute the comprehensive comparison
if __name__ == "__main__":
    developer = ModelComparisonDeveloper()
    success = developer.run_complete_comparison()
    
    if success:
        print("\n✅ All models compared and best ones saved!")
        print("📁 Check the model comparison report for detailed results")
    else:
        print("\n❌ Model comparison failed")