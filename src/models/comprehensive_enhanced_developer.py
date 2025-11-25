# src/models/final_fixed_developer.py
import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class FinalFixedDeveloper:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.processed_path = self.project_root / "data/processed/"
        self.modules_path = self.project_root / "src/modules/"
        self.results = {}
        self.best_models = {}
        
    def load_all_datasets(self):
        """Load ALL 7 processed datasets"""
        print("📊 LOADING ALL 7 DATASETS...")
        datasets = {}
        
        try:
            datasets['D1'] = pd.read_csv(self.processed_path / "D1_processed.csv")
            datasets['D2_meta'] = pd.read_csv(self.processed_path / "D2_metadata.csv")
            datasets['D3'] = pd.read_csv(self.processed_path / "D3_processed.csv")
            datasets['D4'] = pd.read_csv(self.processed_path / "D4_processed.csv")
            datasets['D5'] = pd.read_csv(self.processed_path / "D5_processed.csv")
            datasets['D6'] = pd.read_csv(self.processed_path / "D6_processed.csv")
            datasets['D7'] = pd.read_csv(self.processed_path / "D7_processed.csv")
            
            print("✅ ALL 7 datasets loaded successfully")
            return datasets
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return None

    def develop_final_pesticide_models_fixed(self, d5_data, d6_data):
        """FIXED pest detection with proper multi-class handling"""
        print("\n" + "="*60)
        print("🐛 FIXED PESTICIDE RECOMMENDATION")
        print("="*60)
        
        # Use both D5 and D6 for comprehensive analysis
        d5_enhanced = self.final_feature_engineering(d5_data, 'D5')
        d6_enhanced = self.final_feature_engineering(d6_data, 'D6')
        
        print("🔧 Using D5 (indoor) + D6 (farming) for comprehensive pest analysis")
        
        # FIX: Check if we should use D6 instead or combine datasets
        if 'Pest_Presence' in d5_enhanced.columns:
            print("🔧 Using D5 for pest classification...")
            
            # Prepare features - exclude non-predictive columns
            exclude_cols = ['Pest_Presence', 'Pest_Severity', 'Plant_ID', 'Health_Notes']
            feature_cols = [col for col in d5_enhanced.columns if col not in exclude_cols]
            
            X = d5_enhanced[feature_cols]
            y = d5_enhanced['Pest_Presence']
            
            print(f"🔧 Original class distribution:\n{y.value_counts()}")
            
            # FIX: Keep multiple pest types but handle "Unknown" differently
            mask = y != 'Unknown'
            X_clean = X[mask]
            y_clean = y[mask]
            
            print(f"🔧 After removing 'Unknown':\n{y_clean.value_counts()}")
            
            # Check if we have multiple classes
            unique_classes = y_clean.unique()
            print(f"🔧 Unique classes: {unique_classes}")
            
            if len(unique_classes) <= 1:
                print("⚠️  Only one class remaining. Trying alternative approach...")
                
                # Alternative: Use Pest_Severity or create synthetic data
                if 'Pest_Severity' in d5_enhanced.columns:
                    print("🔧 Using Pest_Severity as target instead...")
                    y_clean = d5_enhanced.loc[mask, 'Pest_Severity']
                    print(f"🔧 Pest_Severity distribution:\n{y_clean.value_counts()}")
                else:
                    print("❌ Cannot proceed with pest classification - insufficient class diversity")
                    return {}
            
            # Advanced feature engineering
            X_engineered = self.create_final_features(X_clean, 'D5')
            
            # Handle class imbalance only if we have multiple classes
            if len(y_clean.unique()) > 1:
                # Use SMOTE-like manual balancing
                from sklearn.utils import resample
                
                # Combine features and target
                df_combined = pd.concat([X_engineered, y_clean], axis=1)
                
                # Get value counts for each class
                class_counts = y_clean.value_counts()
                max_size = class_counts.max()
                
                # Resample each class
                dfs_resampled = []
                for class_name in y_clean.unique():
                    class_df = df_combined[df_combined['Pest_Presence'] == class_name]
                    if len(class_df) < max_size:
                        # Upsample minority class
                        class_resampled = resample(
                            class_df,
                            replace=True,
                            n_samples=max_size,
                            random_state=42
                        )
                        dfs_resampled.append(class_resampled)
                    else:
                        dfs_resampled.append(class_df)
                
                # Combine resampled classes
                df_balanced = pd.concat(dfs_resampled)
                X_balanced = df_balanced.drop(columns=['Pest_Presence'])
                y_balanced = df_balanced['Pest_Presence']
                
                print(f"🔧 Balanced class distribution:\n{y_balanced.value_counts()}")
            else:
                X_balanced, y_balanced = X_engineered, y_clean
                print("⚠️  Using original data (single class or no balancing needed)")
            
            # Only proceed if we have multiple classes
            if len(y_balanced.unique()) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
                )
                
                # FINAL OPTIMIZED models for pest detection
                models = {
                    'Random Forest': RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        class_weight='balanced',
                        random_state=42
                    ),
                    'Gradient Boosting': GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42
                    ),
                    'Logistic Regression': LogisticRegression(
                        max_iter=2000,
                        class_weight='balanced',
                        random_state=42,
                        solver='liblinear',
                        C=0.5
                    )
                }
                
                pest_results = {}
                
                for name, model in models.items():
                    print(f"🔧 Training {name} for pest detection...")
                    start_time = time.time()
                    
                    try:
                        if name == 'Logistic Regression':
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
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        pest_results[name] = {
                            'accuracy': accuracy,
                            'f1_score': f1,
                            'precision': precision,
                            'recall': recall,
                            'training_time': training_time,
                            'model': model
                        }
                        
                        print(f"   ✅ Accuracy: {accuracy:.4f}")
                        print(f"   🎯 F1-Score: {f1:.4f}")
                        print(f"   📊 Precision: {precision:.4f}")
                        print(f"   📈 Recall: {recall:.4f}")
                        
                    except Exception as e:
                        print(f"   ❌ {name} failed: {e}")
                        continue
                
                if not pest_results:
                    print("❌ All pest classification models failed")
                    return {}
                
                best_pest_model = max(pest_results.keys(), key=lambda x: pest_results[x]['f1_score'])
                best_accuracy = pest_results[best_pest_model]['accuracy']
                
                print(f"\n🎯 BEST PEST CLASSIFIER: {best_pest_model}")
                print(f"📊 Best Accuracy: {best_accuracy:.4f}")
                print(f"🎯 Best F1-Score: {pest_results[best_pest_model]['f1_score']:.4f}")
                
                # Enhanced treatment rules
                treatment_rules = self.create_comprehensive_treatment_rules()
                
                # Save models and rules
                module_path = self.modules_path / "pesticide_recommendation" / "models"
                module_path.mkdir(parents=True, exist_ok=True)
                
                with open(module_path / "final_pest_classifier.pkl", 'wb') as f:
                    pickle.dump(pest_results[best_pest_model]['model'], f)
                
                with open(module_path / "treatment_rules.pkl", 'wb') as f:
                    pickle.dump(treatment_rules, f)
                
                # Save feature names
                with open(module_path / "feature_names.pkl", 'wb') as f:
                    pickle.dump(list(X_engineered.columns), f)
                
                self.best_models['pesticide_recommendation'] = pest_results[best_pest_model]['model']
                self.results['pesticide_recommendation'] = {
                    'pest_classification': pest_results,
                    'treatment_rules': treatment_rules
                }
                
                return self.results['pesticide_recommendation']
            else:
                print("❌ Cannot train classifier with only one class")
                return {}
        
        return {}

    def create_comprehensive_treatment_rules(self):
        """Create comprehensive pesticide treatment rules"""
        return {
            'Aphids': {
                'Low': {
                    'treatment': 'Neem oil spray', 
                    'dosage': '2-3ml/L', 
                    'frequency': 'Weekly', 
                    'prevention': 'Introduce ladybugs, use reflective mulch',
                    'organic': True
                },
                'Moderate': {
                    'treatment': 'Insecticidal soap + Neem oil', 
                    'dosage': '5ml/L each', 
                    'frequency': 'Every 5 days', 
                    'prevention': 'Remove affected leaves, improve air circulation',
                    'organic': True
                },
                'High': {
                    'treatment': 'Pyrethrin-based insecticide', 
                    'dosage': '1ml/L', 
                    'frequency': 'Every 3 days', 
                    'prevention': 'Use row covers, remove heavily infested plants',
                    'organic': False
                }
            },
            'Spider Mites': {
                'Low': {
                    'treatment': 'Water spray (underside of leaves)', 
                    'dosage': 'N/A', 
                    'frequency': 'Daily', 
                    'prevention': 'Increase humidity, mist regularly',
                    'organic': True
                },
                'Moderate': {
                    'treatment': 'Horticultural oil', 
                    'dosage': '10-15ml/L', 
                    'frequency': 'Weekly', 
                    'prevention': 'Introduce predatory mites, maintain humidity',
                    'organic': True
                },
                'High': {
                    'treatment': 'Miticide (Abamectin)', 
                    'dosage': '2ml/L', 
                    'frequency': 'Every 5-7 days', 
                    'prevention': 'Isolate infected plants, improve ventilation',
                    'organic': False
                }
            },
            'Whiteflies': {
                'Low': {
                    'treatment': 'Yellow sticky traps', 
                    'dosage': 'N/A', 
                    'frequency': 'Continuous', 
                    'prevention': 'Companion planting with marigolds',
                    'organic': True
                },
                'Moderate': {
                    'treatment': 'Insecticidal soap + Sticky traps', 
                    'dosage': '5ml/L', 
                    'frequency': 'Weekly', 
                    'prevention': 'Use reflective mulch, remove weeds',
                    'organic': True
                },
                'High': {
                    'treatment': 'Systemic insecticide', 
                    'dosage': 'As per label', 
                    'frequency': 'As needed', 
                    'prevention': 'Greenhouse screening, crop rotation',
                    'organic': False
                }
            },
            'Fungus Gnats': {
                'Low': {
                    'treatment': 'Reduce watering frequency', 
                    'dosage': 'N/A', 
                    'frequency': 'Immediate', 
                    'prevention': 'Allow soil to dry between watering',
                    'organic': True
                },
                'Moderate': {
                    'treatment': 'Beneficial nematodes', 
                    'dosage': 'As per instructions', 
                    'frequency': 'Once', 
                    'prevention': 'Use sterile potting mix, avoid overwatering',
                    'organic': True
                },
                'High': {
                    'treatment': 'Bacillus thuringiensis (Bt)', 
                    'dosage': 'As per label', 
                    'frequency': 'Weekly', 
                    'prevention': 'Bottom watering, sand topping',
                    'organic': True
                }
            },
            'General Prevention': {
                'tips': [
                    'Regularly inspect plants for early signs of pests',
                    'Maintain proper plant spacing for air circulation',
                    'Avoid over-fertilization which attracts pests',
                    'Use companion planting strategies (marigolds, basil, mint)',
                    'Keep growing area clean and free of debris',
                    'Introduce beneficial insects (ladybugs, lacewings)',
                    'Use row covers for vulnerable plants',
                    'Practice crop rotation in farming systems',
                    'Monitor humidity and temperature levels',
                    'Quarantine new plants before introducing to main area'
                ],
                'early_detection': [
                    'Check undersides of leaves weekly',
                    'Look for discoloration or spots on leaves',
                    'Monitor for sticky residue (honeydew)',
                    'Watch for wilting or stunted growth',
                    'Use yellow sticky traps for flying insects'
                ]
            }
        }

    def improve_crop_recommendation(self, d1_data, d4_data):
        """Further improved crop recommendation with feature selection"""
        print("\n" + "="*60)
        print("🌱 IMPROVED CROP RECOMMENDATION")
        print("="*60)
        
        # Enhanced feature engineering
        d1_enhanced = self.final_feature_engineering(d1_data, 'D1')
        
        target_col = self._find_target_column(d1_enhanced, ['label', 'crop'])
        X = d1_enhanced.drop(columns=[target_col])
        y = d1_enhanced[target_col]
        
        print(f"🔧 Dataset shape: {X.shape}")
        print(f"🔧 Target classes: {len(y.unique())}")
        
        # Advanced feature selection
        X_engineered = self.create_final_features(X, 'D1')
        
        # Use feature selection to improve performance
        from sklearn.feature_selection import SelectFromModel
        
        # Use RandomForest for feature selection
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )
        X_selected = selector.fit_transform(X_engineered, y)
        selected_features = X_engineered.columns[selector.get_support()]
        
        print(f"🔧 Selected {len(selected_features)} most important features")
        
        X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_engineered.index)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Focus on best performing models
        models = {
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            training_time = time.time() - start_time
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_final, y, cv=5, scoring='accuracy')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   🎯 F1-Score: {f1:.4f}")
            print(f"   🔁 CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        
        print(f"\n🎯 BEST MODEL: {best_model_name}")
        print(f"📊 Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        # Save model
        module_path = self.modules_path / "crop_recommendation" / "models"
        module_path.mkdir(parents=True, exist_ok=True)
        
        with open(module_path / "improved_crop_model.pkl", 'wb') as f:
            pickle.dump(results[best_model_name]['model'], f)
        
        # Save feature names
        with open(module_path / "feature_names.pkl", 'wb') as f:
            pickle.dump(list(X_final.columns), f)
        
        self.results['crop_recommendation'] = results
        return results

    def final_feature_engineering(self, data, dataset_name):
        """Final optimized feature engineering"""
        data_clean = self._clean_data_final(data)
        data_eng = data_clean.copy()
        
        # Enhanced feature engineering based on dataset
        if dataset_name == 'D1':
            # Advanced soil and climate features
            if all(col in data_clean.columns for col in ['N', 'P', 'K']):
                data_eng['soil_health_score'] = (
                    data_clean['N'] * 0.4 + data_clean['P'] * 0.35 + data_clean['K'] * 0.25
                )
                data_eng['nutrient_balance_score'] = 100 - (
                    abs(data_clean['N'] - data_clean['P']) + 
                    abs(data_clean['P'] - data_clean['K']) + 
                    abs(data_clean['K'] - data_clean['N'])
                ) / 3
                
        elif dataset_name == 'D5':
            # Enhanced plant health features
            if all(col in data_clean.columns for col in ['Height_cm', 'Leaf_Count', 'Health_Score']):
                data_eng['plant_vigor'] = (
                    data_clean['Height_cm'] * 0.4 + 
                    data_clean['Leaf_Count'] * 0.3 + 
                    data_clean['Health_Score'] * 0.3
                )
        
        return data_eng

    def create_final_features(self, X, dataset_name):
        """Create final optimized features"""
        X_encoded = self._encode_features_final(X)
        
        # Ensure no NaN or infinite values
        X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
        X_encoded = X_encoded.fillna(X_encoded.median())
        
        return X_encoded

    def _clean_data_final(self, data):
        """Final optimized data cleaning"""
        data_clean = data.copy()
        
        # Handle infinite values
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data_clean[col] = data_clean[col].replace([np.inf, -np.inf], np.nan)
        
        # Use median for numeric, mode for categorical
        for col in data_clean.columns:
            if data_clean[col].isna().sum() > 0:
                if data_clean[col].dtype in ['int64', 'float64']:
                    data_clean[col].fillna(data_clean[col].median(), inplace=True)
                else:
                    data_clean[col].fillna(data_clean[col].mode()[0] if not data_clean[col].mode().empty else 'Unknown', inplace=True)
        
        return data_clean

    def _encode_features_final(self, X):
        """Final optimized feature encoding"""
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if X_encoded[col].nunique() <= 15:  # One-hot for low cardinality
                dummies = pd.get_dummies(X_encoded[col], prefix=col)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded.drop(columns=[col], inplace=True)
            else:  # Label encode for high cardinality
                le = LabelEncoder()
                X_encoded[col] = X_encoded[col].fillna('Unknown')
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        # Ensure all columns are numeric
        X_encoded = X_encoded.select_dtypes(include=[np.number])
        
        return X_encoded

    def _find_target_column(self, data, possible_names):
        """Find target column in data"""
        for name in possible_names:
            if name in data.columns:
                return name
        return data.columns[-1]

    def run_final_fixed_development(self):
        """Run final fixed model development"""
        print("🚀 FINAL FIXED MODEL DEVELOPMENT")
        print("="*60)
        print("🔧 Fixing pest detection and improving crop recommendation")
        print("="*60)
        
        datasets = self.load_all_datasets()
        if not datasets:
            return False
        
        try:
            # Run improved models
            self.improve_crop_recommendation(datasets['D1'], datasets['D4'])
            self.develop_final_pesticide_models_fixed(datasets['D5'], datasets['D6'])
            
            # Save disease metadata
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
            
            print(f"\n🦠 Disease Detection: {len(disease_mapping)} diseases mapped")
            
            # Print performance summary
            print("\n" + "="*60)
            print("📊 FINAL PERFORMANCE SUMMARY")
            print("="*60)
            
            for module, results in self.results.items():
                if module == 'crop_recommendation':
                    best_acc = max([r['accuracy'] for r in results.values()])
                    best_f1 = max([r['f1_score'] for r in results.values()])
                    print(f"🌱 {module.replace('_', ' ').title()}:")
                    print(f"   ✅ Best Accuracy: {best_acc:.4f}")
                    print(f"   🎯 Best F1-Score: {best_f1:.4f}")
                elif module == 'pesticide_recommendation':
                    pest_results = results['pest_classification']
                    best_acc = max([r['accuracy'] for r in pest_results.values()])
                    best_f1 = max([r['f1_score'] for r in pest_results.values()])
                    print(f"🐛 {module.replace('_', ' ').title()}:")
                    print(f"   ✅ Best Accuracy: {best_acc:.4f}")
                    print(f"   🎯 Best F1-Score: {best_f1:.4f}")
            
            print("\n🎉 FINAL FIXED MODELS COMPLETED!")
            print("📁 All models saved with fixes and improvements")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

# Run final fixed development
if __name__ == "__main__":
    developer = FinalFixedDeveloper()
    success = developer.run_final_fixed_development()
    
    if success:
        print("\n✅ ALL FINAL FIXED MODELS CREATED SUCCESSFULLY!")
        print("🎯 Key improvements:")
        print("   🐛 Fixed pest detection multi-class issue")
        print("   🌱 Improved crop recommendation with feature selection")
        print("   📋 Enhanced treatment rules with organic options")
    else:
        print("\n❌ Final fixed development failed")