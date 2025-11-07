import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from Heart_Disease_Prediction.logger.log import log

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.create_directories()
    
    def create_directories(self):
        """Create all necessary directories for model training"""
        try:
            # Create main directories (skip if they exist)
            os.makedirs(self.config.root_dir, exist_ok=True)
            os.makedirs(self.config.trained_data_dir, exist_ok=True)
            os.makedirs(self.config.tested_data_dir, exist_ok=True)
            
            # For transformed_data_path, check if it's a directory or file
            transformed_path = self.config.transformed_data_path
            if transformed_path.exists():
                if transformed_path.is_file():
                    # If it's a file, get its parent directory
                    os.makedirs(transformed_path.parent, exist_ok=True)
                else:
                    # If it's a directory, create it
                    os.makedirs(transformed_path, exist_ok=True)
            else:
                # If path doesn't exist, create as directory
                os.makedirs(transformed_path, exist_ok=True)
                
            log.info("All directories created/verified successfully")
            
        except Exception as e:
            log.warning(f"Directory creation warning: {e}")
    
    def load_and_split_data(self):
        """Load transformed data and perform train-test split"""
        try:
            # Load the transformed data - use the correct path structure
            data_path = self.config.transformed_data_path / "transformed_data.csv"
            log.info(f"Loading transformed data from: {data_path}")
            
            # Check if file exists
            if not data_path.exists():
                raise FileNotFoundError(f"Transformed data file not found at: {data_path}")
            
            data = pd.read_csv(data_path)
            log.info(f"Data loaded successfully. Shape: {data.shape}")
            
            # Separate features and target
            X = data.drop('HeartDisease', axis=1)
            y = data['HeartDisease']
            
            log.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            log.info(f"Target distribution:\n{y.value_counts()}")
            
            # Perform train-test split (80-20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42,
                stratify=y  # Maintain class distribution
            )
            
            log.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Create data directory for split datasets
            data_dir = self.config.root_dir / "data"
            os.makedirs(data_dir, exist_ok=True)
            
            # Save the split datasets
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            train_data.to_csv(data_dir / "train.csv", index=False)
            test_data.to_csv(data_dir / "test.csv", index=False)
            
            log.info("Train-test split completed and datasets saved")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            log.error(f"Error in loading and splitting data: {str(e)}")
            raise e
    
    def load_best_params(self):
        """Load best parameters from param.yaml"""
        try:
            from Heart_Disease_Prediction.constant import PARAM_FILE_PATH
            from Heart_Disease_Prediction.utils.util import read_yaml
            
            params = read_yaml(PARAM_FILE_PATH)
            # Updated to use logistic regression parameters
            best_params = params.get('logistic_regression_params', {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'liblinear'
            })
            
            log.info(f"Loaded Logistic Regression parameters: {best_params}")
            return best_params
            
        except Exception as e:
            log.error(f"Error loading parameters from param.yaml: {str(e)}")
            # Return default Logistic Regression parameters if loading fails
            default_params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'liblinear'
            }
            log.info(f"Using default Logistic Regression parameters: {default_params}")
            return default_params
    
    def cross_validate_model(self, X, y):
        """Perform cross-validation to check model stability"""
        try:
            best_params = self.load_best_params()
            model = LogisticRegression(**best_params)
            
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            log.info(f"Cross-Validation Scores: {cv_scores}")
            log.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if cv_scores.std() > 0.05:
                log.warning("High variance in CV scores - model may be unstable")
            else:
                log.info("CV scores show good stability")
            
            return cv_scores
            
        except Exception as e:
            log.error(f"Error in cross-validation: {str(e)}")
            raise e
    
    def analyze_feature_importance(self, model, feature_names):
        """Analyze which features are most important for Logistic Regression"""
        try:
            # For Logistic Regression, use absolute coefficients as importance
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                # Fallback: use random feature importance
                importances = np.ones(len(feature_names)) / len(feature_names)
                log.warning("Using uniform feature importance - model coefficients not available")
            
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            log.info("Top 5 Most Important Features (based on coefficients):")
            for _, row in feature_imp.head().iterrows():
                log.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Check for dominant features
            top_feature_importance = feature_imp['importance'].iloc[0]
            if top_feature_importance > (2 * feature_imp['importance'].iloc[1]):
                log.warning(f"Single feature dominating (importance: {top_feature_importance:.4f}) - consider regularization")
            
            return feature_imp
            
        except Exception as e:
            log.error(f"Error in feature importance analysis: {str(e)}")
            raise e
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression using best parameters from param.yaml"""
        try:
            log.info("Starting Logistic Regression training...")
            
            # Load best parameters from param.yaml
            best_params = self.load_best_params()
            
            # Create and train Logistic Regression model
            logreg = LogisticRegression(**best_params)
            logreg.fit(X_train, y_train)

            # Make predictions on both train and test sets
            train_pred = logreg.predict(X_train)
            test_pred = logreg.predict(X_test)
            
            # Calculate accuracies for both sets
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            accuracy_gap = train_accuracy - test_accuracy
            
            log.info(f"Training Accuracy: {train_accuracy:.4f}")
            log.info(f"Test Accuracy: {test_accuracy:.4f}")
            log.info(f"Accuracy Gap (Train - Test): {accuracy_gap:.4f}")
            
            # Overfitting detection
            overfitting_status = ""
            overfitting_log_status = ""
            
            if accuracy_gap > 0.10:
                overfitting_status = "SIGNIFICANT OVERFITTING DETECTED"
                overfitting_log_status = "🚨 SIGNIFICANT OVERFITTING DETECTED"
                log.warning(overfitting_log_status)
                log.warning("Training accuracy is much higher than test accuracy")
            elif accuracy_gap > 0.05:
                overfitting_status = "MODERATE OVERFITTING DETECTED"
                overfitting_log_status = "⚠️ MODERATE OVERFITTING DETECTED"
                log.warning(overfitting_log_status)
                log.warning("Model may be memorizing training data")
            else:
                overfitting_status = "Good generalization - minimal overfitting"
                overfitting_log_status = "✅ Good generalization - minimal overfitting"
                log.info(overfitting_log_status)
            
            # Generate detailed classification report
            class_report = classification_report(y_test, test_pred)
            log.info(f"Classification Report:\n{class_report}")
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, test_pred)
            log.info(f"Confusion Matrix:\n{cm}")
            
            # Also get prediction probabilities for potential threshold tuning
            train_proba = logreg.predict_proba(X_train)
            test_proba = logreg.predict_proba(X_test)
            
            log.info("Logistic Regression training completed successfully")
            
            return logreg, best_params, test_accuracy, train_accuracy, class_report, cm, accuracy_gap, overfitting_status
            
        except Exception as e:
            log.error(f"Error in Logistic Regression training: {str(e)}")
            raise e
    
    def save_model_and_results(self, model, best_params, test_accuracy, train_accuracy, 
                             class_report, cm, accuracy_gap, overfitting_status, cv_scores, feature_imp):
        """Save the trained model and evaluation results"""
        try:
            # Save the trained model
            model_path = self.config.trained_data_dir / self.config.model_name
            joblib.dump(model, model_path)
            log.info(f"Model saved at: {model_path}")
            
            # Save model parameters
            params_path = self.config.trained_data_dir / "best_params.joblib"
            joblib.dump(best_params, params_path)
            
            # Save evaluation results
            results = {
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'accuracy_gap': accuracy_gap,
                'overfitting_status': overfitting_status,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'best_params': best_params,
                'cv_scores': cv_scores,
                'feature_importance': feature_imp.to_dict()
            }
            
            results_path = self.config.tested_data_dir / "evaluation_results.joblib"
            joblib.dump(results, results_path)
            
            # Save a comprehensive text report
            report_path = self.config.tested_data_dir / "model_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== LOGISTIC REGRESSION TRAINING RESULTS ===\n\n")
                f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
                f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
                f.write(f"Accuracy Gap: {accuracy_gap:.4f}\n")
                f.write(f"Overfitting Status: {overfitting_status}\n\n")
                
                f.write("=== CROSS-VALIDATION RESULTS ===\n")
                f.write(f"CV Scores: {list(cv_scores)}\n")
                f.write(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
                
                f.write("=== TOP 5 FEATURE IMPORTANCES (Absolute Coefficients) ===\n")
                top_features = feature_imp.head()
                for _, row in top_features.iterrows():
                    f.write(f"{row['feature']}: {row['importance']:.4f}\n")
                f.write("\n")
                
                f.write("=== CLASSIFICATION REPORT ===\n")
                f.write(class_report)
                f.write(f"\n=== CONFUSION MATRIX ===\n")
                f.write(f"{cm}\n")
            
            log.info(f"Results saved at: {results_path}")
            log.info(f"Comprehensive text report saved at: {report_path}")
            
        except Exception as e:
            log.error(f"Error saving model and results: {str(e)}")
            # Create a basic report even if detailed one fails
            try:
                basic_report_path = self.config.tested_data_dir / "basic_model_report.txt"
                with open(basic_report_path, 'w', encoding='utf-8') as f:
                    f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
                    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
                    f.write(f"Accuracy Gap: {accuracy_gap:.4f}\n")
                log.info(f"Basic report saved at: {basic_report_path}")
            except:
                log.error("Failed to save basic report")
            raise e
        
    def train_model(self):
        """Main method to execute the complete model training pipeline"""
        try:
            log.info("Starting Logistic Regression model training pipeline...")
            
            # Step 1: Load and split data
            X_train, X_test, y_train, y_test = self.load_and_split_data()
            
            # Step 2: Cross-validation check
            cv_scores = self.cross_validate_model(
                pd.concat([X_train, X_test]), 
                pd.concat([y_train, y_test])
            )
            
            # Step 3: Train Logistic Regression
            model, best_params, test_accuracy, train_accuracy, class_report, cm, accuracy_gap, overfitting_status = self.train_logistic_regression(
                X_train, X_test, y_train, y_test
            )
            
            # Step 4: Feature importance analysis
            feature_imp = self.analyze_feature_importance(model, X_train.columns)
            
            # Step 5: Save model and results
            self.save_model_and_results(
                model, best_params, test_accuracy, train_accuracy, 
                class_report, cm, accuracy_gap, overfitting_status, cv_scores, feature_imp
            )
            
            log.info("Logistic Regression model training pipeline completed successfully!")
            
            return model
            
        except Exception as e:
            log.error(f"Logistic Regression model training pipeline failed: {str(e)}")
            raise e