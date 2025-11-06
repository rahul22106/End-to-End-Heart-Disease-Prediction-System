import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
            best_params = params['best_params']
            
            log.info(f"Loaded best parameters from param.yaml: {best_params}")
            return best_params
            
        except Exception as e:
            log.error(f"Error loading parameters from param.yaml: {str(e)}")
            raise e
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest using best parameters from param.yaml"""
        try:
            log.info("Starting Random Forest training with parameters from param.yaml...")
            
            # Load best parameters from param.yaml
            best_params = self.load_best_params()
            
            # Create and train model with best parameters
            rfctree = RandomForestClassifier(**best_params)
            rfctree.fit(X_train, y_train)

            # Make predictions and evaluate
            rfc_pred = rfctree.predict(X_test)
            accuracy = accuracy_score(y_test, rfc_pred)
            
            log.info(f"RandomForestClassifier's Accuracy: {accuracy:.4f}")
            
            # Generate detailed classification report
            class_report = classification_report(y_test, rfc_pred)
            log.info(f"Classification Report:\n{class_report}")
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, rfc_pred)
            log.info(f"Confusion Matrix:\n{cm}")
            
            return rfctree, best_params, accuracy, class_report, cm
            
        except Exception as e:
            log.error(f"Error in Random Forest training: {str(e)}")
            raise e
    
    def save_model_and_results(self, model, best_params, accuracy, class_report, cm):
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
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'best_params': best_params
            }
            
            results_path = self.config.tested_data_dir / "evaluation_results.joblib"
            joblib.dump(results, results_path)
            
            # Save a readable text report
            report_path = self.config.tested_data_dir / "model_report.txt"
            with open(report_path, 'w') as f:
                f.write("=== MODEL TRAINING RESULTS ===\n\n")
                f.write(f"Best Parameters: {best_params}\n\n")
                f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(class_report)
                f.write(f"\nConfusion Matrix:\n{cm}")
            
            log.info(f"Results saved at: {results_path}")
            log.info(f"Text report saved at: {report_path}")
            
        except Exception as e:
            log.error(f"Error saving model and results: {str(e)}")
            raise e
    
    def train_model(self):
        """Main method to execute the complete model training pipeline"""
        try:
            log.info("Starting model training pipeline...")
            
            # Step 1: Load and split data
            X_train, X_test, y_train, y_test = self.load_and_split_data()
            
            # Step 2: Train Random Forest with parameters from param.yaml
            model, best_params, accuracy, class_report, cm = self.train_random_forest(
                X_train, X_test, y_train, y_test
            )
            
            # Step 3: Save model and results
            self.save_model_and_results(model, best_params, accuracy, class_report, cm)
            
            log.info("✅ Model training pipeline completed successfully!")
            
            return model
            
        except Exception as e:
            log.error(f"Model training pipeline failed: {str(e)}")
            raise e