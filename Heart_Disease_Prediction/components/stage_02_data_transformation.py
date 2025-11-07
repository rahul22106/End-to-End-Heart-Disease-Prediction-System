import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler  # ADD THIS IMPORT
import joblib  # ADD THIS IMPORT
from Heart_Disease_Prediction.entity.config_entity import DataTransformationConfig
from Heart_Disease_Prediction.logger.log import log

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.heart = None
        self.categorical_columns = None
        
    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            log.info(f"Loading data from: {self.config.data_path}")
            
            # Check if file exists
            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(f"Data file not found at: {self.config.data_path}")
            
            self.heart = pd.read_csv(self.config.data_path)
            log.info(f"Dataset loaded successfully with {len(self.heart)} rows and {len(self.heart.columns)} columns")
            return self.heart
            
        except Exception as e:
            log.error(f"Error loading data: {str(e)}")
            raise e
    
    def explore_data(self):
        """Perform initial data exploration"""
        if self.heart is None:
            raise ValueError("No data loaded. Please load data first.")
        
        log.info("=== Data Exploration ===")
        log.info(f"Shape: {self.heart.shape}")
        log.info(f"Columns: {list(self.heart.columns)}")
        log.info(f"Missing values:\n{self.heart.isnull().sum()}")
        
    def encode_categorical_variables(self):
        """Convert categorical variables to numeric"""
        if self.heart is None:
            raise ValueError("No data loaded. Please load data first.")
        
        self.categorical_columns = self.heart.select_dtypes(include='object').columns
        
        if len(self.categorical_columns) == 0:
            log.info("No categorical columns found to encode")
            return {}
        
        log.info("=== Encoding Categorical Variables ===")
        encoding_map = {}
        
        for col in self.categorical_columns:
            unique_values = list(self.heart[col].unique())
            encoded_values = list(range(len(unique_values)))
            encoding_map[col] = dict(zip(unique_values, encoded_values))
            
            log.info(f"{col}: {unique_values} -> {encoded_values}")
            self.heart[col].replace(unique_values, encoded_values, inplace=True)
        
        log.info("Categorical encoding completed")
        return encoding_map
    
    def scale_features(self):  # NEW METHOD
        """Apply StandardScaler to all features except target"""
        if self.heart is None:
            raise ValueError("No data loaded. Please load data first.")
        
        log.info("=== Applying StandardScaler ===")
        
        # Separate features and target
        features = self.heart.drop('HeartDisease', axis=1)
        target = self.heart['HeartDisease']
        
        # Get feature names for reconstruction
        feature_names = features.columns.tolist()
        
        # Apply StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Convert back to DataFrame with original column names
        scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
        
        # Recombine with target
        self.heart = pd.concat([scaled_df, target], axis=1)
        
        # Save the scaler for future use
        scaler_path = os.path.join(self.config.root_dir, "standard_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        log.info(f"StandardScaler saved to: {scaler_path}")
        
        log.info("Feature scaling completed successfully")
        return scaler
    
    def handle_zero_values(self):
        """Handle zero values in Cholesterol and RestingBP columns"""
        if self.heart is None:
            raise ValueError("No data loaded. Please load data first.")
        
        log.info("=== Handling Zero Values ===")
        
        # Check if required columns exist
        if 'Cholesterol' not in self.heart.columns or 'RestingBP' not in self.heart.columns:
            log.warning("Required columns for zero value handling not found")
            return
        
        cholesterol_zero_count = (self.heart['Cholesterol'] == 0).sum()
        restingbp_zero_count = (self.heart['RestingBP'] == 0).sum()
        
        log.info(f"Zero values - Cholesterol: {cholesterol_zero_count}, RestingBP: {restingbp_zero_count}")
        
        if cholesterol_zero_count > 0 or restingbp_zero_count > 0:
            # Replace zeros with NaN
            if cholesterol_zero_count > 0:
                self.heart['Cholesterol'].replace(0, np.nan, inplace=True)
            if restingbp_zero_count > 0:
                self.heart['RestingBP'].replace(0, np.nan, inplace=True)
            
            # Apply KNN imputer
            imputer = KNNImputer(n_neighbors=3)
            after_impute = imputer.fit_transform(self.heart)
            self.heart = pd.DataFrame(after_impute, columns=self.heart.columns)
            
            log.info("Zero values imputed successfully")
    
    def optimize_data_types(self):
        """Optimize data types for memory efficiency"""
        if self.heart is None:
            raise ValueError("No data loaded. Please load data first.")
        
        log.info("=== Optimizing Data Types ===")
        
        # Convert features to float32 (scaled values) and target to int32
        features = self.heart.drop('HeartDisease', axis=1)
        target = self.heart['HeartDisease']
        
        # Convert features to float32 for scaled values
        features = features.astype('float32')
        target = target.astype('int32')
        
        self.heart = pd.concat([features, target], axis=1)
        
        log.info("Data types optimized: features to float32, target to int32")
    
    def save_transformed_data(self):
        """Save the transformed dataset"""
        if self.heart is None:
            raise ValueError("No data loaded. Cannot save.")
        
        # Ensure the directory exists
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        output_path = os.path.join(self.config.root_dir, "transformed_data.csv")
        self.heart.to_csv(output_path, index=False)
        log.info(f"Transformed data saved to: {output_path}")
        return output_path
    
    def transform_data(self):
        """
        Execute the complete data transformation pipeline
        
        Returns:
            tuple: (transformed_dataframe, output_file_path)
        """
        log.info("Starting data transformation pipeline")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Explore data
            self.explore_data()
            
            # Step 3: Encode categorical variables
            self.encode_categorical_variables()
            
            # Step 4: Handle zero values
            self.handle_zero_values()
            
            # Step 5: Scale features (NEW STEP - ADDED HERE)
            self.scale_features()
            
            # Step 6: Optimize data types
            self.optimize_data_types()
            
            # Step 7: Save transformed data
            output_path = self.save_transformed_data()
            
            log.info("Data transformation pipeline completed successfully")
            
            return self.heart, output_path
            
        except Exception as e:
            log.error(f"Data transformation pipeline failed: {str(e)}")
            raise e