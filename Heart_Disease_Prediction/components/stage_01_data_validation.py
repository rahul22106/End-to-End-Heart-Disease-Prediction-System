import os
import pandas as pd
from Heart_Disease_Prediction.logger.log import log
from Heart_Disease_Prediction.entity.config_entity import DataValidationConfig
from Heart_Disease_Prediction.exception.exception_handler import AppException
import sys

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validate that all columns in the dataset match the schema
        Returns True if all columns are valid, False otherwise
        """
        try:
            validation_status = True
            missing_columns = []
            extra_columns = []
            
            log.info("Starting column validation...")
            
            # Read the dataset
            data = pd.read_csv(self.config.unzip_data_dir)
            dataset_columns = set(data.columns)
            log.info(f"Dataset columns: {list(dataset_columns)}")
            
            # Get schema columns
            schema_columns = set(self.config.all_schema.keys())
            log.info(f"Schema columns: {list(schema_columns)}")
            
            # Check for missing columns
            missing_columns = schema_columns - dataset_columns
            if missing_columns:
                validation_status = False
                log.error(f"Missing columns: {list(missing_columns)}")
            
            # Check for extra columns
            extra_columns = dataset_columns - schema_columns
            if extra_columns:
                validation_status = False
                log.warning(f"Extra columns found: {list(extra_columns)}")
            
            # Write validation results to file
            self._write_validation_status(validation_status, missing_columns, extra_columns)
            
            if validation_status:
                log.info("✅ All columns validated successfully!")
            else:
                log.error("❌ Column validation failed!")
                
            return validation_status
            
        except Exception as e:
            log.error(f"Error during column validation: {str(e)}")
            raise AppException(e, sys) from e

    def _write_validation_status(self, status: bool, missing_cols: set, extra_cols: set):
        """
        Write validation status and details to file
        """
        try:
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {status}\n")
                f.write(f"Missing columns: {list(missing_cols)}\n")
                f.write(f"Extra columns: {list(extra_cols)}\n")
                f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            
            log.info(f"Validation status written to: {self.config.STATUS_FILE}")
            
        except Exception as e:
            log.error(f"Error writing validation status: {str(e)}")
            raise

    def validate_data_types(self) -> bool:
        """
        Validate that data types match the schema
        """
        try:
            validation_status = True
            type_mismatches = []
            
            log.info("Starting data type validation...")
            
            # Read the dataset
            data = pd.read_csv(self.config.unzip_data_dir)
            
            for column, expected_type in self.config.all_schema.items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    expected_type_str = str(expected_type)
                    
                    # Handle type comparisons (you might need to adjust this logic)
                    if not self._check_type_compatibility(actual_type, expected_type_str):
                        validation_status = False
                        type_mismatches.append({
                            'column': column,
                            'expected': expected_type_str,
                            'actual': actual_type
                        })
                        log.warning(f"Type mismatch in {column}: expected {expected_type_str}, got {actual_type}")
            
            # Write type validation results
            if type_mismatches:
                with open(self.config.STATUS_FILE, 'a') as f:
                    f.write(f"\nType mismatches: {type_mismatches}\n")
            
            if validation_status:
                log.info("✅ All data types validated successfully!")
            else:
                log.error("❌ Data type validation failed!")
                
            return validation_status
            
        except Exception as e:
            log.error(f"Error during data type validation: {str(e)}")
            raise AppException(e, sys) from e

    def _check_type_compatibility(self, actual_type: str, expected_type: str) -> bool:
        """
        Check if actual data type is compatible with expected type
        You might need to customize this based on your needs
        """
        type_mapping = {
            'int64': ['int64', 'int32', 'float64', 'float32'],
            'float64': ['float64', 'float32', 'int64', 'int32'],
            'object': ['object']
        }
        
        expected_base = expected_type.lower()
        actual_base = actual_type.lower()
        
        if expected_base in type_mapping:
            return actual_base in type_mapping[expected_base]
        
        return actual_base == expected_base

    def validate_all_files(self) -> bool:
        """
        Main method to run all validation checks
        """
        try:
            log.info("Starting comprehensive data validation...")
            
            # Check if data file exists
            if not os.path.exists(self.config.unzip_data_dir):
                log.error(f"Data file not found: {self.config.unzip_data_dir}")
                self._write_validation_status(False, set(), set())
                return False
            
            # Run column validation
            columns_valid = self.validate_all_columns()
            
            # Run data type validation
            types_valid = self.validate_data_types()
            
            overall_status = columns_valid and types_valid
            
            if overall_status:
                log.info("🎉 All data validation checks passed!")
            else:
                log.error("💥 Data validation failed!")
                
            return overall_status
            
        except Exception as e:
            log.error(f"Error during comprehensive validation: {str(e)}")
            raise AppException(e, sys) from e