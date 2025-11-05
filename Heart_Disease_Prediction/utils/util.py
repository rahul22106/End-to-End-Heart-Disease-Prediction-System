import yaml
import os
import sys
from typing import Any, Dict
from Heart_Disease_Prediction.exception.exception_handler import AppException

def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns the contents as a dictionary.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Parsed YAML content as dictionary
        
    Raises:
        AppException: If file not found, invalid YAML, or permission issues
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        # Check if it's a file (not directory)
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file extension
        if not file_path.lower().endswith(('.yaml', '.yml')):
            raise ValueError(f"File is not a YAML file: {file_path}")
        
        # Check file size (avoid reading huge files accidentally)
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError(f"YAML file too large: {file_size} bytes")
        
        # Read and parse YAML file
        with open(file_path, 'r', encoding='utf-8') as yaml_file:
            yaml_content = yaml.safe_load(yaml_file)
            
            # Check if YAML content is valid
            if yaml_content is None:
                return {}  # Return empty dict for empty YAML files
                
            if not isinstance(yaml_content, dict):
                raise ValueError(f"YAML file does not contain a dictionary: {file_path}")
                
            return yaml_content
            
    except yaml.YAMLError as e:
        raise AppException(f"Invalid YAML syntax in {file_path}: {str(e)}", sys) from e
    except Exception as e:
        raise AppException(f"Error reading YAML file {file_path}: {str(e)}", sys) from e