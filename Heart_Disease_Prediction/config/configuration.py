from Heart_Disease_Prediction.constant import *
from Heart_Disease_Prediction.utils.util import read_yaml, create_directories
from Heart_Disease_Prediction.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH, 
        param_filepath = PARAM_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.schema = read_yaml(schema_filepath)
        self.param  = read_yaml(param_filepath)

        
    

        create_directories([self.config.artifacts_root])
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        # Create directories
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            transformed_data_dir=Path(config.transformed_data_dir)
        )

        return data_transformation_config
   

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        # Create directories
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            transformed_data_path=Path(config.transformed_data_path),
            trained_data_dir=Path(config.trained_data_dir),
            tested_data_dir=Path(config.tested_data_dir),
            model_name=config.model_name
        )

        return model_trainer_config