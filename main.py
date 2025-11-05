from Heart_Disease_Prediction.logger.log import log
from Heart_Disease_Prediction.pipeline.training_pipeline import DataIngestionTrainingPipeline
#from Heart_Disease_Prediction.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
#from Heart_Disease_Prediction.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
#from Heart_Disease_Prediction.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
#from Heart_Disease_Prediction.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception("An error occurred in the training pipeline", exc_info=True)
        raise