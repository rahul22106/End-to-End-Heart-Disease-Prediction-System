from Heart_Disease_Prediction.logger.log import log
from Heart_Disease_Prediction.pipeline.training_pipeline import DataIngestionTrainingPipeline
from Heart_Disease_Prediction.pipeline.training_pipeline import DataValidationTrainingPipeline
from Heart_Disease_Prediction.pipeline.training_pipeline import DataTransformationTrainingPipeline
from Heart_Disease_Prediction.pipeline.training_pipeline import ModelTrainerTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception("An error occurred in the training pipeline", exc_info=True)
        raise

STAGE_NAME = "Data Validation stage"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataValidationTrainingPipeline()
   data_transformation.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(e)
        raise e

STAGE_NAME ="Model Training stage"
try:
    log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<\n\nx==========x")
    model_trainer= ModelTrainerTrainingPipeline()
    model_trainer.main()
    log.info(f" >>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
     log.exception(e)   
     raise e 
