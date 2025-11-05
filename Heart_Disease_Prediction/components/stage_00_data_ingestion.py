import os
import zipfile
from urllib import request
from pathlib import Path
from Heart_Disease_Prediction.logger.log import log
from Heart_Disease_Prediction.exception.exception_handler import AppException
import sys

class DataIngestion:
    def __init__(self, config):
        self.config = config
    
    def download_file(self):
        """
        Download file from URL and save to local directory
        """
        try:
            log.info("Starting file download...")
            
            # Convert to Path objects
            local_file_path = Path(self.config.local_data_file)
            download_dir = local_file_path.parent
            
            # Create directory if it doesn't exist
            download_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Ensured directory exists: {download_dir}")
            
            # Check if file already exists
            if local_file_path.exists():
                log.info(f"File already exists, skipping download: {local_file_path}")
                return
            
            # Download the file
            log.info(f"Downloading from: {self.config.source_URL}")
            log.info(f"Saving to: {local_file_path}")
            
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=str(local_file_path)  # Convert to string for urlretrieve
            )
            
            log.info(f"✅ File downloaded successfully: {filename}")
            log.info(f"Download size: {os.path.getsize(filename)} bytes")
            
        except Exception as e:
            log.error(f"❌ Failed to download file: {str(e)}")
            raise AppException(e, sys) from e
    
    def extract_zip_file(self):
        """
        Extract zip file to specified directory
        """
        try:
            log.info("Starting file extraction...")
            
            zip_file_path = Path(self.config.local_data_file)
            extract_dir = Path(self.config.unzip_dir)
            
            # Create extraction directory
            extract_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Ensured extraction directory exists: {extract_dir}")
            
            # Check if zip file exists
            if not zip_file_path.exists():
                raise FileNotFoundError(f"Zip file not found: {zip_file_path}")
            
            # Extract the file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            log.info(f"✅ File extracted successfully to: {extract_dir}")
            
            # List extracted files
            extracted_files = list(extract_dir.rglob('*'))
            log.info(f"Extracted {len(extracted_files)} files")
            
        except Exception as e:
            log.error(f"❌ Failed to extract file: {str(e)}")
            raise AppException(e, sys) from e