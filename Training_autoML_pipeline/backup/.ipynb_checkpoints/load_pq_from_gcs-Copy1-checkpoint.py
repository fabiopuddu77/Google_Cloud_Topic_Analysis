import pandas as pd
from google.cloud import storage
from google.cloud import storage
import pandas as pd
from io import BytesIO
from google.cloud import bigquery
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import RobustScaler
import logging

class GCSParquetLoader:
    def __init__(self, bucket_name: str, file_path: str):
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.client: storage.Client = storage.Client()
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create a file handler and set level to INFO
        file_handler = logging.FileHandler('gcs_parquet_loader.log')
        file_handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        return logger

    def load_parquet_from_gcs(self) -> pd.DataFrame:
        self.logger.info(f"Start loading DataFrame from {self.file_path}")
        bucket: storage.Bucket = self.client.bucket(self.bucket_name)
        blob: storage.Blob = bucket.blob(self.file_path)
        byte_stream: BytesIO = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)
        df: pd.DataFrame = pd.read_parquet(byte_stream, engine='pyarrow')
        self.logger.info(f"Loaded DataFrame from {self.file_path}")
        return df
