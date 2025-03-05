import logging
import os
from google.cloud import storage
from io import BytesIO
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TokenizationProcessor:
    def __init__(self, bucket: str, file_path: str, folder: str, parquet_file_name: str):
        self.bucket = bucket
        self.file_path = file_path
        self.folder = folder
        self.parquet_file_name = parquet_file_name
        self.client: storage.Client = storage.Client()
        self.logger = self._configure_logger()
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('italian'))


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
    
    def file_exists_in_gcs(self) -> bool:
        print("Check if the file in exists in gcs")
        self.logger.info(f"Check if the file in exists in gcs {self.parquet_file_name}")
        bucket = self.client.bucket(self.bucket)
        blob = bucket.blob(f"{self.folder}/{self.parquet_file_name}")
        return blob.exists()

    def load_parquet_from_gcs(self) -> pd.DataFrame:
        try:
            print(f"Start loading DataFrame from {self.file_path}")
            self.logger.info(f"Start loading DataFrame from {self.file_path}")
            bucket: storage.Bucket = self.client.bucket(self.bucket)
            blob: storage.Blob = bucket.blob(f"{self.folder}/{self.file_path}")
            byte_stream = BytesIO()
            blob.download_to_file(byte_stream)
            byte_stream.seek(0)
            df: pd.DataFrame = pd.read_parquet(byte_stream, engine='pyarrow')
            self.logger.info(f"Loaded DataFrame from {self.file_path}")
            print(f"Loaded DataFrame from {self.file_path}")
            return df
        
        except Exception as e:
            print(f"Error loading Parquet file from GCS: {e}")
            self.logger.error(f"Error loading Parquet file from GCS: {e}")
            # You can raise an exception or return None depending on how you want to handle this situation
            return None

    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        cleaned_text = ' '.join(filtered_tokens)
        return cleaned_text

    def preprocess_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Starting Tokenization DataFrame from {self.file_path}")
        print(f"Starting Tokenization DataFrame from {self.file_path}")
        df['body_pre'] = df['body'].apply(self.preprocess_text)
        return df

    def save_df_to_gcs_parquet(self, df: pd.DataFrame):
        destination_blob_name = f'{self.folder}/{self.parquet_file_name}'
        byte_stream = BytesIO()
        df.to_parquet(byte_stream, engine='pyarrow')
        byte_stream.seek(0)
        bucket = self.client.bucket(self.bucket)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(byte_stream)
    
    def process(self):
        if self.file_exists_in_gcs():
            print(f"{self.parquet_file_name} already exists in GCS. Skipping process.")
            self.logger.info(f"{self.parquet_file_name}  already exists in GCS. Skipping process.")
            return self.parquet_file_name
        else:
            df = self.load_parquet_from_gcs()
            if df is None:
                print(f"File {self.file_path} doesn't exists")
                self.logger.info(f"File {self.file_path} doesn't exists")
                # Exit from the class method if the file does not exist
                return
            
            print("Loading Data Tokenized")
            self.logger.info("Loading Data Tokenized")
            df_tokenized = self.preprocess_articles(df)
            
            self.save_df_to_gcs_parquet(df_tokenized)
            print("Dataframe tokenized - process completed and file saved to GCS.")
            self.logger.info("Dataframe tokenized - process completed and file saved to GCS.")
            return self.parquet_file_name
