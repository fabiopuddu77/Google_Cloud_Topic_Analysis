import pandas as pd
import logging
from io import BytesIO
from google.cloud import storage, language_v1
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import RobustScaler

class GCSSentimentAnalyzer:
    def __init__(self, bucket_name: str, file_path: str, folder: str, parquet_file_name: str, 
                 text_column: str, num_doc: int):
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.folder = folder
        self.parquet_file_name = parquet_file_name
        self.client: storage.Client = storage.Client()
        self.text_column = text_column
        self.num_doc = num_doc
        self.client = storage.Client()
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

    def file_exists_in_gcs(self) -> bool:
        print("Check if the file in exists in gcs")
        self.logger.info(f"Check if the file in exists in gcs {self.parquet_file_name}")
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(f"{self.folder}/{self.parquet_file_name}")
        return blob.exists()

    def load_parquet_from_gcs(self) -> pd.DataFrame:
        try:
            print(f"Start loading DataFrame from {self.file_path}")
            self.logger.info(f"Start loading DataFrame from {self.file_path}")
            bucket: storage.Bucket = self.client.bucket(self.bucket_name)
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

          
    def analyze_text_sentiment(self, row) -> dict:
        client = language_v1.LanguageServiceClient()
        text = row[self.text_column]
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = client.analyze_sentiment(request={'document': document})
        return {
            'score': response.document_sentiment.score,
            'magnitude': response.document_sentiment.magnitude
        }

    def analyze_and_merge_sentiments(self, df: pd.DataFrame) -> pd.DataFrame:
        df_merged = df.sample(n=self.num_doc)
        # Assuming 'text_column' is the name of the column containing the text data
        # Create new columns 'score' and 'magnitude' using the analyze_text_sentiment function
        df_merged[['score', 'magnitude']] = df_merged[self.text_column].apply(lambda x: pd.Series(self.analyze_text_sentiment({self.text_column: x})))
        return df_merged

    def save_df_to_gcs_parquet(self, df: pd.DataFrame):
        destination_blob_name = f'{self.folder}/{self.parquet_file_name}'
        byte_stream = BytesIO()
        df.to_parquet(byte_stream, engine='pyarrow')
        byte_stream.seek(0)
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(byte_stream)

    def process(self):
        if self.file_exists_in_gcs():
            print("File already exists in GCS. Skipping process.")
            self.logger.info("File already exists in GCS. Skipping process.")
            return self.parquet_file_name
        else:
            df = self.load_parquet_from_gcs()
            if df is None:
                print(f"File {self.file_path} doesn't exists")
                self.logger.info(f"File {self.file_path} doesn't exists")
                # Exit from the class method if the file does not exist
                return
         
            print("Extracting Sentiment")
            self.logger.info("Extracting Sentiment")
            df_sentiment = self.analyze_and_merge_sentiments(df)
            self.save_df_to_gcs_parquet(df_sentiment)
            print("Sentiment extracted - process completed and file saved to GCS.")
            self.logger.info("Sentiment extracted - process completed and file saved to GCS.")
            return self.parquet_file_name



# analyzer = GCSSentimentAnalyzer(bucket_name='ccai-storage', input_file_path='pipeline/final_df1.parquet',
#                                 output_folder='pipeline', output_file_name='final_df2.parquet', text_column='body_pre')
# analyzer.process()
