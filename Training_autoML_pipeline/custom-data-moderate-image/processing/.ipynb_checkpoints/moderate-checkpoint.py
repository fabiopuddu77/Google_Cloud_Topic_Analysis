import pandas as pd
import logging
from io import BytesIO
from google.cloud import storage, language_v1 as language
from html.parser import HTMLParser

class GCSTextModerationLoader:
    def __init__(self, bucket_name: str, file_path: str, folder: str, parquet_file_name: str, text_column: str, num_doc=int):
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.folder = folder
        self.parquet_file_name = parquet_file_name
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

    @staticmethod
    def strip_html_tags(html: str) -> str:
        class MLStripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.reset()
                self.strict = False
                self.convert_charrefs = True
                self.text = []
            def handle_data(self, d):
                self.text.append(d)
            def get_data(self):
                return ''.join(self.text)

        s = MLStripper()
        s.feed(html)
        return s.get_data()


    def moderate_text(self, text: str) -> language.ModerateTextResponse:
        client = language.LanguageServiceClient()
        document = language.Document(
            content=text,
            type_=language.Document.Type.PLAIN_TEXT,
        )
        
        return client.moderate_text(document=document)
        
   

    def analyze_and_moderate_documents(self, df: pd.DataFrame):
        documents = []
        for index, row in df.iterrows():
            linetext_cleaned = row.get(self.text_column, '')
            if linetext_cleaned:
                # Perform moderation analysis on the cleaned text
                response = self.moderate_text(linetext_cleaned)
                categories = []
                confidences = []
                for category in response.moderation_categories:
                    categories.append(category.name)
                    confidences.append(category.confidence)
                document = row.to_dict()
                for category, confidence in zip(categories, confidences):
                    # Create new columns for each category with confidence scores
                    document[category] = confidence
                documents.append(document)
                if len(documents) >= self.num_doc:
                    break  # Stop after processing the specified number of documents

        df_moderated = pd.DataFrame(documents)
        return df_moderated


    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.replace('&', '').str.replace(r'\s+', ' ').str.replace(',', '').str.replace(' ', '_')
        return df

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
            
            print("Extracting Moderate")
            self.logger.info("Extracting Moderate")
            df_moderated = self.analyze_and_moderate_documents(df)
            df_cleaned = self.clean_column_names(df_moderated)
            self.save_df_to_gcs_parquet(df_cleaned)
            print("Moderate extracted - process completed and file saved to GCS.")
            self.logger.info("Moderate extracted - process completed and file saved to GCS.")
            return self.parquet_file_name


# analyzer = GCSTextModerationLoader(bucket_name='ccai-storage', file_path='pipeline/final_df2.parquet',
#                                 folder='pipeline', parquet_file_name='final_df3.parquet', column='body_pre')
# analyzer.process()