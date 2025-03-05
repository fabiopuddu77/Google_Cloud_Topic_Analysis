

import logging
from processing.data_preparation_inference import GCSParquetLoader
from processing.tokenization_inference import TokenizationProcessor
from processing.sentiment_inference import GCSSentimentAnalyzer
from processing.moderate_inference import GCSTextModerationLoader
from processing.entities_inference import GCSCEntityAnalyzer
from processing.bigquery_inference import GCS_Bigquery

class DataProcessor:
    def __init__(self, bucket: str, folder: str, num_doc: int, random_seed: int):
        self.bucket = bucket
        self.folder = folder
        self.num_doc = num_doc
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s\n')
        
    def data_preprocessing(self, file_path: str, parquet_file_name: str,  num_doc: int, random_seed:int ) -> str:
        processor = GCSParquetLoader(self.bucket, self.folder, file_path, parquet_file_name, self.num_doc, random_seed)
        return processor.process()
    
    def data_tokenization(self, file_path: str, parquet_file_name: str) -> str:
        processor = TokenizationProcessor(self.bucket, file_path, self.folder, parquet_file_name)
        return processor.process()
    
    def data_sentiment(self, file_path: str, parquet_file_name: str, text_column: str, num_doc: int) -> str:
        processor = GCSSentimentAnalyzer(self.bucket, file_path, self.folder, parquet_file_name, text_column, num_doc)
        return processor.process()
    
    def data_moderate(self, file_path: str, parquet_file_name: str, text_column: str, num_doc: int) -> str:
        processor = GCSTextModerationLoader(self.bucket, file_path, self.folder, parquet_file_name, text_column, num_doc)
        return processor.process()
    
    def data_entities(self, file_path: str, parquet_file_name: str, text_column: str, num_doc: int) -> str:
        processor = GCSCEntityAnalyzer(self.bucket, file_path, self.folder, parquet_file_name, text_column, num_doc)
        return processor.process()
    
    def data_bigquery(self, file_path: str, parquet_file_name: str, project_id: str, dataname: str, table_id: str, location: str) -> str:
        processor = GCS_Bigquery(self.bucket, file_path, self.folder, parquet_file_name, project_id, dataname, table_id, location)
        return processor.upload_dataframe_to_bigquery()

