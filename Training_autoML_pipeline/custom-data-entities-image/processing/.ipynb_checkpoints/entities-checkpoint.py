import pandas as pd
from google.cloud import storage, language_v1
from google.cloud import language
from io import BytesIO
import logging

class GCSCEntityAnalyzer:
    def __init__(self, bucket_name: str, file_path: str, folder: str, parquet_file_name: str, text_column: str, num_doc: int):
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
        
    
    
    def analyze_text_entities(self, text: str) -> language.AnalyzeEntitiesResponse:
        client = language.LanguageServiceClient()
        document = language.Document(
            content=text,
            type_=language.Document.Type.PLAIN_TEXT,
        )
        return client.analyze_entities(document=document)
    
    
    def analyze_document_entities(self, df: pd.DataFrame):
        documents = []

        for index, row in df.iterrows():
            if len(documents) >= self.num_doc:
                break  # Stop after processing the specified number of documents

            linetext_cleaned = row.get(self.text_column, '')

            if linetext_cleaned:
                # Perform entity analysis on the cleaned text
                response = self.analyze_text_entities(linetext_cleaned)
                entity_type_counts = {}
                entities_data = {}

                for entity in response.entities:
                    entity_type = entity.type_.name
                    salience = entity.salience

                    # Update entity counts
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

                    # Append salience to the list corresponding to the entity type
                    if entity_type not in entities_data:
                        entities_data[entity_type] = []
                        #print(entities_data)
                    entities_data[entity_type].append(salience)

                # Calculate mean salience for each entity type
                mean_salience_per_entity_type = {}
                for entity_type, salience_list in entities_data.items():
                    mean_salience_per_entity_type[entity_type] = sum(salience_list) / len(salience_list)
                    #print(mean_salience_per_entity_type)

                # Create a copy of the current row as a dict and update it with entity counts
                document = row.to_dict()
                document.update(entity_type_counts)

                # Add mean salience for each entity type to the document
                for entity_type, mean_salience in mean_salience_per_entity_type.items():
                    document[f"{entity_type}_mean_salience"] = mean_salience

                documents.append(document)


        # Create a new DataFrame that includes the original data and the new entity count columns
        df_entity = pd.DataFrame(documents)

        # Fill missing entity count values with 0
        entity_columns = [col for col in df_entity.columns if col not in df.columns]
        df_entity[entity_columns] = df_entity[entity_columns].fillna(0).astype(float)

        return df_entity

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

            print("Extracting Entity")
            self.logger.info("Extracting Entity")
            df_entity = self.analyze_document_entities(df)
            self.save_df_to_gcs_parquet(df_entity)
            print("Entity extracted - process completed and file saved to GCS.")
            self.logger.info("Entity extracted - process completed and file saved to GCS.")
            return self.parquet_file_name
        
