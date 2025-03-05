import pandas as pd
from google.cloud import storage, bigquery
from io import BytesIO
import logging
import os

class GCS_Bigquery:
    def __init__(self, bucket_name: str, file_path: str, folder: str, parquet_file_name: str, project_id: str, dataset_id: str, table_id: str, location: str):
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.folder = folder
        self.parquet_file_name = parquet_file_name
        self.project_id = project_id
        self.client = storage.Client()
        self.location = location
        self.logger = self._configure_logger()
        self.dataset_id = dataset_id
        self.table_id = table_id

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('gcs_parquet_loader.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

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
           
        
    def upload_dataframe_to_bigquery(self):
        # Create a BigQuery client object
        client = bigquery.Client(project=self.project_id, location=self.location)  # Change location if necessary

        # Convert pandas DataFrame to CSV
        csv_filename = "temp_data.csv"
    
        try:
            df = self.load_parquet_from_gcs()
            df = df.drop(columns="DATE")
            df.to_csv(csv_filename, index=False, sep = '|')  # Quote all fields to ensure proper handling of special characters
            # Define the Google Cloud Storage (GCS) URI for the CSV file
            gcs_uri = f"gs://{self.bucket_name}/{csv_filename}"

            # Define the BigQuery table ID
            bq_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

            # Define the job configuration for loading the CSV into BigQuery
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,
                allow_quoted_newlines=True,
                field_delimiter="|",
                autodetect=True  # Automatically detect schema from CSV
            )

            # Replace the file CSV
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

            # Load the CSV file into BigQuery
            with open(csv_filename, "rb") as source_file:
                job = client.load_table_from_file(
                    source_file,
                    bq_table_id,
                    job_config=job_config,
                )

            # Wait for the job to complete
            job.result()

            # Get the destination table
            destination_table = client.get_table(bq_table_id)

            # Print the number of rows loaded
            print(f"Loaded {destination_table.num_rows} rows into BigQuery table: {bq_table_id}")
            
            return self.table_id
            print(self.table_id)
            
        except Exception as e:
            print(f"Error uploading DataFrame to BigQuery: {e}")

        finally:
            # Clean up: delete the temporary CSV file
            if os.path.exists(csv_filename):
                os.remove(csv_filename)
