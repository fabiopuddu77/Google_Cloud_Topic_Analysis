import logging
import pandas as pd
from google.cloud import storage
from io import BytesIO
import numpy as np
from sklearn.preprocessing import RobustScaler
import random

class GCSParquetLoader:
    def __init__(self, bucket: str, folder: str, file_path: str, parquet_file_name: str, num_doc: int, random_seed: int):
        self.bucket = bucket
        self.file_path = file_path
        self.client: storage.Client = storage.Client()
        self.folder = folder
        self.parquet_file_name = parquet_file_name
        self.num_doc = num_doc
        self.random_seed = random_seed
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
    
    def analyze_articles(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        print("Analyze articles")
        random.seed(self.random_seed)
        articles_df = articles_df.sample(n=self.num_doc, random_state=self.random_seed)
        
        #articles_df=articles_df.head(self.num_doc)
        # Extract the year and month components
        articles_df['date_column'] = pd.to_datetime(articles_df['date'], errors='coerce') 
        articles_df['year'] = articles_df['date_column'].dt.year
        articles_df['month'] = articles_df['date_column'].dt.month

        # Concatenate year and month into a single column
        articles_df['year_month'] = articles_df['year'].astype(str) + '-' + articles_df['month'].astype(str)

        # Filter the DataFrame to keep only the observations for the year 2023
        df_2023 = articles_df[articles_df['date_column'].dt.year == 2023]
        
        # Drop duplicates by the 'uri' column
        df_no_dups = df_2023.drop_duplicates(subset=['uri'])

        # Count the occurrences of each unique value in the 'uri' column
        uri_value_counts: pd.Series = df_no_dups['uri'].value_counts()

        print("Replace empty strings with NaN")
        df_no_dups.replace('', np.nan, inplace=True)
        #df_no_dups.loc[df_no_dups[''] == '', :] = np.nan
        
        print("Remove rows where both 'title' and 'authors' are NaN")
        df_no_dups_remov: pd.DataFrame = df_no_dups.dropna(subset=['categoryLabels', 'authors'])
        
        # Keep just the first topic in each entry
        df_no_dups_remov.loc[:,'topic'] = df_no_dups_remov.loc[:,'categoryLabels'].str.split(';').str[0]
        df_no_dups_remov = df_no_dups_remov[df_no_dups_remov['topic'].str.startswith('news')]
        # Remove 'news/' prefix from the 'topic' column
        df_no_dups_remov.loc[:,'topic'] = df_no_dups_remov.loc[:,'topic'].str.replace('news/', '')
        
        final_df: pd.DataFrame = df_no_dups_remov.copy()


        # Create a RobustScaler object
        scaler: RobustScaler = RobustScaler()

        # Apply the scaler to the 'sharesFacebook' column
        final_df['shares_scaled'] = scaler.fit_transform(final_df['sharesFacebook'].values.reshape(-1, 1))

        return final_df
        
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
            
            print("Loading Data Preparation")
            self.logger.info("Loading Data Preparation")
            df_preprocessed = self.analyze_articles(df)
            self.save_df_to_gcs_parquet(df_preprocessed)
            print("Dataset preprocessed - process completed and file saved to GCS.")
            self.logger.info("Dataset preprocessed- process completed and file saved to GCS.")
            return self.parquet_file_name