import os
import logging
import pandas as pd
from google.cloud import storage
from io import BytesIO
import numpy as np
from sklearn.preprocessing import RobustScaler

class GCSParquetLoader:
    def __init__(self, bucket: str, file_path: str, folder: str, parquet_file_name: str):
        self.bucket = bucket
        self.file_path = file_path
        self.client: storage.Client = storage.Client()
        self.folder = folder
        self.parquet_file_name = parquet_file_name
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
        
        df_scaled = df_no_dups_remov.copy()
        
        # Create a RobustScaler object
        scaler: RobustScaler = RobustScaler()

        # Apply the scaler to the 'sharesFacebook' column
        df_scaled['shares_scaled'] = scaler.fit_transform(df_scaled['sharesFacebook'].values.reshape(-1, 1))

        return df_scaled


        # Define a function to assign labels based on date ranges
    def split(self, df: pd.DataFrame) -> pd.DataFrame:
    
        df_2023_cl: pd.DataFrame = self.analyze_articles(df)
        
        def assign_labels(group: pd.DataFrame) -> pd.Series:

            total_count: int = len(group)
            train_count: int = int(0.95 * total_count)  # 95% of observations for TRAIN
            validate_test_count: int = (total_count - train_count) // 2  # Remaining observations for VALIDATE and TEST

            # Assign labels based on date ranges
            group.loc[group.index[:train_count], 'split'] = 'TRAIN'
            group.loc[group.index[train_count:train_count + validate_test_count], 'split'] = 'VALIDATE'
            group.loc[group.index[train_count + validate_test_count:], 'split'] = 'TEST'

            return group['split']  # Return the 'split' column as a Series


        # Sort the DataFrame by date
        df_sorted: pd.DataFrame = df_2023_cl.sort_values(by='date_column')

        # Group the DataFrame by topic
        grouped: pd.DataFrameGroupBy = df_sorted.groupby('topic')
        # Apply the function to each group and concatenate the results
        split_series: pd.Series = pd.concat([assign_labels(group) for _, group in grouped])

        # Assign the resulting Series to a new column in the original DataFrame
        df_2023_cl['split'] = split_series

        # Count the number of different splits for each topic
        split_counts = df_2023_cl.groupby('topic')['split'].value_counts()

        # Print the result
        print(split_counts)

        # Filter the DataFrame to include only "TRAIN" observations
        train_df: pd.DataFrame = df_2023_cl[df_2023_cl['split'] == 'TRAIN']

        # Downsample the "TRAIN" section
        minority_class_size = min(train_df['topic'].value_counts())

        downsampled_train_df = train_df.reset_index(drop=True).groupby('topic').apply(lambda x: x.sample(minority_class_size)).reset_index(drop=True)

        print(downsampled_train_df['topic'].value_counts())

        # Concatenate downsampled DataFrames for "TEST" and "VALIDATION" splits
        final_df = pd.concat([downsampled_train_df, df_2023_cl[df_2023_cl['split'] == 'VALIDATE'], df_2023_cl[df_2023_cl['split'] == 'TEST']])
        final_df.reset_index(drop=True)
        print(final_df['split'].value_counts())
        print("Dataset splitted - process completed and file saved to GCS.")
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
            df_splitted = self.split(df_preprocessed)
            
            self.save_df_to_gcs_parquet(df_preprocessed)
            print("Dataset preprocessed - process completed and file saved to GCS.")
            self.logger.info("Dataset preprocessed- process completed and file saved to GCS.")
            return self.parquet_file_name
