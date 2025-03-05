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
        print(f"Start loading DataFrame from {self.file_path}")
        self.logger.info(f"Start loading DataFrame from {self.file_path}")
        bucket: storage.Bucket = self.client.bucket(self.bucket_name)
        blob: storage.Blob = bucket.blob(self.file_path)
        byte_stream: BytesIO = BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)
        df: pd.DataFrame = pd.read_parquet(byte_stream, engine='pyarrow')
        self.logger.info(f"Loaded DataFrame from {self.file_path}")
        print(f"Loaded DataFrame from {self.file_path}")
        return df
    
    def analyze_articles(self) -> pd.DataFrame:
        print("Analyze articles")
        articles_df: pd.DataFrame = self.load_parquet_from_gcs()

        # Drop duplicates by the 'uri' column
        df_no_dups: pd.DataFrame = articles_df.drop_duplicates(subset=['uri'])

        # Count the occurrences of each unique value in the 'uri' column
        uri_value_counts: pd.Series = df_no_dups['uri'].value_counts()

        print("Replace empty strings with NaN")
        df_no_dups.replace('', np.nan, inplace=True)

        print("Remove rows where both 'title' and 'authors' are NaN")
        df_no_dups_remov: pd.DataFrame = df_no_dups.dropna(subset=['categoryLabels', 'authors'])

        # Keep just the first topic in each entry
        df_no_dups_remov['topic'] = df_no_dups_remov['categoryLabels'].str.split(';').str[0]
        df_no_dups_remov = df_no_dups_remov[df_no_dups_remov['topic'].str.startswith('news')]
        
        print("Remove 'news/' prefix from the 'topic' column")
        df_no_dups_remov['topic'] = df_no_dups_remov['topic'].str.replace('news/', '')

        df_2023_cl: pd.DataFrame = df_no_dups_remov.copy()

        # Sort the DataFrame by date
        df_sorted: pd.DataFrame = df_2023_cl.sort_values(by='date_column')

        # Group the DataFrame by topic
        grouped: pd.DataFrameGroupBy = df_sorted.groupby('topic')

        # Define a function to assign labels based on date ranges
        def assign_labels(group: pd.DataFrame) -> pd.Series:
            
            print("Assign labels")
            total_count: int = len(group)
            train_count: int = int(0.95 * total_count)  # 95% of observations for TRAIN
            validate_test_count: int = (total_count - train_count) // 2  # Remaining observations for VALIDATE and TEST

            # Assign labels based on date ranges
            group.loc[group.index[:train_count], 'split'] = 'TRAIN'
            group.loc[group.index[train_count:train_count + validate_test_count], 'split'] = 'VALIDATE'
            group.loc[group.index[train_count + validate_test_count:], 'split'] = 'TEST'

            return group['split']  # Return the 'split' column as a Series

        # Apply the function to each group and concatenate the results
        split_series: pd.Series = pd.concat([assign_labels(group) for _, group in grouped])

        # Assign the resulting Series to a new column in the original DataFrame
        df_2023_cl['split'] = split_series

        # Filter the DataFrame to include only "TRAIN" observations
        train_df: pd.DataFrame = df_2023_cl[df_2023_cl['split'] == 'TRAIN']

        # Concatenate downsampled DataFrames for "TEST" and "VALIDATION" splits
        final_df: pd.DataFrame = pd.concat([downsampled_train_df, df_2023_cl[df_2023_cl['split'] == 'VALIDATE'], df_2023_cl[df_2023_cl['split'] == 'TEST']])
        final_df.reset_index(drop=True)

        # Create a RobustScaler object
        scaler: RobustScaler = RobustScaler()

        # Apply the scaler to the 'sharesFacebook' column
        final_df['shares_scaled'] = scaler.fit_transform(final_df['sharesFacebook'].values.reshape(-1, 1))

        return final_df

    def save_df_to_gcs_parquet(self, df: pd.DataFrame, folder: str, parquet_file_name: str):
        
        print("Save df to gcs")
        # Define the destination path in the GCS bucket
        destination_blob_name: str = os.path.join(folder, parquet_file_name)

        # Convert DataFrame to Parquet format and write it to BytesIO object
        byte_stream: BytesIO = BytesIO()
        df.to_parquet(byte_stream, engine='pyarrow')
        byte_stream.seek(0)

        # Upload the Parquet file to GCS
        bucket: storage.Bucket = self.client.bucket(self.bucket_name)
        blob: storage.Blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(byte_stream)