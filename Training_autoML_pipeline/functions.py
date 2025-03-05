
from google.cloud import bigquery
import pandas as pd
import os
import csv 

def save_df_to_gcs_parquet(df, bucket_name, folder, destination_blob_name):
    """Save a DataFrame to a Parquet file in GCS."""
    parquet_byte_stream = BytesIO()
    df.to_parquet(parquet_byte_stream, index=False, engine='pyarrow')
    parquet_byte_stream.seek(0)
    
    path = folder + "/" + destination_blob_name
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    
    blob.upload_from_file(parquet_byte_stream, content_type='application/octet-stream')
    
    # Function to download a Parquet file from GCS and load it into a pandas DataFrame
def load_parquet_from_gcs(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)
    df = pd.read_parquet(byte_stream, engine='pyarrow')
    return df


def upload_dataframe_to_bigquery(dataframe: pd.DataFrame, table_id: str, project_id: str, dataset_id: str, bucket_name: str, location: str) -> None:
    # Create a BigQuery client object
    client = bigquery.Client(project=project_id, location=location)  # Change location if necessary

    # Convert pandas DataFrame to CSV
    csv_filename = "temp_data.csv"
    dataframe.to_csv(csv_filename, index=False, sep = '|')  # Quote all fields to ensure proper handling of special characters

    try:
        # Define the Google Cloud Storage (GCS) URI for the CSV file
        gcs_uri = f"gs://{bucket_name}/{csv_filename}"

        # Define the BigQuery table ID
        bq_table_id = f"{project_id}.{dataset_id}.{table_id}"

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

    except Exception as e:
        print(f"Error uploading DataFrame to BigQuery: {e}")

    finally:
        # Clean up: delete the temporary CSV file
        if os.path.exists(csv_filename):
            os.remove(csv_filename)