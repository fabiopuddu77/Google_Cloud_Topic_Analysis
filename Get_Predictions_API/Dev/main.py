


def main():
    
    from pipeline_preprocessing import DataProcessor

    import kfp
    from kfp import compiler
    from kfp.dsl import component, pipeline, Artifact, ClassificationMetrics, Input, Output, Model, Metrics

    from google.cloud import aiplatform as aip
    from typing import NamedTuple

    from datetime import datetime

    #import kfp.v2.dsl as dsl
    #import google_cloud_pipeline_components as gcc_aip
    from google_cloud_pipeline_components.v1.dataset import TabularDatasetCreateOp
    from google_cloud_pipeline_components.v1.automl.training_job import AutoMLTabularTrainingJobRunOp
    from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

    from google.cloud import bigquery 
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    import json
    import numpy as np
    
    
    PROJECT_ID = 'gcp-ccai-auto-ml-contactcenter'
    REGION= "europe-west3"
    REPO_NAME = "repo-demo3"
    SERVICE_ACCOUNT = "944308723981-compute@developer.gserviceaccount.com"
    BUCKET = "ccai-storage"
    PIPELINE_NAME = "automl_pipeline"
    YAML_NAME = f"{PIPELINE_NAME}.yml"
    PIPELINE_ROOT = f"gs://{BUCKET}/pipeline_root/"
    DISPLAY_NAME = PIPELINE_NAME.replace("_", "-")
    NOTEBOOK = "automl"
    DATANAME = "datasetnlp"
    FILE_PATH = 'test_file.parquet'
    FOLDER = 'make_prediction'
    PROJECT_ID = 'gcp-ccai-auto-ml-contactcenter'
    TABLE_ID = "testdatabq"
    TEXT_COLUMN = 'body_pre'
    LOCATION = "europe-west3"
    NUM_DOC = 20
    RANDOM_SEED=123
    #BQ_SOURCE = "bq://gcp-ccai-auto-ml-contactcenter.datasetnlp.stepfinalbq"
    OUTPUT_PROCESSING = 'output_processing.parquet'
    OUTPUT_TOKENIZATION = 'output_tokenized.parquet'
    OUTPUT_SENTIMENT = 'output_sentiment.parquet'
    OUTPUT_MODERATE = 'output_moderate.parquet'
    OUTPUT_ENTITIES = 'output_entities.parquet'
    OUTPUT_FINAL = 'step_final_bq.parquet'

    # Resources
    DEPLOY_COMPUTE = 'n1-standard-4'
    
    aip.init(project=PROJECT_ID, staging_bucket=PIPELINE_ROOT, location=REGION)
    bq = bigquery.Client()
    
    # Initialize DataProcessor object
    data_processor = DataProcessor(bucket=BUCKET, folder=FOLDER, num_doc=NUM_DOC, random_seed=RANDOM_SEED)

    # Step 1: Data Preprocessing
    processed_data_path = data_processor.data_preprocessing(file_path=FILE_PATH,
                                                            parquet_file_name=OUTPUT_PROCESSING,
                                                            num_doc=NUM_DOC, random_seed=RANDOM_SEED)
    print("Data Preprocessing completed. Processed data saved at:", processed_data_path)

    # Step 2: Tokenization
    tokenized_data_path = data_processor.data_tokenization(file_path=processed_data_path,
                                                           parquet_file_name=OUTPUT_TOKENIZATION)
    print("Tokenization completed. Tokenized data saved at:", tokenized_data_path)

    # Step 3: Sentiment Analysis
    sentiment_data_path = data_processor.data_sentiment(file_path=tokenized_data_path, 
                                                        parquet_file_name=OUTPUT_SENTIMENT,
                                                        text_column=TEXT_COLUMN, num_doc=NUM_DOC)
    print("Sentiment Analysis completed. Sentiment data saved at:", sentiment_data_path)

    # Step 4: Text Moderation
    moderated_data_path = data_processor.data_moderate(file_path=sentiment_data_path, 
                                                       parquet_file_name=OUTPUT_MODERATE, 
                                                       text_column=TEXT_COLUMN,
                                                       num_doc=NUM_DOC)
    print("Text Moderation completed. Moderated data saved at:", moderated_data_path)

    # Step 5: Entity Analysis
    entity_data_path = data_processor.data_entities(file_path=moderated_data_path,
                                                    parquet_file_name=OUTPUT_ENTITIES,
                                                    text_column=TEXT_COLUMN,
                                                    num_doc=NUM_DOC)
    print("Entity Analysis completed. Entity data saved at:", entity_data_path)

    # Step 6: BigQuery Upload
    bigquery_upload_status = data_processor.data_bigquery(file_path=entity_data_path, 
                                                          parquet_file_name=OUTPUT_FINAL, 
                                                          project_id=PROJECT_ID, 
                                                          dataname=DATANAME, 
                                                          table_id=TABLE_ID, 
                                                          location=LOCATION)
    print("BigQuery Upload completed. Status:", bigquery_upload_status)

    
    # Model Training
    VAR_TARGET = 'topic'
    VAR_OMIT = (
        'uri,url,date,body,time,dateTime,dateTimePub,lang,isDuplicate,dataType,sentiment,' +
        'eventUri,image,sharesFacebook,' +
        'sourceLocationLabel,categoryLabels,' +
        'categoryWeights,' +
        'alexaCountryRank,date_column,year,year_month,' +
        'num_documents,' +
        'PERSON,OTHER,ORGANIZATION,' +
        'EVENT,LOCATION,WORK_OF_ART,CONSUMER_GOOD,NUMBER,DATE,' +
        'NUMBER_mean_salience,' +
        'DATE_mean_salience,PRICE,ADDRESS,' +
        'ADDRESS_mean_salience,PHONE_NUMBER,PHONE_NUMBER_mean_salience'
    )
    
    # Remove the duplicate date from the string
    VAR_OMIT_rev = VAR_OMIT.replace("DATE,", "")
    
    print("Create Dataframe prediction")
    pred = bq.query(
    query = f"""
        SELECT * EXCEPT({VAR_TARGET}, {VAR_OMIT_rev})
        FROM {DATANAME}.{TABLE_ID}
        LIMIT 10
    """
).to_dataframe()
    
    print("Adapt the variables to the autoML")
    pred['relevance'] = pred['relevance'].astype(str)
    pred['importanceRank'] = pred['importanceRank'].astype(str)
    pred['alexaGlobalRank'] = pred['alexaGlobalRank'].astype(str)
    pred['month'] = pred['month'].astype(str)

    newobs = pred.to_dict(orient='records')
    
    print("Create instances")
    instances = [json_format.ParseDict(newob, Value()) for newob in newobs]
    
    return newobs, instances, NOTEBOOK


