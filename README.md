# Data Processing Pipeline README

## Directory Structure

- **Notebooks Pipeline Preprocessing/**: This directory contains Jupyter Notebooks for the pipeline modules

  - `1.pipeline_token.ipynb`: Notebook for loading and preprocessing data from Google Cloud Storage (GCS).
  - `1.pipeline_token.ipynb`: Notebook for tokenizing data.
  - `2.pipeline_sentiment.ipynb`: Notebook for performing sentiment analysis on text data.
  - `3.pipeline_moderate.ipynb`: Notebook for moderating text data.
  - `4.pipeline_entities.ipynb`: Notebook for extracting entities from text data.
  - `5.pipeline_save_BQ.ipynb`: Notebook for uploading data to BigQuery.
  - `functions.ipynb`: Contains the functions to communicate with the GCS and BigQuery 



- **Get Prediction API /**: This directory contains RUN_make_prediction.ipynb to run preprocessing and call the API from a test sample. 

  - `processing/data_preparation.py`: Defines a class `GCSParquetLoader` for loading and preprocessing data from Google Cloud Storage (GCS).
  - `processing/tokenization.py`: Defines a class `TokenizationProcessor` for tokenizing data.
  - `processing/sentiment.py`: Defines a class `GCSSentimentAnalyzer` for performing sentiment analysis on text data.
  - `processing/moderate.py`: Defines a class `GCSTextModerationLoader` for moderating text data.
  - `processing/entities.py`: Defines a class `GCSCEntityAnalyzer` for extracting entities from text data.
  - `processing/bigquery.py`: Defines a class `GCS_Bigquery` for uploading data to BigQuery.
  
  - `Get Predictions Client.ipynb`: Contains the script to get the predicions from a test dataset by Client
  
  - `Get Predictions REST.ipynb`: Contains the script to get the predicions from a test dataset by REST
  
   - `pipeline_preprocessing.py`: Contains a class `DataProcessor` that orchestrates the data processing pipeline by calling different processing steps.
   
    - `main.py`: Script to execute the data processing pipeline.
    - `request.json`: File json output of the AutoML model trained.

- **Training autoML pipeline /**: 

        - 1.RUN_pipeline_prepr_and_train_autoML.ipynb to create the pipeline and the autoML api. 
        - custom folders contains docker images

## Usage

1. **Setting up Google Cloud Storage (GCS)**:
    - Make sure you have access to a GCS bucket where your data is stored.

2. **Running the Training Pipeline**:
    - in the folder Training autoML pipeline you will find the notebook RUN_pipeline_prepr_and_train_autoML that creates a pipeline of preprocessing and deploying the model in an EndPoint.
    - Get predictions API will get the predictions from a test data.


## Dependencies

- Python 3.x
- Google Cloud Storage (google-cloud-storage)
- DataFlow
- BigQuery
- Language API
- Vertex AI
- Kubeflow
- pandas
- numpy
- scikit-learn

## Contributors

