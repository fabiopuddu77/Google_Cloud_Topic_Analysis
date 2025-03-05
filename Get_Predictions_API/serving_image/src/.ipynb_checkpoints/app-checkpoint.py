from fastapi import FastAPI, HTTPException
from typing import List, Dict
import pandas as pd
from processing.inference_preprocessing import GCS_preprocessing  # Import your preprocessing module
from google.cloud import aiplatform as aip
from tabulate import tabulate

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
FOLDER = 'make_prediction'
PROJECT_ID = 'gcp-ccai-auto-ml-contactcenter'
TABLE_ID = "testdatabq"
TEXT_COLUMN = 'body_pre'
LOCATION = "europe-west3"
RANDOM_SEED=123
#BQ_SOURCE = "bq://gcp-ccai-auto-ml-contactcenter.datasetnlp.stepfinalbq"

OUTPUT_FINAL = 'step_final_bq.parquet'

# Resources
DEPLOY_COMPUTE = 'n1-standard-4'


#FILE_PATH = 'test_file.parquet'
#NUM_DOC = 20
app = FastAPI()

# Initialize your AIP endpoint
aip.init(project=PROJECT_ID, location=LOCATION)

# Define your preprocessing function
def preprocess_data(file_path: str, num_doc: int) -> List[Dict]:
    # Perform preprocessing using your GCS_preprocessing class
    processor = GCS_preprocessing(bucket=BUCKET,
                                  folder=FOLDER,
                                  file_path=file_path,
                                  parquet_file_name=OUTPUT_FINAL,
                                  num_doc=num_doc, 
                                  random_seed=RANDOM_SEED,
                                  project_id=PROJECT_ID,
                                  dataset_id=DATANAME,
                                  table_id=TABLE_ID,
                                  location=LOCATION,
                                  text_column=TEXT_COLUMN,
                                  pipeline_root=PIPELINE_ROOT,
                                  overwrite=True)
    
    instances = processor.process()
    return instances

@app.post("/predict/")

def predict(file_path: str, num_doc: int):
    print("----- Running Predict ---------")
    try:
        # Preprocess the data
        instances = preprocess_data(file_path, num_doc)
        print("----- Instances created ---------")
        # Get the AIP endpoint
        endpoint = aip.Endpoint.list(filter=f'labels.notebook={NOTEBOOK}')[0]
        print("----- Endopoint list ---------")
        # Make predictions
        prediction = endpoint.predict(instances=instances)
        print("----- Prediction Done ---------")
        # Process the prediction results and return them
        # (Assuming prediction.predictions[2] contains the relevant data)
        print("prediction.predictions -------- ",prediction.predictions)
        predictions = prediction.predictions

        table_data = []

        for i, prediction_data in enumerate(predictions):
            classes = prediction_data['classes']
            scores = prediction_data['scores']
            for class_, score in zip(classes, scores):
                table_data.append((class_, score))

        # Print the table
        table = tabulate(table_data, headers=['Class', 'Score'], tablefmt='grid')
        print(table)
        return table
        #return {"predictions": table_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__=='__main__':
    app.run(host='0.0.0.0', port=8081)
