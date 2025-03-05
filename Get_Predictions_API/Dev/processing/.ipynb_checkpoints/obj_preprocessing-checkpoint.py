import logging
import pandas as pd
from google.cloud import storage, bigquery
from io import BytesIO
import numpy as np
from sklearn.preprocessing import RobustScaler
import random
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from html.parser import HTMLParser
from google.cloud import language_v1 as language
from google.cloud import aiplatform as aip
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import json
    
class GCS_preprocessing:
    def __init__(self, bucket: str, folder: str, file_path: str, parquet_file_name: str, num_doc: int, random_seed: int, project_id: str, dataset_id: str, table_id: str, location: str, text_column: str, pipeline_root: str, overwrite: bool = False):
        self.bucket = bucket
        self.file_path = file_path
        self.client: storage.Client = storage.Client()
        self.folder = folder
        self.parquet_file_name = parquet_file_name
        self.num_doc = num_doc
        self.random_seed = random_seed
        self.location = location
        self.project_id = project_id
        self.logger = self._configure_logger()
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.text_column = text_column
        self.overwrite = overwrite
        self.pipeline_root = pipeline_root
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('italian'))

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler('gcs_parquet_loader.log')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger
    
    def file_exists_in_gcs(self) -> bool:
        if self.overwrite:
            return False
        self.logger.info(f"Check if the file in exists in gcs {self.parquet_file_name}")
        bucket = self.client.bucket(self.bucket)
        blob = bucket.blob(f"{self.folder}/{self.parquet_file_name}")
        return blob.exists()




    def load_parquet_from_gcs(self) -> pd.DataFrame:
        try:
            print(f"Start loading DataFrame from {self.file_path}")
            self.logger.info(f"Start loading DataFrame from {self.file_path}")
            print(f"Start loading DataFrame from {self.file_path}\n")
            bucket: storage.Bucket = self.client.bucket(self.bucket)
            blob: storage.Blob = bucket.blob(f"{self.folder}/{self.file_path}")
            byte_stream = BytesIO()
            blob.download_to_file(byte_stream)
            byte_stream.seek(0)
            df: pd.DataFrame = pd.read_parquet(byte_stream, engine='pyarrow')
            print("INPUT DATAFRAME\n",df.head())
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

        #print("Replace empty strings with NaN")
        df_no_dups.replace('', np.nan, inplace=True)
        #df_no_dups.loc[df_no_dups[''] == '', :] = np.nan
        
        #print("Remove rows where both 'title' and 'authors' are NaN")
        df_no_dups_remov: pd.DataFrame = df_no_dups.dropna(subset=['categoryLabels', 'authors'])
        
        # Keep just the first topic in each entry
        df_no_dups_remov.loc[:,'topic'] = df_no_dups_remov.loc[:,'categoryLabels'].str.split(';').str[0]
        df_no_dups_remov = df_no_dups_remov[df_no_dups_remov['topic'].str.startswith('news')]
        # Remove 'news/' prefix from the 'topic' column
        df_no_dups_remov.loc[:,'topic'] = df_no_dups_remov.loc[:,'topic'].str.replace('news/', '')
        
        df: pd.DataFrame = df_no_dups_remov.copy()
        
        # Create a RobustScaler object
        scaler: RobustScaler = RobustScaler()

        # Apply the scaler to the 'sharesFacebook' column
        df['shares_scaled'] = scaler.fit_transform(df['sharesFacebook'].values.reshape(-1, 1))

        return df
    
    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        cleaned_text = ' '.join(filtered_tokens)
        return cleaned_text

    def preprocess_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Starting Tokenization DataFrame from {self.file_path}")
        print(f"Starting Tokenization DataFrame from {self.file_path}")
        df['body_pre'] = df['body'].apply(self.preprocess_text)
        return df
    
    
    def analyze_text_sentiment(self, row) -> dict:
        client = language.LanguageServiceClient()
        text = row[self.text_column]
        document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
        response = client.analyze_sentiment(request={'document': document})
        return {
            'score': response.document_sentiment.score,
            'magnitude': response.document_sentiment.magnitude
        }

    def analyze_and_merge_sentiments(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Extracting Sentiment")
        df_merged = df.sample(n=self.num_doc)
        # Assuming 'text_column' is the name of the column containing the text data
        # Create new columns 'score' and 'magnitude' using the analyze_text_sentiment function
        df_merged[['score', 'magnitude']] = df_merged[self.text_column].apply(lambda x: pd.Series(self.analyze_text_sentiment({self.text_column: x})))
        return df_merged
 

    @staticmethod
    def strip_html_tags(html: str) -> str:
        class MLStripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.reset()
                self.strict = False
                self.convert_charrefs = True
                self.text = []
            def handle_data(self, d):
                self.text.append(d)
            def get_data(self):
                return ''.join(self.text)

        s = MLStripper()
        s.feed(html)
        return s.get_data()


    def moderate_text(self, text: str) -> language.ModerateTextResponse:
        client = language.LanguageServiceClient()
        document = language.Document(
            content=text,
            type_=language.Document.Type.PLAIN_TEXT,
        )
        
        return client.moderate_text(document=document)
        
   

    def analyze_and_moderate_documents(self, df: pd.DataFrame):
        print("Starting Data Moderation\n")
        documents = []
        for index, row in df.iterrows():
            linetext_cleaned = row.get(self.text_column, '')
            if linetext_cleaned:
                # Perform moderation analysis on the cleaned text
                response = self.moderate_text(linetext_cleaned)
                categories = []
                confidences = []
                for category in response.moderation_categories:
                    categories.append(category.name)
                    confidences.append(category.confidence)
                document = row.to_dict()
                for category, confidence in zip(categories, confidences):
                    # Create new columns for each category with confidence scores
                    document[category] = confidence
                documents.append(document)
                if len(documents) >= self.num_doc:
                    break  # Stop after processing the specified number of documents

        df_moderated = pd.DataFrame(documents)
        
        return df_moderated


    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.replace('&', '').str.replace(r'\s+', ' ').str.replace(',', '').str.replace(' ', '_')
        return df

    
    
    def analyze_text_entities(self, text: str) -> language.AnalyzeEntitiesResponse:
        client = language.LanguageServiceClient()
        document = language.Document(
            content=text,
            type_=language.Document.Type.PLAIN_TEXT,
        )
        return client.analyze_entities(document=document)
    
    
    def analyze_document_entities(self, df: pd.DataFrame):
        print("Extracting Entities")
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

        
        # Ensure all required columns are present
        required_columns = ['uri', 'url', 'title', 'body', 'date', 'time', 'dateTime',
       'dateTimePub', 'lang', 'isDuplicate', 'dataType', 'sentiment',
       'eventUri', 'relevance', 'image', 'authors', 'sharesFacebook',
       'sourceTitle', 'sourceLocationLabel', 'categoryLabels',
       'categoryWeights', 'importanceRank', 'alexaGlobalRank',
       'alexaCountryRank', 'date_column', 'year', 'month', 'year_month',
       'topic', 'split', 'shares_scaled', 'body_pre', 'score', 'magnitude',
       'num_documents', 'Toxic', 'Insult', 'Profanity', 'Derogatory', 'Sexual',
       'Death_Harm__Tragedy', 'Violent', 'Firearms__Weapons', 'Public_Safety',
       'Health', 'Religion__Belief', 'Illicit_Drugs', 'War__Conflict',
       'Politics', 'Finance', 'Legal', 'PERSON', 'OTHER', 'ORGANIZATION',
       'EVENT', 'LOCATION', 'WORK_OF_ART', 'CONSUMER_GOOD', 'NUMBER',
       'PERSON_mean_salience', 'OTHER_mean_salience',
       'ORGANIZATION_mean_salience', 'EVENT_mean_salience',
       'LOCATION_mean_salience', 'WORK_OF_ART_mean_salience',
       'CONSUMER_GOOD_mean_salience', 'NUMBER_mean_salience',
       'DATE_mean_salience', 'PRICE', 'PRICE_mean_salience', 'ADDRESS',
       'ADDRESS_mean_salience', 'PHONE_NUMBER', 'PHONE_NUMBER_mean_salience']

        for column in required_columns:
            if column not in df_entity.columns:
                df_entity[column] = 0

        print("DATAFRAME PROCESSED\n",df_entity.head())
        return df_entity

     
    def upload_dataframe_to_bigquery(self, df: pd.DataFrame):
        # Create a BigQuery client object
        client = bigquery.Client(project=self.project_id, location=self.location)  # Change location if necessary

        # Convert pandas DataFrame to CSV
        csv_filename = "temp_data.csv"
    
        try:
            # Check if the column "DATE" exists
            if "DATE" in df.columns:
                # Drop the column "DATE" if it exists
                df = df.drop(columns="DATE")
                df.to_csv(csv_filename, index=False, sep = '|')  # Quote all fields to ensure proper handling of special characters
            else:
                
                df.to_csv(csv_filename, index=False, sep = '|')  # Quote all fields to ensure proper handling of special characters
                
            #df.to_csv(csv_filename, index=False, sep = '|')   
            # Define the Google Cloud Storage (GCS) URI for the CSV file
            gcs_uri = f"gs://{self.bucket}/{csv_filename}"

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
            self.logger.info(f"Error uploading DataFrame to BigQuery: {e}")
            print(f"Error uploading DataFrame to BigQuery: {e}")

        finally:
            # Clean up: delete the temporary CSV file
            if os.path.exists(csv_filename):
                os.remove(csv_filename)

                
    def save_df_to_gcs_parquet(self, df: pd.DataFrame):
        destination_blob_name = f'{self.folder}/{self.parquet_file_name}'
        byte_stream = BytesIO()
        df.to_parquet(byte_stream, engine='pyarrow')
        byte_stream.seek(0)
        bucket = self.client.bucket(self.bucket)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_file(byte_stream)
        
    def pred(self):
    
        aip.init(project=self.project_id, staging_bucket=self.pipeline_root, location=self.location)
        bq = bigquery.Client()
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
            SELECT * EXCEPT( topic, {VAR_OMIT_rev})
            FROM {self.dataset_id}.{self.table_id}
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
        
        return instances

        
    def process(self):
        if isinstance(self.file_path, dict):
            # JSON object provided, load directly into DataFrame
            df = pd.DataFrame.from_dict(self.file_path, orient='index').T
        elif "_JSON" in self.file_path:
            # Load JSON file into DataFrame
            try:
                df = pd.read_json(self.file_path)
            except Exception as e:
                self.logger.error(f"Error loading JSON file: {e}")
                print(f"Error loading JSON file: {e}")
                return None
            
        else:
            if not self.overwrite and self.file_exists_in_gcs():
                self.logger.info(f"{self.parquet_file_name} already exists in GCS. Skipping process.")
                return self.pred()

            elif self.file_path.endswith('.parquet'):
                # Load Parquet file from GCS
                df = self.load_parquet_from_gcs()
                if df is None:
                    self.logger.info(f"File {self.file_path} doesn't exist.")
                    return None
            else:
                self.logger.error("Unsupported file format. File must be either JSON or Parquet.")
                return None
            
            
            self.logger.info("Start Data preprocessing")
     
            df_preprocessed = self.analyze_articles(df)
            
            self.logger.info("Start Data tokenize")
            print("Start Data tokenize")
            df_tokenized = self.preprocess_articles(df_preprocessed)
                  
            self.logger.info("Extracting Sentiment")
            print("Extracting Sentiment")
            df_sentiment = self.analyze_and_merge_sentiments(df_tokenized)          
            
            self.logger.info("Extracting Moderate")
            print("Extracting Moderate")
            df_moderated = self.analyze_and_moderate_documents(df_sentiment)
            df_cleaned = self.clean_column_names(df_moderated)           
            
            self.logger.info("Extracting Entity")
            print("Extracting Entity")
            df_entity = self.analyze_document_entities(df_cleaned)
            
            self.save_df_to_gcs_parquet(df_entity)

            self.upload_dataframe_to_bigquery(df_entity)

            return self.pred()
