import os
import pandas as pd
from openai import AzureOpenAI

df=pd.read_csv(os.path.join(os.getcwd(),r"D:\hackathon\TamilNadu_maws_municipality_DataSet.csv"))

api_key = "7282c43ed0b549118f59d015ccdc467d"
api_base = "https://openapiazconf.openai.azure.com/"
api_type = 'azure'
api_version = '2023-05-15'
embedding_model_name = 'text-embedding-ada-002'
embedding_deployment_name='embedding'
deployment_name='prohackathon2-1'

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint = api_base
)

def generate_embeddings(text, model="text-embedding-ada-002"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def test():
    df_bills = df[['District', 'typeOfWork']]
    df_bills['ada_v2'] = df_bills["District"].apply(lambda x : generate_embeddings (x, model = 'text-embedding-ada-002')) # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    print(df_bills)