import os
 
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

embedding_model_name='text-embedding-ada-002'
embedding_deployment_name='embedding'
deployment_name='prohackathon-2'#AI-Avengers
values={
'api_key' : "7282c43ed0b549118f59d015ccdc467d",
'api_version' : "2023-05-15",
'api_base' : "https://openapiazconf.openai.azure.com/",
'api_type' : 'azure'
}

client = AzureOpenAI(
    api_key=values['api_key'],
    api_version=values['api_version'],
    azure_endpoint=values['api_base']
)

def extract_data(file_path: str) -> str:
  df = pd.read_csv(file_path)
  raw_data = []
  for index, row in df.iterrows():
      raw_data.append(row.to_string())
  return raw_data

file_path = r"D:\hackathon\TamilNadu_maws_municipality_DataSet_Test.csv"
data_chunks = extract_data(file_path)

embeddings = AzureOpenAIEmbeddings(model=embedding_deployment_name,
                        openai_api_base=values['api_base'],
                        openai_api_key=values['api_key'],
                        openai_api_type=values['api_type'])

test_chunks = data_chunks[0:20]
print(type(test_chunks))

knowledge_hub = FAISS.from_texts(test_chunks, embeddings)

retriever = knowledge_hub.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
)

llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_base=values['api_base'],
            openai_api_key=values['api_key'],
            openai_api_type=values['api_type'],
            openai_api_version=values['api_version'],
            temperature=0.3)

chain_type= 'stuff'
chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
    )

def Get_RAG_Answer(question):
    result = chain({"query": question})
    print(result['source_documents'])