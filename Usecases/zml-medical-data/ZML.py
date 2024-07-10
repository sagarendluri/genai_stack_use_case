import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.embeddings import AzureAIEmbeddings
from beyondllm.llms import AzureOpenAIModel
from beyondllm import source
import secrets
import os
pdf = [ ]
pdf_paths = "/Doctor_prescription_files/"

pdfs = ['Doctor_prescription_files/Prescription_1.pdf', 'Doctor_prescription_files/Prescription_3.pdf', 'Doctor_prescription_files/Prescription_2.pdf']
# dir = os.walk(pdf_paths)
# for root, dir,files  in dir:
#     for file in files:
#         if file.endswith('.pdf'):
#             paths = pdf_paths + file
#             pdf.append(paths)
st.title("Chat with ZML file Patient data file.")



endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY
api_version = st.secrets.azure_embeddings_credentials.API_VERSION
deployment_name = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
BASE_URL = st.secrets.azure_embeddings_credentials.BASE_URL
# DEPLOYMENT_NAME = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
API_KEY = st.secrets.azure_embeddings_credentials.API_KEY

embed_model = AzureAIEmbeddings(
    endpoint_url = endpoint_url,
    azure_key = azure_key,
    api_version= api_version,
    deployment_name=deployment_name
)
data = source.fit(path=pdfs, dtype="pdf",chunk_size=512,chunk_overlap=20)
# text_embedding = embed_model.embed_text(str(data))
retriever = retrieve.auto_retriever(data,embed_model=embed_model,type="normal",top_k=4)
llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name="gpt-4-32k" ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})
# option = st.selectbox( 'Please Select the Patient name?', ('Bobby Jackson', 'Leslie Terry','Danny Smith'))
question = st.text_input("Enter your question")
# question = "what is the Bobby Jackson condition?"

system_prompt = "You are acting like a chat...."

submit=st.button("Get the data")
if submit:
    print(question)
    pipeline = generator.Generate(question=question, retriever=retriever,system_prompt=system_prompt, llm=llm)
    response = pipeline.call()
    st.write(response)
    

