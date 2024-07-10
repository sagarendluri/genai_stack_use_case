import os
from beyondllm import source,retrieve,embeddings,llms,generator
import streamlit as st
from beyondllm.llms import AzureOpenAIModel
from beyondllm import embeddings
from streamlit_chat import message

# pdf = [ ]
# pdf_paths = "/tbank/"
# dir = os.walk(pdf_paths)
# for root, dir,files  in dir:
#     for file in files:
#         if file.endswith('.pdf'):
#             paths = pdf_paths + file
#             pdf.append(paths)
# print(pdf)
pdfs = ['tbank/About_page_tbank.pdf', 'tbank/About_page_tbank.pdf', 'tbank/About_page_tbank.pdf']


data = source.fit(path=pdfs, dtype="pdf", chunk_size=512,chunk_overlap=0)

endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY
api_version = st.secrets.azure_embeddings_credentials.API_VERSION
deployment_name = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
BASE_URL = st.secrets.azure_embeddings_credentials.BASE_URL
# DEPLOYMENT_NAME = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
API_KEY = st.secrets.azure_embeddings_credentials.API_KEY
embed_model = embeddings.AzureAIEmbeddings(
                endpoint_url=endpoint_url,
                azure_key=api_version,
                api_version=api_version,
                deployment_name=deployment_name)


retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)

BASE_URL = BASE_URL
DEPLOYMENT_NAME= "gpt-4-32k" 
API_KEY = API_KEY
llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name=DEPLOYMENT_NAME ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})


system_prompt = """ You are a Customer support Assistant who answers user query from the given CONTEXT, sound like a customer service\
You are honest, coherent and don't halluicnate \
If the user query is not in context, simply tell `We are sorry, we don't have information on this` \
"""
# query = "What about this bank?"
# pipeline = generator.Generate(question=query,system_prompt=system_prompt,retriever=retriever,llm=llm)

def conversational_chat(question):
    pipeline = generator.Generate(question=question, system_prompt=system_prompt, retriever=retriever, llm=llm)
    response = pipeline.call()
    return response
    
def main():
    st.set_page_config(page_title="ai Planet seemless process ")

    st.title("tbank chatbot")

    
    if 'history' not in st.session_state:
            st.session_state['history'] = []

    if 'generated' not in st.session_state:
      st.session_state['generated'] = ["Ask your question"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Welcome!"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
        
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True)
                message(st.session_state["generated"][i], key=str(i))
                
if __name__ == '__main__':
    main()


 
