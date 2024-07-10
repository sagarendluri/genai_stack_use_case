import os
import json
import pydot
import subprocess
import streamlit as st
from getpass import getpass
from beyondllm.llms import AzureOpenAIModel
from beyondllm.embeddings import AzureAIEmbeddings
from beyondllm import source,retrieve,embeddings,llms,generator

# endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
# azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY
# api_version = st.secrets.azure_embeddings_credentials.API_VERSION
# deployment_name = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
# BASE_URL = st.secrets.azure_embeddings_credentials.BASE_URL
# # DEPLOYMENT_NAME = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
# API_KEY = st.secrets.azure_embeddings_credentials.API_KEY
# st.title("Respectus decision tree generator")


# api_key = st.text_input("API Key:", type="password")
# os.environ['OPENAI_API_KEY'] = api_key

st.title("Respectus decision tree generator")

def embeddings_llm(data):
    embed_model = embeddings.AzureAIEmbeddings(
                    endpoint_url="https://marketplace.openai.azure.com/",
                    azure_key="d6d9522a01c74836907af2f3fd72ff85",
                    api_version="2024-02-01",
                    deployment_name="text-embed-marketplace")
    BASE_URL = "https://gpt-res.openai.azure.com/"
    DEPLOYMENT_NAME = "gpt-4-32k" 
    API_KEY = "a20bc67dbd7c47ed8c978bbcfdacf930"

    retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)
    llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name=DEPLOYMENT_NAME ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})
    return retriever, llm
uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

def uploaded_files(uploaded_file):
    if uploaded_file is not None:
        save_path = "./uploaded_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0)
            return data
data = uploaded_files(uploaded_file)
retriever,llm  = embeddings_llm(data)
question = st.text_input(label='Type your question')
submit=st.button("Generate results")
if submit:
    question = question
    system_prompt = '''You are a business analyst with extensive knowledge of legal documents and regulatory documentation.
    Your expertise is in helping people understand and extract key information from such documents. 
    Your task is to extract the rules and exceptions in a way that enables the creation of a decision tree, facilitating integration into the proper flow.
    Legal Document Context: {context}
    Response Structure:
    Go through all the documents and understand the context. Next , step you have write questions from in a sequence.
    [Generate multiple questions ]
    Question :1
    If answer No  end of the process -> “Not restricted” 
    If question Res, Result : « Not restricted (+articles from the document)
    Write another question in sequence, if the first question is yes. And  go for the second question,.

    If Answer is yes, should create two responses as mentioned below
    If Yes,  -> Result : Not restricted (+articles from the document)”
    If No, ->  Result ; Prohibited (+articles from the document)”
    If Answer is no, should create two responses as mentioned below 
    If Yes  -> Result : Not restricted (+articles from the document)”
    If No , -> Result ;  Prohibited (+articles from the document)”
    write a Question1 and then go with pattern.
    The below one is json example of stucture: use this same flow of key and values:
    {
    "Question1": {
        "No": "Not restricted",
        "Yes": {
            "Question2": "Are you acting through or on behalf of one of the persons, entities or bodies referred to in point (a)?",
            "Yes": {
                "Result": "Not restricted (Source: Document)"
            },
            "No": {
                "Result": "Prohibited (Source: Document)"
            }
        },
        "No": {
            "Question3": "Are you acting through or on behalf of one of the persons, entities or bodies referred to in point (a)?",
            "Yes": {
                "Result": "Not restricted (Source: Document)"
            },
            "No": {
                "Result": "Prohibited (Source: Document)"
            }
        }
    }
    }

    
   
    
    Create Json in this format only

    Additional Instructions:
    
    Analyze the entire document to identify all relevant rules and exceptions.
    Ensure that the descriptions of rules and exceptions are clear and concise.
    Include relevant dates, jurisdictions, and specific regulations where applicable.
    Structure the questions and answers to facilitate the creation of a logical decision tree or workflow.
    If the regulation mentions specific products, territories, or operations, include them in the appropriate sections.
    Aim to simplify legal language while maintaining accuracy and intent.
    [Provide your answer in JSON form. Reply with only the answer in JSON form and include no other commentary]:
    Provide your answer in JSON form. Reply with only the answer in JSON form and include no other commentary
    Return Valid Json to create Tree 
    '''
        # embed_model = AzureAIEmbeddings(
        #       endpoint_url = endpoint_url,
        #       azure_key = azure_key,
        #       api_version= api_version,
        #       deployment_name=deployment_name
        #   )

        # retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)
        # llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name="gpt-4-32k" ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})
        # pipeline = generator.Generate(question=question, system_prompt=system_prompt, retriever=retriever, llm=llm)
        # -----------------------------------------------------------------------------

    pipeline = generator.Generate(question=question, system_prompt=system_prompt, retriever=retriever, llm=llm)
    decision_tree_json = pipeline.call()
    print(decision_tree_json)
    data = json.loads(decision_tree_json)
    print(data)
    # For user queiry control 

    # def json_to_graph(graph, parent_label, parent_node):
    #     if isinstance(parent_node, dict):
    #         for key, value in parent_node.items():
    #             if key in ["Yes", "No"]:
    #                 option_label = parent_label + "_" + key
    #                 node = pydot.Node(option_label, label=key, shape='box', style='filled', fillcolor='lightgreen' if key == "Yes" else 'lightcoral')
    #                 graph.add_node(node)
    #                 edge = pydot.Edge(parent_label, option_label)
    #                 graph.add_edge(edge)
    #                 if isinstance(value, dict):
    #                     for sub_key, sub_value in value.items():
    #                         if sub_key.startswith("Question"):
    #                             question_label = option_label + "_" + sub_key
    #                             question_text = "\n".join(sub_value[i:i+30] for i in range(0, len(sub_value), 30))
    #                             node = pydot.Node(question_label, label=question_text, shape='box', style='filled', fillcolor='lightblue')
    #                             graph.add_node(node)
    #                             edge = pydot.Edge(option_label, question_label)
    #                             graph.add_edge(edge)
    #                             json_to_graph(graph, question_label, value)
    #                         else:
    #                             json_to_graph(graph, option_label, value)
    #             elif key == "Result":
    #                 result_label = parent_label + "_" + key
    #                 result_str = f"{key}: {value}"
    #                 node = pydot.Node(result_label, label=result_str, shape='box', style='filled', fillcolor='lightgrey')
    #                 graph.add_node(node)
    #                 edge = pydot.Edge(parent_label, result_label)
    #                 graph.add_edge(edge)
    #             elif key.startswith("Question"):
    #                 continue

    # graph = pydot.Dot(graph_type='graph')

    # start_label = data['Question1']["Question"]
    # start_question = data[start_label]
    # question_text = "\n".join(start_label[i:i+100] for i in range(0, len(start_label), 100))
    # start_node = pydot.Node(start_label, label=question_text, shape='box', style='filled', fillcolor='lightblue')
    # graph.add_node(start_node)

    # json_to_graph(graph, start_label, start_question)

    # graph.write_png('decision_tree.png')
    # Function to recursively create nodes and edges in the grap
    # Create a new graph
    # Create a new directed graph

    def json_to_graph(graph, parent_label, parent_node):
        if isinstance(parent_node, dict):
            for key, value in parent_node.items():
                if key in ["Yes", "No"]:
                    option_label = parent_label + "_" + key
                    node = pydot.Node(option_label, label=key, shape='box', style='filled', fillcolor='lightgreen' if key == "Yes" else 'lightcoral')
                    graph.add_node(node)
                    edge = pydot.Edge(parent_label, option_label)
                    graph.add_edge(edge)
                    
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key.startswith("Question"):
                                question_label = option_label + "_" + sub_key
                                question_text = "\n".join(sub_value[i:i+30] for i in range(0, len(sub_value), 30))
                                node = pydot.Node(question_label, label=question_text, shape='box', style='filled', fillcolor='lightblue')
                                graph.add_node(node)
                                edge = pydot.Edge(option_label, question_label)
                                graph.add_edge(edge)
                                json_to_graph(graph, question_label, value)
                            elif sub_key == "Result":
                                result_label = option_label + "_" + sub_key
                                result_str = f"{sub_key}: {sub_value}"
                                node = pydot.Node(result_label, label=result_str, shape='box', style='filled', fillcolor='lightgrey')
                                graph.add_node(node)
                                edge = pydot.Edge(option_label, result_label)
                                graph.add_edge(edge)
                    else:
                        json_to_graph(graph, option_label, value)
                elif key == "Result":
                    result_label = parent_label + "_" + key
                    result_str = f"{key}: {value}"
                    node = pydot.Node(result_label, label=result_str, shape='box', style='filled', fillcolor='lightgrey')
                    graph.add_node(node)
                    edge = pydot.Edge(parent_label, result_label)
                    graph.add_edge(edge)
    graph = pydot.Dot(graph_type='graph')

    start_label = data['Question1']["Question"]
    print(start_label)
    start_question = data["Question1"]
    start_node = pydot.Node(start_label, label=start_label, shape='box', style='filled', fillcolor='lightblue')
    graph.add_node(start_node)

    json_to_graph(graph, start_label, start_question)
    image_name = "decision_tree.png"
    graph.write_png(image_name)

    # Display the graph
    # from PIL import Image
    # img = Image.open(uploaded_file.name+'_tree.png')
    # img.show()

    with open(image_name, "rb") as file:
        btn = st.download_button(
                label="Download image",
                data=file,
                file_name=image_name,
                mime="image/png"
                )
    with st.chat_message(""):
        st.write("")
        st.image(image_name, caption="tree_from_json")


# uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
# submit=st.button("Generate results")
# if submit:
#     question = "Give Decision taken in the document"
#     system_prompt = '''You are a business analyst with extensive knowledge of legal documents and regulatory documentation.
#     Your expertise is in helping people understand and extract key information from such documents. 
#     Your task is to extract the rules and exceptions in a way that enables the creation of a decision tree, facilitating integration into the proper flow.
#     Legal Document Context: {context}
#     Response Structure:
#     [Generate multiple questions ]
#     Question :1
#     Write a Question1 and then go with pattern.
    
#     If answer No  end of the process -> “Not restricted” 
#     If question Res, Result : « Not restricted (+articles from the document)
#     Write another question in sequence, if the first question is yes. And  go for the second question,.

#     If Answer is yes, should create two responses as mentioned below
#     If Yes,  -> Result : Not restricted (+articles from the document)”
#     If No, ->  Result ; Prohibited (+articles from the document)”
#     If Answer is no, should create two responses as mentioned below 
#     If Yes  -> Result : Not restricted (+articles from the document)”
#     If No , -> Result ;  Prohibited (+articles from the document)”
#     {
#     "Question1": {
#         "No": "Not restricted",
#         "Yes": {
#             "Question2": ,
#             "Yes": {
#                 "Result": 
#                                         },
#             "No": {
#                 "Result": 
#             }
#         },
#         "No": {
#             "Question3": ,
#             "Yes": {
#                 "Result": 
#             },
#             "No": {
#                 "Result": 
#             }
#         }
#     }
#     }

#     Additional Instructions:
    
#     Analyze the entire document to identify all relevant rules and exceptions.
#     Ensure that the descriptions of rules and exceptions are clear and concise.
#     Include relevant dates, jurisdictions, and specific regulations where applicable.
#     Structure the questions and answers to facilitate the creation of a logical decision tree or workflow.
#     If the regulation mentions specific products, territories, or operations, include them in the appropriate sections.
#     Aim to simplify legal language while maintaining accuracy and intent.
#     [Provide your answer in JSON form. Reply with only the answer in JSON form and include no other commentary]:
#     Provide your answer in JSON form. Reply with only the answer in JSON form and include no other commentary
#     Return Valid Json to create Tree 
#     '''


#     if uploaded_file is not None and question:
#         save_path = "./uploaded_files"
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         file_path = os.path.join(save_path, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
      
#         data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0)
#         embed_model = embeddings.AzureAIEmbeddings(
#                 endpoint_url="https://marketplace.openai.azure.com/",
#                 azure_key="d6d9522a01c74836907af2f3fd72ff85",
#                 api_version="2024-02-01",
#                 deployment_name="text-embed-marketplace")


#         retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)

#         BASE_URL = "https://gpt-res.openai.azure.com/"
#         DEPLOYMENT_NAME= "gpt-4-32k" 
#         API_KEY = "a20bc67dbd7c47ed8c978bbcfdacf930"
#         llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name=DEPLOYMENT_NAME ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})
#         pipeline = generator.Generate(question=question, system_prompt=system_prompt, retriever=retriever, llm=llm)
#         decision_tree_json = pipeline.call()
#         data = json.loads(decision_tree_json)
#         # Function to recursively create nodes and edges in the grap
#         # Create a new graph
#         # Create a new directed graph

        
#         def json_to_graph(graph, parent_label, parent_node):
#             if isinstance(parent_node, dict):
#                 for key, value in parent_node.items():
#                     if key in ["Yes", "No"]:
#                         option_label = parent_label + "_" + key
#                         node = pydot.Node(option_label, label=key, shape='box', style='filled', fillcolor='lightgreen' if key == "Yes" else 'lightcoral')
#                         graph.add_node(node)
#                         edge = pydot.Edge(parent_label, option_label)
#                         graph.add_edge(edge)
                        
#                         if isinstance(value, dict):
#                             for sub_key, sub_value in value.items():
#                                 if sub_key.startswith("Question"):
#                                     question_label = option_label + "_" + sub_key
#                                     question_text = "\n".join(sub_value[i:i+30] for i in range(0, len(sub_value), 30))
#                                     node = pydot.Node(question_label, label=question_text, shape='box', style='filled', fillcolor='lightblue')
#                                     graph.add_node(node)
#                                     edge = pydot.Edge(option_label, question_label)
#                                     graph.add_edge(edge)
#                                     json_to_graph(graph, question_label, value)
#                                 elif sub_key == "Result":
#                                     result_label = option_label + "_" + sub_key
#                                     result_str = f"{sub_key}: {sub_value}"
#                                     node = pydot.Node(result_label, label=result_str, shape='box', style='filled', fillcolor='lightgrey')
#                                     graph.add_node(node)
#                                     edge = pydot.Edge(option_label, result_label)
#                                     graph.add_edge(edge)
#                         else:
#                             json_to_graph(graph, option_label, value)
#                     elif key == "Result":
#                         result_label = parent_label + "_" + key
#                         result_str = f"{key}: {value}"
#                         node = pydot.Node(result_label, label=result_str, shape='box', style='filled', fillcolor='lightgrey')
#                         graph.add_node(node)
#                         edge = pydot.Edge(parent_label, result_label)
#                         graph.add_edge(edge)
        
#         graph = pydot.Dot(graph_type='graph')

#         start_label = "Is the entity listed in the sanctions list?"
#         start_question = data["Question1"]
#         start_node = pydot.Node(start_label, label=start_label, shape='box', style='filled', fillcolor='lightblue')
#         graph.add_node(start_node)

#         json_to_graph(graph, start_label, start_question)
#         image_name = "decision_tree.png"
#         graph.write_png(image_name)

        
#         # Display the graph
#         from PIL import Image
#         img = Image.open(image_name)
#         img.show()

#         with open(image_name, "rb") as file:
#             btn = st.download_button(
#                     label="Download image",
#                     data=file,
#                     file_name=image_name,
#                     mime="image/png"
#                   )
#         with st.chat_message(""):
#             st.write("")
#             st.image(image_name, caption="tree_from_json")
