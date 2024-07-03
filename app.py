import streamlit as st
from langchain_openai import ChatOpenAI
import os
import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader

# Set page config
st.set_page_config(page_title="Tbank Assistant", layout="wide")

# Streamlit app header
st.title("Tbank Customer Support Chatbot")


os.environ["OPENAI_API_KEY"] = "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"

# Main app logic
if "OPENAI_API_KEY" in os.environ:
    # Initialize components
    @st.cache_resource
    def initialize_components():
        dotenv.load_dotenv()
        chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

        #loader1 = WebBaseLoader("https://www.tbankltd.com/")
        loader1 = PyPDFLoader("Tbank_resources_1.pdf")
        loader2 = PyPDFLoader("Tbank_resources_2.pdf")
        data1 = loader1.load()
        data2 = loader2.load()
        data = data1 + data2
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(k=4)

        SYSTEM_TEMPLATE = """
        You are Tbank's AI assistant, a chatbot whose knowledge comes exclusively from Tbank's website content and provided PDF documents. Follow these guidelines:
        1. Greet users warmly, e.g., "Hello! Welcome to Tbank. How can I assist you today?"
        2. If asked about your identity, state you're Tbank's AI assistant and ask how you can help.
        3. Use only information from the website content and provided PDFs. Do not infer or make up information.
        4. Provide clear, concise responses using only the given information. Keep answers brief and relevant.
        5. For questions outside your knowledge base, respond:
          "I apologize, but I don't have information about that. My knowledge is limited to Tbank's products/services and our website/document content. Is there anything specific about Tbank I can help with?"
        6. Maintain a friendly, professional tone.
        7. If unsure, say:
          "I'm not certain about that. For accurate information, please check our website or contact our customer support team."
        8. For requests for opinions or subjective information, remind users you're an AI that provides only factual information from Tbank sources.
        9. End each interaction by asking if there's anything else you can help with regarding Tbank.
        10. Do not hallucinate or provide information from sources other than the website and provided PDFs.
        11. If the information isn't in your knowledge base, clearly state that you don't have that information rather than guessing.
        12. Regularly refer to the provided PDFs for accurate, up-to-date information about Tbank's products and services.
        13. Check for the basic Grammar and Spellings and understand if the spellings or grammar is slightly incorrect.
        14. Understand the user query with different angle, analyze properly, check through the possible answers and then give the answer.
        15. Be forgiving of minor spelling mistakes and grammatical errors in user queries. Try to understand the intent behind the question.
        16. Maintain context from previous messages in the conversation. If a user asks about a person or topic mentioned earlier, refer back to that information.
        17. If a user asks about a person using only a name or title, try to identify who they're referring to based on previous context or your knowledge base.
        18. When answering questions about specific people, provide their full name and title if available.
        
        Your primary goal is to assist users with information directly related to Tbank, using only the website content and provided PDF documents. Avoid speculation and stick strictly to the provided information.
        <context>
        {context}
        </context>
        Chat History:
        {chat_history}
        """

        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

        return retriever, document_chain

    # Load components
    with st.spinner("Initializing Tbank Assistant..."):
        retriever, document_chain = initialize_components()

    # Initialize memory for each session
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Chat interface
    st.subheader("Chat with Tbank Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know about Tbank?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(prompt)

            # Generate response
            response = document_chain.invoke(
                {
                    "context": docs,
                    "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"],
                    "messages": [
                        HumanMessage(content=prompt)
                    ],
                }
            )

            # The response is already a string, so we can use it directly
            full_response = response
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Update memory
        st.session_state.memory.save_context({"input": prompt}, {"output": full_response})

else:
    st.warning("Please enter your OpenAI API Key in the sidebar to start the chatbot.")

# Add a footer
st.markdown("---")
st.markdown("By AI Planet")