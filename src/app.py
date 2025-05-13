# src/app.py

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
from src.main import Assistente
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.config import IMAGES_DIR

# Configure the page
st.set_page_config(
    page_title="RAG Publicações",
    page_icon="📚",
    layout="wide",
)
st.title("RAG Publicações")

@st.cache_resource
def get_assistant():
    return Assistente()

# Initialize chat history
def initialize_chat_history():
    """
    Initialize the chat history using StreamlitChatMessageHistory.
    """
    # Use Streamlit's built-in chat message history
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    # Initialize messages if empty
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Como posso te ajudar hoje?")
    
    return msgs

def display_chat_history(msgs):
    """
    Display the chat history messages.
    
    Args:
        msgs (StreamlitChatMessageHistory): Chat message history object
    """
    # Display chat messages
    for msg in msgs.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content)

def main():
    """
    Main function to run the Streamlit RAG chat application.
    """
    # Initialize chat history
    msgs = initialize_chat_history()

    main_assistant = get_assistant()
    
    # Display existing chat history
    display_chat_history(msgs)
    
    # User input
    if prompt := st.chat_input("Digite sua pergunta"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        
        # Add user message to history
        msgs.add_user_message(prompt)
        
        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Processando sua pergunta..."):
                try:
                    # Call RAG function
                    resposta = main_assistant.consultar_rag(prompt)
                    print(resposta)

                    # Display and store RAG response
                    st.markdown(resposta['resposta'])
                    msgs.add_ai_message(resposta['resposta'])

                    # Display relevant results in a second message
                    st.markdown("### Resultados relevantes:")
                    for i, doc in enumerate(resposta["contexto"]):
                        metadados = doc.get("metadados", {})
                        page = metadados.get("page_number", "desconhecida")
                        content_type = metadados.get("content_type", "texto")
                        filename = metadados.get("filename", "desconhecido")
                        texto = doc.get("texto", "Texto não disponível")
                        # Display relevant result
                        if content_type == "text":
                            st.markdown(f"**{i + 1}.** Fonte: {filename} (Página {page}) | Texto")

                        if content_type == "table":
                            st.markdown(f"**{i + 1}.** Fonte: {filename} (Página {page}) | Tabela")
                            st.markdown(f"{texto}")

                        # if image, display the image in chat
                        if content_type == "image":

                            st.markdown(f"**{i + 1}.** Fonte: {filename} (Página {metadados.get("page", "desconhecida")}) | Imagem")
                            image_path = os.path.join(IMAGES_DIR, metadados.get("ref", ""))
                            st.image(image_path, caption=metadados.get("caption", ""), width=300)
                
                except Exception as e:
                    error_message = f"Erro ao processar a consulta: {str(e)}"
                    st.error(error_message)
                    msgs.add_ai_message(error_message)

if __name__ == "__main__":
    main()