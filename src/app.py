# src/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.main import Assistente
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from src.config import IMAGES_DIR
from src.graph.pipeline_graph import *

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

    main_assistant = get_rag_graph()
    
    # Display existing chat history
    display_chat_history(msgs)
    
    # User input
    if prompt := st.chat_input("Digite sua pergunta"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        
        # Add user message to history
        msgs.add_user_message(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Processando sua pergunta..."):
            try:
                # Call RAG function
                resultado = main_assistant.answer_question(prompt)
                
                # Extract the final response text from the result dictionary
                resposta = resultado.get("final_response", "Resposta não disponível")
                
                # Display and store RAG response
                st.markdown(resposta)
                msgs.add_ai_message(resposta)
                
                # Optional: Display query details in an expandable section
                with st.expander("Ver detalhes da consulta", expanded=False):
                    # Show the queries that were generated
                    if "queries" in resultado:
                        st.subheader("Consultas geradas:")
                        for i, query in enumerate(resultado["queries"]):
                            st.markdown(f"**{i+1}.** `{query}`")
                    
                    # Show query results if available
                    if "queries_results" in resultado and resultado["queries_results"]:
                        st.subheader("Resultados por consulta:")
                        for i, result in enumerate(resultado["queries_results"]):
                            st.markdown(f"### Resultado {i+1}:")
                            
                            # Display response part
                            if hasattr(result, "resposta"):
                                st.markdown("**Resposta:**")
                                st.markdown(result.resposta)
                            
                            # Display context if available
                            if hasattr(result, "contexto") and result.contexto:
                                st.markdown("**Contexto:**")
                                for j, contexto_item in enumerate(result.contexto):
                                    with st.container():
                                        # Extract metadata
                                        metadados = contexto_item.metadados if hasattr(contexto_item, "metadados") else {}
                                        filename = getattr(metadados, "filename", "desconhecido")
                                        page = getattr(metadados, "page", "desconhecida")
                                        content_type = getattr(metadados, "content_type", "texto")
                                        
                                        # Display source information
                                        st.markdown(f"**Fonte {j+1}:** {filename} (Página {page}) | {content_type.capitalize()}")
                                        
                                        # Display text content
                                        if hasattr(contexto_item, "texto") and contexto_item.texto:
                                            st.text_area(f"Texto {j+1}", contexto_item.texto, height=100, key=f"text_{i}_{j}")
                                        
                                        # Handle different content types
                                        if content_type == "table" and hasattr(contexto_item, "texto"):
                                            try:
                                                # Try to display as a table if it's structured data
                                                st.markdown(contexto_item.texto)
                                            except:
                                                pass
                                        
                                        # If it's an image and has a reference, try to display it
                                        if content_type == "image" and hasattr(metadados, "ref"):
                                            image_ref = getattr(metadados, "ref", "")
                                            if image_ref:
                                                try:
                                                    image_path = os.path.join(IMAGES_DIR, image_ref)
                                                    st.image(image_path, 
                                                            caption=getattr(metadados, "caption", ""), 
                                                            width=300)
                                                except Exception as img_err:
                                                    st.error(f"Erro ao carregar imagem: {str(img_err)}")
            
            except Exception as e:
                error_message = f"Erro ao processar a consulta: {str(e)}"
                st.error(error_message)
                msgs.add_ai_message(error_message)

if __name__ == "__main__":
    main()