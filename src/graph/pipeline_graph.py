
import logging
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from src.config import *
from src.main import Assistente
from .schemas_graph import *
from .prompts_graph import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RAGGraph:
    def __init__(self, model=OLLAMA_MODEL, base_url=OLLAMA_API_URL):
        """Initialize the RAG Graph with model and base URL."""
        self.llm = ChatOllama(model=model, base_url=base_url)
        self.rag = Assistente()
        self.graph = self._build_graph()

    def _build_first_queries(self, state: ReportState):
        """
        1. Get user question
        2. Apply prompt
        3. Structure response as a list
        4. Call LLM with prompt that returns a list
        """
        user_input = state.user_input
        prompt = build_queries.format(user_input=user_input)
        query_llm = self.llm.with_structured_output(QueryList)
        result = query_llm.invoke(prompt)
        return {"queries": result.queries}

    def _spaw_researchers(self, state: ReportState):
        """Node connecting RAG to graph"""
        return [Send("single_search", query) for query in state.queries]

    def _single_search(self, query: str):
        """
        Function to search in RAG and return structured result with preserved metadata
        1. Query RAG
        2. Extract only text
        3. Pass only text to LLM
        4. Preserve original metadata without modification
        """
        resultado_rag = self.rag.consultar_rag(query)
        contexto_items = [ContextoItem(**item) for item in resultado_rag["contexto"]]
        textos_contexto = "\n\n".join(item.texto for item in contexto_items)
        prompt = resume_search.format(user_input=query, single_search=textos_contexto)
        resposta_llm = self.llm.invoke(prompt)
        queries_result = QueryResult(
            resposta=resposta_llm.content,
            contexto=contexto_items
        )
        return {"queries_results": [queries_result]}

    def _final_writer(self, state: ReportState):
        """Generate the final response with references"""
        search_results = ""
        references = ""
        for idx, result in enumerate(state.queries_results):
            search_results += f"[{idx+1}]\n\n"
            search_results += f"resposta: {result.resposta}\n"
            search_results += f"contexto: {result.contexto}\n"
            search_results += f"=============================\n\n"

            references += f"[{idx+1}] {result.contexto[0].metadados.filename}\n"
        prompt = build_final_response.format(user_input=state.user_input, search_results=search_results)
        llm_result = self.llm.invoke(prompt)
        final_response = llm_result.content + "\n\n References: \n" + references
        return {'final_response': final_response}

    def _build_graph(self):
        """Build and compile the graph"""
        builder = StateGraph(ReportState)

        builder.add_node("build_first_queries", self._build_first_queries)
        builder.add_node("single_search", self._single_search)
        builder.add_node("final_writer", self._final_writer)

        builder.add_edge(START, "build_first_queries")
        builder.add_conditional_edges("build_first_queries",
                                     self._spaw_researchers,
                                     ["single_search"])

        builder.add_edge("single_search", "final_writer")
        builder.add_edge("final_writer", END)

        return builder.compile()

    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG graph system
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing the final response
        """
        return self.graph.invoke({"user_input": query})

# Function to get a configured RAG graph instance
def get_rag_graph(model=OLLAMA_MODEL, base_url=OLLAMA_API_URL):
    """
    Returns a configured RAG graph instance for use in external modules
    
    Args:
        model: The LLM model to use
        base_url: The base URL for the Ollama API
        
    Returns:
        Configured RAGGraph instance
    """
    return RAGGraph(model=model, base_url=base_url)

# For testing locally
def teste_reason(model=OLLAMA_MODEL, base_url=OLLAMA_API_URL):
    llm = ChatOllama(model=model, base_url=base_url)
    resp = llm.invoke("oi tudo bem?")
    print(resp)

# Example usage
if __name__ == "__main__":
    rag_graph = get_rag_graph()
    query = "quais as porcentagens de receita per-capita de trasnferencias para os estados em 1968 e 1970"
    resultado_final = rag_graph.answer_question(query)
    print(resultado_final)