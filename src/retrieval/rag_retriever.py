# src/retrieval/rag_retriever.py

from typing import Dict, Any, List, Optional
import requests

from src.vector_db.chroma_client import ChromaDB
from src.config import OLLAMA_API_URL, OLLAMA_MODEL, VECTOR_DB
from src.modelos import manager

class RAGRetriever:
    def __init__(self):
        self.modelo_embedding_texto = manager.modelo_texto
        self.modelo_embedding_imagem = manager.modelo_imagem
        self.vector_db = manager.vector_db

        
    def perguntar_ollama(self, prompt: str, modelo: str = OLLAMA_MODEL) -> str:
        """
        Faz uma pergunta ao modelo de linguagem Ollama.
        
        Args:
            prompt: Pergunta a ser feita.
            modelo: Nome do modelo de linguagem a ser utilizado.
            
        Returns:
            Resposta do modelo de linguagem.
        """
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={
                    "model": modelo,
                    "prompt": prompt,
                    "stream": False,
                }
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Erro: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Erro ao conectar com o modelo de linguagem: {e}"
           
    def recuperar(
            self,
            query: str,
            limite: int = 5,
            content_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recupera documentos relevantes para uma consulta usando o modelo de linguagem e o banco de dados vetorial.
        
        Args:
            query: Consulta a ser feita.
            limite: Número máximo de resultados a serem retornados.
            content_types: Lista de tipos de conteúdo a serem filtrados (opcional).
            
        Returns:
            Lista de documentos relevantes encontrados.
        """
        todos_resultados = []

        todos_resultados = self.vector_db.busca_completa(
            query=self.modelo_embedding_texto.gerar_embedding(query).tolist(),
            query_imagem=self.modelo_embedding_imagem.gerar_embedding_textual(query)[0].tolist(),
            top_k=limite,
            content_types=content_types
        )
    
        # Ordenar e limitar resultados
        todos_resultados.sort(key=lambda x: x["distancia"], reverse=True)
        return todos_resultados
    
    def criar_prompt(self, query: str, contexto: List[Dict[str, Any]]) -> str:
        """
        Cria um prompt para o modelo de linguagem com base na consulta e no contexto.
        
        Args:
            query: Consulta a ser feita.
            contexto: Lista de documentos relevantes encontrados.
            
        Returns:
            Prompt formatado para o modelo de linguagem.
        """
        partes_contexto = []
        for i, doc in enumerate(contexto):
            metadados: dict = doc.get("metadados", {})

            content_type = metadados.get("content_type", "text")
            arquivo_fonte = metadados.get("filename", "desconhecido")
            pagina = metadados.get("page", "")

            if content_type == "text":
                partes_contexto.append(f"[Texto {i + 1}] (Fonte: {arquivo_fonte}, Página: {pagina}):\n{doc.get('texto', '')}")
            elif content_type == "image":
                caption = metadados.get("caption", "sem legenda")
                caminho_imagem = metadados.get("ref", "desconhecido")
                classificacao = metadados.get("classification", [])
                partes_contexto.append(f"[Imagem {i + 1}] (Fonte: {arquivo_fonte}, Caminho: {caminho_imagem}:\n{caption}\nClassificação: {', '.join(classificacao)}")
            elif content_type == "table":
                partes_contexto.append(f"[Tabela {i + 1}] (Fonte: {arquivo_fonte}, Página: {pagina}):\n{doc.get('texto', '')}")

        contexto_formatado = "\n\n".join(partes_contexto)
        contexto_formatado += "\n\n"
        prompt = f"""
Você é um assistente de pesquisa especializado em responder perguntas com base nas informações fornecidas.
        
Contexto:
{contexto_formatado}
        
Pergunta:
{query}
        
Com base APENAS nas informações fornecidas no CONTEXTO acima, responda à PERGUNTA de forma completa e detalhada.
Mencione as fontes das informações na sua resposta.
Se o CONTEXTO não contiver informações suficientes para responder completamente, indique quais partes você não consegue responder e por quê.
"""
        return prompt

    def responder(self, query: str, limite: int = 5, content_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Responde a uma consulta usando o modelo de linguagem e o banco de dados vetorial.
        
        Args:
            query: Consulta a ser feita.
            limite: Número máximo de resultados a serem retornados.
            content_types: Lista de tipos de conteúdo a serem filtrados (opcional).
            
        Returns:
            Resposta do modelo de linguagem com base no contexto encontrado.
        """
        
        documentos_recuperados = self.recuperar(
            query=query,
            limite=limite,
            content_types=content_types
        )

        if not documentos_recuperados:
            return {
                "resposta": "Desculpe, não consegui encontrar informações relevantes para sua pergunta.",
                "contexto": []
            }
        
        prompt = self.criar_prompt(query, documentos_recuperados)
        resposta = self.perguntar_ollama(prompt)
        return {
            "resposta": resposta,
            "contexto": documentos_recuperados
        }
