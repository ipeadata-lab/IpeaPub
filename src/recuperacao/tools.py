"""
Tools para busca vetorial nas coleções do Qdrant.
Essas ferramentas são utilizadas pelos agentes para recuperar informações.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.db.banco_vetorial import QdrantVectorDB


# Instância global do embedder e banco vetorial
_embedder = None
_db_vetorial = None


def _get_embedder() -> SentenceTransformer:
    """Retorna instância singleton do embedder."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    return _embedder


def _get_db() -> QdrantVectorDB:
    """Retorna instância singleton do banco vetorial."""
    global _db_vetorial
    if _db_vetorial is None:
        _db_vetorial = QdrantVectorDB()
    return _db_vetorial


def search_recommendations(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Busca documentos recomendados com base em uma query semântica.
    
    Útil para: recomendação de leituras, encontrar documentos relacionados,
    buscar por título/resumo/palavras-chave.
    
    Args:
        query: Texto de busca (título, tema, palavras-chave)
        top_k: Número máximo de resultados
        
    Returns:
        Lista de documentos com título, resumo, keywords, handle e score
    """
    embedder = _get_embedder()
    db = _get_db()
    
    query_vector = embedder.encode(query).tolist()
    results = db.search_recommendations(query_vector, top_k=top_k)
    
    # Formatar resultados para facilitar leitura
    formatted = []
    for r in results:
        payload = r.get("payload", {})
        formatted.append({
            "doc_id": payload.get("doc_id", ""),
            "titulo": payload.get("titulo", ""),
            "resumo": payload.get("resumo", "")[:500] + "..." if len(payload.get("resumo", "")) > 500 else payload.get("resumo", ""),
            "keywords": payload.get("keywords", []),
            "handle": payload.get("handle", ""),
            "score": round(r.get("score", 0), 4)
        })
    
    return formatted


def search_chunks(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Busca chunks de texto relevantes para RAG.
    
    Útil para: responder perguntas específicas, buscar trechos de documentos,
    encontrar informações detalhadas em texto.
    
    Args:
        query: Pergunta ou texto de busca
        top_k: Número máximo de chunks retornados
        
    Returns:
        Lista de chunks com texto, documento de origem, página e score
    """
    embedder = _get_embedder()
    db = _get_db()
    
    query_vector = embedder.encode(query).tolist()
    results = db.search_chunks(query_vector, top_k=top_k)
    
    formatted = []
    for r in results:
        payload = r.get("payload", {})
        formatted.append({
            "chunk_id": payload.get("pid", ""),
            "doc_id": payload.get("doc_id", ""),
            "texto": payload.get("texto", ""),
            "handle": payload.get("handle", ""),
            "pagina": payload.get("pagina", -1),
            "score": round(r.get("score", 0), 4)
        })
    
    return formatted


def search_tables(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Busca tabelas relevantes nos documentos.
    
    Útil para: encontrar dados numéricos, estatísticas, séries temporais,
    informações tabulares.
    
    Args:
        query: Descrição da tabela ou dados procurados
        top_k: Número máximo de tabelas retornadas
        
    Returns:
        Lista de tabelas em formato markdown com metadados
    """
    embedder = _get_embedder()
    db = _get_db()
    
    query_vector = embedder.encode(query).tolist()
    results = db.search_tables(query_vector, top_k=top_k)
    
    formatted = []
    for r in results:
        payload = r.get("payload", {})
        formatted.append({
            "table_id": payload.get("pid", ""),
            "doc_id": payload.get("doc_id", ""),
            "tabela": payload.get("tabela", ""),
            "descricao": payload.get("descricao_llm", ""),
            "handle": payload.get("handle", ""),
            "pagina": payload.get("pagina", -1),
            "score": round(r.get("score", 0), 4)
        })
    
    return formatted


def search_images(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Busca imagens e gráficos relevantes nos documentos.
    
    Útil para: encontrar gráficos, figuras, visualizações de dados,
    imagens explicativas.
    
    Args:
        query: Descrição da imagem ou gráfico procurado
        top_k: Número máximo de imagens retornadas
        
    Returns:
        Lista de imagens com caption, descrição e metadados
    """
    embedder = _get_embedder()
    db = _get_db()
    
    query_vector = embedder.encode(query).tolist()
    results = db.search_images(query_vector, top_k=top_k)
    
    formatted = []
    for r in results:
        payload = r.get("payload", {})
        formatted.append({
            "image_id": r.get("id", ""),
            "doc_id": payload.get("doc_id", ""),
            "caption": payload.get("caption", ""),
            "descricao": payload.get("descricao_llm", ""),
            "pagina": payload.get("pagina", -1),
            "score": round(r.get("score", 0), 4)
        })
    
    return formatted


def search_all_collections(query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Busca em todas as coleções simultaneamente.
    
    Útil para: busca exploratória, quando não se sabe onde a informação está,
    primeira camada de recuperação.
    
    Args:
        query: Texto de busca
        top_k: Número máximo de resultados por coleção
        
    Returns:
        Dicionário com resultados de cada coleção
    """
    return {
        "recomendacoes": search_recommendations(query, top_k),
        "chunks": search_chunks(query, top_k),
        "tabelas": search_tables(query, top_k),
        "imagens": search_images(query, top_k)
    }
