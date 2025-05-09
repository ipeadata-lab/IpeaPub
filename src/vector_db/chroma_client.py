import chromadb
from typing import List, Dict, Any, Optional
import os

from src.config import VECTOR_DB_DIR
from src.embeddings.modelo_texto import ModeloEmbeddingTexto
from src.embeddings.modelo_imagem import ModeloEmbeddingImagem

class ChromaDB:
    def __init__(self, collection_name: str = "rag_pub", persist_directory: str = VECTOR_DB_DIR):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))

        self.collection_texto = self.client.get_or_create_collection(
            name=f"{collection_name}_text",
            metadata={"description": "Textos de documentos PDF"}
        )
        
        self.collection_imagem = self.client.get_or_create_collection(
            name=f"{collection_name}_image",
            metadata={"description": "Imagens com legendas geradas"}
        )
        
        self.collection_tabela = self.client.get_or_create_collection(
            name=f"{collection_name}_table",
            metadata={"description": "Tabelas em formato markdown"}
        )

    def inserir_documento(self, doc: Dict[str, Any]) -> None:
        """
        Insere um documento no banco de dados.
        
        Args:
            doc: Documento com texto, metadados e embedding.
        """
        doc_id = f"text_{hash(doc['texto'])}"

        # formatar metadados para o ChromaDB
        metadados = {}
        for key, value in doc["metadados"].items():
            if isinstance(value, (str, int, float, bool)):
                metadados[key] = value
            else:
                metadados[key] = str(value)

        # Inserir o documento na coleção de texto
        self.collection_texto.upsert(
            ids=[doc_id],
            documents=[doc["texto"]],
            metadatas=[metadados],
            embeddings=doc["embedding"],
        )
    
    def inserir_imagem(self, doc: Dict[str, Any]) -> None:
        """
        Insere uma imagem no banco de dados.
        
        Args:
            doc: Documento com imagem, metadados e embedding.
        """
        doc_id = f"image_{hash(doc['texto'])}"

        # formatar metadados para o ChromaDB
        metadados = {}
        for key, value in doc["metadados"].items():
            if isinstance(value, (str, int, float, bool)):
                metadados[key] = value
            else:
                metadados[key] = str(value)

        # Inserir a imagem na coleção de imagem
        self.collection_imagem.upsert(
            ids=[doc_id],
            documents=[doc["texto"]],
            metadatas=[metadados],
            embeddings=doc["embedding_imagem"],
        )

        # Inserir a legenda na coleção de texto
        self.collection_texto.upsert(
            ids=[doc_id],
            documents=[doc["texto"]],
            metadatas=[metadados],
            embeddings=doc["embedding_texto"],
        )

    def inserir_tabela(self, doc: Dict[str, Any]) -> None:
        """
        Insere uma tabela no banco de dados.

        Args:
            doc: Documento com tabela, metadados e embedding.
        """
        doc_id = f"table_{hash(doc['texto'])}"

        # formatar metadados para o ChromaDB
        metadados = {}
        for key, value in doc["metadados"].items():
            if isinstance(value, (str, int, float, bool)):
                metadados[key] = value
            else:
                metadados[key] = str(value)

        # Inserir a tabela na coleção de tabela
        self.collection_tabela.upsert(
            ids=[doc_id],
            documents=[doc["texto"]],
            metadatas=[metadados],
            embeddings=doc["embedding"],
        )

    def buscar(self,
               query: List[float],
               top_k: int = 5,
               tipo_collection: str = "text") -> List[Dict[str, Any]]:
        
        """
        Busca por documentos semelhantes no banco de dados.

        Args:
            query: Lista de embeddings da consulta.
            top_k: Número máximo de resultados a serem retornados.
            collection: Nome da coleção a ser pesquisada (text, image, table).
        
        Returns:
            Lista de documentos semelhantes encontrados.
        """

        if tipo_collection == "text":
            collection = self.collection_texto
        elif tipo_collection == "image":
            collection = self.collection_imagem
        elif tipo_collection == "table":
            collection = self.collection_tabela
        else:
            raise ValueError("Tipo de coleção inválido. Use 'text', 'image' ou 'table'.")

        resultados = collection.query(
            query_embeddings=query,
            n_results=top_k,
        )

        processados = []

        if resultados['ids'] and len(resultados['ids'][0]) > 0:
            for i, doc_id in enumerate(resultados['ids'][0]):
                metadados = resultados['metadatas'][0][i]
                documento = resultados['documents'][0][i]
                distancia = resultados['distances'][0][i]

                resultado = {
                    "distancia" : distancia,
                    "metadados" : metadados,
                    "texto" : documento,
                    "tipo" : metadados.get("content_type", tipo_collection),
                }

                if tipo_collection == "image":
                    resultado["caminho_imagem"] = metadados.get("ref", "")

                processados.append(resultado)

        return processados
    
    def busca_completa(self, 
                      query: str, 
                      modelo_embedding_texto: ModeloEmbeddingTexto ,
                      modelo_embedding_imagem= ModeloEmbeddingImagem | None,
                      top_k: int = 5,
                      content_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Realiza uma pesquisa híbrida por texto
        
        Args:
            query: Texto da consulta
            modelo_embedding_texto: Modelo de embedding de texto
            modelo_embedding_imagem: Modelo de embedding de imagem (opcional)
            top_k: Número máximo de resultados
            content_types: Lista de tipos de conteúdo para incluir na pesquisa
            
        Returns:
            Lista de documentos similares
        """

        embedding_texto: List[float] = modelo_embedding_texto.gerar_embedding(query).tolist()

        resultados = []

        if not content_types or "text" in content_types:
            resultados_texto = self.buscar(query=embedding_texto, top_k=top_k, tipo_collection="text")
            resultados.extend(resultados_texto[:top_k])
        if not content_types or "table" in content_types:
            resultados_tabela = self.buscar(query=embedding_texto, top_k=top_k, tipo_collection="table")
            resultados.extend(resultados_tabela[:top_k])
        
        if (not content_types or "image" in content_types) and modelo_embedding_imagem:
            embedding_imagem: List[float] = modelo_embedding_imagem.gerar_embedding_textual(query)[0].tolist()

            resultados_imagem = self.buscar(query=embedding_imagem, top_k=top_k, tipo_collection="image")
            resultados.extend(resultados_imagem[:top_k])
            
        resultados.sort(key=lambda x: x["distancia"], reverse=True)
        return resultados