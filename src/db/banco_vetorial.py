from typing import Any, Dict, Iterable, List, Tuple
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

    # ============================================================ #
    # Configuração e inicialização
    # ============================================================ #
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "banco1.db"

COLLECTIONS = {
    "recomendacoes" : {
        "vector_size": None,
        "distance": Distance.COSINE
    },
    "chunks": {
        "vector_size": None,
        "distance": Distance.COSINE
    },
    "imagens": {
        "vector_size": None,
        "distance": Distance.COSINE
    },
    "tabelas": {
        "vector_size": None,
        "distance": Distance.COSINE
    }
}

def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

class QdrantVectorDB:
    """Adapter simples para Qdrant local.


    Parâmetros importantes:
    - host, port: conexão com qdrant local
    - prefer_grpc: se True (prefira gRPC), senão HTTP
    - vector_size: dimensão padrão usada para criar coleções (pode ser sobrescrita)


    Métodos principais:
    - ensure_collections()
    - upsert_recommendation / upsert_chunk / upsert_image / upsert_table
    - upsert_{collection}_batch
    - search_{collection}
    - delete_collection / count_points


    Nota: os métodos aceitam `embedding` como lista de floats.
    """


    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False,
        vector_size: int = 768,
    ) -> None:
        self.host = host
        self.port = port
        self.prefer_grpc = prefer_grpc
        self.vector_size = vector_size

        # self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)
        self.client = QdrantClient(path=str(DB_PATH.parent / "qdrant"))  # Qdrant local em arquivo

        for k in COLLECTIONS:
            COLLECTIONS[k]["vector_size"] = vector_size

    def ensure_collections(self) -> None:
        for name, params in COLLECTIONS.items():
            try:
                # get_collection raises if not found (local Qdrant), so catch and create
                self.client.get_collection(name)
            except Exception:
                print(f"[Qdrant] Criando coleção '{name}'...")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=params["vector_size"],
                        distance=params["distance"],
                    )
                )

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)
    
    def count_points(self, name: str) -> int:
        stats = self.client.count(collection_name=name)
        return int(stats.count) if stats else 0
    
    def show_points(self, name: str, limit: int = 5) -> None:
        points, _ = self.client.scroll(
            collection_name=name,
            limit=limit,
        )
        results = []
        for hit in points:
            print(f"ID: {hit.id}, Payload: {hit.payload}, Vector: {hit.vector}\n\n")

    # ============================================================ #
    # Funções internas de upsert
    # ============================================================ #
    def _upsert_point(
            self,
            collection: str,
            point_id: str,
            payload: Dict[str, Any],
            embedding: List[float],
    ) -> None:
        point = PointStruct(id=point_id, payload=payload, vector=embedding)

        self.client.upsert(
            collection_name=collection,
            points=[point]
        )

    def _upsert_batch(
            self,
            collection: str,
            items: Iterable[Tuple[str, Dict[str, Any], List[float]]], 
            batch_size: int = 128) -> None:
        batch: List[PointStruct] = []
        for pid, payload, vector in items:
            batch.append(PointStruct(id=pid, payload=payload, vector=vector))
            if len(batch) >= batch_size:
                self.client.upsert(collection_name=collection, points=batch)
                batch = []
        if batch:
            self.client.upsert(collection_name=collection, points=batch)


    # ============================================================ #
    # Funções específicas de upsert
    # ============================================================ #
    def upsert_recommendation(
            self,
            doc_meta: Dict[str, Any],
            embedding: List[float],
    ) -> None:
        """
        Insere ou atualiza uma recomendação vetorial.
        O embedding deve juntar os campos do payload (conforme doc_meta).
        
        Args:
            doc_meta: metadados do documento (deve conter 'id')
            embedding: vetor de embedding
        """
        pid = doc_meta.get("pid") or doc_meta.get("chunk_id")
        if not pid:
            raise ValueError("doc_meta deve conter 'id' ou 'doc_id'")
        
        payload = {
            "doc_id": pid,
            "titulo": doc_meta.get("titulo", ""),
            "keywords": _ensure_list(doc_meta.get("palavras_chave")),
            "resumo": doc_meta.get("resumo", ""),
            "handle": doc_meta.get("handle", ""),
        }

        self._upsert_point("recomendacoes", pid, payload, embedding)

    def upsert_chunk(
            self,
            chunk_meta: Dict[str, Any],
            embedding: List[float],
    ) -> None:
        """
        Insere ou atualiza um chunk vetorial.
        O embedding deve juntar os campos do payload (conforme chunk_meta).
        
        Args:
            chunk_meta: metadados do chunk (deve conter 'id', 'doc_id', 'texto')
            embedding: vetor de embedding
        """
        pid = chunk_meta.get("pid") or chunk_meta.get("chunk_id")
        if not pid:
            raise ValueError("chunk_meta deve conter 'pid' ou 'chunk_id'")
        
        payload = {
            "pid": pid,
            "doc_id": chunk_meta.get("doc_id", ""),
            "texto": chunk_meta.get("texto", ""),
            "handle": chunk_meta.get("handle", ""),
            "pagina": chunk_meta.get("pagina", -1),
        }

        self._upsert_point("chunks", pid, payload, embedding)

    def upsert_image(self, image_meta: Dict[str, Any], embedding: List[float]) -> None:
        """image_meta esperado: {doc_id, pagina, caption, descricao_llm, bytes_imagem}
        Atenção: bytes_imagem idealmente deve ser base64 string ou pequeno.
        """
        pid = image_meta.get("id") or f"img::{image_meta.get('doc_id')}::{image_meta.get('pagina')}::{abs(hash(image_meta.get('caption') or ''))}"
        
        payload = {
        "doc_id": image_meta.get("doc_id"),
        "pagina": image_meta.get("pagina"),
        "caption": image_meta.get("legenda"),
        "descricao_llm": image_meta.get("descricao_llm"),
        "bytes_imagem": image_meta.get("bytes_imagem"),
        }
        self._upsert_point("imagens", pid, payload, embedding)
        
    def upsert_table(self, table_meta: Dict[str, Any], embedding: List[float]) -> None:
        """
        Insere ou atualiza uma tabela vetorial.
        O embedding deve juntar os campos do payload (conforme table_meta).
        Args:
            table_meta: metadados da tabela (deve conter 'pid' ou 'chunk_id')
            embedding: vetor de embedding
        """
        pid = table_meta.get("pid") or table_meta.get("chunk_id")
        if not pid:
            raise ValueError("table_meta deve conter 'pid' ou 'chunk_id'")

        payload = {
            "pid": pid,
            "doc_id": table_meta.get("doc_id"),
            "pagina": table_meta.get("pagina"),
            "descricao_llm": table_meta.get("descricao_llm"),
            "tabela": table_meta.get("tabela"),
            "handle": table_meta.get("handle"),
        }
        self._upsert_point("tabelas", pid, payload, embedding)

    def upsert_recommendation_batch(
            self,
            items: Iterable[Tuple[str, Dict[str, Any], List[float]]],
            batch_size: int = 128) -> None:
        self._upsert_batch("recomendacoes", items, batch_size)

    def upsert_chunk_batch(
            self,
            items: Iterable[Tuple[str, Dict[str, Any], List[float]]],
            batch_size: int = 128) -> None:
        self._upsert_batch("chunks", items, batch_size)

    def upsert_image_batch(
            self,
            items: Iterable[Tuple[str, Dict[str, Any], List[float]]],
            batch_size: int = 128) -> None:
        self._upsert_batch("imagens", items, batch_size)
    
    def upsert_table_batch(
            self,
            items: Iterable[Tuple[str, Dict[str, Any], List[float]]],
            batch_size: int = 128) -> None:
        self._upsert_batch("tabelas", items, batch_size)


    # ============================================================ #
    # Suportes para busca
    # ============================================================ #
    def _search(
            self,
            collection: str,
            query_vector: List[float],
            top_k: int = 5,
    ):
        
        hits = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        results = []
        for hit in hits.points:
            results.append({
                "id": hit.id,
                "payload": hit.payload,
                "score": hit.score,
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def search_recommendations(
            self,
            query_vector: List[float],
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._search("recomendacoes", query_vector, top_k)
    
    def search_chunks(
            self,
            query_vector: List[float],
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._search("chunks", query_vector, top_k)
    
    def search_images(
            self,
            query_vector: List[float],
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._search("imagens", query_vector, top_k)
    
    def search_tables(
            self,
            query_vector: List[float],
            top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        return self._search("tabelas", query_vector, top_k)