from sentence_transformers import SentenceTransformer
from typing import List, Union
import torch
import numpy as np
from src.config import MODELO_EMBEDDING_TEXTO

class ModeloEmbeddingTexto:
    def __init__(self, modelo: str = MODELO_EMBEDDING_TEXTO):
        self.modelo = SentenceTransformer(modelo)
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo.to(self.dispositivo)
        print(f"Modelo de embedding de texto carregado: {modelo}")

    def gerar_embedding(self, textos: Union[str, List[str]]) -> np.ndarray:
        """
        Gera embeddings para um ou mais textos.

        Args:
            textos (Union[str, List[str]]): Texto ou lista de textos para gerar embeddings.
        
        Returns:
            np.ndarray: Embeddings gerados.
        """

        if isinstance(textos, str):
            textos = [textos]

        # Gerar embeddings
        embeddings = self.modelo.encode(textos)
        return embeddings
    
