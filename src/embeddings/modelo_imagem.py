import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union
import numpy as np
from pathlib import Path

from src.config import MODELO_EMBEDDING_IMAGEM

class ModeloEmbeddingImagem:
    def __init__(self, modelo: str = MODELO_EMBEDDING_IMAGEM):
        self.modelo = CLIPModel.from_pretrained(modelo)
        self.processador = CLIPProcessor.from_pretrained(modelo)
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo.to(self.dispositivo)
        print(f"Modelo de embedding de imagem carregado: {modelo}")
        
    def gerar_embedding(self, imagens: Union[str, List[str], Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Gera embeddings para imagem ou lista de imagens
        
        Args:
            imagens: Caminho para a imagem, lista de caminhos, objeto PIL.Image ou lista de objetos PIL.Image
            
        Returns:
            Array numpy com os embeddings
        """

        imagens_processadas = []

        if isinstance(imagens, (str, Path)):
            imagens = [Image.open(imagens)]
        elif isinstance(imagens, list) and all(isinstance(img, (str, Path)) for img in imagens):
            imagens = [Image.open(img) for img in imagens]
        elif isinstance(imagens, Image.Image):
            imagens = [imagens]

        # Processar as imagens
        inputs = self.processador(images=imagens, return_tensors="pt", padding=True).to(self.dispositivo)

        with torch.no_grad():
            outputs = self.modelo.get_image_features(**inputs)
        embeddings = outputs.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def gerar_embedding_textual(self, textos: Union[str, List[str]]) -> np.ndarray:
        """
        Gera embeddings para texto no espaço conjunto de texto-imagem do CLIP
        Útil para realizar buscas multimodais (texto -> imagem)
        
        Args:
            text: Texto ou lista de textos
            
        Returns:
            Array numpy com os embeddings
        """
        if isinstance(textos, str):
            textos = [textos]
            
        inputs = self.processador(text=textos, return_tensors="pt", padding=True).to(self.dispositivo)
        
        with torch.no_grad():
            text_embeddings = self.modelo.get_text_features(**inputs)
            
        # Normalizar os embeddings
        embeddings = text_embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings
    