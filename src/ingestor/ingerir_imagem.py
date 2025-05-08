import json
from typing import Dict, Any, List, Union
import os
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
import torch

from src.embeddings.modelo_imagem import ModeloEmbeddingImagem
from src.ner.ner import ModeloNER
from src.config import IMAGES_DIR, MODELO_VLM

class IngestorImagem:
    def __init__(self):

        self.processador = AutoProcessor.from_pretrained(MODELO_VLM)
        self.legendar = AutoModelForImageTextToText.from_pretrained(MODELO_VLM)
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.legendar.to(self.dispositivo)
        self.modelo_embedding = ModeloEmbeddingImagem()
        self.modelo_ner = ModeloNER()
        print(f"Modelo VLM carregado no dispositivo {self.dispositivo}: {MODELO_VLM}")

    def gerar_legendas(self, caminho_imagem: str) -> str:
        """
        Gera legendas para uma imagem usando o modelo VLM.
        
        Args:
            caminho_imagem: Caminho para a imagem.
            
        Returns:
            Legenda gerada para a imagem.
        """
        try:
            imagem = Image.open(caminho_imagem).convert("RGB")
        
            # Processar a imagem
            inputs = self.processador(images=imagem, return_tensors="pt").to(self.dispositivo)
            # Gerar a legenda
            with torch.no_grad():
                output_ids = self.legendar.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    return_dict_in_generate=True,
                ).sequences
            # Decodificar a legenda
            legenda = self.processador.batch_decode(output_ids, skip_special_tokens=True)[0]
            return legenda # Preciso Traduzir essa legenda para o português
        except Exception as e:
            print(f"Erro ao gerar legenda para a imagem {caminho_imagem}: {e}")
            return None
        