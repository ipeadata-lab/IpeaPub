import json
from typing import Dict, Any, List, Union
import os
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
import torch

from src.embeddings.modelo_imagem import ModeloEmbeddingImagem
from src.embeddings.modelo_texto import ModeloEmbeddingTexto
from src.ner.ner import ModeloNER
from src.config import IMAGES_DIR, MODELO_VLM

class IngestorImagem:
    def __init__(self):
        self.processador = AutoProcessor.from_pretrained(MODELO_VLM)
        self.legendar = AutoModelForImageTextToText.from_pretrained(MODELO_VLM)
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.legendar.to(self.dispositivo)
        self.modelo_embedding = ModeloEmbeddingImagem()
        self.modelo_embedding_texto = ModeloEmbeddingTexto()
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
            # Adicionar legenda: no inicio da string
            legenda = "Legenda: " + legenda
            return legenda # Preciso Traduzir essa legenda para o português
        except Exception as e:
            print(f"Erro ao gerar legenda para a imagem {caminho_imagem}: {e}")
            return None
        
    def processar_json(self, caminho_arquivo: str) -> List[Dict[str, Any]]:
        """
        Processa as imagens dentro de um arquivo JSON conforme a estrutura do projeto.
        
        Args:
            caminho_arquivo: Caminho para o arquivo JSON.
            
        Returns:
            Lista de dicionários com os dados processados.
        """
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        # Extrair informações dos metadados do json
        if isinstance(dados, dict):
            metadados = dados.get("metadata", {})
            if isinstance(metadados, dict):
                metadados_base = {
                    "format": metadados.get("format", None),
                    "title": metadados.get("title", None),
                    "modDate": metadados.get("modDate", None),
                    "filename": os.path.basename(caminho_arquivo),
                }
            else:
                metadados_base = {}

        documentos_processados = []
        # Processar cada imagem
        imagens = dados.get("images", [])
        for idx, imagem in enumerate(imagens):
            if isinstance(imagem, dict) and "ref" in imagem:
                caminho_imagem = os.path.join(IMAGES_DIR, imagem["ref"])
                legenda = self.gerar_legendas(caminho_imagem)

                if legenda is not None:
                    metadados_imagem = metadados_base.copy()
                    metadados_imagem.update({
                        "ref" : imagem.get("ref", ""),
                        "self_ref" : imagem.get("self_ref", []),
                        "classification" : imagem.get("classification"),
                        "confidence" : imagem.get("confidence", ""),
                        "page" : imagem.get("page"),
                        "content_type" : "image",
                        "references" : imagem.get("references", []),
                        "footnotes" : imagem.get("footnotes", []),
                        "image_id" : idx + 1,
                        "caption" : imagem.get("caption", "") + " " + legenda
                    })

                    metadados_enriquecidos = self.modelo_ner.enriquecer_metadados(legenda, metadados_imagem)
                    embedding_imagem = self.modelo_embedding.gerar_embedding(caminho_imagem).tolist()
                    embedding_texto = self.modelo_embedding_texto.gerar_embedding(legenda).tolist()
                    documentos_processados.append({
                        "texto": legenda,
                        "metadados": metadados_enriquecidos,
                        "embedding_imagem": embedding_imagem,
                        "embedding_texto": embedding_texto
                    })

        return documentos_processados
    
    def processar_diretorio(self, diretorio: str = IMAGES_DIR) -> List[Dict[str, Any]]:
        """
        Processa todas as imagens dentro de um diretório conforme a estrutura do projeto.
        
        Args:
            diretorio: Caminho para o diretório contendo as imagens.
            
        Returns:
            Lista de dicionários com os dados processados.
        """
        documentos_processados = []
        for arquivo in os.listdir(diretorio):
            if arquivo.endswith(".json"):
                caminho_arquivo = os.path.join(diretorio, arquivo)
                documentos_processados.extend(self.processar_json(caminho_arquivo))
        return documentos_processados