import json
from typing import Dict, Any, List
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embeddings.modelo_texto import ModeloEmbeddingTexto
from src.ner.ner import ModeloNER
from src.config import TAMANHO_CHUNK, SOBREPOR_CHUNK

class IngestorTexto:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TAMANHO_CHUNK,
            chunk_overlap=SOBREPOR_CHUNK,
            length_function=len,
        )
        self.modelo_embedding_texto = ModeloEmbeddingTexto()
        self.modelo_ner = ModeloNER()

    def processar_json(self, caminho: str) -> List[Dict[str, Any]]:
        """
        Processa um arquivo JSON e retorna uma lista de dicionários com os dados.
        
        Args:
            caminho: Caminho para o arquivo JSON.
            
        Returns:
            Lista de dicionários com os dados do JSON.
        """
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Extrair informações dos metadados do json
        if isinstance(dados, dict):
            metadados = dados.get("metadata", {})
            if isinstance(metadados, dict):
                metadados_base = {
                    "format": metadados.get("format", None),
                    "title": metadados.get("title", None),
                    "modDate": metadados.get("modDate", None),
                    "filename": os.path.basename(caminho),
                    "content_type" : "text"
                }
            else:
                metadados_base = {
                    "content_type" : "text"
                }

        conteudo_paginas = []
        if "pages" in dados and isinstance(dados["pages"], list):
            for i, pagina in enumerate(dados["pages"]):
                if isinstance(pagina, dict) and "content" in pagina:
                    metadados_pagina = metadados_base.copy()
                    metadados_pagina["page_number"] = pagina.get("page_number", i + 1)
                    conteudo_paginas.append({
                        "metadados": metadados_pagina,
                        "conteudo": pagina["content"]
                    })

        # Processar cada página individualmente
        documentos_processados = []

        for conteudo in conteudo_paginas:
            texto = conteudo["conteudo"]
            metadados = conteudo["metadados"]

            chunks = self.text_splitter.split_text(texto)

            for i, chunk in enumerate(chunks):
                metadados_chunk = metadados.copy()
                metadados_chunk["chunk_id"] = i

                metadados_enriquecidos = self.modelo_ner.enriquecer_metadados(chunk, metadados_chunk)

                # Gerar embedding para o chunk
                embedding = self.modelo_embedding_texto.gerar_embedding(chunk)
                # Criar documento processado
                documento_processado = {
                    "texto": chunk,
                    "embedding": embedding.tolist(),  # Converter para lista
                    "metadados": metadados_enriquecidos
                }

                documentos_processados.append(documento_processado)

        return documentos_processados

    def processar_diretorio(self, diretorio: str) -> List[List[Dict[str, Any]]]:
        """
        Processa todos os arquivos JSON em um diretório e retorna uma lista de dicionários com os dados.
        
        Args:
            diretorio: Caminho para o diretório.
            
        Returns:
            Lista de dicionários com os dados dos arquivos JSON.
        """
        documentos_processados = []

        for arquivo in os.listdir(diretorio):
            if arquivo.endswith('.json'):
                caminho_arquivo = Path(os.path.join(diretorio, arquivo))
                documentos_processados.extend(self.processar_json(caminho_arquivo))

        return documentos_processados