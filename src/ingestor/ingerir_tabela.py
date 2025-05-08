import json
import re
from typing import Dict, List, Any
import os
from pathlib import Path

from src.embeddings.modelo_texto import ModeloEmbeddingTexto
from src.ner.ner import ModeloNER
from src.config import DATA_DIR

class IngestorTabela:
    def __init__(self):
        self.modelo_embedding = ModeloEmbeddingTexto()
        self.modelo_ner = ModeloNER()

    def processar_json(self, caminho_arquivo: str) -> List[Dict[str, Any]]:
        """
        Processa as tabelas dentro de um arquivo JSON conforme a estrutura do projeto.
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

        tabelas = []
        if isinstance(dados, dict) and "tables" in dados and isinstance(dados["tables"], list):
            for i, tabela in enumerate(dados["tables"]):
                if isinstance(tabela, dict) and "table" in tabela:
                    metadados_tabela = metadados_base.copy()
                    metadados_tabela["tabela_id"] = i + 1

                    for chave, valor in tabela.items():
                        if chave != "table" and isinstance(valor, (str, int, float, bool)):
                            metadados_tabela[chave] = valor

                    # Limpeza de espaços em números: "1. 234. 567" -> "1.234.567"
                    texto_tabela = re.sub(r'(\d)\.\s(\d)', r'\1.\2', tabela["table"])

                    tabelas.append({
                        "texto": texto_tabela,
                        "metadados": metadados_tabela
                    })

        documentos_processados = []

        for tabela in tabelas:
            markdown = tabela["texto"]
            metadados = tabela["metadados"]
            metadados = self.modelo_ner.enriquecer_metadados(markdown, metadados)
            embedding = self.modelo_embedding.gerar_embedding(markdown).tolist()
            documentos_processados.append({
                "texto": markdown,
                "metadados": metadados,
                "embedding": embedding
            })

        return documentos_processados

    def processar_diretorio(self, diretorio: str = DATA_DIR) -> List[Dict[str, Any]]:
        """
        Processa todas as tabelas nos arquivos JSON dentro de um diretório.
        """
        documentos_processados = []
        for arquivo in os.listdir(diretorio):
            if arquivo.endswith('.json'):
                caminho_arquivo = Path(os.path.join(diretorio, arquivo))
                documentos_processados.extend(self.processar_json(caminho_arquivo))

        return documentos_processados
