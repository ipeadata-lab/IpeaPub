import json
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

        documentos_processados = []
        # Processar cada tabela
        tabelas = dados.get("tables", [])
        for idx, tabela in enumerate(tabelas):
            if isinstance(tabela, dict) and "table" in tabela:
                metadados_tabela = metadados_base.copy()
                tabela_markdown = tabela["table"]
                metadados_tabela.update({
                    "self_ref": tabela.get("self_ref", ""),
                    "caption" : tabela.get("caption", ""),
                    "references" : tabela.get("references", []),
                    "footnotes" : tabela.get("footnotes", []),
                    "page" : tabela.get("page"),
                    "content_type" : "table",
                    "table_id" : idx + 1,
                })

                metadados_enriquecidos = self.modelo_ner.enriquecer_metadados(tabela_markdown, metadados_tabela)
                embedding = self.modelo_embedding.gerar_embedding(tabela_markdown).tolist()
                documentos_processados.append({
                    "texto": tabela_markdown,
                    "metadados": metadados_enriquecidos,
                    "embedding": embedding
                })

        return documentos_processados

    def processar_diretorio(self, diretorio: str = DATA_DIR) -> List[List[Dict[str, Any]]]:
        """
        Processa todas as tabelas nos arquivos JSON dentro de um diretório.
        """
        documentos_processados = []
        for arquivo in os.listdir(diretorio):
            if arquivo.endswith('.json'):
                caminho_arquivo = Path(os.path.join(diretorio, arquivo))
                documentos_processados.extend(self.processar_json(caminho_arquivo))

        return documentos_processados
