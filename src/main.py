# src/main.py

import time
from typing import List, Optional, Dict, Any

from src.vector_db.chroma_client import ChromaDB
from src.retrieval.rag_retriever import RAGRetriever
from src.config import DATA_DIR, VECTOR_DB
from src.modelos import manager

class Assistente:
    """
    Classe para gerenciar a ingestão de dados e consultas usando o modelo RAG.
    """

    def __init__(self):
        self.rag_retriever = RAGRetriever()
        self.ingestor_texto = manager.ingestor_texto
        self.ingestor_imagem = manager.ingestor_imagem
        self.ingestor_tabela = manager.ingestor_tabela

    def ingerir_dados(self, dir: str = DATA_DIR) -> None:
        """
        Ingerir dados de texto, imagem e tabela a partir de diretórios especificados.
        
        Args:
            dir: Diretório onde os arquivos de texto e tabela estão localizados.
            imagens_dir: Diretório onde as imagens estão localizadas.
        """


        # Inicializar banco de dados vetorial
        if VECTOR_DB == "chroma":
            banco = ChromaDB()
        else:
            raise ValueError(f"Banco de dados vetorial {VECTOR_DB} ainda não implementado.")

        # Ingestão de texto
        tempo = time.time()
        print("\nProcessando textos dos arquivos...")
        textos_ingeridos = self.ingestor_texto.processar_diretorio(dir)

        # Inserir documentos no banco de dados
        print("Inserindo documentos no banco de dados...")
        for documento in textos_ingeridos:
                banco.inserir_documento(documento)
        print(f"Textos processados em {time.time() - tempo:.2f} segundos.")

        # Ingestão de imagem
        tempo = time.time()
        print("\nProcessando imagens dos arquivos...")
        imagens_ingeridas = self.ingestor_imagem.processar_diretorio(dir)

        # Inserir documentos de imagem
        print("Inserindo documentos de imagem no banco vetorial...")
        for documento in imagens_ingeridas:
            banco.inserir_imagem(documento)
        print(f"Imagens processadas em {time.time() - tempo:.2f} segundos.")

        # Ingestão de tabela
        tempo = time.time()
        print("\nProcessando tabelas dos arquivos...")
        tabelas_ingeridas = self.ingestor_tabela.processar_diretorio(dir)

        # Inserir documentos de tabela
        print("Inserindo documentos de tabela no banco vetorial...")
        for documento in tabelas_ingeridas:
            banco.inserir_tabela(documento)
        print(f"Tabelas processadas em {time.time() - tempo:.2f} segundos.")

        print("\nIngestão de dados concluída!")

    def consultar_rag(self, query: str, limite: int = 5, content_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Realiza uma consulta usando o modelo RAG.
        
        Args:
            query: Consulta a ser realizada.
            limite: Número máximo de resultados a serem retornados.
            content_types: Lista de tipos de conteúdo a serem filtrados (opcional).
        """

        resultados = self.rag_retriever.responder(query, limite, content_types)
        
        #print("\nResultados da consulta:")
        #print(resultados["resposta"])

        #print("\nResultados relevantes:")
        #for i, doc in enumerate(resultados["contexto"]):
        #    content_type: str = doc.get("metadados", {}).get("content_type", "text")
        #    fonte = doc.get("metadados", {}).get("filename", "desconhecido")
        #    print(f"{i+1}. [{content_type.upper()}] Fonte: {fonte}, Score: {doc['distancia']:.4f}")

        return resultados


