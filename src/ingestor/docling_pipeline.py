import os
import uuid
import torch
import tempfile
from transformers import AutoTokenizer
from typing import Iterable, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from docling_core.types.doc.labels import DocItemLabel
from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from src.db.banco_metadados import MetadataDB
from src.ingestor.utils import baixar_pdf_real
from src.db.banco_vetorial import QdrantVectorDB

class DoclingPipeline:
    """
    Pipeline para processar documentos com Docling.
    Inclui as seguintes etapas:
    1. Busca de documento pendente no banco de metadados
    2. Criação de embedding resumido (coleção de recomendação)
    3. Conversão do PDF para Docling Document
    4. Hybrid Chunking para segmentação (coleção de chunks)
    5. Extração de tabelas inteligente (coleção de tabelas)
    6. Extração de imagens, legendagem com IA (coleção de imagens)
    """

    def __init__(self,
                 embed_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 max_tokens: int = 512 
                 ) -> None:
        self.db_metadata = MetadataDB()
        self.db_vetorial = QdrantVectorDB()

        self.converter = self._setup_converter()
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embed_model),
            max_tokens=max_tokens
        )
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True
        )
        self.embedder = SentenceTransformer(embed_model)

        self.db_vetorial.ensure_collections()

    def _setup_converter(self) -> DocumentConverter:
        """Configura o DocumentConverter para PDFs com opções otimizadas."""

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 0.7

        device_type = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=device_type,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        return converter

    def _get_page_no(self, chunk: DocChunk) -> Optional[int]:
        page_nos = set()
        for doc_item in getattr(chunk.meta, "doc_items", []) or []:
            for prov in getattr(doc_item, "prov", []) or []:
                if getattr(prov, "page_no", None) is not None:
                    page_nos.add(prov.page_no)
        page_no = min(page_nos) if page_nos else None  # ou escolher outro critério

        return page_no

    def processar_documento(self) -> bool:
        """
        Pipeline completa de processamento de documento

        Returns:
            bool: True se um documento foi processado, False se falhar
        """

        # ============================================================ #
        # Buscar e converter inicialmente o documento
        # ============================================================ #
        metadata = self.db_metadata.buscar_pendente()
        if not metadata:
            print("[Docling] Nenhum documento pendente para processar.")
            return False
        
        print(f"[Docling] Processando documento {metadata['titulo']}...")
        self.db_metadata.atualizar_status(metadata["id"], "em processamento")

        link_pagina = metadata.get("link_pdf")
        if not link_pagina:
            print(f"[Docling] Documento {metadata['id']} sem link_pdf.")
            return False
        pdf_bytes = baixar_pdf_real(link_pagina)
        if not pdf_bytes:
            print("[Docling] Falha no download, abortando.")
            return False
        
        if not pdf_bytes.startswith(b"%PDF"):
            print("[Docling] PDF inválido, abortando.")
            return False
        
        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = tmp_pdf.name
        try:
            tmp_pdf.write(pdf_bytes)
            tmp_pdf.flush()
            tmp_pdf.close()  # fechar antes de passar o caminho ao conversor (necessário no Windows)

            try:
                docling_doc = self.converter.convert(tmp_path).document
            except Exception as e:
                print(f"[Docling] ERRO ao converter PDF {tmp_path}: {e}")
                return False
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                print(f"[Docling] Aviso: não foi possível deletar o arquivo temporário {tmp_path}: {e}")
        
        print(f"[Docling] Documento {metadata['id']} convertido com sucesso.")
        print("Iniciando pipeline completa de processamento...")

        chunk_iter = self.chunker.chunk(docling_doc)

        # Inserir os metadados na coleção de recomendação
        self._processar_recomendacao(metadata)

        # Inserir os chunks na coleção de chunks
        self._processar_chunks(metadata, chunk_iter)

        # Inserir as tabelas na coleção de tabelas
        self._processar_tabelas(metadata, chunk_iter)

        print(f"[Docling] Documento {metadata['id']} processado com sucesso.\n")
        print("Contagem de pontos no banco vetorial:")
        print(f" - Recomendação: {self.db_vetorial.count_points('recomendacoes')} pontos")
        print(f" - Chunks: {self.db_vetorial.count_points('chunks')} pontos")
        print(f" - Tabelas: {self.db_vetorial.count_points('tabelas')} pontos")
        print(f" - Imagens: {self.db_vetorial.count_points('imagens')} pontos")
        self.db_metadata.atualizar_status(metadata["id"], "processado")

        return True

    def _processar_recomendacao(self, metadata: dict) -> None:
        """
        Processa e insere o documento na coleção de recomendação.
        Args:
            metadata (dict): Metadados do documento.
        """
        pid = metadata.get("id", "")

        titulo = metadata.get("titulo", "")
        resumo = metadata.get("resumo", "")
        keywords = metadata.get("palavras_chave", "")
        handle = metadata.get("link_pdf", "")

        payload = {
            "pid": pid,
            "titulo": titulo,
            "keywords": keywords,
            "resumo": resumo,
            "handle": handle,
        }

        embed_text = f"{titulo}\n\n{resumo}\n\n{keywords}"
        embedding = self.embedder.encode(embed_text).tolist()

        self.db_vetorial.upsert_recommendation(payload, embedding)

    def _processar_chunks(self, metadata: dict, docling_iter) -> None:
        """
        Processa e insere os chunks do documento na coleção de chunks.
        Args:
            metadata (dict): Metadados do documento.
            docling_iter (Iterable): Iterador de chunks do documento.
        """
        chunks: List[DocChunk] = list(docling_iter)

        doc_id = metadata.get("id", "")
        handle = metadata.get("link_pdf", "")

        processed = 0
        for _, chunk in enumerate(chunks):

            doc_items = getattr(getattr(chunk, "meta", None), "doc_items", []) or []
            labels = [getattr(it, "label", None) for it in doc_items]
            for label in labels:
                # Se encontrar uma tabela, pular este chunk (será processado na coleção de tabelas)
                if label == DocItemLabel.TABLE:
                    print("[Docling] Chunk com tabela detectado, pulando para coleção de tabelas.")
                    continue

            context_chunk = self.chunker.contextualize(chunk=chunk)
            pid = uuid.uuid4().hex  # Gerar um ID único para o chunk

            # Processo de embedding do contexto_chunk
            embedding = self.embedder.encode(context_chunk).tolist()

            # Preparar payload para inserção
            payload = {
                "pid": pid,
                "doc_id": doc_id,
                "texto": context_chunk,
                "handle": handle,
                "pagina": self._get_page_no(chunk)
            }

            self.db_vetorial.upsert_chunk(payload, embedding)
            processed += 1

        if processed == 0:
            print("[Docling] Nenhum chunk processado no documento.\n")

    def _processar_tabelas(self, metadata: dict, docling_iter) -> None:
        """
        Processa e insere as tabelas do documento na coleção de tabelas.
        Args:
            metadata (dict): Metadados do documento.
            docling_iter (Iterable): Iterador de chunks do documento.
        """
        chunks: List[DocChunk] = list(docling_iter)

        doc_id = metadata.get("id", "")
        handle = metadata.get("link_pdf", "")

        processed = 0
        for _, chunk in enumerate(chunks):

            has_table = False
            doc_items = getattr(getattr(chunk, "meta", None), "doc_items", []) or []
            labels = [getattr(it, "label", None) for it in doc_items]
            for label in labels:
                # Procura apenas por chunks que contenham tabelas
                if label == DocItemLabel.TABLE:
                    print("[Docling] Chunk com tabela detectado, pulando para coleção de tabelas.")
                    has_table = True
            if not has_table:
                continue

            context_chunk = self.chunker.contextualize(chunk=chunk)
            pid = uuid.uuid4().hex  # ID único para a tabela

            # placeholder para embedding (use seu gerador real de embeddings aqui)
            embedding = self.embedder.encode(context_chunk).tolist()

            payload = {
                "pid": pid,
                "doc_id": doc_id,
                "tabela": context_chunk,
                "handle": handle,
                "pagina": self._get_page_no(chunk),
                "descricao_llm": "",  # placeholder para descrição gerada por LLM
            }

            self.db_vetorial.upsert_table(payload, embedding)
            processed += 1

        if processed == 0:
            print("[Docling] Nenhuma tabela encontrada no documento.\n")



