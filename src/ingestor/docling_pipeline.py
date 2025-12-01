"""
Aqui vai vir toda a lógica de extração de texto, imagens e tals

"""

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

import os
import torch
import tempfile

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.document import ConversionResult

from src.db.banco_metadados import MetadataDB
from src.db.banco_vetorial import QdrantVectorDB
from src.ingestor.utils import baixar_pdf_real


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

    def __init__(self) -> None:
        self.db_metadata = MetadataDB()
        self.db_vectorial = QdrantVectorDB()

        self.converter = self._setup_converter()

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

    def processar_documento(self) -> bool:
        """
        Pipeline completa de processamento de documento

        Returns:
            bool: True se um documento foi processado, False se falhar
        """

        # ============================================================ #
        # Buscar e converter inicialmente o documento
        # ============================================================ #
        documento = self.db_metadata.buscar_pendente()
        if not documento:
            print("[Docling] Nenhum documento pendente para processar.")
            return False
        
        print(f"[Docling] Processando documento {documento['id']}...")
        
        link_pagina = documento.get("link_pdf")
        if not link_pagina:
            print(f"[Docling] Documento {documento['id']} sem link_pdf.")
            return False
        pdf_bytes = baixar_pdf_real(link_pagina)
        if not pdf_bytes:
            print("[Docling] Falha no download, abortando.")
            return False
        
        if not pdf_bytes.startswith(b"%PDF"):
            print("[Docling] PDF inválido, abortando.")
            return False
        
        # with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        #     tmp_pdf.write(pdf_bytes)
        #     tmp_path = tmp_pdf.name

        #     try:
        #         doc = self.converter.convert(tmp_path)
        #     except Exception as e:
        #         print(f"[Docling] ERRO ao converter PDF {tmp_path}: {e}")
        #         return False
        #     finally:
        #         try:
        #             os.unlink(tmp_path)
        #         except:
        #             print(f"[Docling] Aviso: não foi possível deletar o arquivo temporário {tmp_path}.")
        #             pass
        
        print(f"[Docling] Documento {documento['id']} convertido com sucesso.")
        print("Iniciando pipeline completa de processamento...")

        """
        Modelo do payload dos metadados:
        {
            "id": str,
            "titulo": str,
            "autores": str,
            "ano": int,
            "tipo_conteudo": str,
            "resumo": str,
            "palavras_chave": str,
            "link_pdf": str,
            "status_ingestao": str,
            "data_ingestao": str
        }
        """

        return True