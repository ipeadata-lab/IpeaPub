"""
Aqui vai vir toda a lógica de extração de texto, imagens e tals

"""
from src.db.banco_metadados import atualizar_status
from src.ingestor.docling_setup import converter_docling

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

import os
import torch
import tempfile
import requests


from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.document import ConversionResult

# ============================================================
#   1. CONFIGURAÇÕES GERAIS
# ============================================================



# ============================================================
#   3. DOCLING SETUP
# ============================================================

def setup_converter() -> DocumentConverter:
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


# ============================================================
#   4. DOCLING CONVERSÃO
# ============================================================

def converter_docling() -> tuple[ConversionResult, str] | None:
    """
    Converte o próximo PDF pendente em Docling Document.
    
    Returns:
        tuple(ConversionResult, doc_id) | None
    """
    doc_bruto = buscar_pendente()
    if not doc_bruto:
        return None
    
    doc_id = doc_bruto["id"]

    link_pagina = doc_bruto.get("link_pdf")
    if not link_pagina:
        print(f"[Docling] Documento {doc_bruto['id']} sem link_pdf.")
        return None

    pdf_bytes = baixar_pdf_real(link_pagina)
    if not pdf_bytes:
        print("[Docling] Falha no download, abortando.")
        return None

    if not pdf_bytes.startswith(b"%PDF"):
        # Salvar para debug
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(pdf_bytes)
            path = tmp.name
        print(f"[Docling] PDF inválido. Dump salvo em:\n  {path}")
        return None

    converter = setup_converter()

    # Criar PDF temporário
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        tmp_path = tmp_pdf.name

    try:
        doc = converter.convert(tmp_path)
    except Exception as e:
        print(f"[Docling] ERRO ao converter PDF {tmp_path}: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

    # Docling pode retornar doc ou wrapper
    return doc, doc_id






# ============================================================
#   5. PIPELINE COMPLETO
# ============================================================

def processar_documento() -> bool:
    """
    converte com Docling,
    processa os resultados
    atualiza status.
    """

    doc_tuple = converter_docling()
    if not doc_tuple:
        print("[Pipeline] Nenhum documento pendente para conversão.")
        return False
    
    doc, doc_id = doc_tuple

    if hasattr(doc, 'document'):
        doc = doc.document
    else:
        print("[Pipeline] Aviso: objeto retornado não possui atributo 'document'.")
        return False
    

    

    print(f"\n=== PROCESSANDO DOCUMENTO {doc_id} ===")


    # Aqui virá a lógica de processamento do documento
    print("placeholder para a lógica de extração e processamento do documento...")

    # No final, se tudo der certo, atualizar o status do documento para processado
    atualizar_status(doc_id, "processado")
    print(f"Documento processado para os bancos de dados!\n ID: {doc_id}\n")

    return True

