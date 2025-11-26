import os
import torch
import tempfile
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from src.db.banco1 import buscar_pendente

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.document import ConversionResult

# ============================================================
#   1. CONFIGURAÇÕES GERAIS
# ============================================================

BASE_URL = "https://repositorio.ipea.gov.br"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


# ============================================================
#   2. CRAWLER DO PDF (IPEA)
# ============================================================

def baixar_pdf_real(link_pagina: str) -> bytes | None:
    """
    Recebe o link da página do documento no repositório do IPEA.
    Identifica o botão de download, resolve redirecionamentos e retorna bytes do PDF.
    """

    print(f"[Crawler] Acessando página do documento:\n  {link_pagina}")

    try:
        resp = requests.get(link_pagina, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Crawler] ERRO ao acessar página: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Procurar botão do tipo /bitstreams/<uuid>/download
    a_tags = soup.find_all("a", href=True)
    download_links = []
    for a in a_tags:
        href = a.get("href")
        if not href:
            continue
        href = str(href)
        if "bitstreams" in href and "download" in href:
            download_links.append(href)

    if not download_links:
        print("[Crawler] Nenhum link de download encontrado na página.")
        return None

    download_url = urljoin(BASE_URL, download_links[0])
    print(f"[Crawler] Botão de download encontrado:\n  {download_url}")

    # Resolver download efetivo (pode virar /content ou baixar direto)
    try:
        r = requests.get(download_url, headers=HEADERS, allow_redirects=True, timeout=40)
        r.raise_for_status()
    except Exception as e:
        print(f"[Crawler] ERRO ao baixar PDF real: {e}")
        return None

    # Validar PDF
    if not r.content.startswith(b"%PDF"):
        print("[Crawler] Conteúdo baixado não parece ser PDF real. Pode ser página 'Baixando...'")
        # algumas páginas fazem meta refresh → tentar novamente
        # ou acessar o botão final no HTML (mas em geral o requests já resolve)
        pass

    return r.content


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


