"""
Módulo de ingestão de documentos.

- scraper: Coleta metadados do repositório IPEA
- docling_pipeline: Processamento de PDFs e extração de conteúdo
- utils: Funções auxiliares
"""

from src.ingestao.scraper import Scraper
from src.ingestao.docling_pipeline import DoclingPipeline

__all__ = ["Scraper", "DoclingPipeline"]
