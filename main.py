from src.ingestao.docling_pipeline import DoclingPipeline
from src.ingestao.scraper import Scraper

import time
import os

TOTAL_PAGES = 860

if __name__ == "__main__":

    scraper = Scraper()
    for i in range(1, 20):
        scraper.processar_pagina(i)
    
    time.sleep(5)

    if os.name == "nt":
        os.system('cls')

    pipeline = DoclingPipeline()
    while True:
        sucesso = pipeline.processar_documento()
        if not sucesso:
            break
    print("Pipeline concluído.")




