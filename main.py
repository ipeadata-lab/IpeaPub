from src.ingestor.docling_pipeline import DoclingPipeline
from src.ingestor.scraper import Scraper
import time
import os

TOTAL_PAGES = 860

if __name__ == "__main__":

    scraper = Scraper()
    for i in range(0, 5):
        scraper.processar_pagina(i)
        print(f"Página {i} processada.")

    time.sleep(10)

    if os.name == "nt":
        os.system('cls')
    else:
        os.system('clear')

    pipeline = DoclingPipeline()
    while True:
        sucesso = pipeline.processar_documento()
        if not sucesso:
            break
    print("Pipeline concluído.")
