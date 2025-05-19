from .processor import PdfProcessor
from .scraper import scrape_pdf

import multiprocessing

output_path = "Z:/rag_publicacoes/data1"

processor = PdfProcessor(output_path=output_path)
temp_path = "Z:/rag_publicacoes/src/pdf_scraping/temp"

if __name__ == "__main__":
    print("Iniciando o scraping...")

    for i in range(954, 960):
        handle_id = str(i)
        filepath = scrape_pdf(handle_id, temp_path)
        print(f"Filename: {filepath}")

processor.process_pdf()