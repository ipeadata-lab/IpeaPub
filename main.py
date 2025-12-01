from src.ingestor.docling_pipeline import DoclingPipeline


if __name__ == "__main__":

    # for pagina in range(0, 2):  # Exemplo: processar as primeiras 2 páginas
    #     processar_pagina(pagina)
    #     print(f"Página {pagina} processada e dados inseridos no banco de dados banco1.db")

    pipeline = DoclingPipeline()
    sucesso = pipeline.processar_documento()

