from src.ingestor.scraper import processar_pagina
from src.ingestor.pipeline_extracao import processar_documento

if __name__ == "__main__":
    print("Tabela criada com sucesso no banco de dados banco1.db")

    for pagina in range(0, 2):  # Exemplo: processar as primeiras 2 páginas
        processar_pagina(pagina)
        print(f"Página {pagina} processada e dados inseridos no banco de dados banco1.db")

    while True:
        sucesso = processar_documento()
        if not sucesso:
            break

