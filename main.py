from src.db.banco1 import *
from src.ingestao.scraper import processar_pagina

if __name__ == "__main__":
    criar_tabela()
    print("Tabela criada com sucesso no banco de dados banco1.db")


    for pagina in range(0, 10):  # Exemplo: processar as primeiras 10 páginas
        processar_pagina(pagina)
        print(f"Página {pagina} processada e dados inseridos no banco de dados banco1.db")