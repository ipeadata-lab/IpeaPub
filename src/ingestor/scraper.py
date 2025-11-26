import requests
from datetime import datetime, timezone
from src.db.banco1 import inserir_documento, buscar_documento, atualizar_documento
from src.ingestor.utils import clean_item

BASE = "https://repositorio.ipea.gov.br/server/api/discover/browses/dateissued/items"
TOTAL_PAGES = 860

def _buscar_pagina(page_number: int):
    """Busca os itens brutos da API para a página fornecida."""
    url = f"{BASE}?page={page_number}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data["_embedded"]["items"]

def _extrair_campos(item):
    """Extrai campos brutos da API, NÃO limpa."""
    metadata = item.get("metadata", {})

    def get(field):
        arr = metadata.get(field, [])
        if not arr:
            return None
        if len(arr) == 1:
            return arr[0]["value"]
        return [v["value"] for v in arr]

    titulo = get("dc.title.scholar") or item.get("name")

    return {
        "id": item.get("id"),
        "handle": get("dc.identifier.uri"),
        "titulo": titulo or "",
        "autores": get("dc.contributor.author") or [],
        "ano": get("dc.date.issued"),
        "resumo": get("dc.description.abstract") or [],
        "palavras_chave": get("dc.subject.keyword") or [],
        "tipo": get("dc.type") or [],
        "last_modified": item.get("lastModified"),
    }

def processar_pagina(pagina: int):
    """Extrai, limpa e salva todos itens de uma página no Banco 1."""

    itens_raw = _buscar_pagina(pagina)
    count = 0

    for raw in itens_raw:
        try:
            bruto = _extrair_campos(raw)
            item = clean_item(bruto)

            # prepara estrutura SQLite (Banco 1)
            doc = {
                "id": item["id"],
                "titulo": item["titulo"],
                "autores": item["autores"],
                "ano": item["ano"],
                "tipo_conteudo": item.get("tipo") or item.get("tipo_conteudo") or "",
                "resumo": item["resumo"],
                "palavras_chave": item["palavras_chave"],
                "link_pdf": item["handle"],
                "status_ingestao": "pendente",
                "data_ingestao": datetime.now(timezone.utc).isoformat(),
            }


            print(f"Inserindo novo documento {item['id']} no banco de dados.")
            inserir_documento(doc)
    
            count += 1

        except Exception as e:
            print(f"Falha ao processar item {raw.get('id', 'desconhecido')}: {e}", flush=True)

    return count
