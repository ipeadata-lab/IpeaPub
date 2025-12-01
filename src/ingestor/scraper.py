import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timezone
from src.ingestor.utils import clean_item
from typing import Any, Dict, List, Optional
from src.db.banco_metadados import MetadataDB

BASE = "https://repositorio.ipea.gov.br/server/api/discover/browses/dateissued/items"
TOTAL_PAGES = 860
BASE_URL = "https://repositorio.ipea.gov.br"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class Scraper:
    """
    Scraper para o repositório IPEA.
    Agrupa operações de listagem (API), extração/limpeza de metadados e download de PDF.
    """

    def __init__(
        self,
        base_api: str = BASE,
        base_url: str = BASE_URL,
        headers: Dict[str, str] = HEADERS,
    ) -> None:
        self.base_api = base_api
        self.base_url = base_url
        self.headers = headers
        self.db = MetadataDB()

    def _buscar_pagina(self, page_number: int) -> List[Dict[str, Any]]:
        """Busca os itens brutos da API para a página fornecida."""
        url = f"{self.base_api}?page={page_number}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data["_embedded"]["items"]

    def _extrair_campos(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai campos brutos da API (não limpa)."""
        metadata = item.get("metadata", {})

        def get(field: str):
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

    def processar_pagina(self, pagina: int) -> int:
        """
        Extrai, limpa e salva todos itens de uma página no banco de metadados.
        Retorna a quantidade de itens processados.
        """
        itens_raw = self._buscar_pagina(pagina)
        count = 0

        for raw in itens_raw:
            try:
                bruto = self._extrair_campos(raw)
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
                self.db.inserir_documento(doc)

                count += 1

            except Exception as e:
                print(f"Falha ao processar item {raw.get('id', 'desconhecido')}: {e}", flush=True)

        return count

    def baixar_pdf_real(self, link_pagina: str) -> Optional[bytes]:
        """
        Recebe o link da página do documento no repositório do IPEA.
        Identifica o botão de download, resolve redirecionamentos e retorna bytes do PDF.
        """
        print(f"[Crawler] Acessando página do documento:\n  {link_pagina}")

        try:
            resp = requests.get(link_pagina, headers=self.headers, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Crawler] ERRO ao acessar página: {e}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Procurar botão do tipo /bitstreams/<uuid>/download
        a_tags = soup.find_all("a", href=True)
        download_links: List[str] = []
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

        download_url = urljoin(self.base_url, download_links[0])
        print(f"[Crawler] Botão de download encontrado:\n  {download_url}")

        try:
            r = requests.get(download_url, headers=self.headers, allow_redirects=True, timeout=40)
            r.raise_for_status()
        except Exception as e:
            print(f"[Crawler] ERRO ao baixar PDF real: {e}")
            return None

        # Validar PDF básico
        if not r.content.startswith(b"%PDF"):
            print("[Crawler] Conteúdo baixado não parece ser PDF real. Pode ser página 'Baixando...'")
            # tentar fallbacks se necessário
            pass

        return r.content

