"""
Todas as funções utilitárias para o ingestor

- modelos de visão para legendar imagens e tabelas
- cleaner de texto extraído
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List, Optional
from datetime import datetime, timezone

CRAWLER_URL = "https://repositorio.ipea.gov.br"
CRAWLER_HEADER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def _normalize_string(value):
    if value is None:
        return None
    if isinstance(value, list):
        value = "; ".join(str(v) for v in value)
    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None

def _join_field(value, sep="; "):
    if value is None:
        return None
    if isinstance(value, list):
        flat = []
        for v in value:
            if isinstance(v, list):
                flat.extend(str(x) for x in v)
            else:
                flat.append(str(v))
        return _normalize_string(sep.join(flat))
    return _normalize_string(value)

def _parse_year(ano):
    if ano is None:
        return None
    match = re.search(r"(\d{4})", str(ano))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None

def _parse_last_modified(iso_str):
    if not iso_str:
        return None, None

    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.isoformat().replace("+00:00", "Z"), int(dt_utc.timestamp())
    except Exception:
        return None, None

def clean_item(raw):
    cleaned = {}

    cleaned["id"] = _normalize_string(raw.get("id"))
    cleaned["handle"] = _normalize_string(raw.get("handle"))

    cleaned["titulo"] = _normalize_string(raw.get("titulo")) or ""
    cleaned["resumo"] = _join_field(raw.get("resumo"), sep="\n\n") or ""

    cleaned["autores"] = _join_field(raw.get("autores"), sep="; ") or ""
    palavras = raw.get("palavras_chave") or raw.get("palavras-chave")
    cleaned["palavras_chave"] = _join_field(palavras, sep=", ") or ""

    cleaned["tipo"] = _join_field(raw.get("tipo"), sep="; ") or ""

    cleaned["ano"] = _parse_year(raw.get("ano"))

    iso_utc, epoch = _parse_last_modified(_normalize_string(raw.get("last_modified")))
    cleaned["last_modified"] = iso_utc
    cleaned["last_modified_ts"] = epoch

    return cleaned

def baixar_pdf_real(link_pagina: str) -> Optional[bytes]:
    """
    Recebe o link da página do documento no repositório do IPEA.
    Identifica o botão de download, resolve redirecionamentos e retorna bytes do PDF.
    """
    print(f"[Crawler] Acessando página do documento:\n  {link_pagina}")

    try:
        resp = requests.get(link_pagina, headers=CRAWLER_HEADER, timeout=30)
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

    download_url = urljoin(CRAWLER_URL, download_links[0])
    print(f"[Crawler] Botão de download encontrado:\n  {download_url}")

    try:
        r = requests.get(download_url, headers=CRAWLER_HEADER, allow_redirects=True, timeout=40)
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