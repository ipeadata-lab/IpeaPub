"""
Todas as funções utilitárias para o ingestor

- modelos de visão para legendar imagens e tabelas
- cleaner de texto extraído
"""

import re
from datetime import datetime, timezone

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
