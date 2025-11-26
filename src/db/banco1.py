import sqlite3
from pathlib import Path

# Banco 1: SQLite para os metadados principais dos documentos

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "banco1.db"

def conectar():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # return rows as dict-like objects
    conn.row_factory = sqlite3.Row
    return conn

def criar_tabela():
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id TEXT PRIMARY KEY,
            titulo TEXT,
            autores TEXT,
            ano INTEGER,
            tipo_conteudo TEXT,
            resumo TEXT,
            palavras_chave TEXT,
            link_pdf TEXT,
            status_ingestao TEXT,
            data_ingestao TEXT
        );
    """)

    conn.commit()
    conn.close()

def inserir_documento(document: dict):
    conn = conectar()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO documentos (
            id, titulo, autores, ano, tipo_conteudo,
            resumo, palavras_chave, link_pdf,
            status_ingestao, data_ingestao
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        document.get("id"),
        document.get("titulo"),
        document.get("autores"),
        document.get("ano"),
        document.get("tipo_conteudo"),
        document.get("resumo"),
        document.get("palavras_chave"),
        document.get("link_pdf"),
        document.get("status_ingestao"),
        document.get("data_ingestao"),
    ))

    conn.commit()
    conn.close()

def buscar_documento(id: str):
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documentos WHERE id = ?", (id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

def atualizar_documento(document: dict):
    conn = conectar()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE documentos
        SET titulo = ?, autores = ?, ano = ?, tipo_conteudo = ?,
            resumo = ?, palavras_chave = ?, link_pdf = ?,
            status_ingestao = ?, data_ingestao = ?
        WHERE id = ?
    """, (
        document.get("titulo"),
        document.get("autores"),
        document.get("ano"),
        document.get("tipo_conteudo"),
        document.get("resumo"),
        document.get("palavras_chave"),
        document.get("link_pdf"),
        document.get("status_ingestao"),
        document.get("data_ingestao"),
        document.get("id"),
    ))

    conn.commit()
    conn.close()

def buscar_pendente():
    """Buscar um documento com ingestão pendente. Retorna None se não houver."""
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documentos WHERE status_ingestao = 'pendente' LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)

def atualizar_status(id: str, status: str):
    conn = conectar()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE documentos
        SET status_ingestao = ?
        WHERE id = ?
    """, (status, id))

    conn.commit()
    conn.close()

criar_tabela()