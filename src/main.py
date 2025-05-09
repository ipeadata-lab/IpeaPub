import argparse
from typing import List, Optional, Dict, Any

from src.ingestor.ingerir_texto import IngestorTexto
from src.ingestor.ingerir_imagem import IngestorImagem
from src.ingestor.ingerir_tabela import IngestorTabela
from src.vector_db.chroma_client import ChromaDB
from src.retrieval.rag_retriever import RAGRetriever
from src.config import DATA_DIR, VECTOR_DB

def ingerir_dados(dir: str = DATA_DIR) -> None:
    """
    Ingerir dados de texto, imagem e tabela a partir de diretórios especificados.
    
    Args:
        dir: Diretório onde os arquivos de texto e tabela estão localizados.
        imagens_dir: Diretório onde as imagens estão localizadas.
    """
    ingestor_texto = IngestorTexto()
    ingestor_imagem = IngestorImagem()
    ingestor_tabela = IngestorTabela()

    # Ingestão de texto
    print("\nProcessando textos dos arquivos...")
    textos_ingeridos = ingestor_texto.processar_diretorio(dir)

    # Ingestão de imagem
    print("\nProcessando imagens dos arquivos...")
    imagens_ingeridas = ingestor_imagem.processar_diretorio(dir)

    # Ingestão de tabela
    print("\nProcessando tabelas dos arquivos...")
    tabelas_ingeridas = ingestor_tabela.processar_diretorio(dir)

    # Inicializar banco de dados vetorial
    if VECTOR_DB == "chroma":
        banco = ChromaDB()
    else:
        raise ValueError(f"Banco de dados vetorial {VECTOR_DB} ainda não implementado.")
    

    # Inserir documentos de texto
    print("Inserindo documentos de texto no banco vetorial...")
    for arquivo in textos_ingeridos:
        for documento in arquivo:
            banco.inserir_documento(documento)

    # Inserir documentos de tabela
    print("Inserindo documentos de tabela no banco vetorial...")
    for arquivo in tabelas_ingeridas:
        for documento in arquivo:
            banco.inserir_documento(documento)
    
    # Inserir documentos de imagem
    print("Inserindo documentos de imagem no banco vetorial...")
    for arquivo in imagens_ingeridas:
        for documento in arquivo:
            banco.inserir_documento(documento)
    
    print("\nIngestão de dados concluída!")

def consultar_rag(query: str, limite: int = 5, content_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Realiza uma consulta usando o modelo RAG.
    
    Args:
        query: Consulta a ser realizada.
        limite: Número máximo de resultados a serem retornados.
        content_types: Lista de tipos de conteúdo a serem filtrados (opcional).
    """

    rag_retriever = RAGRetriever()
    resultados = rag_retriever.responder(query, limite, content_types)
    
    print("\nResultados da consulta:")
    print(resultados["resposta"])

    print("\nResultados relevantes:")
    for i, doc in enumerate(resultados["contexto"]):
        content_type: str = doc.get("metadados", {}).get("content_type", "text")
        fonte = doc.get("metadados", {}).get("filename", "desconhecido")
        print(f"{i+1}. [{content_type.upper()}] Fonte: {fonte}, Score: {doc['distancia']:.4f}")

    return resultados

def main():
    parser = argparse.ArgumentParser(description="Sistema de RAG multimodal.")
    subparsers = parser.add_subparsers(dest="comando", help="Comandos disponíveis")

    # comandos de ingestão
    parser_ingestao = subparsers.add_parser("ingestao", help="Realiza a ingestão de dados.")
    parser_ingestao.add_argument("--diretorio", type=str, default=DATA_DIR, help="Diretório de ingestão.")

    # comandos de consulta
    parser_consulta = subparsers.add_parser("consulta", help="Realiza uma consulta.")
    parser_consulta.add_argument("--query", type=str, required=True, help="Consulta a ser realizada.")
    parser_consulta.add_argument("--limite", type=int, default=5, help="Número máximo de resultados a serem retornados.")
    parser_consulta.add_argument("--content-types", type=str, nargs="+", choices=["text", "image", "table"], 
                            help="Tipos de conteúdo a considerar")

    args = parser.parse_args()

    if args.comando == "ingestao":
        ingerir_dados(args.diretorio)
    elif args.comando == "consulta":
        consultar_rag(args.query, args.limite, args.content_types)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
