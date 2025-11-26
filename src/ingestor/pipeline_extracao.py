"""
Aqui vai vir toda a lógica de extração de texto, imagens e tals

"""
from src.db.banco1 import atualizar_status
from src.ingestor.docling_setup import converter_docling

# ============================================================
#   5. PIPELINE COMPLETO
# ============================================================

def processar_documento() -> bool:
    """
    converte com Docling,
    processa os resultados
    atualiza status.
    """

    doc_tuple = converter_docling()
    if not doc_tuple:
        print("[Pipeline] Nenhum documento pendente para conversão.")
        return False
    
    doc, doc_id = doc_tuple
    

    print(f"\n=== PROCESSANDO DOCUMENTO {doc_id} ===")


    # Aqui virá a lógica de processamento do documento
    print("placeholder para a lógica de extração e processamento do documento...")

    # No final, se tudo der certo, atualizar o status do documento para processado
    atualizar_status(doc_id, "processado")
    print(f"Documento processado para os bancos de dados!\n ID: {doc_id}\n")

    return True

