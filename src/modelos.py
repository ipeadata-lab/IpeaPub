from src.ingestor.ingerir_texto import IngestorTexto
from src.ingestor.ingerir_imagem import IngestorImagem
from src.ingestor.ingerir_tabela import IngestorTabela
from src.vector_db.chroma_client import ChromaDB
from src.ner.ner import ModeloNER
from src.embeddings.modelo_texto import ModeloEmbeddingTexto
from src.embeddings.modelo_imagem import ModeloEmbeddingImagem

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            # Carregamento inicial dos modelos
            cls._instance.modelo_texto = ModeloEmbeddingTexto()
            cls._instance.modelo_imagem = ModeloEmbeddingImagem()
            cls._instance.modelo_ner = ModeloNER()
            cls._instance.ingestor_texto = IngestorTexto()
            cls._instance.ingestor_imagem = IngestorImagem()
            cls._instance.ingestor_tabela = IngestorTabela()
            cls._instance.vector_db = ChromaDB()
        return cls._instance

manager = ModelManager()
print("Modelos e ingestors carregados com sucesso.")