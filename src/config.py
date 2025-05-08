# Configuração principal do projeto

from pathlib import Path

# Diretórios do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.absolute()
DATA_DIR = BASE_DIR / 'data' # onde os Json estão
IMAGES_DIR = DATA_DIR / 'images'

# Configuração do banco de dados vetorial
VECTOR_DB = "chroma"
VECTOR_DB_DIR = BASE_DIR / 'src' / 'vector_db' / 'vector_database' # onde o banco de dados vetorial está armazenado

# Configuração do modelo de linguagem
OLLAMA_API_URL = "https://ollama-api.ipea.gov.br/"
OLLAMA_MODEL = "llama3.2" # modelo de linguagem a ser utilizado

MODELO_EMBEDDING_TEXTO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # modelo de embedding para texto
MODELO_EMBEDDING_IMAGEM = "openai/clip-vit-base-patch16" # modelo de embedding para imagem
MODELO_NER = "lfcc/bert-portuguese-ner" # modelo de NER (Reconhecimento de Entidades Nomeadas)
MODELO_VLM = "microsoft/git-base"

# Parâmetros de chunking
TAMANHO_CHUNK = 1000
SOBREPOR_CHUNK = 200

# Parâmetros de pré-processamento
DIMENSÃO_EMBEDDING_TEXTO = 768 # dimensão do embedding de texto
DIMENSÃO_EMBEDDING_IMAGEM = 512 # dimensão do embedding de imagem
