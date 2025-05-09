# RAG Multimodal de publicações do IPEA
## Sistema de Recuperação Aumentada por Geração (RAG) Multimodal
Este projeto implementa um sistema RAG multimodal para processamento e consulta dos arquivos e documentos presentes no repositório do conhecimento do IPEA, permitindo a recuperação de informações a partir de textos, imagens e tabelas.

## Arquitetura
O sistema é composto por quatro componentes principais:
1. Extração de informações
- Uso do [pdfplucker](https://github.com/ipeadata-lab/pdfplucker) para extração das informações de PDFS
- Resultados como JSONs personalizados para uso no RAG multimodal
- Módulo em construção

2. Ingestão de dados
- **Textos** : Processamento dos textos com chunking inteligente
- **Imagens** : Geração de legendas e extração de características visuais
- **Tabelas** : Extração e indexação de dados

3. Indexação vetorial
- Armazenamento de embeddings em coleções de dados vetoriais
- Suporte à consultas semânticas separadas

4. Recuperação e Resposta
- Busca híbrida de contexto relevante
- Geração de respostas contextualizadas
- Citação de fontes de informação

## Tecnologias utilizadas
| Componente | Tecnologia |
|------------|------------|
| Banco Vetorial | ChromaDB |
| Chunking | Langchain RecursiveCharacterTextSplitter |
| Reconhecimento de entidades | lfcc/bert-portuguese-ner |
| Embeddings de texto | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| Modelo de visão-linguagem | microsoft/git-base |
| Embeddings de imagem | openai/clip-vit-base-patch16 |
| Modelo de linguagem | Ollama APi (llama3.2) |

## Instalação
```bash
git clone https://github.com/ipeadata-lab/rag_publicacoes
cd rag_publicacoes
poetry install
```

## Uso
### Via linha de comando
```bash
# Ingestão de dados
ragpub ingestao --diretorio caminho/para/dados

# Consulta
ragpub consulta --query "Sua pergunta" --limite 5 --content-types text image table
```

### Como biblioteca

```python
from src.main import consultar_rag, ingerir_dados

# Para ingestão de dados no banco, caminho padrão DATA_DIR = ./data
diretorio = "caminho/para/dados"
ingerir_dados(
    dir=diretorio | DATA_DIR # Caminho para a pasta contendo os dados processados
)

# Realizando a consulta
pergunta = "Quais são as prioridades do projeto Rio Grande durante 1971 e 1974?"
resultado = consultar_rag(
    query=pergunta,          # Pergunta a ser respondida
    limite=5,                # Número máximo de respostas relevantes
    content_types=["text", "table", "image"]  # Tipos de conteúdo a serem considerados
)

# Exibindo o resultado
print(resultado)
```

### Estrutura Esperada dos Dados
Vale notar que a pasta de dados deve ser a mesma que o pdfplucker cria sem `--folder-separation` ativado. Certifique-se de que a estrutura da pasta de dados esteja conforme o esperado:

```
data/
  ├── arquivo1.json        # resultado de conversão
  ├── arquivo2.json        # resultado de conversão
  ├── images/              # pasta com as imagens das conversões
  │   ├── arquivo1_1.png   # as imagens tem o nome dos arquivos
```

Com isso, você pode integrar o sistema RAG multimodal em seus fluxos de trabalho e realizar consultas avançadas em textos, imagens e tabelas.

## Estrutura do projeto
```
src/
  ├── config.py                # Configurações globais
  ├── main.py                  # CLI e funções principais
  ├── embeddings/              # Modelos de embedding
  │   ├── modelo_texto.py      # Embeddings de texto
  │   └── modelo_imagem.py     # Embeddings de imagem
  ├── ingestor/                # Processamento de dados
  │   ├── ingerir_texto.py     # Processamento de texto
  │   ├── ingerir_imagem.py    # Processamento de imagem
  │   └── ingerir_tabela.py    # Processamento de tabelas
  ├── ner/                     # Reconhecimento de entidades
  │   └── ner.py               # Extração de entidades nomeadas
  ├── retrieval/               # Recuperação de informações
  │   └── rag_retriever.py     # Busca e geração de respostas
  └── vector_db/               # Banco de dados vetorial
      └── chroma_client.py     # Cliente para ChromaDB
      └── vector_database/   # Local de armazenamento padrão
```
## Contribuição
Contribuições são bem-vindas! Por favor, consulte o [rastreador de issues](https://github.com/ipeadata-lab/rag_publicacoes/issues) para ver problemas conhecidos ou sugerir melhorias.

Para contribuições em código, por favor:
1. Cheque pull requests e issues existentes
2. Crie um fork do repositório
3. Crie uma branch para sua contribuição
4. Faça suas alterações
5. Crie um pull request
