import os
import json
import torch
import chromadb
import requests
import logging
import traceback
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuração do logging padrão
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MultimodalRAG")

# constantes
DATA_DIR = "./data"
VECTOR_STORE = f"{DATA_DIR}/vector_store"
JSON_DIR = f"{DATA_DIR}/json"
IMAGE_DIR = f"{DATA_DIR}/images"

# Verifica se o diretório de dados existe
os.makedirs(VECTOR_STORE, exist_ok=True)

# API e modelos
OLLAMA_API = "https://ollama-api.ipea.gov.br"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2"

# Parâmetros de chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Parâmetros de recuperação
MAX_HISTORY_MESSAGES = 5
MAX_CONTEXT_DOCS = 5
SIMILARITY_TOP_K = 4

class MultimodalRAG:
    """Sistema RAG multimodal para recuperação de informações e geração de respostas."""

    def __init__(self) -> None:
        """Inicializa o sistema com dados constantes"""
        self.data_dir = DATA_DIR
        self.persist_dir = VECTOR_STORE
        self.embedding_model = EMBEDDING_MODEL
        self.llm_model = LLM_MODEL

        try:
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            logger.info("Modelo CLIP carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo CLIP: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Falha ao carregar os modelos de imagem: {e}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )

        try:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

            # Criar ou obter coleções
            self.text_collection = self.chroma_client.get_or_create_collection("text_collection")
            self.image_collection = self.chroma_client.get_or_create_collection("image_collection")
            self.table_collection = self.chroma_client.get_or_create_collection("table_collection")
            logger.info("Coleções ChromaDB inicializadas")
        except Exception as e:
            logger.error(f"Erro ao conectar ao ChromaDB: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Falha ao inicializar banco de vetores: {str(e)}")
        
        self.documents = {}
        self.images = {}
        self.tables = {}

        logger.info("Sistema RAG multimodal inicializado com sucesso.")

    def _embed_text(self, texts: list[str]) -> list[list[float]]:
        """
        Gera embeddings para uma lista de textos
        
        Args:
            texts (list[str]): Lista de textos a serem embutidos

        Returns:
            list[list[float]]: Lista de embeddings gerados
        """

        if not texts:
            logger.warning("Nenhum texto fornecido para gerar embeddings.")
            return []
        
        embeddings = []

        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    f"{OLLAMA_API}/api/embeddings",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"model": self.embedding_model, "prompt": text}, ensure_ascii=False),
                    timeout=30  # Adicionar timeout
                )

                if response.status_code != 200:
                    logger.error(f"Erro na API de embeddings: {response.status_code}, {response.text}")
                    # Usar embedding vazio como fallback
                    embeddings.append([0.0] * 768)  # dimensão padrão do modelo
                    continue
                    
                data = response.json()
                embeddings.append(data['embedding'])
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erro de requisição ao gerar embeddings para texto {i}: {e}")
                # Usar embedding vazio como fallback
                embeddings.append([0.0] * 768)  # dimensão padrão do modelo
            except Exception as e:
                logger.error(f"Erro inesperado ao gerar embeddings para texto {i}: {e}")
                logger.error(traceback.format_exc())
                # Usar embedding vazio como fallback
                embeddings.append([0.0] * 768)  # dimensão padrão do modelo

        logger.info(f"Gerados {len(embeddings)} embeddings de texto.")
        return embeddings

    def _embed_table(self, tables: list[dict]) -> list[list[float]]:
        """
        Gera embeddings para uma lista de tabelas
        
        Args:
            tables (list[dict]): Lista de tabelas a serem embutidas

        Returns:
            list[list[float]]: Lista de embeddings gerados
        """

        if not tables:
            logger.warning("Lista de tabelas vazia enviada para embedding")
            return []
            
        table_texts = []
        
        for table in tables:
            try:
                # Converter tabela para representação de texto
                if isinstance(table, dict):
                    table_str = json.dumps(table, ensure_ascii=False)
                else:
                    table_str = str(table)
                
                # Limitar o tamanho para evitar tokens muito grandes
                if len(table_str) > 8000:
                    table_str = table_str[:8000] + "..."
                    
                table_texts.append(f"Table: {table_str}")
            except Exception as e:
                logger.error(f"Erro ao converter tabela para texto: {e}")
                logger.error(traceback.format_exc())
                # Adicionar uma string vazia como fallback
                table_texts.append("Table: {}")
        
        # Usar o embedder de texto para gerar embeddings
        embeddings = self._embed_text(table_texts)
        
        logger.info(f"Embeddings de tabela gerados ({len(tables)} tabelas)")
        return embeddings

    def _embed_image(self, images: list[str]) -> list[list[float]]:
        """
        Gera embeddings para uma lista de imagens
        
        Args:
            images (list[str]): Lista de caminhos para as imagens

        Returns:
            list[list[float]]: Lista de embeddings gerados
        """

        if not images:
            logger.warning("Lista de imagens vazia enviada para embedding")
            return []
        
        embeddings = []

        for i, image_path in enumerate(images):
            try:
                if not os.path.exists(image_path):
                    logger.warning(f"Imagem não encontrada: {image_path}")
                    # Adicionar uma lista de zeros como fallback
                    embeddings.append([0.0] * 512)  # dimensão padrão do modelo
                    continue

                image = Image.open(image_path).convert("RGB")
                inputs = self.image_processor(images=image, return_tensors="pt", padding=True)

                with torch.no_grad():
                    outputs = self.image_model.get_image_features(**inputs)
                    image_embeddings = outputs / outputs.norm(dim=-1, keepdim=True)  # Normaliza
                    embeddings.append(image_embeddings[0].cpu().numpy().tolist())
                
            except Exception as e:
                logger.error(f"Erro ao processar imagem {image_path}: {e}")
                logger.error(traceback.format_exc())
                # Adicionar embedding zero como placeholder
                embeddings.append([0.0] * 512)  # CLIP tem 512 dimensões
        
        logger.info(f"Gerados {len(embeddings)} embeddings de imagem.")
        return embeddings

    def _chunk_text(self, text: str) -> list[str]:
        """
        Divide um texto em pedaços menores
        
        Args:
            text (str): Texto a ser dividido

        Returns:
            list[str]: Lista de pedaços de texto
        """
        try:
            if not text:
                logger.warning("Texto vazio enviado para chunking.")
                return []
            
            if len(text) < CHUNK_SIZE:
                logger.info("Texto pequeno demais para ser dividido.")
                return [text]
            
            chunks = self.text_splitter.split_text(text)
            return chunks
        
        except Exception as e:
            logger.error(f"Erro ao dividir texto: {e}")
            logger.error(traceback.format_exc())
            return []

    def _ingest_text(self, sections: str, json_path: str, metadata: dict) -> None:
        """
        Ingere texto, processando e armazenando em ChromaDB
        
        Args:
            sections (str): Texto a ser ingerido
            filename (str): Nome do arquivo de origem
            metadata (dict): Metadados associados ao texto
        """
        try:
            chunks = self._chunk_text(sections)
            logger.info(f"Texto dividido em {len(chunks)} chunks.")

            if not chunks:
                logger.warning("Nenhum chunk de texto para processar.")
                return
            
            # Gerar embeddings para os chunks
            embeddings = self._embed_text(chunks)
            if len(embeddings) != len(chunks):
                logger.error("Número de embeddings gerados não corresponde ao número de chunks.")
                return
            
            logger.info(f"Armazenando {len(chunks)} chunks de texto na coleção.")

            # Criar metadados para cada chunk
            ids = [f"{json_path}_text_{i}" for i in range(len(chunks))]
            metadatas = []
            
            for i in range(len(chunks)):
                metadata_chunk = {k: v for k, v in metadata.items() if v is not None}
                metadata_chunk.update({
                    "source_file": json_path,
                    "chunk_id": i,
                    "modality": "text",
                    "chunk_size": len(chunks[i]),
                    })
                metadatas.append(metadata_chunk)

            self.text_collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(f"Ingestão de texto concluída: {json_path} com {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Erro ao ingerir texto de {json_path}: {e}")
            logger.error(traceback.format_exc())

    def _ingest_images(self, images: list[dict], image_dir: str, metadata: dict) -> None:
        """
        Ingere imagens, processando e armazenando em ChromaDB
        
        Args:
            images (list[dict]): Lista de imagens a serem ingeridas
            image_dir (str): Diretório onde as imagens estão armazenadas
            metadata (dict): Metadados associados às imagens
        """
        
        try:
            image_paths = []
            image_docs = []
            image_metadatas = []
            image_ids = []

            for i, img_data in enumerate(images):
                try:
                    ref = img_data.get("ref")
                    if not ref:
                        logger.warning(f"Imagem sem referência encontrada: {img_data}")
                        continue

                    img_path = os.path.join(image_dir, ref)
                    img_path = img_path.replace("\\", "/")  # Normalizar caminho

                    if not os.path.exists(img_path):
                        logger.warning(f"Imagem não encontrada: {img_path}")
                        continue

                    image_paths.append(img_path)

                    # Criar representação textual da imagem
                    classification = img_data.get("classification", {})
                    class_name = classification.get('class_name', 'unknown')
                    confidence = classification.get('confidence', 0)
                    subtitle = img_data.get('subtitle', '')

                    img_text = f"Imagem de tipo: {class_name}, legenda: {subtitle}"
                    image_docs.append(img_text)

                    img_metadata = {k: v for k, v in metadata.items() if v is not None}
                    img_metadata.update({
                        "source_file": img_path,
                        "image_id": i,
                        "image_ref": ref,
                        "class_name": class_name,
                        "confidence": confidence,
                        "subtitle": subtitle,
                        "modality": "image",
                    })
                    image_metadatas.append(img_metadata)

                    img_id = f"{os.path.basename(img_path)}_image_{i}"
                    image_ids.append(img_id)

                    self.images[img_id] = img_path

                except Exception as e:
                    logger.error(f"Erro ao processar imagem {i}: {e}")
                    logger.error(traceback.format_exc())

            if not image_paths:
                logger.warning("Nenhuma imagem válida encontrada para ingestão.")
                return
            
            logger.info(f"Processando embeddings de {len(image_paths)} imagens.")

            embeddings = self._embed_image(image_paths)
            if len(embeddings) != len(image_paths):
                logger.error(f"Número de embeddings ({len(embeddings)}) não corresponde ao número de imagens ({len(image_paths)})")
                return

            # Adicionar ao Chroma
            self.image_collection.add(
                embeddings=embeddings,
                documents=image_docs,
                ids=image_ids,
                metadatas=image_metadatas
            )

            logger.info(f"Ingestão de imagens concluída ({len(image_paths)} imagens)")

        except Exception as e:
            logger.error(f"Erro ao ingerir imagens de {img_path}: {e}")
            logger.error(traceback.format_exc())

    def _ingest_tables(self, tables: list[dict], json_path: str, metadata: dict) -> None:
        """
        Ingere tabelas no ChromaDB.
        
        Args:
            tables: Lista de dados de tabelas
            json_path: Caminho do arquivo fonte
            metadata: Metadados associados
        """
        try:
            table_docs = []
            table_metadatas = []
            table_ids = []

            for i, table_data in enumerate(tables):
                try:
                    subtitle = table_data.get("subtitle", "")
                    content = table_data.get("table", {})
                    table_ref = table_data.get("self_ref", "")

                    if not content:
                        logger.warning(f"Tabela vazia encontrada: {table_data}")
                        content = {}
                        continue

                    table_str = json.dumps(content, ensure_ascii=False)
                    table_docs.append(f"Tabela: {subtitle if subtitle else "Sem título"} \n\n Conteúdo: {table_str}")

                    # criar metadados
                    table_metadata = {k: v for k, v in metadata.items() if v is not None}
                    table_metadata.update({
                        "source_file": json_path,
                        "table_id": i,
                        "table_ref": table_ref,
                        "subtitle": subtitle,
                        "modality": "table",
                    })
                    table_metadatas.append(table_metadata)

                    # gerar ID
                    table_id = f"{os.path.basename(json_path)}_table_{i}"
                    table_ids.append(table_id)

                    self.tables[table_id] = content

                except Exception as e:
                    logger.error(f"Erro ao processar tabela {i}: {e}")
                    logger.error(traceback.format_exc())

            if not table_docs:
                logger.warning(f"Nenhuma tabela válida encontrada em {json_path}.")
                return
            
            logger.info(f"Processando embeddings de {len(table_docs)} tabelas.")

            embeddings = self._embed_table(tables)
            if len(embeddings) != len(table_docs):
                logger.error(f"Número de embeddings ({len(embeddings)}) não corresponde ao número de tabelas ({len(table_docs)})")
                return
            
            # Adicionar ao Chroma
            self.table_collection.add(
                embeddings=embeddings,
                documents=table_docs,
                ids=table_ids,
                metadatas=table_metadatas
            )
            logger.info(f"Ingestão de tabelas concluída ({len(table_docs)} tabelas)")

        except Exception as e:
            logger.error(f"Erro ao ingerir tabelas de {json_path}: {e}")
            logger.error(traceback.format_exc())

    def ingest_json(self, json_path: str) -> bool:
        """
        Ingere um JSON customizado completo, processando textos, imagens e tabelas

        Args:
            json_path (str): Caminho para o arquivo JSON a ser ingerido

        Returns:
            bool: True se a ingestão for bem-sucedida, False caso contrário
        """
        logger.info(f"Iniciando ingestão do JSON: {json_path}")

        # replace \ with /
        json_path = json_path.replace("\\", "/")

        try:
            if not os.path.exists(json_path):
                logger.error(f"Arquivo JSON não encontrado: {json_path}")
                return False
            
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Erro ao decodificar JSON: {e}")
                    return False
            
            metadata = data.get("metadata", {})
            filename = os.path.basename(json_path)

            if 'sections' in data:
                logger.info(f"Processando seções do JSON.")
                self._ingest_text(data['sections'], filename, metadata)
            else:
                logger.info(f"Arquivo {filename} não contém seções de texto")

            # Processar imagens
            if 'images' in data:
                logger.info(f"Processando {len(data['images'])} imagens do arquivo {filename}")
                self._ingest_images(data['images'], IMAGE_DIR, metadata)
            else:
                logger.info(f"Arquivo {filename} não contém imagens")
            
            # Processar tabelas
            if 'tables' in data:
                logger.info(f"Processando {len(data['tables'])} tabelas do arquivo {filename}")
                self._ingest_tables(data['tables'], json_path, metadata)
            else:
                logger.info(f"Arquivo {filename} não contém tabelas")

            self.documents[json_path] = data

            logger.info(f"Ingestão do JSON {json_path} concluída com sucesso.")
            return True

        except Exception as e:
            logger.error(f"Erro ao ingerir JSON {json_path}: {e}")
            logger.error(traceback.format_exc())
            return False

    def _format_results(self, results: dict) -> list[dict]:
        """
        Formata os resultados da consulta.
        
        Args:
            results: Resultados brutos do ChromaDB
            
        Returns:
            Lista formatada de resultados
        """
        formatted = []

        try:
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            for i in range(len(documents)):
                result = {
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i],
                }
                formatted.append(result)
                
        except Exception as e:
            logger.error(f"Erro ao formatar resultados: {e}")
            logger.error(traceback.format_exc())

        return formatted

    def _clean_query(self, results: dict) -> list:
        """
        Realiza a limpeza da busca do rag
        """

        clean = []

        all_items = []
        for modality_list in results.values():
            all_items.extend(modality_list)

        for item in all_items:
            if not isinstance(item, dict):
                continue

            metadata = item.get("metadata", {})
            modality = metadata.get('modality', 'unknown')
            doc = item.get('document', "")

            if modality == 'text':
                clean.append({
                    "modality": modality,
                    "source": metadata.get('source_file', 'unknown'),
                    "content": doc,
                    "distance": item.get('distance', float('inf')),
                })

            elif modality == 'image':
                clean.append({
                    "modality": modality,
                    "source": metadata.get('source_file', 'unknown'),
                    "content": doc,
                    "class" : metadata.get('class_name', 'unknown'),
                    "distance": item.get('distance', float('inf')),
                })

            elif modality == 'table':
                clean.append({
                    "modality": modality,
                    "source": metadata.get('source_file', 'unknown'),
                    "content": doc,
                    "distance": item.get('distance', float('inf')),
                })

        return clean

    def query(self, query: str, modalities: list[str] = ['text', 'image', 'tables'], top_k: int = 3) -> dict:
        """
        Realiza uma consulta multimodal no sistema RAG.

        Args:
            query (str): Consulta a ser realizada
            modalities (list[str]): Modalidades a serem consultadas (texto, imagem, tabela)
            top_k (int): Número de resultados a serem retornados

        Returns:
            dict: Resultados da consulta
        """
        # Implementar lógica de consulta
        results = {}

        if 'text' in modalities:
            try:
                logger.debug(f"Gerando embedding para consulta de texto: '{query}'")
                text_embedding = self._embed_text([query])[0]
                
                text_results = self.text_collection.query(
                    query_embeddings=[text_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                results['text'] = self._format_results(text_results)
                logger.info(f"Busca de texto concluída: {len(results['text'])} resultados")
            except Exception as e:
                logger.error(f"Erro na busca de texto: {e}")
                logger.error(traceback.format_exc())
                results['text'] = []

        # Buscar em imagens
        if 'image' in modalities:
            try:
                logger.debug(f"Gerando embedding para consulta de imagem: 'Image of {query}'")
                # Usamos uma descrição textual para buscar imagens relevantes
                image_query = f"Imagem de {query}"
                inputs = self.image_processor(text=[image_query], return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = self.image_model.get_text_features(**inputs)
                    image_embedding = outputs[0] / outputs[0].norm()  # Normaliza
                    image_embedding = image_embedding.cpu().numpy().tolist()
                
                image_results = self.image_collection.query(
                    query_embeddings=[image_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                results['image'] = self._format_results(image_results)
                logger.info(f"Busca de imagem concluída: {len(results['image'])} resultados")
            except Exception as e:
                logger.error(f"Erro na busca de imagem: {e}")
                logger.error(traceback.format_exc())
                results['image'] = []

        # Buscar em tabelas
        if 'tables' in modalities:
            try:
                logger.debug(f"Gerando embedding para consulta de tabela: 'Table containing {query}'")
                table_query = f"Table containing {query}"
                table_embedding = self._embed_text([table_query])[0]
                
                table_results = self.table_collection.query(
                    query_embeddings=[table_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                results['table'] = self._format_results(table_results)
                logger.info(f"Busca de tabela concluída: {len(results['table'])} resultados")
            except Exception as e:
                logger.error(f"Erro na busca de tabela: {e}")
                logger.error(traceback.format_exc())
                results['table'] = []

        logger.info(f"Busca concluída")
        return results

    def answer(self, query: str, top_k: int = 3) -> str:
        """
        Answer a question using the rag system

        Args:
            query (str): Question to be answered
            top_k (int): Number of top results to consider
        
        Returns:
            str: Answer to the question with evidence
        """

        search_results = self.query(query, top_k=top_k)
        all_results = self._clean_query(search_results)
        all_results.sort(key=lambda x: x['distance'])

        # Ajuda de um LLM
        prompt = f"""Responda a pergunta a seguir com base nas evidências fornecidas.

        Pergunta: {query}

        Evidências:
        {json.dumps(all_results, indent=4, ensure_ascii=False)}

        Responda de forma clara, utilizando as informações das evidências. Se não houver informações suficientes, diga que não sabe.
        """

        response = requests.post(
            f"{OLLAMA_API}/api/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model" : self.llm_model,
                "messages" : [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream" : False
            }),
            timeout=300
        )
        if response.status_code != 200:
            logger.error(f"Erro na API de geração: {response.status_code}, {response.text}")
            return "Desculpe, não consegui gerar uma resposta."
        
        data = response.json()

        llm_response = data['message']['content']

        answer = f"Aqui está o que eu encontrei para responder: '{query}' \n\n"

        # elimina content e distance de all_results
        all_results = [
            {
                "modalidade": r['modality'],
                "fonte": r['source'],
                **({"classe": r['class']} if r['modality'] == 'image' and 'class' in r else {})
            }
            for r in all_results
        ]

        # juntado tudo em uma resposta só

        answer += f"Resposta: {llm_response} \n\n Fontes: {json.dumps(all_results, indent=4, ensure_ascii=False)}"

        return answer

if __name__ == "__main__":
    rag = MultimodalRAG()

    for file in os.listdir(JSON_DIR):
        if file.endswith('.json'):
            json_path = os.path.join(JSON_DIR, file)
            rag.ingest_json(json_path)

    #print(rag.answer("Mostre dados a respeito das importações dos estados unidos e participação do brasil"))

