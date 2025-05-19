import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
import time
import random
import logging
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter

def scrape_pdf(handle_id: str, output: str) -> str:
    """
    Faz o download de um PDF e extrai os metadados de documentos presentes no repositório do conhecimento do IPEA.
    Adiciona os metadados ao PDF antes de salvá-lo no diretório de saída.

    Args:
        handle_id (str): id do documento.
        output (str): Diretório onde o PDF será salvo.

    Returns:
        tuple: Tupla contendo nome do PDF baixado e os metadados extraídos. 
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.FileHandler('logs/scrape.log', encoding='utf-8')
    logger = logging.getLogger(__name__)

    os.makedirs(output, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    })

    base_url = "https://repositorio.ipea.gov.br"
    prefix = "11058"
    url = f"{base_url}/handle/{prefix}/{handle_id}?mode=full"

    # Pega o conteúdo da página
    try:
        response = requests.get(url)
        response.raise_for_status()
        pagina: BeautifulSoup = BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        pagina = None
        logger.error(f"Erro ao acessar {url}: {e}")

    if pagina is None:
        logger.error(f"Não foi possível acessar a página: {url}")
        return None

    # Encontrar todos os links para arquivos PDF na página
    links = []
    for link in pagina.find_all('a', href=True):
        href: str = link['href']
        if 'bitstream' in href and href.endswith('.pdf'):
            links.append(base_url + href)

    links = list(set(links))

    time.sleep(random.uniform(1, 3))
    
    # Extrair os metadados
    metadata = {}

    def _get_dc_values(soup: BeautifulSoup, field: str) -> list:
        """Retorna os valores de um campo DC específico."""

        values = []
        for tag in soup.find_all('td', class_='metadataFieldLabel', string=field):
            value = tag.find_next_sibling('td', class_='metadataFieldValue')
            if value:
                values.append(value.get_text(strip=True))
        
        if len(values) == 1:
            return values[0]
        else:
            return values

    try:
        metadata['handle_id'] = handle_id
        metadata['titulo'] = (_get_dc_values(pagina, 'dc.title'))
        metadata['autor(res)'] = (_get_dc_values(pagina, 'dc.contributor.author'))
        metadata['resumo'] = (_get_dc_values(pagina, 'dc.description.abstract'))
        metadata['link'] = (_get_dc_values(pagina, 'dc.identifier.uri'))
        metadata['tipo'] = (_get_dc_values(pagina, 'dc.type'))
        metadata['keywords'] = (_get_dc_values(pagina, 'dc.subject.keyword'))
        metadata['idioma'] = (_get_dc_values(pagina, 'dc.language.iso'))
        metadata['data'] = (_get_dc_values(pagina, 'dc.date.issued'))
        metadata['pdf_url'] = links[0] if links else None
        filename = metadata['pdf_url'].split('/')[-1] if metadata['pdf_url'] else None
        if filename:
            filename = urllib.parse.unquote(filename)
        metadata['pdf_filename'] = filename
    except Exception as e:
        logger.error(f"Erro ao extrair metadados: {e}")
        metadata = None

    time.sleep(random.uniform(1, 3))

    # Baixar o PDF e adicionar metadados
    try:
        if not metadata or not metadata['pdf_url']:
            return None
            
        filename = metadata['pdf_filename']
        filepath: str = os.path.join(output, filename)
        pdf_url = metadata['pdf_url']
        logger.info(f"Baixando de {pdf_url}...")

        response = session.get(pdf_url, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
            logger.warning(f"Aviso: O conteúdo pode não ser um PDF. Content-Type: {content_type}")

        # Ler o PDF em memória
        pdf_bytes = BytesIO(response.content)
        
        try:
            # Abrir o PDF com PyPDF2
            reader = PdfReader(pdf_bytes)
            writer = PdfWriter()

            # Copiar todas as páginas do PDF original
            for page in reader.pages:
                writer.add_page(page)
                
            # Função auxiliar para converter valores em strings seguras para PDF
            def sanitize_metadata_value(value: str):
                if value is None:
                    return ''
                elif isinstance(value, list):
                    # Se for uma lista, junta os elementos com ponto e vírgula
                    return '; '.join(str(item) for item in value if item is not None)
                else:
                    # remove caracteres especiais a\r\n
                    value = value.replace('\r', '').replace('\n', '').replace('\t', '')
                    # Converte para string e trata caracteres especiais
                    return str(value)
                    
            # Adicionar metadados ao PDF
            pdf_metadata = {
                '/Title': sanitize_metadata_value(metadata.get('titulo')),
                '/Author': sanitize_metadata_value(metadata.get('autor(res)')),
                '/Subject': sanitize_metadata_value(metadata.get('resumo')),
                '/Keywords': sanitize_metadata_value(metadata.get('keywords')),
                '/Link' : sanitize_metadata_value(metadata.get('link')),
                '/Type': sanitize_metadata_value(metadata.get('tipo')),
                '/Language': sanitize_metadata_value(metadata.get('idioma')),
                '/CreationDate': sanitize_metadata_value(metadata.get('data')),
                '/Url' : sanitize_metadata_value(metadata.get('pdf_url')),
                '/Filename': sanitize_metadata_value(filename),
                '/Producer': 'IPEA Repositório',
                '/CreationDate': sanitize_metadata_value(metadata.get('data')),
                '/URI': sanitize_metadata_value(metadata.get('link'))
            }
    
            # Adicionar metadados personalizados para campos adicionais
            for key, value in metadata.items():
                if key not in [
                    'handle_id', 'titulo', 'autor(res)', 'resumo', 'link', 'tipo',
                    'keywords', 'idioma', 'data', 'pdf_url', 'pdf_filename'
                ]:
                    pdf_metadata[f'/IPEA_{key}'] = sanitize_metadata_value(value)

            writer.add_metadata(pdf_metadata)
            
            # Salvar o PDF com os metadados atualizados
            with open(filepath, 'wb') as output_file:
                writer.write(output_file)
                
            logger.info(f"Arquivo salvo: temp/{filename}")
            
        except Exception as e:
            logger.error(f"Erro ao modificar metadados do PDF: {e}")
            # Em caso de erro na modificação de metadados, salva o PDF original sem modificações
            logger.warning("Salvando arquivo original sem modificação de metadados...")
            with open(filepath, 'wb') as f:
                f.write(response.content)
            logger.info(f"Arquivo original salvo em: temp/{filename}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao baixar {pdf_url}: {e}")
        filepath = None
    except Exception as e:
        logger.error(f"Erro inesperado ao processar o PDF: {e}")
        filepath = None

    return filepath.replace('\\', '/')

