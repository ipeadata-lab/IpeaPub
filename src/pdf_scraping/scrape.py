import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
from pdfplucker import pdfplucker
import time
import random
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.FileHandler('logs/scrape.log', encoding='utf-8')
logger = logging.getLogger(__name__)


def scrape_pdf(handle_id: str, output: str) -> tuple:
    """
    Faz o download de um PDF e extrai os metadados de documentos presentes no repositório do conhecimento do IPEA.

    Args:
        url (str): id do documento.
        output (str): Diretório onde o PDF será salvo.

    Returns:
        tuple: Tupla contendo nome do PDF baixado e os metadados extraídos. 
    """

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
        print(f"Erro ao acessar {url}: {e}")

    if pagina is None:
        print(f"Não foi possível acessar a página: {url}")
        return (None, None)

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
        print(f"Erro ao extrair metadados: {e}")
        metadata = None

    time.sleep(random.uniform(1, 3))

    # Baixar o PDF
    try:

        filename = metadata['pdf_filename']
        filepath = os.path.join(output, filename)
        pdf_url = metadata['pdf_url']
        print(f"Baixando de {pdf_url}...")

        response = session.get(pdf_url, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
            print(f"Aviso: O conteúdo pode não ser um PDF. Content-Type: {content_type}")

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Arquivo salvo em: {filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar {pdf_url}: {e}")
        filename = None
    except Exception as e:
        print(f"Erro inesperado ao baixar {url}: {e}")
        filename = None

    return (filename, metadata)

if __name__ == "__main__":

    metadados = []
    arquivos = []

    output = "Z:/rag_publicacoes/src/pdf_scraping/temp"
    i = 954
    while True:
        handle_id = str(i)
        filename, metadado = (scrape_pdf(handle_id, output))
        if filename is not None and metadado is not None:
            arquivos.append(filename)
            metadados.append(metadado)
            logger.info(f"Arquivo {i} ({filename}) baixado com sucesso.")
        
        if len(arquivos) != 0 and len(arquivos) % 6 == 0:
            _ = logger.info(f"Passando para extração...")
            pdfplucker(
                source="Z:/rag_publicacoes/src/pdf_scraping/temp",
                output="Z:/rag_publicacoes/data",
                workers=3,
                force_ocr=True,
                images="Z:/rag_publicacoes/data/images",
                device="cuda",
                timeout=12000,
            )
            logger.info(f"Passando para a alteração de metadados...")

            # find the files to change its metadatas
            for i in range(len(arquivos)):
                for file in os.listdir("Z:/rag_publicacoes/data"):
                    if file == arquivos[i]:
                        with open(f"Z:/rag_publicacoes/data/{file}", "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if "metadata" in data and data["metadata"]:
                            data["metadata"] = metadados[i]
                            with open(f"Z:/rag_publicacoes/data/{file}", "w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
                        else:
                            logger.info(f"Arquivo {file} não possui metadados para serem alterados.")

            logger.info(f"Metadados alterados com sucesso.")
            logger.info(f"Deletando arquivos temporários...")
            for file in os.listdir("Z:/rag_publicacoes/src/pdf_scraping/temp"):
                os.remove(f"Z:/rag_publicacoes/src/pdf_scraping/temp/{file}")

        i += 1