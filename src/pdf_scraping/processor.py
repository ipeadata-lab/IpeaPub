import os
import json
import PyPDF2
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.exceptions import ConversionError
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    EasyOcrOptions,
)
from docling_core.types.doc import (
    PictureItem,
    TableItem,
    TextItem,
    DocItemLabel,
)
import traceback
from typing import TypedDict, List, Dict, Any
import logging

class Data(TypedDict):
    metadata: Dict[str, Any]
    pages: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    captions: List[Dict[str, Any]]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.FileHandler('logs/processor.log', encoding='utf-8')
logger = logging.getLogger(__name__)


class PdfProcessor:
    def __init__(self, output_path: str, ocr_lang: list[str] = ["en", "pt"], force_ocr: bool = True):

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.image_path = self.output_path / "images"
        self.image_path.mkdir(parents=True, exist_ok=True)

        self.converter: DocumentConverter = self.create_converter(ocr_lang, force_ocr)

    def create_converter(ocr_lang: list[str] = ["en", "pt"], force_ocr: bool = True) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True 
        pipeline_options.ocr_options.lang = ocr_lang
        pipeline_options.generate_picture_images = True
        pipeline_options.do_picture_classification = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.images_scale = 1

        if force_ocr:
            ocr_options = EasyOcrOptions(force_full_page_ocr=True, lang=ocr_lang)
            pipeline_options.ocr_options = ocr_options

        device_type = AcceleratorDevice.CUDA
        pipeline_options.accelerator_options = AcceleratorOptions(
            device_type=device_type,
            num_threads=1
        )

        converter = DocumentConverter(
            format_options= {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        return converter

    def json_serializable(obj):
        """Função auxiliar para tornar objetos personalizados serializáveis em JSON."""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    def format_results(conv: ConversionResult, data: Data, filename: str, image_path: str) -> bool:
        ''' Uses the docling document to format a readable JSON result '''
        
        try:
            # Pré-inicializa o dicionário de páginas para evitar verificações repetidas
            pages_dict = {}
            for page_dict in data['pages']:
                pages_dict[page_dict['page_number']] = page_dict
                if 'content' not in page_dict:
                    page_dict['content'] = ""
            
            # Processa captions antecipadamente para uso posterior
            caption_dict = {}
            
            counter = 0
            for idx, (item, _) in enumerate(conv.document.iterate_items()):
                if isinstance(item, TextItem):
                    page = item.prov[0].page_no
                    label = item.label
                    text = item.text
                    
                    # Criar ou obter a página uma única vez
                    if page not in pages_dict:
                        new_page = {'page_number': page, 'content': ""}
                        data['pages'].append(new_page)
                        pages_dict[page] = new_page
                    
                    # Determina o conteúdo baseado no label
                    match label:
                        case DocItemLabel.SECTION_HEADER:
                            content = f"\n# {text}\n"
                        case DocItemLabel.FORMULA:
                            content = f" Equation: {text}\n"
                        case DocItemLabel.REFERENCE:
                            content = f"\nReference: {text}\n"
                        case DocItemLabel.LIST_ITEM:
                            content = f"\n- {text}\n"
                        case DocItemLabel.CAPTION:
                            content = f" _{text}_\n"
                            # Armazena caption para uso posterior
                            data['captions'].append({
                                'self_ref': item.self_ref,
                                'cref': item.parent.cref,
                                'text': text
                            })
                            # Também pré-processa para uso posterior
                            caption_dict[item.parent.cref] = text
                        case DocItemLabel.FOOTNOTE:
                            content = f"\nFootnote: {text}\n"
                        case DocItemLabel.TITLE:
                            content = f"\n## {text}\n"
                        case DocItemLabel.TEXT:
                            content = f" {text}"
                        case DocItemLabel.PARAGRAPH:
                            content = f"\n{text}\n"
                        case DocItemLabel.PAGE_FOOTER:
                            content = f"\n{text}\n"
                        case DocItemLabel.CHECKBOX_SELECTED:
                            content = f"\n- {text}\n"
                        case DocItemLabel.CHECKBOX_UNSELECTED:
                            content = f"\n- {text}\n"
                        case _:
                            content = f" {text}"
                    
                    # Adiciona o conteúdo à página
                    pages_dict[page]['content'] += content
                        
                elif isinstance(item, TableItem):
                    table = item.export_to_markdown(doc=conv.document)
                    self_ref = item.self_ref
                    page = item.prov[0].page_no
                    
                    # Criar ou obter a página uma única vez
                    if page not in pages_dict:
                        new_page = {'page_number': page, 'content': f" <{self_ref}>"}
                        data['pages'].append(new_page)
                        pages_dict[page] = new_page
                    else:
                        pages_dict[page]['content'] += f" <{self_ref}>"
                    
                    data['tables'].append({
                        'self_ref': self_ref,
                        'captions': item.captions,
                        'caption': "",
                        'references': item.references,
                        'footnotes': item.footnotes,
                        'page': page,
                        'table': table
                    })
        
                elif isinstance(item, PictureItem):
                    self_ref = item.self_ref
                    page = item.prov[0].page_no
                    
                    # Extrair classificação, se disponível
                    classification = None
                    confidence = None
                    if item.annotations:
                        for annotation in item.annotations:
                            if annotation.kind == 'classification':
                                best_class = max(
                                    annotation.predicted_classes,
                                    key=lambda cls: cls.confidence
                                )
                                classification = best_class.class_name
                                confidence = best_class.confidence
                                break
                    
                    # Salva a imagem
                    image_filename = (image_path / f"{filename}_{counter}.png")
                    placeholder = f"{filename}_{counter}.png"
                    with image_filename.open('wb') as file:
                        item.get_image(conv.document).save(file, "PNG")
                    
                    # Criar ou obter a página uma única vez
                    if page not in pages_dict:
                        new_page = {'page_number': page, 'content': f" <{placeholder}>"}
                        data['pages'].append(new_page)
                        pages_dict[page] = new_page
                    else:
                        pages_dict[page]['content'] += f" <{placeholder}>"
                    
                    data['images'].append({
                        'ref': placeholder,
                        'self_ref': self_ref,
                        'captions': item.captions,
                        'caption': "",
                        'classification': classification,
                        'confidence': confidence,
                        'references1': item.references,
                        'references': [],
                        'footnotes1': item.footnotes,
                        'footnotes': [],
                        'page': page,
                    })
                    counter += 1

            # Processa as referências de texto e aplica captions
            text_refs = {}
            for text in conv.document.texts:
                if text.label == DocItemLabel.TEXT:
                    text_refs[text.self_ref] = text.text
            
            # Aplica captions, referências e notas de rodapé em uma única iteração
            for image in data.get("images", []):
                # Aplica caption
                self_ref = image.get("self_ref")
                if self_ref in caption_dict:
                    image["caption"] += caption_dict[self_ref]
                
                # Aplica referências
                for ref in image.get("references1", []):
                    ref_key = getattr(ref, 'self_ref', str(ref))
                    if ref_key in text_refs:
                        image['references'].append(text_refs[ref])
                
                # Aplica notas de rodapé
                for footnote in image.get("footnotes1", []):
                    footnote_key = getattr(footnote, 'self_ref', str(footnote))
                    if footnote_key in text_refs:
                        image['footnotes'].append(text_refs[footnote])
                
                # Remove campos temporários
                image.pop('captions')
                image.pop('references1')
                image.pop('footnotes1')

            # Mesmo processo para tabelas
            for table in data.get("tables", []):
                # Aplica caption
                self_ref = table.get("self_ref")
                if self_ref in caption_dict:
                    table["caption"] += caption_dict[self_ref]
                
                # Aplica referências
                for ref in table.get("references", []):
                    ref_key = getattr(ref, 'self_ref', str(ref))
                    if ref_key in text_refs:
                        table['references'].append(text_refs[ref])
                
                # Aplica notas de rodapé
                for footnote in table.get("footnotes", []):
                    footnote_key = getattr(footnote, 'self_ref', str(footnote))
                    if footnote_key in text_refs:
                        table['footnotes'].append(text_refs[footnote])
                
                # Remove campos temporários
                table.pop('captions')
                if 'references1' in table:
                    table.pop('references1')
                if 'footnotes1' in table:
                    table.pop('footnotes1')

            # Remove captions temporárias
            data.pop('captions')
            
            return True
        except Exception as e:
            logger.error(f"Error formatting result: {e}")
            traceback.print_exc()
            return False

    def process_pdf(self, source: Path, image_path: Path) -> bool:
        
        filename = os.path.basename(source)
        base_name = os.path.splitext(filename)[0]

        logger.info(f"Processando arquivo: {base_name}: {source}")

        data: Data = {
            "metadata": {},
            "pages" : [],
            "images": [],
            "tables": [],
            "captions" : []
        }

        reader = PyPDF2.PdfReader(source)
        for key, value in reader.metadata.items():
            data["metadata"][key] = value
        if reader.numPages > 100:
            logger.warning(f"Arquivo grande detectado, {source.name} com {reader.numPages} páginas.")

        try:
            result: ConversionResult = self.converter.convert(str(source))

            success = self.format_results(result, data, base_name, image_path)

            if not success:
                logger.error(f"Erro ao processar o arquivo {source.name}.")
                return False
            
            with open(self.output_path / f"{base_name}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4, default=self.json_serializable)

            logger.info(f"Arquivo {source.name} processado com sucesso.")

            return True

        except MemoryError:
            logger.error(f"Out of memory while converting '{filename}'")
            return False
        except IOError as e:      
            logger.error(f"I/O error while processing '{filename}': {e}")
            return False
        except ConversionError as e:
            logger.error(f"Conversion error for '{filename}': {e}")
            return False
        except Exception as e:    
            import traceback
            logger.error(f"Error processing '{filename}': {str(e)}\n{traceback.format_exc()}")
            return False


