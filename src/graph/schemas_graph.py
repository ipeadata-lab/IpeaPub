import operator
from typing_extensions import Annotated
from typing import Optional, Union, List, Dict
from pydantic import BaseModel, Field

class Metadados(BaseModel):
    self_ref: Optional[str] = None
    filename: Optional[str] = None
    modDate: Optional[str] = None
    caption: Optional[str] = None
    format: Optional[str] = None
    references: Optional[str] = None
    footnotes: Optional[str] = None
    page: Optional[Union[int, str]] = None
    title: Optional[str] = None
    entidades: Optional[str] = None
    content_type: Optional[str] = None
    table_id: Optional[Union[int, str]] = None

class ContextoItem(BaseModel):
    distancia: float
    metadados: Metadados
    texto: str
    tipo: str

class RagResumo(BaseModel):
    resumo: str

class QueryResult(BaseModel):
    resposta: str
    contexto: List[ContextoItem]

class SearchRAGResponse(BaseModel):
    query_result: List[QueryResult]

class QueryList(BaseModel):
    '''estrutura o dado como lista'''
    queries: List[str]

class ReportState(BaseModel):
    user_input: Optional[str] = None
    final_response: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    queries_results: Annotated[List[QueryList], operator.add]
