"""
Schemas/Modelos de dados para a pipeline de agentes.
Define estruturas para comunicação entre agentes.
"""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from pydantic import ConfigDict


class IntentClassification(BaseModel):
    """Classificação da intenção do usuário."""
    
    intent_type: Literal[
        "simple_response",      # Resposta simples sem RAG
        "rag_textual",          # RAG com chunks de texto
        "table_search",         # Busca por tabelas/dados
        "image_search",         # Busca por gráficos/imagens
        "recommendation"        # Recomendação de leituras
    ] = Field(description="Tipo de intenção identificada")
    
    detail_level: Literal["baixo", "medio", "alto"] = Field(
        default="medio",
        description="Nível de detalhamento esperado na resposta"
    )
    
    requires_data: bool = Field(
        default=False,
        description="Se a resposta requer dados numéricos/tabelas"
    )
    
    requires_images: bool = Field(
        default=False,
        description="Se a resposta requer gráficos/imagens"
    )
    
    original_query: str = Field(description="Query original do usuário")
    
    reasoning: str = Field(
        description="Explicação breve do motivo da classificação"
    )


class ContextExtraction(BaseModel):
    """Extração de contexto semântico da query."""
    
    main_topic: str = Field(description="Tema principal da consulta")
    
    keywords: List[str] = Field(
        default_factory=list,
        description="Termos-chave extraídos"
    )
    
    temporal_filter: Optional[str] = Field(
        default=None,
        description="Filtro temporal se mencionado (ex: '2020-2023')"
    )
    
    document_types: List[str] = Field(
        default_factory=list,
        description="Tipos de documento preferidos"
    )
    
    query_for_recommendations: str = Field(
        description="Query otimizada para busca em recomendações"
    )
    
    query_for_chunks: str = Field(
        description="Query otimizada para busca em chunks textuais"
    )
    
    query_for_tables: str = Field(
        description="Query otimizada para busca em tabelas"
    )
    
    query_for_images: str = Field(
        description="Query otimizada para busca em imagens"
    )


class RefinedQueries(BaseModel):
    """Queries refinadas após primeira recuperação."""
    
    query_recommendations: str = Field(
        description="Query refinada para recomendações"
    )
    
    query_chunks: str = Field(
        description="Query refinada para chunks"
    )
    
    query_tables: str = Field(
        description="Query refinada para tabelas"
    )
    
    query_images: str = Field(
        description="Query refinada para imagens"
    )
    
    expansion_terms: List[str] = Field(
        default_factory=list,
        description="Termos adicionados para expansão de query"
    )
    
    refinement_reasoning: str = Field(
        description="Explicação das mudanças realizadas"
    )


class Evidence(BaseModel):
    """Uma evidência recuperada."""
    
    source_type: Literal["chunk", "table", "image", "recommendation"] = Field(
        description="Tipo de fonte"
    )
    
    content: str = Field(description="Conteúdo da evidência")
    
    doc_id: str = Field(description="ID do documento de origem")
    
    handle: Optional[str] = Field(
        default=None,
        description="Link para o documento"
    )
    
    page: Optional[int] = Field(
        default=None,
        description="Página no documento"
    )
    
    relevance_score: float = Field(
        description="Score de relevância da busca"
    )


class FusedContext(BaseModel):
    """Contexto unificado após fusão de evidências."""
    
    main_evidences: List[Evidence] = Field(
        description="Evidências principais consolidadas"
    )
    
    supporting_evidences: List[Evidence] = Field(
        default_factory=list,
        description="Evidências de suporte"
    )
    
    data_evidences: List[Evidence] = Field(
        default_factory=list,
        description="Evidências com dados/tabelas"
    )
    
    image_evidences: List[Evidence] = Field(
        default_factory=list,
        description="Evidências com imagens/gráficos"
    )
    
    consolidated_text: str = Field(
        description="Texto consolidado das evidências principais"
    )
    
    sources_summary: str = Field(
        description="Resumo das fontes utilizadas"
    )


class ExtractedValue(BaseModel):
    """Item estruturado extraído de tabelas/dados."""
    
    label: str = Field(description="Rótulo ou nome do indicador")
    
    value: Union[str, float, int, bool, None] = Field(description="Valor do indicador")
    
    unit: Optional[str] = Field(default=None, description="Unidade do valor, se houver")
    
    source_doc_id: Optional[str] = Field(default=None, description="ID do documento fonte")
    
    table_id: Optional[str] = Field(default=None, description="ID da tabela fonte")

    # Schema estrito para objetos aninhados no strict mode
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["label", "value", "unit", "source_doc_id", "table_id"],
            "additionalProperties": False
        }
    )


class DataInterpretation(BaseModel):
    """Interpretação de dados extraídos."""
    
    has_data: bool = Field(
        default=False,
        description="Se há dados estruturados"
    )
    
    extracted_values: List[ExtractedValue] = Field(
        default_factory=list,
        description="Valores extraídos das tabelas"
    )
    
    time_series: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Séries temporais identificadas"
    )
    
    key_metrics: List[str] = Field(
        default_factory=list,
        description="Métricas/indicadores chave"
    )
    
    data_summary: str = Field(
        description="Resumo textual dos dados"
    )

    # Pydantic v2: schema estrito no nível raiz
    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "has_data",
                "extracted_values",
                "time_series",
                "key_metrics",
                "data_summary"
            ],
            "additionalProperties": False
        }
    )


class FinalResponse(BaseModel):
    """Resposta final gerada."""
    
    response_text: str = Field(
        description="Texto da resposta"
    )
    
    sources: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Lista de fontes citadas"
    )
    
    confidence: Literal["alta", "media", "baixa"] = Field(
        description="Nível de confiança na resposta"
    )
    
    data_included: bool = Field(
        default=False,
        description="Se dados numéricos foram incluídos"
    )


class FactCheckResult(BaseModel):
    """Resultado da verificação de fatos."""
    
    is_valid: bool = Field(
        description="Se a resposta é válida"
    )
    
    issues_found: List[str] = Field(
        default_factory=list,
        description="Problemas encontrados"
    )
    
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Afirmações sem suporte nas evidências"
    )
    
    corrections_needed: bool = Field(
        default=False,
        description="Se correções são necessárias"
    )
    
    verification_notes: str = Field(
        description="Notas sobre a verificação"
    )


def build_openai_response_format(model: type[BaseModel]) -> Dict[str, Any]:
    """
    Retorna um response_format válido para OpenAI Responses API (strict)
    a partir de um BaseModel do Pydantic.
    Uso: response_format = build_openai_response_format(DataInterpretation)
    """
    try:
        schema = model.model_json_schema()  # Pydantic v2
    except AttributeError:
        schema = model.schema()  # Pydantic v1

    props = schema.get("properties", {})
    if props:
        schema["required"] = list(props.keys())
        schema["additionalProperties"] = False

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": schema,
        },
    }
