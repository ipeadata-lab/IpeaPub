"""
Módulo de Agentes para o sistema RAG do IPEA.

Este módulo implementa a pipeline de recuperação com 9 agentes especializados:
1. Agente de Classificação de Intenção
2. Agente de Extração de Contexto
3. Primeira Camada de Recuperação
4. Agente de Refinamento de Query
5. Segunda Camada de Recuperação
6. Agente de Fusão de Contexto
7. Agente de Interpretação de Dados
8. Agente Gerador de Resposta Final
9. Agente Verificador de Fatos
"""

from src.recuperacao.pipeline import (
    RAGPipeline,
    create_coordinator_agent,
    quick_search,
    run_full_pipeline
)

from src.recuperacao.tools import (
    search_recommendations,
    search_chunks,
    search_tables,
    search_images,
    search_all_collections
)

from src.recuperacao.schemas import (
    IntentClassification,
    ContextExtraction,
    RefinedQueries,
    FusedContext,
    DataInterpretation,
    FinalResponse,
    FactCheckResult,
    Evidence
)

__all__ = [
    # Pipeline
    "RAGPipeline",
    "create_coordinator_agent",
    "quick_search",
    "run_full_pipeline",
    
    # Tools
    "search_recommendations",
    "search_chunks",
    "search_tables",
    "search_images",
    "search_all_collections",
    
    # Schemas
    "IntentClassification",
    "ContextExtraction",
    "RefinedQueries",
    "FusedContext",
    "DataInterpretation",
    "FinalResponse",
    "FactCheckResult",
    "Evidence",
]
