"""
Pipeline de Agentes para Recuperação de Informações - IPEA RAG
Utiliza o framework Agno com OpenAI como LLM.
"""

import os
import json
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from time import perf_counter
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from src.recuperacao.prompts import *
from src.recuperacao.schemas import *

from src.recuperacao.tools import (
    search_recommendations,
    search_chunks,
    search_tables,
    search_images,
    search_all_collections
)

# Carregar variáveis de ambiente
load_dotenv()


# ============================================================ #
# Definição das Tools para os Agentes
# ============================================================ #

@tool
def tool_search_recommendations(query: str, top_k: int = 5) -> str:
    """
    Busca documentos recomendados no repositório do IPEA.
    Use para: recomendações de leitura, documentos relacionados, busca por tema.
    
    Args:
        query: Texto de busca (tema, título, palavras-chave)
        top_k: Número máximo de resultados (padrão: 5)
    """
    results = search_recommendations(query, top_k)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def tool_search_chunks(query: str, top_k: int = 10) -> str:
    """
    Busca trechos de texto relevantes nos documentos do IPEA.
    Use para: responder perguntas específicas, encontrar informações detalhadas.
    
    Args:
        query: Pergunta ou texto de busca
        top_k: Número máximo de chunks (padrão: 10)
    """
    results = search_chunks(query, top_k)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def tool_search_tables(query: str, top_k: int = 5) -> str:
    """
    Busca tabelas e dados numéricos nos documentos do IPEA.
    Use para: encontrar estatísticas, indicadores, séries temporais.
    
    Args:
        query: Descrição dos dados procurados
        top_k: Número máximo de tabelas (padrão: 5)
    """
    results = search_tables(query, top_k)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def tool_search_images(query: str, top_k: int = 5) -> str:
    """
    Busca gráficos e imagens nos documentos do IPEA.
    Use para: encontrar visualizações, gráficos de tendências.
    
    Args:
        query: Descrição do gráfico ou imagem procurada
        top_k: Número máximo de imagens (padrão: 5)
    """
    results = search_images(query, top_k)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def tool_search_all(query: str, top_k: int = 5) -> str:
    """
    Busca em todas as coleções simultaneamente.
    Use para: busca exploratória inicial, quando não sabe onde está a informação.
    
    Args:
        query: Texto de busca
        top_k: Número máximo de resultados por coleção (padrão: 5)
    """
    results = search_all_collections(query, top_k)
    return json.dumps(results, ensure_ascii=False, indent=2)


# ============================================================ #
# Configuração do Modelo
# ============================================================ #

def get_openai_model(model_id: str = "gpt-4o-mini") -> OpenAIChat:
    """
    Retorna o modelo OpenAI configurado.
    
    Args:
        model_id: ID do modelo (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, etc.)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY não encontrada. "
            "Configure a variável de ambiente ou crie um arquivo .env"
        )
    
    return OpenAIChat(id=model_id, api_key=api_key)


# ============================================================ #
# Definição dos Agentes Especializados
# ============================================================ #

def create_intent_agent(model: OpenAIChat) -> Agent:
    """Agente 1: Classificação de Intenção"""
    return Agent(
        name="Classificador de Intenção",
        model=model,
        instructions=INTENT_CLASSIFICATION_PROMPT,
        output_schema=IntentClassification,
        markdown=True
    )


def create_context_agent(model: OpenAIChat) -> Agent:
    """Agente 2: Extração de Contexto"""
    return Agent(
        name="Extrator de Contexto",
        model=model,
        instructions=CONTEXT_EXTRACTION_PROMPT,
        output_schema=ContextExtraction,
        markdown=True
    )


def create_refinement_agent(model: OpenAIChat) -> Agent:
    """Agente 4: Refinamento de Query"""
    return Agent(
        name="Refinador de Query",
        model=model,
        instructions=QUERY_REFINEMENT_PROMPT,
        output_schema=RefinedQueries,
        markdown=True
    )


def create_fusion_agent(model: OpenAIChat) -> Agent:
    """Agente 6: Fusão de Contexto"""
    return Agent(
        name="Fusionador de Contexto",
        model=model,
        instructions=CONTEXT_FUSION_PROMPT,
        markdown=True
    )


def create_data_agent(model: OpenAIChat) -> Agent:
    """Agente 7: Interpretação de Dados"""
    return Agent(
        name="Interpretador de Dados",
        model=model,
        instructions=DATA_INTERPRETATION_PROMPT,
        output_schema=DataInterpretation,
        markdown=True
    )


def create_response_agent(model: OpenAIChat) -> Agent:
    """Agente 8: Gerador de Resposta"""
    return Agent(
        name="Gerador de Resposta",
        model=model,
        instructions=RESPONSE_GENERATION_PROMPT,
        # output_schema=FinalResponse,  # removido para evitar erro de JSON schema strict
        markdown=True
    )


def create_verification_agent(model: OpenAIChat) -> Agent:
    """Agente 9: Verificador de Fatos"""
    return Agent(
        name="Verificador de Fatos",
        model=model,
        instructions=FACT_VERIFICATION_PROMPT,
        output_schema=FactCheckResult,
        markdown=True
    )


def create_retrieval_agent(model: OpenAIChat) -> Agent:
    """
    Agente de Recuperação (Camadas 3 e 5)
    Executa buscas nas coleções vetoriais.
    """
    return Agent(
        name="Recuperador",
        model=model,
        instructions="""
        Você é um agente de recuperação de informações.
        Use as ferramentas disponíveis para buscar informações relevantes
        nas coleções vetoriais do IPEA.
        
        Siga as instruções sobre quais coleções buscar e quais queries usar.
        Retorne todos os resultados encontrados de forma estruturada.
        """,
        tools=[
            tool_search_recommendations,
            tool_search_chunks,
            tool_search_tables,
            tool_search_images,
            tool_search_all
        ],
        markdown=True
    )


# ============================================================ #
# Pipeline Principal
# ============================================================ #

class RAGPipeline:
    """
    Pipeline completa de RAG com 9 etapas de agentes especializados.
    """
    
    def __init__(self, model_id: str = "gpt-4o-mini", verbose: bool = True) -> None:
        """
        Inicializa a pipeline com o modelo especificado.
        
        Args:
            model_id: ID do modelo OpenAI (padrão: gpt-4o-mini)
        """
        self.verbose = verbose
        self.model = get_openai_model(model_id)
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}][Init] Modelo OpenAI: {self.model.id} | Verbose: {self.verbose}")
        
        # Criar agentes
        self.intent_agent = create_intent_agent(self.model)
        self.context_agent = create_context_agent(self.model)
        self.retrieval_agent = create_retrieval_agent(self.model)
        self.refinement_agent = create_refinement_agent(self.model)
        self.fusion_agent = create_fusion_agent(self.model)
        self.data_agent = create_data_agent(self.model)
        self.response_agent = create_response_agent(self.model)
        self.verification_agent = create_verification_agent(self.model)
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}][Init] Agentes criados: intent, context, retrieval, refinement, fusion, data, response, verification")
        
        # Estado da pipeline
        self.state: Dict[str, Any] = {}
    
    def _log(self, step: str, message: str):
        """Log de progresso da pipeline."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}][{step}] {message}")
    
    def _get_content(self, response) -> Any:
        """Extrai conteúdo da resposta do agente."""
        if hasattr(response, 'content'):
            if self.verbose:
                self._log("Debug", f"Tipo de resposta: {type(response).__name__} (com 'content')")
            return response.content
        if self.verbose:
            self._log("Debug", f"Tipo de resposta: {type(response).__name__} (sem 'content')")
        return response
    
    def step1_classify_intent(self, query: str) -> IntentClassification:
        """Passo 1: Classificar intenção do usuário."""
        start = perf_counter()
        self._log("Passo 1", "Classificando intenção...")
        
        response = self.intent_agent.run(query)
        intent = self._get_content(response)
        
        # Se não for o schema esperado, criar um padrão
        if not isinstance(intent, IntentClassification):
            self._log("Passo 1", "Aviso: Resposta não corresponde ao schema. Aplicando padrão.")
            intent = IntentClassification(
                intent_type="rag_textual",
                detail_level="medio",
                requires_data=False,
                requires_images=False,
                original_query=query,
                reasoning="Classificação padrão"
            )
        
        self._log("Passo 1", f"Intenção: {intent.intent_type}")
        if self.verbose:
            self._log("Passo 1", f"Detalhes: {intent}")
        self._log("Passo 1", f"Duração: {int((perf_counter()-start)*1000)} ms")
        self.state["intent"] = intent
        return intent
    
    def step2_extract_context(self, query: str, intent: IntentClassification) -> ContextExtraction:
        """Passo 2: Extrair contexto semântico."""
        start = perf_counter()
        self._log("Passo 2", "Extraindo contexto...")
        
        prompt = f"""
        Query do usuário: {query}
        
        Intenção classificada:
        - Tipo: {intent.intent_type}
        - Nível de detalhe: {intent.detail_level}
        - Requer dados: {intent.requires_data}
        - Requer imagens: {intent.requires_images}
        
        Extraia o contexto e gere queries otimizadas.
        """
        
        response = self.context_agent.run(prompt)
        context = self._get_content(response)
        
        # Se não for o schema esperado, criar um padrão
        if not isinstance(context, ContextExtraction):
            self._log("Passo 2", "Aviso: Resposta não corresponde ao schema. Aplicando padrão.")
            context = ContextExtraction(
                main_topic=query,
                keywords=query.split()[:5],
                query_for_recommendations=query,
                query_for_chunks=query,
                query_for_tables=query,
                query_for_images=query
            )
        
        self._log("Passo 2", f"Tema principal: {context.main_topic}")
        if self.verbose:
            self._log("Passo 2", f"Contexto extraído: {context}")
        self._log("Passo 2", f"Duração: {int((perf_counter()-start)*1000)} ms")
        self.state["context"] = context
        return context
    
    def step3_first_retrieval(self, context: ContextExtraction) -> Dict[str, Any]:
        """Passo 3: Primeira camada de recuperação."""
        start = perf_counter()
        self._log("Passo 3", "Executando primeira recuperação...")
        
        results = {}
        intent = self.state.get("intent")
        
        # Buscar em recomendações
        results["recomendacoes"] = search_recommendations(
            context.query_for_recommendations, top_k=5
        )
        
        # Buscar em chunks
        results["chunks"] = search_chunks(
            context.query_for_chunks, top_k=10
        )
        
        # Buscar em tabelas se necessário
        if intent and intent.requires_data:
            results["tabelas"] = search_tables(
                context.query_for_tables, top_k=5
            )
        else:
            results["tabelas"] = search_tables(
                context.query_for_tables, top_k=3
            )
        
        # Buscar imagens se necessário
        if intent and intent.requires_images:
            results["imagens"] = search_images(
                context.query_for_images, top_k=5
            )
        else:
            results["imagens"] = []
        
        self._log("Passo 3", f"Recuperados: {len(results['recomendacoes'])} recomendações, {len(results['chunks'])} chunks, "
                  f"{len(results['tabelas'])} tabelas e {len(results['imagens'])} imagens")
        if self.verbose:
            self._log("Passo 3", f"Resultados: {results}")
        self._log("Passo 3", f"Duração: {int((perf_counter()-start)*1000)} ms")
        self.state["first_retrieval"] = results
        return results
    
    def step4_refine_queries(
        self, 
        query: str, 
        context: ContextExtraction, 
        first_results: Dict[str, Any]
    ) -> RefinedQueries:
        """Passo 4: Refinar queries com base nos resultados iniciais."""
        start = perf_counter()
        self._log("Passo 4", "Refinando queries...")
        
        # Resumir resultados para o agente
        chunks_summary = [
            f"- {r['texto'][:200]}..." for r in first_results.get("chunks", [])[:3]
        ]
        
        prompt = f"""
        Query original: {query}
        
        Contexto extraído:
        - Tema: {context.main_topic}
        - Keywords: {', '.join(context.keywords)}
        
        Queries iniciais:
        - Recomendações: {context.query_for_recommendations}
        - Chunks: {context.query_for_chunks}
        - Tabelas: {context.query_for_tables}
        - Imagens: {context.query_for_images}
        
        Amostra dos resultados da primeira busca:
        {chr(10).join(chunks_summary)}
        
        Refine as queries para melhorar a segunda recuperação.
        """
        
        response = self.refinement_agent.run(prompt)
        refined = self._get_content(response)
        
        # Se não for o schema esperado, criar um padrão
        if not isinstance(refined, RefinedQueries):
            self._log("Passo 4", "Aviso: Resposta não corresponde ao schema. Aplicando padrão.")
            refined = RefinedQueries(
                query_recommendations=context.query_for_recommendations,
                query_chunks=context.query_for_chunks,
                query_tables=context.query_for_tables,
                query_images=context.query_for_images,
                expansion_terms=[],
                refinement_reasoning="Refinamento padrão"
            )
        
        self._log("Passo 4", f"Termos expandidos: {refined.expansion_terms}")
        if self.verbose:
            self._log("Passo 4", f"Queries refinadas: {refined}")
        self._log("Passo 4", f"Duração: {int((perf_counter()-start)*1000)} ms")
        self.state["refined_queries"] = refined
        return refined
    
    def step5_second_retrieval(self, refined: RefinedQueries) -> Dict[str, Any]:
        """Passo 5: Segunda camada de recuperação com queries refinadas."""
        start = perf_counter()
        self._log("Passo 5", "Executando segunda recuperação...")
        
        results = {}
        
        # Buscar com queries refinadas
        results["recomendacoes"] = search_recommendations(
            refined.query_recommendations, top_k=5
        )
        results["chunks"] = search_chunks(
            refined.query_chunks, top_k=10
        )
        results["tabelas"] = search_tables(
            refined.query_tables, top_k=5
        )
        
        intent = self.state.get("intent")
        if intent and intent.requires_images:
            results["imagens"] = search_images(
                refined.query_images, top_k=5
            )
        else:
            results["imagens"] = []
        
        self._log("Passo 5", f"Recuperados: {len(results['chunks'])} chunks refinados")
        if self.verbose:
            self._log("Passo 5", f"Resultados: {results}")
        self._log("Passo 5", f"Duração: {int((perf_counter()-start)*1000)} ms")
        self.state["second_retrieval"] = results
        return results
    
    def step6_fuse_context(
        self, 
        first_results: Dict[str, Any], 
        second_results: Dict[str, Any]
    ) -> str:
        """Passo 6: Fundir contexto das duas camadas de recuperação."""
        start = perf_counter()
        self._log("Passo 6", "Fundindo contexto...")
        
        # Combinar resultados únicos
        all_chunks = {}
        for chunk in first_results.get("chunks", []) + second_results.get("chunks", []):
            cid = chunk.get("chunk_id", "")
            if cid and cid not in all_chunks:
                all_chunks[cid] = chunk
        
        all_tables = {}
        for table in first_results.get("tabelas", []) + second_results.get("tabelas", []):
            tid = table.get("table_id", "")
            if tid and tid not in all_tables:
                all_tables[tid] = table
        
        if self.verbose:
            self._log("Passo 6", f"Únicos: {len(all_chunks)} chunks, {len(all_tables)} tabelas")
        
        # Preparar evidências para o agente de fusão
        evidence_text = "## Chunks de Texto:\n"
        for chunk in list(all_chunks.values())[:10]:
            evidence_text += f"\n### Chunk {chunk.get('chunk_id', '')}\n"
            evidence_text += f"Documento: {chunk.get('doc_id', '')}\n"
            evidence_text += f"Página: {chunk.get('pagina', 'N/A')}\n"
            evidence_text += f"Score: {chunk.get('score', 0)}\n"
            evidence_text += f"Texto:\n{chunk.get('texto', '')}\n"
            evidence_text += "---\n"
        
        evidence_text += "\n## Tabelas:\n"
        for table in list(all_tables.values())[:5]:
            evidence_text += f"\n### Tabela {table.get('table_id', '')}\n"
            evidence_text += f"Documento: {table.get('doc_id', '')}\n"
            evidence_text += f"Página: {table.get('pagina', 'N/A')}\n"
            evidence_text += f"Conteúdo:\n{table.get('tabela', '')}\n"
            evidence_text += "---\n"
        
        # Adicionar recomendações
        evidence_text += "\n## Documentos Recomendados:\n"
        for rec in second_results.get("recomendacoes", [])[:5]:
            evidence_text += f"\n### {rec.get('titulo', 'Sem título')}\n"
            evidence_text += f"Resumo: {rec.get('resumo', '')}\n"
            evidence_text += f"Link: {rec.get('handle', '')}\n"
            evidence_text += "---\n"
        
        prompt = f"""
        Analise e funda as seguintes evidências recuperadas:
        
        {evidence_text}
        
        Produza um contexto consolidado, removendo duplicatas e organizando
        as informações de forma coerente.
        """
        
        response = self.fusion_agent.run(prompt)
        fused_context = self._get_content(response)
        
        # Garantir que é string
        if not isinstance(fused_context, str):
            fused_context = str(fused_context) if fused_context else evidence_text
        
        self._log("Passo 6", "Contexto fundido com sucesso")
        if self.verbose:
            self._log("Passo 6", f"Contexto fundido: {fused_context[:500]}...")

        self.state["fused_context"] = fused_context
        self.state["all_evidence"] = {
            "chunks": list(all_chunks.values()),
            "tables": list(all_tables.values()),
            "recommendations": second_results.get("recomendacoes", [])
        }
        self._log("Passo 6", f"Duração: {int((perf_counter()-start)*1000)} ms")
        return fused_context
    
    def step7_interpret_data(self, tables: List[Dict]) -> Optional[DataInterpretation]:
        """Passo 7: Interpretar dados das tabelas (quando aplicável)."""
        start = perf_counter()
        intent = self.state.get("intent")
        
        if not intent or not intent.requires_data:
            self._log("Passo 7", "Interpretação de dados não necessária")
            return None
        
        if not tables:
            self._log("Passo 7", "Nenhuma tabela para interpretar")
            return None
        
        self._log("Passo 7", f"{len(tables)} tabela(s) recebidas para interpretação")
        
        tables_text = "\n\n".join([
            f"Tabela {t.get('table_id', '')}:\n{t.get('tabela', '')}"
            for t in tables[:5]
        ])
        
        prompt = f"""
        Interprete os seguintes dados e tabelas:
        
        {tables_text}
        
        Extraia valores relevantes, identifique séries temporais
        e produza um resumo dos dados.
        """
        
        response = self.data_agent.run(prompt)
        interpretation = self._get_content(response)
        
        # Se não for o schema esperado, criar um padrão
        if not isinstance(interpretation, DataInterpretation):
            self._log("Passo 7", "Aviso: Resposta não corresponde ao schema. Aplicando padrão.")
            interpretation = DataInterpretation(
                has_data=True,
                extracted_values=[],
                key_metrics=[],
                data_summary="Dados disponíveis nas tabelas acima."
            )
        
        self._log("Passo 7", f"Métricas identificadas: {len(interpretation.key_metrics)}")
        if self.verbose:
            self._log("Passo 7", f"Interpretação de dados: {interpretation}")
        self.state["data_interpretation"] = interpretation
        self._log("Passo 7", f"Duração: {int((perf_counter()-start)*1000)} ms")
        return interpretation
    
    def step8_generate_response(
        self, 
        query: str, 
        intent: IntentClassification,
        fused_context: str,
        data_interpretation: Optional[DataInterpretation]
    ) -> FinalResponse:
        """Passo 8: Gerar resposta final."""
        start = perf_counter()
        self._log("Passo 8", "Gerando resposta...")
        
        prompt = f"""
        Query do usuário: {query}
        
        Intenção: {intent.intent_type}
        Nível de detalhe: {intent.detail_level}
        
        Contexto consolidado:
        {fused_context}
        """
        
        if data_interpretation and data_interpretation.has_data:
            prompt += f"""
            
            Dados interpretados:
            {data_interpretation.data_summary}
            
            Métricas principais: {', '.join(data_interpretation.key_metrics)}
            """
        
        prompt += """
        
        Gere uma resposta completa e bem fundamentada.

        IMPORTANTE:
        - Responda **APENAS** com um JSON válido, sem texto extra.
        - O JSON deve ter exatamente estas chaves:
          {
            "response_text": "texto da resposta em markdown",
            "sources": [
              {
                "titulo": "Título do documento",
                "handle": "URL ou identificador do documento"
              }
            ],
            "confidence": "alta" | "media" | "baixa",
            "data_included": true | false
          }
        """
        
        response = self.response_agent.run(prompt)
        raw = self._get_content(response)
        # Tentar converter o retorno em FinalResponse
        final_response: FinalResponse
        try:
            if isinstance(raw, str):
                data = json.loads(raw)
            elif isinstance(raw, dict):
                data = raw
            else:
                data = json.loads(str(raw))

            final_response = FinalResponse(**data)
        except Exception as e:
            # Fallback se o modelo não respeitar o formato
            self._log("Passo 8", f"Aviso: falha ao parsear JSON ({e}). Aplicando padrão.")
            response_text = str(raw) if raw else "Não foi possível gerar resposta."
            final_response = FinalResponse(
                response_text=response_text,
                sources=[],
                confidence="media",
                data_included=bool(
                    data_interpretation and data_interpretation.has_data
                )
            )
        
        self._log("Passo 8", f"Resposta gerada (confiança: {final_response.confidence})")
        if self.verbose:
            self._log("Passo 8", f"Tamanho da resposta: {len(final_response.response_text)} caracteres")
        self._log("Passo 8", f"Duração: {int((perf_counter()-start)*1000)} ms")
        self.state["response"] = final_response
        return final_response
    
    def step9_verify_facts(
        self, 
        response: FinalResponse, 
        evidence: Dict[str, Any]
    ) -> FactCheckResult:
        """Passo 9: Verificar fatos da resposta."""
        start = perf_counter()
        self._log("Passo 9", "Verificando fatos...")
        
        # Preparar evidências para verificação
        evidence_summary = "## Evidências disponíveis:\n\n"
        
        for chunk in evidence.get("chunks", [])[:5]:
            evidence_summary += f"- {chunk.get('texto', '')[:300]}...\n\n"
        
        prompt = f"""
        Resposta gerada:
        {response.response_text}
        
        Fontes citadas: {response.sources}
        
        {evidence_summary}
        
        Verifique se todas as afirmações na resposta estão suportadas
        pelas evidências disponíveis.
        """
        
        verification = self.verification_agent.run(prompt)
        result = self._get_content(verification)
        
        # Se não for o schema esperado, criar um padrão
        if not isinstance(result, FactCheckResult):
            self._log("Passo 9", "Aviso: Resposta não corresponde ao schema. Aplicando padrão.")
            result = FactCheckResult(
                is_valid=True,
                issues_found=[],
                unsupported_claims=[],
                corrections_needed=False,
                verification_notes="Verificação automática não disponível"
            )
        if result.is_valid:
            self._log("Passo 9", "✓ Resposta verificada com sucesso")
        else:
            self._log("Passo 9", f"⚠ Problemas encontrados: {len(result.issues_found)}")

        if self.verbose:
            self._log("Passo 9", f"Detalhes da verificação: {result}")
        
        self.state["verification"] = result
        self._log("Passo 9", f"Duração: {int((perf_counter()-start)*1000)} ms")
        return result
    
    def run(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Executa a pipeline completa.
        
        Args:
            query: Consulta do usuário
            max_retries: Número máximo de tentativas se verificação falhar
            
        Returns:
            Dicionário com resposta e metadados
        """
        print("\n" + "="*60)
        print(f"Pipeline RAG - Query: {query[:50]}...")
        print("="*60 + "\n")
        if self.verbose:
            self._log("Run", f"Modelo: {self.model.id} | Retries: {max_retries}")
        # Reset estado
        self.state = {"query": query}
        
        # Executar pipeline
        intent = self.step1_classify_intent(query)
        context = self.step2_extract_context(query, intent)
        first_results = self.step3_first_retrieval(context)
        refined = self.step4_refine_queries(query, context, first_results)
        second_results = self.step5_second_retrieval(refined)
        fused_context = self.step6_fuse_context(first_results, second_results)
        
        # Interpretação de dados (opcional)
        tables = self.state.get("all_evidence", {}).get("tables", [])
        data_interpretation = self.step7_interpret_data(tables)
        
        # Gerar resposta
        response = self.step8_generate_response(
            query, intent, fused_context, data_interpretation
        )
        
        # Verificar fatos
        evidence = self.state.get("all_evidence", {})
        verification = self.step9_verify_facts(response, evidence)
        
        # Retry se necessário
        retries = 0
        while not verification.is_valid and retries < max_retries:
            self._log("Retry", f"Tentativa {retries + 1} de correção...")
            
            # Adicionar feedback de verificação ao prompt
            correction_prompt = f"""
            A resposta anterior teve problemas:
            {verification.issues_found}
            
            Afirmações sem suporte:
            {verification.unsupported_claims}
            
            Por favor, corrija a resposta removendo informações não suportadas.
            """
            
            response = self.step8_generate_response(
                query + "\n\n" + correction_prompt,
                intent, fused_context, data_interpretation
            )
            verification = self.step9_verify_facts(response, evidence)
            retries += 1
        
        print("\n" + "="*60)
        print("Pipeline concluída!")
        print("="*60 + "\n")
        
        return {
            "query": query,
            "intent": intent.model_dump() if hasattr(intent, 'model_dump') else str(intent),
            "response": response.response_text if hasattr(response, 'response_text') else str(response),
            "sources": response.sources if hasattr(response, 'sources') else [],
            "confidence": response.confidence if hasattr(response, 'confidence') else "unknown",
            "verification": verification.model_dump() if hasattr(verification, 'model_dump') else str(verification),
            "data_included": response.data_included if hasattr(response, 'data_included') else False
        }


# ============================================================ #
# Agente Coordenador (Alternativa com Team)
# ============================================================ #

def create_coordinator_agent(model_id: str = "gpt-4o-mini") -> Agent:
    """
    Cria um agente coordenador que pode ser usado diretamente.
    """
    model = get_openai_model(model_id)
    
    return Agent(
        name="Coordenador RAG IPEA",
        model=model,
        instructions=COORDINATOR_PROMPT,
        tools=[
            tool_search_recommendations,
            tool_search_chunks,
            tool_search_tables,
            tool_search_images,
            tool_search_all
        ],
        markdown=True
    )


# ============================================================ #
# Funções de Conveniência
# ============================================================ #

def quick_search(query: str, model_id: str = "gpt-4o-mini") -> str:
    """
    Executa uma busca rápida usando o agente coordenador.
    
    Args:
        query: Consulta do usuário
        model_id: Modelo OpenAI a usar
        
    Returns:
        Resposta do agente
    """
    agent = create_coordinator_agent(model_id)
    response = agent.run(query)
    content = response.content if hasattr(response, 'content') else response
    return str(content) if content else ""


def run_full_pipeline(query: str, model_id: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Executa a pipeline completa de 9 passos.
    
    Args:
        query: Consulta do usuário
        model_id: Modelo OpenAI a usar
        
    Returns:
        Dicionário com resposta e metadados
    """
    pipeline = RAGPipeline(model_id=model_id)
    return pipeline.run(query)
