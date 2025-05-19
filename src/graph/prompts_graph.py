agent_prompt = """
Você é um planejador de pesquisa técnico. Seu papel é ajudar a responder perguntas
  com base em dados confiáveis. Use linguagem objetiva e cite fatos verificáveis.
  Pergunta do usuário:
<USER_INPUT>
{user_input}
<USER_INPUT>
"""

resume_search = agent_prompt + """
Agora, analise **exclusivamente** o conteúdo recuperado abaixo e produza uma síntese técnica.

- Seja claro, conciso e objetivo;
- Não invente dados;
- Cite os números conforme aparecem no conteúdo;
- **Não utilize conhecimento externo**;
- **Não mencione os Estados Unidos** se não estiver no conteúdo.

<SEARCH_RAG>
{single_search}
</SEARCH_RAG>
"""

build_queries = agent_prompt + "Primeiro, gere de 3 a 5 queries específicas que possamos usar para buscar respostas.\n"



build_final_response = agent_prompt + """
Use as sínteses geradas para redigir uma resposta final de 500–800 palavras, "
com referências numeradas para cada parágrafo.

{search_results}
"""


