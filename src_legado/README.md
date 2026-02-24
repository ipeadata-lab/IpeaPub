Documentação: projeto RagPub

# **1. Contexto**  
O projeto RagPub visa o desenvolvimento de um sistema de agentes assistentes inteligentes para o repositório de conhecimento do Ipea. A proposta central é criar uma solução robusta, baseada em IA, capaz de processar e interpretar todos os documentos do repositório, permitindo aos usuários interagir com o conteúdo por meio de um agente conversacional. Os principais recursos incluem:

* Recomendação personalizada de leituras;  
* Respostas a perguntas com base no contexto específico dos documentos (RAG);  
* Interação conversacional com fontes de dados (gráficos, imagens e tabelas) presentes nos documentos.

Para atingir esses objetivos, o projeto utilizará uma arquitetura de sistema multi-agentes, bibliotecas Python especializadas e técnicas avançadas de Recuperação Aumentada por Geração (RAG). O detalhamento do funcionamento interno será apresentado nas próximas seções.

# **2. Tecnologias**  
O projeto está sendo desenvolvido em Python, com código fonte disponível em [IpeaPub](https://github.com/ipeadata-lab/IpeaPub) no GitHub do Ipea. Os principais frameworks e bibliotecas utilizadas no projeto são:

* Docling (para parsing de documentos)  
* Qdrant (para banco vetorial)  
* PyTorch/transformers (para rodar modelos de IA localmente)  
* BeautifulSoup (para o crawler do site do Ipea)  
* Agno (framework de agentes de IA)  
* FastAPI (para receber e responder requisições)  
* Docker (Conteinerização do projeto)

# **3. Funcionamento**  
Existem 2 etapas para o funcionamento integral do sistema: ingestão e recuperação. Abaixo, será explicado cada uma delas. Os arquivos Python envolvidos em cada sub etapa do projeto estará explicitado entre parênteses para ajudar o entendimento.

## **3.1. Ingestão**  
Essa etapa serve para inserir as informações dos documentos do repositório na base de dados do projeto. As informações são a base para o funcionamento do sistema multiagente, por isso pode ser considerada a etapa mais importante do projeto. São utilizados bancos de dados vetoriais para realizar a busca semântica de informações. O banco vetorial possui 4 coleções (*banco\_vetorial.py*):

- **recomendações:** Embedding titulo+resumo+keywords+tipo\_conteudo para que o sistema de agentes possa buscar documentos de acordo com o contexto do usuário.  
- **chunks:** Embedding clássico de chunks de texto dos documentos para RAG. Utiliza contextualização disponibilizada pela engine do Docling  
- **tabelas:** Embedding de tabelas markdown/descrição de tabelas para poder buscar fontes de dados reais.  
- **imagens/gráficos:** Embedding de descrição+legenda de imagens para poder buscar fontes de dados reais.

Essas coleções são preenchidas de acordo com a explicação adiante:

1. (*scraper.py* e *banco\_metadados.py*) Todos os metadados de todos os documentos são adicionados em um banco de dados relacional âncora, com um estado de processamento pendente.  
2. (*docling\_pipeline.py* e *banco\_vetorial.py)* A pipeline busca por documentos pendentes no banco relacional e processa as informações dele para inserir nas coleções de pontos vetoriais: chunks, tabelas, imagens e recomendação, utilizando o Docling como ferramenta principal.

O arquivo *utils.py* possui funções auxiliares, como o crawler do site para buscar os arquivos PDFs a serem processados. Na pipeline do Docling, encontra-se também acesso a modelos de LLM e visão computacional para resumir e legendar tabelas e imagens, para que possam ser buscadas na query semântica do banco vetorial. Os arquivos de processamento se encontram na pasta *ingestor*, relacionado justamente com a etapa de ingestão.

## **3.2 Recuperação**  
A etapa de recuperação e geração do sistema é estruturada em uma pipeline composta por nove agentes especializados. Cada agente desempenha uma função específica dentro do fluxo de processamento, garantindo precisão, interpretabilidade e controle sobre o comportamento do sistema. A seguir, descreve-se cada um dos componentes da arquitetura.

1. **Agente de Classificação de Intenção**:
Identifica o propósito da consulta e classifica a intenção do usuário (resposta textual simples, RAG textual, busca por tabelas, busca por imagens/gráficos ou recomendação de leitura). Retorna um objeto estruturado com o tipo de resposta esperado, nível de detalhamento e eventuais requisitos, guiando toda a pipeline.

2. **Agente de Extração de Contexto**:
Extrai o núcleo semântico da pergunta a partir da query e da intenção. Gera termos-chave, filtros temáticos ou temporais e restrições por tipo de documento. Esse contexto orienta a geração de múltiplas queries especializadas para cada coleção vetorial.

3. **Primeira Camada de Recuperação**:
Executa a primeira busca nas coleções vetoriais (recomendações, chunks textuais, tabelas e imagens/gráficos). Cada coleção recebe uma query adaptada ao seu tipo de conteúdo. O resultado inicial serve de base para o refinamento da consulta.

4. **Agente de Refinamento de Query**:
Ajusta e aprimora as queries, sempre sem alterar a intenção original. Expande termos, desambigua conceitos e aplica filtros para melhorar o recall. Para cada coleção vetorial, gera uma versão especializada da query, adequada às características de cada embedding.

5. **Segunda Camada de Recuperação**:
Realiza uma nova rodada de busca usando as queries refinadas. Essa etapa corrige ruídos da primeira recuperação e retorna top-k documentos por coleção, agora com maior precisão e relevância.

6. **Agente de Fusão de Contexto**:
Unifica todas as evidências recuperadas. Deduplica trechos, consolida informações semelhantes, resolve inconsistências entre documentos e produz um contexto coerente e organizado. O resultado é um bloco de evidências limpo, pronto para geração da resposta.

7. **Agente de Interpretação de Dados**:
Ativado quando o usuário solicita tabelas, indicadores ou gráficos. Extrai valores, normaliza dados, identifica séries temporais e prepara estruturas numéricas diretamente derivadas das fontes recuperadas.

8. **Agente Gerador de Resposta Final**:
Produz a resposta final com base na intenção original, no contexto fundido e nos dados estruturados. Garante que o formato e o nível de detalhamento coincidam com o que o usuário solicitou.

9. **Agente Verificador de Fatos**:
Valida a resposta final, verificando se todas as informações estão presentes nas evidências recuperadas e se não há contradições ou alucinações. Caso detecte divergências, devolve o conteúdo para correção pelo gerador de resposta.

O fluxo geral da pipeline então fica da seguinte forma:

1. A consulta do usuário é analisada pelo Agente de Intenção  
2. O Agente de Contexto define palavras-chave e filtros.  
3. O sistema realiza a primeira recuperação de evidências.  
4. O Agente de Refinamento ajusta a query.  
5. Uma segunda recuperação busca evidências mais precisas.  
6. O Agente de Fusão consolida todas as informações.  
7. O Agente de Dados interpreta tabelas e gráficos (quando aplicável).  
8. O Gerador produz a resposta final.  
9. O Verificador garante aderência total às fontes.
