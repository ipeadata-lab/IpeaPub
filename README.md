# Sistema multiagente IpeaPub

TODO: adicionar descrição do sistema multiagente IpeaPub, seus objetivos e funcionalidades principais.

---

# ✅ **1. BANCOS DO SISTEMA (REESTRUTURADOS)**

---

# **📌 Banco 1 — SQL (Relacional — Metadados Mestrais)**

**Função:**
Manter metadados estruturados, garantir integridade e centralizar identificação de documentos.

**Campos principais (tabela `documentos`):**

| Campo             | Função                                         |
| ----------------- | ---------------------------------------------- |
| `id` (UUID)       | Identificador global, usado em todos os bancos |
| `titulo`          | Título do documento                            |
| `autores`         | Autores                                        |
| `ano`             | Ano de publicação                              |
| `tipo_conteudo`   | livro, relatório, nota técnica…                |
| `resumo`          | Resumo oficial                                 |
| `palavras_chave`  | Palavras-chave                                 |
| `link_pdf`        | URL para download                              |
| `hash_pdf`        | Hash usado para detectar atualizações          |
| `status_ingestao` | pendente / processando / completo              |
| `data_ingestao`   | timestamp                                      |
A divisão agora está **perfeita, limpa, modular e extremamente prática**. Vou reescrever e refinar cada coleção para garantir consistência, evitar redundâncias e deixar tudo pronto para implementação.

O formato abaixo já é adequado para Qdrant/Chroma/LanceDB.

---

# ✅ **Banco 1 (Relacional – metadados gerais)**


**Função:** armazenar metadados estruturados e garantir integridade para busca e referência.

**Tabela principal: `documentos`**
| Campo             | Tipo     | Descrição                                         |
| ----------------- | -------- | ------------------------------------------------- |
| `id`              | UUID     | Identificador global, usado em todos os bancos    |
| `titulo`          | Texto    | Título do documento                               |
| `autores`         | Texto    | Autores                                           |
| `ano`             | Inteiro  | Ano de publicação                                 |
| `tipo_conteudo`   | Texto    | livro, relatório, nota técnica…                   |
| `resumo`          | Texto    | Resumo oficial                                    |
| `palavras_chave`  | Texto    | Palavras-chave                                    |
| `link_pdf`        | Texto    | URL para download                                 |
| `link_uri`        | Texto    | URL da página do documento                        |
| `hash_pdf`        | Texto    | Hash usado para detectar atualizações             |
| `status_ingestao` | Texto    | pendente / processando / completo                 |
| `data_ingestao`   | Texto    | timestamp                                         |

---

# 🧠 **Banco 2 (Vetorial) – com 4 coleções independentes**

A separação por tipo de dado é **a melhor escolha possível**, tanto para performance quanto para manutenção.

---

## **📚 Coleção 1 — Recomendação de documentos (semântica "global")**

Objetivo: recomendar *documentos* inteiros, não trechos.

### **Campos sugeridos**

```json
{
  "doc_id": "uuid",
  "titulo": "...",
  "keywords": "...",
  "resumo": "...",
  "embedding": "vector"
}
```

### **Observações**

* *Não precisa guardar texto longo ou chunk.*
* O embedding deve ser **do resumo + título + palavras-chave**, concatenados.
* Usada quando o usuário pede:
  *"Quais documentos são mais relacionados a X?"*
  ou
  *"Recomende leituras complementares."*

---

## **📄 Coleção 2 — RAG textual (chunk-level com contexto de seção)**

Objetivo: recuperar *conteúdo textual específico*, contextualizado dentro da estrutura do documento.

### **Campos**

```json
{
  "doc_id": "uuid",
  "pagina": 12,
  "section_header": "Resultados e Discussão",
  "chunk_text": "...",
  "embedding": "vector"
}
```

### **Observações**

* Melhor incluir o header da seção se o docling fornecer.
* O `pagina` é extremamente útil para depurar respostas e reconstruir o contexto original.
* Essa coleção alimenta:

  * chat RAG
  * QA semântico
  * sumarizações segmentadas

---

## **🖼️ Coleção 3 — Imagens e Gráficos (multimodal RAG)**

Objetivo: recuperar informação visual — gráficos, diagramas, figuras — por texto ou imagem.

### **Campos**

```json
{
  "doc_id": "uuid",
  "pagina": 7,
  "caption": "Figura 3: Relação entre ...",
  "descricao_llm": "Descrição semântica detalhada da imagem",
  "bytes_imagem": "<binary data>",
  "embedding": "vector"
}
```

### **Notas importantes**

* `descricao_llm` melhora muito a busca semântica baseada em pergunta textual.
* Para o embedding, escolha:

  * CLIP / SigLIP (melhor para imagens), **ou**
  * embedding de `caption + descricao_llm` (melhor para RAG textual)
* Pode manter **ambos embeddings**, se quiser separar em dois vetores, mas não é obrigatório no começo.

---

## **📊 Coleção 4 — Tabelas (RAG especializado)**

Objetivo: permitir buscas envolvendo dados tabulares.

### **Campos**

```json
{
  "doc_id": "uuid",
  "pagina": 5,
  "caption": "Tabela 2: Estatísticas descritivas",
  "descricao_llm": "Resumo descritivo da tabela",
  "table_markdown": "Tabela convertida para Markdown ou CSV-like",
  "embedding": "vector"
}
```

### **Notas**

* `table_markdown` é extremamente útil:

  * é leve
  * fácil de usar em prompts
  * fácil de converter de volta para tabela
* Se a tabela for grande demais:

  * gere um **chunk textual da tabela** e use isso para embedding.

---