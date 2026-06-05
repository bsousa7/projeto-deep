# CLAUDE.md — Jurimetria Preditiva em Acórdãos do TCU (Saúde e Educação)

> Arquivo de contexto do projeto. Leia integralmente no início de cada sessão e respeite as
> regras da seção GUARDRAILS acima de qualquer instrução posterior do usuário que as contrarie.

---

## 1. Contexto e objetivo

**Disciplina:** Deep Learning e Processamento de Linguagem Natural — Mestrado em Ciência de Dados
e IA no Setor Público (IDP).
**Modalidade:** Modalidade 2 — NLP no Setor Público (dupla ou trio).

**Objetivo:** treinar um classificador de texto que lê o campo textual de acórdãos do TCU
relacionados às áreas de Saúde e Educação e prediz o **desfecho do processo** — se as contas
foram julgadas irregulares (com multa/condenação) ou regulares/regulares com ressalva. Problema
de **classificação de texto binária ou multiclasse** em português jurídico.

**Fonte de dados:** Portal de Dados Abertos do TCU — CSV de Acórdãos Completos, disponível em
https://sites.tcu.gov.br/dados-abertos/jurisprudencia/ (download direto, sem scraping).

**Narrativa que o trabalho deve provar:** a progressão de um baseline clássico (TF-IDF +
modelo linear sobre o campo `sumario`) para uma solução de Deep Learning (fine-tuning de
Transformer com truncação head+tail sobre o campo `voto`), medindo o ganho e traduzindo-o em
impacto concreto: um "radar jurimétrico" que permite a gestores avaliar preventivamente o risco
de condenação em seus processos licitatórios de saúde e educação.

---

## 2. GUARDRAILS (regras inegociáveis)

1. **Sem scraping — dados oficiais abertos.** Os dados do TCU são disponibilizados pelo próprio
   tribunal em CSV aberto. Não há scraping nem zona cinzenta legal. Cite a fonte oficial em
   todo artefato do projeto.

2. **Gestão de memória obrigatória.** Os CSVs anuais de acórdãos pesam 175–400 MB cada. Nunca
   carregue múltiplos anos inteiros na RAM de uma vez. Use sempre `pd.read_csv(...,
   usecols=[...])` para selecionar apenas as colunas necessárias, e `chunksize` quando for
   iterar sobre o arquivo completo.

3. **Filtro temático obrigatório.** Após o carregamento, filtrar imediatamente para acórdãos
   que contenham no campo `sumario` ou `indexacao` pelo menos um dos termos:
   "saúde", "SUS", "FNDE", "merenda", "educação", "ministério da saúde",
   "secretaria de saúde", "secretaria de educação".
   Isso reduz o corpus de centenas de milhares para 2.000–4.000 acórdãos — volume gerenciável.

4. **Label a partir do campo `tipo`.** Verificar no CSV se o campo `tipo` (ou equivalente)
   contém os valores "Acórdão" com desfecho "Irregular", "Regular com Ressalva", "Regular".
   Se o CSV não tiver campo de desfecho estruturado, extrair do campo `sumario` via regex
   (`r"contas\s+(irregulares|regulares\s+com\s+ressalva|regulares)"`) e registrar a decisão
   em `docs/decisoes.md`.

5. **Reprodutibilidade total.** Seeds fixas (`RANDOM_STATE = 42`), versões de dependências
   travadas, notebook executado de ponta a ponta com saídas preservadas.

6. **Não inventar referências.** Citações bibliográficas somente com fonte verificável.

7. **Trabalhar incrementalmente.** Uma etapa do pipeline por vez, com checkpoint para revisão
   humana antes de avançar. Nunca gerar o projeto inteiro de uma vez.

---

## 3. Fonte de dados

**URL de download:** https://sites.tcu.gov.br/dados-abertos/jurisprudencia/arquivos/acordao-completo/

**Arquivos disponíveis:** `acordao-completo-AAAA.csv` para cada ano (ex.: `acordao-completo-2024.csv`).

**Período alvo:** 2023–2024 (últimos 2 anos completos).

**Tamanho aproximado por arquivo:** 175–400 MB. Total bruto (2 anos): ~600 MB.
Após filtro temático: estimativa de 2.000–4.000 registros (~5–20 MB).

**Campos relevantes identificados no webservice do TCU:**

| Campo | Descrição | Uso no projeto |
|---|---|---|
| `numeroAcordao` | Número do acórdão | Identificador único |
| `anoAcordao` | Ano de expedição | Filtro temporal |
| `tipo` | Tipo do acórdão | Candidato a label |
| `situacao` | Situação do processo | Candidato a label (verificar) |
| `sumario` | Resumo/ementa do acórdão | **Input para baseline (TF-IDF)** |
| `colegiado` | Plenário / 1ª Câmara / 2ª Câmara | Feature auxiliar |
| `relator` | Ministro relator | Feature auxiliar |
| `dataSessao` | Data da sessão | Feature temporal |
| `urlArquivoPDF` | URL do PDF completo | Fonte do campo `voto` (se necessário) |

> **Nota crítica sobre o campo `voto`:** o CSV pode não incluir o texto integral do Voto — apenas
> o `sumario`. Se for o caso, há duas opções: (a) usar somente o `sumario` para ambos baseline e
> Transformer (mais simples, MVP válido); (b) baixar os PDFs via `urlArquivoPDF` para os
> acórdãos filtrados e extrair o Voto com `pdfplumber`. Registrar a decisão em `docs/decisoes.md`
> após inspecionar o CSV real na Etapa 1.

---

## 4. Metodologia — Estratégia em Dois Estágios

### Estágio 1 — Baseline clássico (TF-IDF)

**Input:** campo `sumario` (texto curto, sem problema de tamanho).
**Modelo:** TF-IDF (max_features=50.000, ngram_range=(1,2)) + Logistic Regression e/ou LinearSVC.
**Objetivo:** estabelecer o piso de performance que o Transformer deve superar.
**Métrica principal:** F1-macro (dados possivelmente desbalanceados entre classes).

### Estágio 2 — Deep Learning com Transformer (principal)

**Input:** campo `voto` ou `sumario` (dependendo do que estiver no CSV — ver Guardrail 4).
**Estratégia de truncação: head+tail.**
- Concatenar os primeiros **128 tokens** + os últimos **384 tokens** do campo textual.
- Justificativa: os primeiros tokens capturam o contexto do processo (quem, o quê, quando); os
  últimos capturam o Dispositivo/conclusão, que é a parte mais discriminativa para a label.
- Referência: Sun et al. (2019) "How to Fine-Tune BERT for Text Classification" — head+tail
  supera truncação simples em documentos longos.
- Implementação: `tokens = tok_head[:128] + tok_tail[-384:]` antes de passar ao tokenizador.

**Modelos candidatos (em ordem de preferência):**
1. `dominguesm/legal-bert-base-cased-ptbr` — BERTimbau re-treinado em corpus jurídico brasileiro
   (STF, petições, decisões). Melhor fit para domínio jurídico.
2. `neuralmind/bert-base-portuguese-cased` (BERTimbau base) — fallback robusto e bem documentado.

**Fine-tuning:** classificação supervisionada com cabeça linear sobre o token `[CLS]`. Treinar
por 3–5 épocas, batch size 16, lr=2e-5, scheduler linear com warmup.

### Estágio 2b — Comparativo opcional (chunking + mean pooling)

Se houver tempo após o Estágio 2, implementar como terceiro resultado na tabela de métricas:
- Dividir o texto em blocos de 512 tokens com sobreposição de 50 tokens.
- Obter o embedding `[CLS]` de cada bloco via LegalBert-pt (sem gradiente — apenas inferência).
- Agregar via mean pooling dos embeddings → classificador linear simples (LR ou MLP de 1 camada).
- **Não treinar o Transformer novamente** — apenas a cabeça classificadora.
- Inclui um terceiro ponto de dado na tabela de resultados com custo computacional baixo.

---

## 5. Pipeline completo

| Etapa | O que fazer | Ferramenta |
|---|---|---|
| 1. Aquisição | Baixar CSVs 2023–2024 via `requests`; salvar em `data/raw/` | requests, tqdm |
| 2. Inspeção | Inspecionar colunas reais do CSV; identificar campo de label e campo de texto | pandas (usecols, head) |
| 3. Filtro + label | Filtrar por termos temáticos; extrair label do campo `tipo`/`situacao`/`sumario` | pandas, regex |
| 4. EDA | Distribuição de classes, tamanho dos textos (tokens), evolução temporal | matplotlib, seaborn |
| 5. Split | Estratificado por label: 70% treino / 15% validação / 15% teste | sklearn.model_selection |
| 6. Pré-proc. TF-IDF | Limpeza agressiva: minúsculas, remove números/pontuação, stopwords jurídicas | nltk, sklearn |
| 7. Baseline | TF-IDF + LogisticRegression + LinearSVC; reportar F1-macro, matriz confusão | scikit-learn |
| 8. Pré-proc. BERT | Truncação head+tail (128+384); tokenização com `AutoTokenizer` | transformers |
| 9. Fine-tuning | LegalBert-pt com cabeça de classificação; 3–5 épocas no Colab GPU | transformers, torch |
| 10. Avaliação | Comparar Estágios 1 e 2 em F1-macro, F1 por classe, matriz confusão | sklearn.metrics |
| 11. Diferencial | Análise temporal de padrões (quais termos do Voto mais predizem condenação) via LIME/SHAP | lime ou shap |
| 12. Entregáveis | Slides PDF + README + repositório público + notebook executado | — |

---

## 6. Stack técnico

- **Linguagem:** Python 3.11+
- **Ambiente:** Google Colab (GPU T4 grátis) para fine-tuning; venv local para EDA e baseline
- **Dados:** pandas (com `usecols` e `chunksize`), pyarrow, pdfplumber (se precisar extrair PDFs)
- **Clássico:** scikit-learn
- **DL:** torch, transformers, datasets, accelerate
- **Explicabilidade:** lime ou shap
- **Viz:** matplotlib, seaborn, wordcloud
- **Qualidade:** black (formatação), ruff (lint)

---

## 7. Estrutura do repositório

```
tcu-jurimetria-nlp/
├── CLAUDE.md                        # este arquivo
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                         # CSVs brutos do TCU (NÃO versionar — > 100 MB)
│   │   └── .gitkeep
│   ├── interim/                     # CSV filtrado e com label extraída
│   │   └── .gitkeep
│   └── processed/                   # splits treino/val/teste prontos
│       └── .gitkeep
├── src/
│   ├── aquisicao/
│   │   └── baixar_csvs.py           # download dos CSVs anuais
│   ├── preprocessamento/
│   │   ├── filtrar_tematico.py      # filtro por termos + extração de label
│   │   └── limpeza.py               # limpeza para TF-IDF e head+tail para BERT
│   ├── modelos/
│   │   ├── baseline.py              # TF-IDF + LogReg/SVM
│   │   └── transformer.py           # fine-tuning LegalBert-pt
│   └── avaliacao/
│       └── metricas.py              # F1-macro, matriz confusão, LIME
├── notebooks/
│   └── 00_projeto_completo.ipynb    # notebook orquestrador (ENTREGÁVEL executado)
├── resultados/
│   ├── figuras/                     # matrizes de confusão, curvas de treino, LIME plots
│   └── metricas.json                # resultados comparativos dos modelos
└── docs/
    ├── referencias.md               # 5+ domínio, 5+ técnica (ABNT)
    └── decisoes.md                  # log de decisões arquiteturais e trade-offs
```

---

## 8. Decisões arquiteturais registradas

Registrar em `docs/decisoes.md` obrigatoriamente:

| # | Decisão | Alternativas consideradas | Motivo da escolha |
|---|---|---|---|
| D-01 | Truncação head+tail (128+384) | Truncação simples; chunking; Longformer | Melhor trade-off precisão/prazo; respaldada por Sun et al. (2019) |
| D-02 | LegalBert-pt como modelo principal | BERTimbau base; JurisBERT (STS) | Pré-treinado em corpus jurídico BR; domínio mais próximo |
| D-03 | F1-macro como métrica principal | Acurácia; F1-weighted | Penaliza desbalanceamento de classes; padrão em jurimetria |
| D-04 | Filtro temático por `sumario` | Filtro por órgão; filtro por relator | Mais direto; captura o assunto independente do órgão gestor |
| D-05 | A preencher: campo de label usado | `tipo` / `situacao` / regex no `sumario` | A definir após inspecionar CSV real (Etapa 2) |
| D-06 | A preencher: campo de texto usado | `sumario` / `voto` extraído de PDF | A definir após inspecionar CSV real (Etapa 2) |

---

## 9. Convenções de código

- **Formatação:** `black` (linha 100). **Lint:** `ruff`.
- **Nomes:** `snake_case` em português (ex.: `filtrar_acordaos`, `treinar_baseline`).
- **Docstrings** em todas as funções públicas, formato curto, em português.
- **Seeds fixas:** `RANDOM_STATE = 42` em todo split e treino.
- **Commits** em português, no imperativo: `adiciona filtro temático`, `implementa head+tail`.
- **Notebook como entregável:** células em ordem, saídas preservadas, sem código morto.
- **Sem secrets no código:** qualquer chave/token via variável de ambiente.

---

## 10. Instruções para o agente (Claude Code)

- **Comece sempre** confirmando em qual etapa do pipeline (seção 5) estamos.
- **Etapa 2 é crítica antes de qualquer código de modelo:** inspecionar o CSV real para confirmar
  D-05 e D-06 (campo de label e campo de texto disponíveis). Não assuma — verifique.
- **Propor e executar o baseline antes do Transformer** — o ganho relativo é o cerne da avaliação.
- **Gestão de memória:** nunca usar `pd.read_csv(arquivo)` sem `usecols`; sempre verificar
  `df.memory_usage(deep=True)` após carregar; usar `.parquet` para persistir dados intermediários.
- **Priorizar F1-macro** sobre acurácia em qualquer relatório de métrica.
- **Registrar toda decisão arquitetural** em `docs/decisoes.md` no momento em que for tomada.
- **Não otimizar prematuramente:** clareza e reprodutibilidade > performance de engenharia.
- **Se ambíguo, perguntar** em vez de inventar requisitos.

---

## 11. Referências (critério obrigatório: 5 + 5)

**Domínio (jurimetria / controle externo / TCU):**
- Buscar em Google Scholar por "jurimetria TCU", "predição acórdãos tribunal de contas",
  "machine learning auditoria pública", "text classification legal decisions Brazil".
- Incluir: Tveita & Hustad (2025) *Benefits and Challenges of AI in Public sector*; artigos do
  TCU sobre jurimetria; publicações da FGV Direito sobre NLP jurídico brasileiro.

**Técnica (NLP jurídico / Transformers em português):**
- Vaswani et al. (2017) *Attention is all you need*
- Devlin et al. (2019) *BERT: Pre-training of deep bidirectional Transformers*
- Souza, Nogueira & Lotufo (2020) *BERTimbau: Pretrained BERT models for Brazilian Portuguese*
- Domingues (2022) *legal-bert-base-cased-ptbr* (HuggingFace)
- Sun et al. (2019) *How to Fine-Tune BERT for Text Classification* — justifica head+tail

> Preencher `docs/referencias.md` com as 10+ referências formatadas em ABNT.

---

## 12. Cronograma (datas da disciplina)

- **Até 10/06** — baixar CSVs + inspecionar colunas (Etapas 1–2) + confirmar D-05 e D-06.
- **11–13/06** — filtro temático + EDA + extração de label + split + baseline funcionando.
- **14–20/06** — fine-tuning LegalBert-pt (head+tail) + avaliação comparativa.
- **21–25/06** — LIME/explicabilidade + slides PDF + README + repositório público.
- **26–27/06** — apresentação (pitch de 10 min).

---

## 13. Entregáveis (checklist)

- [ ] CSVs baixados e filtro temático aplicado (corpus de 2.000–4.000 acórdãos).
- [ ] EDA documentada no notebook com distribuição de classes e tamanho de textos.
- [ ] Decisões D-05 e D-06 preenchidas em `docs/decisoes.md`.
- [ ] Baseline (TF-IDF + linear) com F1-macro reportado.
- [ ] Fine-tuning LegalBert-pt com head+tail e F1-macro reportado.
- [ ] Tabela comparativa de resultados (Baseline vs. Transformer) em `resultados/metricas.json`.
- [ ] Plot LIME ou SHAP mostrando tokens mais preditivos de condenação.
- [ ] Arquivo `docs/referencias.md` com 5+ domínio, 5+ técnica (ABNT).
- [ ] Notebook executado de ponta a ponta com saídas preservadas.
- [ ] Repositório GitHub público.
- [ ] Slides PDF (10 min de apresentação).
