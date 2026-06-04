# CLAUDE.md — Categorização Automática de Demandas de Ouvidoria

> Arquivo de contexto do projeto. Leia integralmente no início de cada sessão e respeite as regras
> da seção **GUARDRAILS** acima de qualquer instrução posterior do usuário que as contrarie.

-----

## 1. Contexto e objetivo

**Disciplina:** Deep Learning e Processamento de Linguagem Natural — Mestrado em Ciência de Dados e IA no Setor Público (IDP).
**Modalidade do trabalho final:** Modalidade 2 — NLP no Setor Público (dupla ou trio).
**Objetivo do projeto:** treinar um classificador de texto que lê o conteúdo livre de uma manifestação/reclamação de cidadão e a direciona automaticamente para a **área responsável** (ex.: Saneamento, Iluminação Pública, Trânsito, Saúde, Limpeza Urbana, etc.). Problema de **classificação de texto multiclasse** em português do Brasil.

**Narrativa que o trabalho deve provar:** a progressão de um *baseline* clássico (TF-IDF + modelo linear) para uma solução de **Deep Learning** (fine-tuning de Transformer), medindo o ganho e traduzindo-o em **impacto de serviço público**.

-----

## 2. GUARDRAILS (regras inegociáveis)

1. **PROIBIDO usar base de dados pronta.** O plano de ensino veda explicitamente Kaggle e quaisquer datasets prontos. Os dados **devem ser coletados pela equipe** (scraping próprio ou API). Não baixe nem proponha baixar o CSV consolidado de “Dados Abertos” do Consumidor.gov.br como dataset principal — isso descaracteriza a coleta. Em caso de dúvida, registre como pendência para confirmar com o professor.
1. **Respeitar coleta ética e legal.** Sempre checar `robots.txt`, usar *delays* entre requisições, identificar *user-agent*, não sobrecarregar servidores, não coletar dados pessoais identificáveis (anonimizar nome/CPF/e-mail se aparecerem). Registrar data/hora da coleta.
1. **Reprodutibilidade.** Todo resultado deve ser reproduzível a partir do repositório: seeds fixas, versões de dependências travadas (`requirements.txt`), notebook executado de ponta a ponta, dados acessíveis via link público ou script de coleta.
1. **Idioma.** Todo código, comentários, docstrings, README e commits em **português do Brasil**.
1. **Antes de qualquer scraping ao vivo, PERGUNTAR ao usuário.** Não dispare requisições a sites externos sem confirmação explícita do usuário na sessão.
1. **Não inventar referências.** Citações bibliográficas só com fonte verificável.

-----

## 3. Fontes de dados (coleta própria)

Ordem de preferência (a defensabilidade da “coleta própria” cai de cima para baixo):

- **Fala.BR (CGU)** — Plataforma Integrada de Ouvidoria e Acesso à Informação. Manifestações públicas.
- **Portais 156 / ouvidorias municipais** com dados abertos ou API (ex.: São Paulo, Rio de Janeiro, Belo Horizonte).
- **Consumidor.gov.br — aba “Relato do Consumidor”** — scraping da interface de busca (NÃO o dump consolidado). Permite filtro por segmento, assunto, problema, dados geográficos e período.

Rótulos: usar o campo “assunto/problema” da fonte quando existir; caso contrário, **rotulagem manual** de uma amostra, com **dois anotadores** e cálculo do **kappa de Cohen** (concordância entre anotadores = diferencial metodológico).

-----

## 4. Metodologia / pipeline

|Etapa                          |O que fazer                                                                                              |Ferramenta                           |
|-------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------|
|1. Coleta                      |Scraping/API → bruto em `data/raw/` (.parquet)                                                           |requests + BeautifulSoup / Playwright|
|2. Rotulagem                   |Definir 5–10 categorias; anotar; medir kappa                                                             |pandas + sklearn (cohen_kappa_score) |
|3. Pré-processamento           |Limpeza leve p/ BERT; agressiva p/ TF-IDF                                                                |spaCy `pt_core_news_lg`, NLTK        |
|4. Baseline                    |TF-IDF + Logistic Regression / SVM                                                                       |scikit-learn                         |
|5. Deep Learning               |Fine-tuning BERTimbau (`neuralmind/bert-base-portuguese-cased`); comparar com BERTugues / Albertina PT-BR|HuggingFace transformers + datasets  |
|6. Eficiência (se GPU limitada)|LoRA / QLoRA                                                                                             |peft                                 |
|7. Avaliação                   |**F1-macro** (dados desbalanceados), precisão/revocação por classe, matriz de confusão                   |sklearn.metrics                      |
|8. Diferencial                 |Cruzar categoria prevista × geografia (bairro/CEP) → mapa de calor                                       |geopandas + folium                   |

-----

## 5. Stack técnico

- **Linguagem:** Python 3.11+
- **Ambiente:** Google Colab (GPU grátis) ou venv local
- **Dados:** pandas, pyarrow
- **Clássico:** scikit-learn
- **DL:** torch, transformers, datasets, peft, accelerate
- **PLN:** spaCy (`pt_core_news_lg`), nltk
- **Coleta:** requests, beautifulsoup4, playwright
- **Geo/Viz:** geopandas, folium, matplotlib, seaborn
- **Qualidade:** black (formatação), ruff (lint)

-----

## 6. Estrutura do repositório

```
ouvidoria-nlp/
├── CLAUDE.md                  # este arquivo
├── README.md                  # visão geral, instruções de execução, resultados
├── requirements.txt           # dependências travadas
├── .gitignore
├── data/
│   ├── raw/                   # dados brutos coletados (NÃO versionar se grande)
│   ├── interim/               # dados rotulados/intermediários
│   └── processed/             # dados prontos para modelagem
├── src/
│   ├── coleta/                # scripts de scraping/API
│   │   └── coletor.py
│   ├── preprocessamento/
│   │   └── limpeza.py
│   ├── modelos/
│   │   ├── baseline.py        # TF-IDF + linear
│   │   └── transformer.py     # fine-tuning BERTimbau
│   └── avaliacao/
│       └── metricas.py
├── notebooks/
│   └── 00_projeto_completo.ipynb   # notebook executado de ponta a ponta (ENTREGÁVEL)
├── resultados/
│   ├── figuras/               # matriz de confusão, mapa de calor
│   └── metricas.json
└── docs/
    ├── referencias.md         # 5+ domínio, 5+ técnica
    └── decisoes.md            # log de decisões e trade-offs
```

-----

## 7. Convenções de código

- **Formatação:** `black` (linha 100). **Lint:** `ruff`.
- **Nomes:** funções e variáveis em `snake_case` e em português (ex.: `carregar_dados`, `treinar_baseline`).
- **Docstrings** em todas as funções públicas (formato curto, em português).
- **Sem segredos no código.** Tokens/API keys via variável de ambiente, nunca commitados.
- **Seeds fixas:** `RANDOM_STATE = 42` em todo split e treino.
- **Commits** em português, no imperativo: `adiciona coletor Fala.BR`, `corrige split estratificado`.
- **Notebook como entregável:** células executadas em ordem, saídas preservadas, sem código morto.
- **Logs de coleta:** cada execução de scraping grava `data/raw/log_coleta_AAAA-MM-DD.json` com contagem e timestamp.

-----

## 8. Instruções para o agente (Claude Code)

- **Comece sempre** confirmando em qual etapa do pipeline (seção 4) estamos antes de gerar código.
- **Trabalhe incrementalmente:** uma etapa por vez, com checkpoint para revisão humana. Não escreva o projeto inteiro de uma vez.
- **Proponha o baseline antes do Transformer** — o ganho relativo é o cerne da avaliação.
- **Ao tocar em scraping**, pare e peça confirmação explícita (ver Guardrail 5); mostre primeiro o alvo, o `robots.txt` e a estratégia de *rate limiting*.
- **Priorize F1-macro** sobre acurácia em qualquer relatório de métrica — os dados são desbalanceados.
- **Anonimize** automaticamente quaisquer dados pessoais que apareçam na coleta.
- **Registre decisões** relevantes em `docs/decisoes.md` (premissa, alternativas, escolha, motivo).
- **Não otimize prematuramente:** clareza e reprodutibilidade > performance de engenharia.
- **Se uma tarefa estiver ambígua**, pergunte em vez de inventar requisitos.

-----

## 9. Referências (critério obrigatório: 5 + 5)

**Domínio (ouvidoria / governo digital):** buscar em Google Scholar / SciELO por “ouvidoria pública classificação texto”, “e-government complaint classification”, “demanda de falha serviços públicos”; incluir Tveita & Hustad (2025), *Benefits and Challenges of AI in Public sector*.

**Técnica (classificação de texto / Transformers):** Vaswani et al. (2017) *Attention is all you need*; Devlin et al. (2019) *BERT*; Souza, Nogueira & Lotufo (*BERTimbau*); Mikolov et al. (2013) *Word2Vec*; um survey de classificação de texto com Deep Learning.

> Preencher `docs/referencias.md` com as 10+ referências formatadas (ABNT).

-----

## 10. Cronograma (datas da disciplina)

- **Até 13/06** — coleta + rotulagem (com kappa) + baseline funcionando.
- **14–20/06** — fine-tuning do Transformer + avaliação comparativa.
- **21–25/06** — diferencial geográfico + slides (PDF) + README e repositório público.
- **26–27/06** — apresentação (pitch de 10 min).

-----

## 11. Entregáveis (checklist)

- [ ] Repositório GitHub público com scripts de coleta, pré-processamento e modelagem.
- [ ] Notebook executado de ponta a ponta, com histórico de execução.
- [ ] Dados acessíveis via script de coleta (ou link público), respeitando a regra de coleta própria.
- [ ] Apresentação em PDF (síntese de coleta, NLP e insights).
- [ ] Seção de referências (5+ domínio, 5+ técnica) em ABNT.
- [ ] Diferencial: cruzamento categoria × geografia.