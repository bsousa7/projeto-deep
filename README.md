# Jurimetria Preditiva em Acórdãos do TCU — Saúde e Educação

**IDP — Mestrado em Ciência de Dados e IA no Setor Público**  
**Disciplina:** Deep Learning e Processamento de Linguagem Natural | Modalidade 2 (NLP no Setor Público)

---

## Hipótese

> Um modelo de Deep Learning (LegalBert-pt com fine-tuning + truncação head+tail) supera o baseline clássico (TF-IDF + modelo linear) na predição do desfecho de acórdãos do TCU, medido por **F1-macro**.

---

## Objetivo

Treinar um classificador de texto que lê acórdãos do Tribunal de Contas da União (TCU) — áreas de Saúde e Educação — e prediz o **desfecho do processo**:

- **Irregular** — contas irregulares com multa/condenação  
- **Regular com Ressalva** — falhas formais sem dano ao erário  
- **Regular** — contas aprovadas com quitação

**Aplicação:** "radar jurimétrico" que permite a gestores públicos avaliar preventivamente o risco de condenação em processos licitatórios de saúde e educação.

---

## Fonte de Dados

**Portal de Dados Abertos do TCU** — download direto de CSV, sem scraping:  
`https://sites.tcu.gov.br/dados-abertos/jurisprudencia/`

| Arquivo | Tamanho aprox. | Período |
|---|---|---|
| `acordao-completo-2023.csv` | ~200 MB | 2023 |
| `acordao-completo-2024.csv` | ~300 MB | 2024 |

Após filtro temático (saúde/SUS/educação/FNDE): **~2.000–4.000 acórdãos**.

---

## Metodologia — Dois Estágios

### Estágio 1 — Baseline Clássico (TF-IDF)

| Componente | Configuração |
|---|---|
| Vetorização | TF-IDF, max_features=50.000, ngram_range=(1,2) |
| Modelos | LogisticRegression e LinearSVC |
| Input | Campo `sumario` (limpeza agressiva) |
| Métrica | **F1-macro** |

### Estágio 2 — Deep Learning (LegalBert-pt)

| Componente | Configuração |
|---|---|
| Modelo base | `dominguesm/legal-bert-base-cased-ptbr` |
| Truncação | **Head+tail**: 128 tokens início + 384 tokens fim = 512 total |
| Fine-tuning | 3–5 épocas, batch=16, lr=2e-5, warmup 10% |
| Input | Campo `sumario` / `voto` |
| Referência | Sun et al. (2019) — head+tail supera truncação simples em docs longos |

---

## Resultados

> Métricas com **CSVs reais do TCU** (2023–2024). Fine-tuning no Google Colab (GPU Tesla T4).

| Modelo | Campo | F1-macro | Acurácia | Obs. |
|---|---|---|---|---|
| TF-IDF + LogisticRegression | `SUMARIO` | **0.6705** | 92.6% | Baseline sólido |
| LegalBert-pt v1 (sem pesos) | `VOTO` | 0.3114 | 87.6% | Colapso para "Regular" |
| **LegalBert-pt v2 (class weights)** | `VOTO` | **0.4972** | 90.1% | +60% vs v1 |

**Análise dos resultados:**

O corpus de acórdãos TCU sobre saúde/educação é fortemente enviesado: **87.6% dos registros são Irregulares** após o filtro temático. Isso explica por que o transformer v1 colapsou para a classe majoritária.

Com class weights balanceados automaticamente, o LegalBert-pt v2 obteve:
- **Irregular**: F1 = 0.96 (excelente — classe bem representada)
- **Regular com Ressalva**: F1 = 0.53 (razoável — 7 amostras de teste)
- **Regular**: F1 = 0.00 (apenas 3 amostras no teste — insuficiente para aprender)

O baseline TF-IDF ainda supera o transformer (F1 0.67 vs 0.50), resultado coerente com a literatura para corpus pequenos (< 500 amostras de treino). O transformer requer ≥ 5.000 amostras para superar modelos lineares em classificação jurídica.

---

## Decisões Arquiteturais

| # | Decisão | Escolha | Motivo |
|---|---|---|---|
| D-01 | Truncação BERT | Head+tail (128+384) | Sun et al. (2019) |
| D-02 | Modelo Transformer | LegalBert-pt | Corpus jurídico BR |
| D-03 | Métrica principal | F1-macro | Penaliza desbalanceamento |
| D-04 | Filtro temático | Termos no `sumario` | Independente do órgão |
| D-05 | Campo de label | Regex em `ACORDAO`/`SUMARIO` | `SITUACAO` contém status processual, não veredicto |
| D-06 | Campo de texto | `SUMARIO` (baseline) / `VOTO` (BERT) | `VOTO` disponível diretamente no CSV (col. 29)! |

Ver detalhes em `docs/decisoes.md`.

---

## Estrutura do Repositório

```
tcu-jurimetria-nlp/
├── CLAUDE.md                        # contexto e guardrails do projeto
├── README.md                        # este arquivo
├── requirements.txt                 # dependências travadas
├── data/
│   ├── raw/                         # CSVs brutos TCU (não versionados — > 100 MB)
│   │   └── .gitkeep
│   ├── interim/                     # acordaos_filtrados.parquet
│   └── processed/                   # dados_processados.parquet + splits
├── src/
│   ├── aquisicao/
│   │   ├── baixar_csvs.py           # download dos CSVs anuais do TCU
│   │   └── gerar_mock.py            # dados sintéticos para testes de pipeline
│   ├── preprocessamento/
│   │   ├── filtrar_tematico.py      # filtro temático + extração de label
│   │   └── limpeza.py               # limpeza para TF-IDF e head+tail BERT
│   ├── modelos/
│   │   ├── baseline.py              # TF-IDF + LogReg / LinearSVC
│   │   └── transformer.py           # fine-tuning LegalBert-pt (requer GPU)
│   └── avaliacao/
│       └── metricas.py              # F1-macro, matrizes de confusão, LIME
├── notebooks/
│   └── 00_projeto_completo.ipynb    # entregável executado (orquestra tudo)
├── resultados/
│   ├── figuras/                     # EDA, matrizes de confusão, LIME plots
│   └── metricas.json                # resultados comparativos dos modelos
└── docs/
    ├── referencias.md               # 5+ domínio, 5+ técnica (ABNT)
    └── decisoes.md                  # log de decisões arquiteturais e trade-offs
```

---

## Como Reproduzir

### Ambiente local (EDA + Baseline)

```bash
git clone https://github.com/bsousa7/projeto-deep
cd projeto-deep
pip install -r requirements.txt

# Opção A — dados reais do TCU (requer conexão)
python src/aquisicao/baixar_csvs.py --anos 2023 2024

# Opção B — dados sintéticos para validar o pipeline
python src/aquisicao/gerar_mock.py

# Executar pipeline completo
python src/preprocessamento/filtrar_tematico.py --anos 2023 2024
jupyter notebook notebooks/00_projeto_completo.ipynb
```

### Fine-tuning do Transformer (Google Colab — GPU T4)

1. Abrir `notebooks/00_projeto_completo.ipynb` no Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Upload de `data/processed/dados_processados.parquet`
4. Executar células marcadas `# COLAB_GPU`

---

## Referências-chave

- Vaswani et al. (2017). *Attention is all you need.* NeurIPS.
- Devlin et al. (2019). *BERT: Pre-training of deep bidirectional Transformers.* NAACL.
- Souza, Nogueira & Lotufo (2020). *BERTimbau: Pretrained BERT models for Brazilian Portuguese.* BRACIS.
- Domingues (2022). *legal-bert-base-cased-ptbr.* HuggingFace Hub.
- Sun et al. (2019). *How to Fine-Tune BERT for Text Classification.* arXiv.

Ver lista completa em `docs/referencias.md`.
