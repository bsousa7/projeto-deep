# Jurimetria Preditiva em Acórdãos do TCU — Saúde e Educação

Classificador de texto que lê acórdãos do Tribunal de Contas da União (TCU) relacionados às
políticas de **Saúde** e **Educação** e prediz automaticamente se o processo resultou em
**contas irregulares** (condenação/multa) ou **contas regulares**.

Trabalho final da disciplina **Deep Learning e PLN** — Mestrado em Ciência de Dados e IA no
Setor Público (IDP), Modalidade 2 (NLP no Setor Público).

---

## Hipótese

> Um modelo de Deep Learning (Transformer com fine-tuning + truncação head+tail) supera o
> baseline clássico (TF-IDF + modelo linear) na predição do desfecho de acórdãos do TCU,
> medido por **F1-macro**.

---

## Fonte de dados

**Portal de Dados Abertos do TCU** — download direto de CSV, sem scraping:
https://sites.tcu.gov.br/dados-abertos/jurisprudencia/

- Arquivos: `acordao-completo-AAAA.csv` (2023–2024)
- Filtro temático aplicado: termos "saúde", "SUS", "FNDE", "educação", "merenda"
- Volume após filtro: ~2.000–4.000 acórdãos

---

## Estratégia em dois estágios

### Estágio 1 — Baseline (TF-IDF)
- Input: campo `sumario` do acórdão
- Modelo: TF-IDF (unigrama + bigrama) + Logistic Regression / LinearSVC
- Objetivo: piso de performance

### Estágio 2 — Deep Learning (principal)
- Input: campo textual do acórdão com **truncação head+tail** (128 primeiros + 384 últimos tokens)
- Modelo: `dominguesm/legal-bert-base-cased-ptbr` (LegalBert-pt) com cabeça de classificação
- Justificativa da truncação: captura o contexto inicial + o Dispositivo final do acórdão,
  as partes mais discriminativas para o desfecho (Sun et al., 2019)

---

## Resultados

| Modelo | F1-macro | F1 Irregular | F1 Regular |
|---|---|---|---|
| TF-IDF + LogReg (baseline) | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| LegalBert-pt head+tail | _(preencher)_ | _(preencher)_ | _(preencher)_ |
| **Ganho DL sobre baseline** | _(preencher)_ | — | — |

---

## Estrutura do repositório

```
tcu-jurimetria-nlp/
├── CLAUDE.md                        # contexto e regras do projeto
├── README.md                        # este arquivo
├── requirements.txt
├── data/
│   ├── raw/                         # CSVs brutos TCU (não versionados)
│   ├── interim/                     # corpus filtrado + label
│   └── processed/                   # splits treino/val/teste
├── src/
│   ├── aquisicao/baixar_csvs.py
│   ├── preprocessamento/
│   │   ├── filtrar_tematico.py
│   │   └── limpeza.py
│   ├── modelos/
│   │   ├── baseline.py
│   │   └── transformer.py
│   └── avaliacao/metricas.py
├── notebooks/
│   └── 00_projeto_completo.ipynb    # entregável principal (executado)
├── resultados/
│   ├── figuras/
│   └── metricas.json
└── docs/
    ├── referencias.md               # 5+ domínio, 5+ técnica (ABNT)
    └── decisoes.md                  # log de decisões arquiteturais
```

---

## Como executar

```bash
# 1. Ambiente virtual
python -m venv .venv && source .venv/bin/activate

# 2. Dependências
pip install -r requirements.txt

# 3. Baixar dados do TCU (executa src/aquisicao/baixar_csvs.py)
python src/aquisicao/baixar_csvs.py --anos 2023 2024

# 4. Notebook completo (de dentro de notebooks/)
jupyter notebook notebooks/00_projeto_completo.ipynb
```

> Fine-tuning do Transformer: executar no **Google Colab** (GPU T4 gratuita).
> Nunca carregar múltiplos CSVs anuais sem `usecols` — cada arquivo tem ~400 MB.

---

## Impacto esperado

Um "radar jurimétrico" preditivo: ao submeter a descrição de um processo licitatório de saúde ou
educação ao modelo, gestores municipais obtêm um **score de risco de condenação futura pelo TCU**,
podendo corrigir irregularidades antes da auditoria.

---

## Referências-chave

- Vaswani et al. (2017). Attention is all you need.
- Devlin et al. (2019). BERT: Pre-training of deep bidirectional Transformers.
- Souza, Nogueira & Lotufo (2020). BERTimbau: Pretrained BERT models for Brazilian Portuguese.
- Domingues (2022). legal-bert-base-cased-ptbr. HuggingFace.
- Sun et al. (2019). How to Fine-Tune BERT for Text Classification.

Ver lista completa em `docs/referencias.md`.

---

## Equipe

- _(nome 1)_
- _(nome 2)_
- _(nome 3)_
