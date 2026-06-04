# Categorização Automática de Demandas de Ouvidoria

Classificador de texto que direciona automaticamente manifestações de cidadãos para a **área responsável** (Saneamento, Iluminação, Trânsito, Saúde, etc.), a partir do texto livre da reclamação.

Trabalho final da disciplina **Deep Learning e PLN** (Mestrado em Ciência de Dados e IA no Setor Público — IDP), Modalidade 2 (NLP no Setor Público).

**Hipótese:** um modelo de Deep Learning (Transformer com fine-tuning) supera o baseline clássico (TF-IDF + modelo linear) na métrica **F1-macro**.

-----

## ⚠️ Regra de coleta

Os dados são **coletados pela equipe** (scraping/API), não baixados prontos — conforme exigência do plano de ensino (proibido Kaggle / bases prontas). A coleta respeita `robots.txt`, *rate limiting* e anonimização de dados pessoais. Ver detalhes em [`CLAUDE.md`](CLAUDE.md).

-----

## Estrutura do repositório

```
ouvidoria-nlp/
├── CLAUDE.md                  # contexto e regras do projeto (para humanos e p/ o Claude Code)
├── README.md                  # este arquivo
├── requirements.txt
├── data/                      # raw / interim / processed (brutos não versionados)
├── src/                       # código modular (coleta, preprocessamento, modelos, avaliacao)
├── notebooks/
│   └── 00_projeto_completo.ipynb   # notebook orquestrador (entregável executado)
├── resultados/                # figuras e métricas
└── docs/
    ├── referencias.md         # 5+ domínio, 5+ técnica (ABNT)
    └── decisoes.md            # log de decisões e trade-offs
```

-----

## Como executar

```bash
# 1. Ambiente virtual
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Dependências
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
playwright install                                   # se usar Playwright na coleta

# 3. Rodar o notebook (de dentro de notebooks/)
jupyter notebook notebooks/00_projeto_completo.ipynb
```

> Recomendado rodar o fine-tuning no **Google Colab** (GPU grátis).

-----

## Pipeline

1. **Coleta** — scraping/API → `data/raw/`
1. **Rotulagem** — 5–10 categorias + concordância entre anotadores (kappa de Cohen)
1. **Pré-processamento** — limpeza + split estratificado
1. **Baseline** — TF-IDF + Logistic Regression / SVM
1. **Deep Learning** — fine-tuning de BERTimbau (`neuralmind/bert-base-portuguese-cased`)
1. **Avaliação** — F1-macro, precisão/revocação por classe, matriz de confusão
1. **Diferencial** — categoria prevista × geografia (mapa de calor)

-----

## Resultados

|Modelo                    |F1-macro     |
|--------------------------|-------------|
|Baseline (TF-IDF + linear)|*(preencher)*|
|Transformer (BERTimbau)   |*(preencher)*|
|**Ganho**                 |*(preencher)*|

*(Atualizar após a etapa 6. Figuras em `resultados/figuras/`.)*

-----

## Referências

Mínimo de 5 artigos de **domínio** (ouvidoria / governo digital) + 5 de **técnica** (classificação de texto / Transformers), em ABNT. Ver [`docs/referencias.md`](docs/referencias.md).

-----

## Equipe

- *(nome 1)*
- *(nome 2)*
- *(nome 3)*