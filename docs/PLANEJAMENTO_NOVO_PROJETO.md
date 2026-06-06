# Planejamento: Jurimetria Preditiva em Acórdãos do TCU (Novo Projeto)

> Documento de referência autossuficiente. Contém tudo que foi aprendido, consolidado e
> evoluído no projeto original, organizado como ponto de partida para uma nova execução
> sem dependência de histórico ou contexto anterior.

---

## 1. O Problema e o Objetivo

**Domínio:** Jurimetria preditiva no controle externo federal brasileiro.

**Tarefa:** Classificação de texto multiclasse — dado o texto de um acórdão do TCU nas
áreas de **Saúde** e **Educação**, predizer o **desfecho do processo**:
- `Irregular` — contas julgadas irregulares (multa e/ou condenação)
- `Regular com Ressalva` — aprovadas com ressalvas formais
- `Regular` — contas aprovadas sem restrições

**Fonte de dados:** Portal de Dados Abertos do TCU — arquivos CSV oficiais, sem scraping.
URL base: `https://sites.tcu.gov.br/dados-abertos/jurisprudencia/arquivos/acordao-completo/`
Arquivos: `acordao-completo-AAAA.csv` para cada ano.

**Impacto prático:** "Radar Jurimétrico" — ferramenta preventiva de risco que permite a
gestores de saúde e educação avaliar a probabilidade de condenação antes da auditoria,
possibilitando correções em contratos e licitações.

---

## 2. O que Aprendemos no Projeto Anterior (Lições Críticas)

### 2.1 Estrutura real do CSV do TCU
| Característica | Valor |
|---|---|
| Separador | pipe (`\|`) |
| Encoding | UTF-8 (sem BOM) |
| Total de colunas | 33 |
| Nomes de colunas | MAIÚSCULO, sem acentos |
| Coluna label | `SITUACAO` (Irregular / Regular com Ressalva / Regular) |
| Coluna texto curto | `SUMARIO` (coluna 22) — para baseline TF-IDF |
| Coluna texto longo | `VOTO` (coluna 29) — para Transformer |
| Coluna dispositivo | `ACORDAO` (coluna 23) — fallback para extração de label |

> **Atenção:** O campo `TIPO` contém "Acórdão" (tipo do documento, não o desfecho).
> O campo `SITUACAO` contém o DESFECHO e é o label correto.

### 2.2 Resultados reais obtidos (CSVs TCU 2020–2024)
| Modelo | Campo | F1-macro | Precisão macro | Revocação macro | Acurácia |
|---|---|---|---|---|---|
| TF-IDF + LogisticRegression | SUMARIO | **0.8404** | 0.9025 | 0.7981 | 96.5% |
| LegalBert-pt head+tail + class weights | VOTO | **0.8686** | 0.8357 | 0.9174 | 95.9% |

Corpus: ~534 acórdãos temáticos após filtro (dominados por 2023–2024 mesmo com 5 anos).
Desbalanceamento: ~88% Irregular, ~6% Regular com Ressalva, ~6% Regular.

### 2.3 Limitação metodológica crítica (assimetria de entradas)
O baseline usou `SUMARIO` e o Transformer usou `VOTO`. O ganho de +0.028 em F1-macro é
um composto de **efeito-arquitetura** e **efeito-campo**. No novo projeto, executar
a ablação 2×2 completa:

| Arquitetura \ Campo | SUMARIO | VOTO |
|---|---|---|
| TF-IDF + LogReg | Realizar ✓ | Realizar ✓ |
| LegalBert-pt head+tail | Realizar ✓ | Realizar ✓ |

### 2.4 Diversidade temporal > Volume
Expandir de 2 para 5 anos **não aumentou** o número de amostras filtradas (~534 em ambos
os casos), mas **melhorou muito** o desempenho (F1 baseline: 0.67 → 0.84; Transformer:
0.50 → 0.87). O motivo é diversidade temporal de vocabulário e contextos regulatórios,
não volume. Testar ablação por ano para confirmar.

### 2.5 Class weights são obrigatórios
Sem `compute_class_weight("balanced")` no Transformer, o modelo colapsa para a classe
majoritária (F1-macro=0.31, F1 de "Regular"=0.00). É condição necessária (mas não
suficiente) com corpus desbalanceado.

---

## 3. Decisões Arquiteturais (D-01 a D-08) — Já Tomadas, Não Revisitar

| # | Decisão | Escolha | Motivo |
|---|---|---|---|
| D-01 | Truncação para documentos longos | Head+tail (128+382) | Captura contexto + dispositivo; Sun et al. (2019) |
| D-02 | Modelo Transformer base | LegalBert-pt (`dominguesm/legal-bert-base-cased-ptbr`) | Pré-treinado em corpus jurídico BR |
| D-03 | Métrica principal | F1-macro | Penaliza desbalanceamento; padrão em jurimetria |
| D-04 | Filtro temático | Filtro por texto (SUMARIO/ASSUNTO) | Captura tema independente do órgão gestor |
| D-05 | Campo de label | `SITUACAO` do CSV | Contém desfecho real; `TIPO` contém tipo do documento |
| D-06 | Campo de texto | `SUMARIO` (baseline) + `VOTO` (Transformer) | VOTO disponível direto no CSV (col. 29) |
| D-07 | Estrutura CSV TCU | Pipe separator, UTF-8, 33 colunas MAIÚSCULAS | Inspecionado no CSV real de 2023 (445 MB) |
| D-08 | Arquiteturas alternativas | Manter LegalBert-pt | Longformer/BigBird/Mamba sem versão PT-BR jurídica |

---

## 4. Guardrails (Regras Inegociáveis)

1. **Sem scraping** — dados CSV oficiais do TCU, fonte aberta.
2. **Gestão de memória** — sempre `pd.read_csv(..., usecols=[...])`. CSVs pesam 175–445 MB.
3. **Filtro temático imediato** após carregamento.
4. **Seeds fixas** — `RANDOM_STATE = 42` em todo split e treino.
5. **Não inventar referências** — só citar fontes verificáveis.
6. **Incremental** — uma etapa por vez.

---

## 5. Stack Técnico (versões testadas e funcionais)

```
pandas>=2.2          pyarrow>=15.0        numpy>=1.26
scikit-learn>=1.4    torch>=2.2           transformers>=4.40
datasets>=2.19       accelerate>=0.30     peft>=0.10
nltk>=3.8            lime>=0.2            matplotlib>=3.8
seaborn>=0.13        wordcloud>=1.9       tqdm>=4.66
requests>=2.31       black>=24.0          ruff>=0.4
```

**Ambiente de treino:** Google Colab T4 GPU (gratuito) para fine-tuning.
**Ambiente local:** Python 3.11+ para EDA, baseline e pré-processamento.

---

## 6. Estrutura de Diretórios

```
projeto-jurimetria-tcu/
├── CLAUDE.md                         ← este arquivo adaptado
├── README.md
├── requirements.txt
├── .gitignore                        ← ignorar data/raw/*.csv, .venv/, __pycache__/
├── data/
│   ├── raw/                          ← CSVs brutos do TCU (NÃO versionar)
│   ├── interim/                      ← acordaos_filtrados.parquet
│   └── processed/                    ← splits treino/val/teste
├── src/
│   ├── aquisicao/
│   │   └── baixar_csvs.py            ← download com retry e retomada
│   ├── preprocessamento/
│   │   ├── filtrar_tematico.py       ← filtro + extração de label
│   │   └── limpeza.py                ← TF-IDF clean + head+tail BERT
│   ├── modelos/
│   │   ├── baseline.py               ← TF-IDF + LogReg/SVM + K-Fold
│   │   └── transformer.py            ← LegalBert-pt + LoRA + kfold_com_lora
│   └── avaliacao/
│       └── metricas.py               ← métricas, custo, LIME, otimizar_threshold
├── notebooks/
│   └── 00_projeto_completo.ipynb     ← notebook orquestrador (entregável executado)
├── resultados/
│   ├── figuras/
│   └── metricas.json
└── docs/
    ├── referencias.md
    └── decisoes.md
```

---

## 7. Pipeline Completo (12 Etapas)

| Etapa | O que fazer | Arquivo | Status recomendado |
|---|---|---|---|
| 1 | Download CSVs 2020–2024 | `src/aquisicao/baixar_csvs.py` | Baixar em Colab — arquivos > 100 MB |
| 2 | Inspecionar colunas reais | `filtrar_tematico.py: inspecionar_colunas()` | Confirmar sep=`\|`, SITUACAO, VOTO |
| 3 | Filtro temático + label | `filtrar_tematico.py: combinar_anos()` | Salvar `data/interim/acordaos_filtrados.parquet` |
| 4 | EDA | notebook células 12–14 | Distribuição de classes, tamanho de textos |
| 5 | Split estratificado | `limpeza.py: dividir_dados()` | 70/15/15, seed=42 |
| 6 | Pré-proc TF-IDF | `limpeza.py: limpar_coluna(modo='tfidf')` | Aplicar em SUMARIO e VOTO |
| 7 | Baseline + K-Fold | `baseline.py: treinar_baseline_kfold()` | F1-macro por fold + IC 95% |
| 8 | Pré-proc BERT | `limpeza.py: limpar_coluna(modo='bert')` | Head+tail via AutoTokenizer |
| 9 | Fine-tuning LegalBert-pt | `transformer.py: treinar_transformer()` | Colab T4, 3–5 épocas, class weights |
| 9b | LoRA + K-Fold (Aula 08) | `transformer.py: kfold_com_lora()` | Opcional — IC 95% para Transformer |
| 10 | Avaliação comparativa | `metricas.py: comparar_modelos()` | F1-macro, F1 por classe, matriz confusão |
| 10b | Otimizar threshold (Aula 07) | `metricas.py: otimizar_threshold()` | Minimizar custo financeiro FN×10 + FP×1 |
| 11 | Explicabilidade LIME | `metricas.py: explicar_com_lime()` | Tokens preditivos de condenação |
| 12 | Entregáveis | slides, artigo, README | PDF + notebook executado + repo público |

---

## 8. Checklist de Entregáveis

### Pipeline de Dados
- [ ] CSVs 2020–2024 baixados em `data/raw/` (confirmar tamanho ~200–445 MB cada)
- [ ] Separador e encoding inspecionados (D-07: pipe, UTF-8)
- [ ] Filtro temático aplicado → `data/interim/acordaos_filtrados.parquet`
- [ ] Label extraída do campo `SITUACAO` (D-05 confirmada)
- [ ] n de acórdãos filtrados documentado (esperado: 300–800 por ano temático)

### EDA
- [ ] Distribuição de classes (esperado: ~88% Irregular, ~6% cada minority)
- [ ] Histograma de comprimento de SUMARIO e VOTO (tokens)
- [ ] Evolução temporal (acórdãos por ano e desfecho)
- [ ] `resultados/figuras/eda_visao_geral.png` gerada

### Ablação Simétrica 2×2 (novo projeto DEVE completar)
- [ ] TF-IDF + SUMARIO — F1-macro reportado
- [ ] TF-IDF + VOTO — F1-macro reportado
- [ ] LegalBert-pt + SUMARIO — F1-macro reportado
- [ ] LegalBert-pt + VOTO — F1-macro reportado
- [ ] Tabela 3 preenchida; efeito-campo vs. efeito-arquitetura isolados

### Modelos
- [ ] Baseline K-Fold (5 folds): F1-macro médio ± std + IC 95% (ambos campos)
- [ ] LegalBert-pt fine-tuning: F1-macro no hold-out (com class weights)
- [ ] LoRA K-Fold (opcional): F1-macro por fold + IC 95%
- [ ] Métricas por classe: sem colapso de "Regular" (F1>0 em todas as classes)
- [ ] `resultados/metricas.json` com comparativo completo

### Alinhamento ao Negócio (Aula 07)
- [ ] Otimização de threshold executada (`otimizar_threshold()`)
- [ ] Threshold ótimo documentado (custo_fn=10, custo_fp=1)
- [ ] FN rate antes/depois do threshold ótimo reportados

### Explicabilidade
- [ ] Plot LIME nos tokens preditivos de condenação (em Colab GPU)
- [ ] `resultados/figuras/lime_explicabilidade.png` com Transformer real

### Documentação e Entregáveis Finais
- [ ] D-05 e D-06 confirmadas em `docs/decisoes.md`
- [ ] `docs/referencias.md` com ≥ 5 domínio + ≥ 5 técnica (ABNT)
- [ ] Notebook executado de ponta a ponta com saídas preservadas
- [ ] Repositório GitHub público
- [ ] Slides PDF (≤ 14 slides, ≈ 10 min)
- [ ] Artigo científico (ABNT NBR 6022:2018, ≥ 6 seções)

---

## 9. Armadilhas Conhecidas (Não Repetir)

| Armadilha | Sintoma | Correção |
|---|---|---|
| Usar `TIPO` como label | Label = "Acórdão" para todos os registros | Usar `SITUACAO` |
| Sem `usecols` no read_csv | MemoryError em CSVs de 400 MB | Sempre especificar `usecols=[...]` |
| Sem class weights | F1-macro=0.31, colapso para "Irregular" | `compute_class_weight("balanced")` |
| K-Fold no Transformer sem LoRA | 5× tempo de treino, inviável no T4 | Usar `kfold_com_lora()` |
| Comparar baseline vs. Transformer com campos diferentes | Ganho = efeito-campo + efeito-arquitetura | Ablação 2×2 obrigatória |
| Corpus de 2 anos sem "Regular" suficiente | F1=0 na classe "Regular" | Usar ≥ 5 anos para diversidade |
| Citar referências sem verificar | Violação do Guardrail 6 | Só citar fontes listadas abaixo |

---

## 10. Parâmetros Recomendados (Testados e Funcionais)

### Filtro Temático
```python
TERMOS = ["saúde", "SUS", "FNDE", "merenda", "educação",
          "ministério da saúde", "secretaria de saúde", "secretaria de educação"]
```

### Baseline TF-IDF
```python
TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
```

### LegalBert-pt Fine-Tuning
```python
modelo_nome = "dominguesm/legal-bert-base-cased-ptbr"  # fallback: neuralmind/bert-base-portuguese-cased
max_head, max_tail = 128, 382                           # soma = 510 + 2 especiais = 512
epocas, batch_size, lr = 5, 16, 1e-5
weight_decay, warmup = 0.01, 0.10
balancear_automatico = True                             # compute_class_weight("balanced")
early_stopping_patience = 2                             # monitorar F1-macro no val
```

### LoRA (PEFT) — Parâmetros para K-Fold no T4
```python
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                        # rank das matrizes
    lora_alpha=16,              # escala
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
)
# Resultado: ~300K parâmetros treináveis de 110M (-99.7%)
```

### Otimização de Threshold
```python
from src.avaliacao.metricas import otimizar_threshold
resultado = otimizar_threshold(
    y_true=y_test,
    proba_irregular=modelo.predict_proba(X_test)[:, idx_irregular],
    custo_fn=10.0,   # FN = 10× mais caro que FP (irregularidade não detectada)
    custo_fp=1.0,
)
# threshold_otimo tipicamente entre 0.25 e 0.40
```

---

## 11. Referências Bibliográficas (ABNT)

### Domínio — Jurimetria / Controle Externo / TCU

BRASIL. Tribunal de Contas da União. **Portal de Dados Abertos do TCU: Acórdãos Completos**. Brasília: TCU, 2024. Disponível em: <https://sites.tcu.gov.br/dados-abertos/jurisprudencia/>. Acesso em: 05 jun. 2026.

TVEITA, Sondre; HUSTAD, Eli. **Benefits and challenges of AI in the public sector**. In: HAWAII INTERNATIONAL CONFERENCE ON SYSTEM SCIENCES, 58., 2025, Maui. *Proceedings...* Honolulu: University of Hawaii, 2025. p. 1–10.

ALETRAS, Nikolaos et al. **Predicting judicial decisions of the European Court of Human Rights: A Natural Language Processing perspective**. *PeerJ Computer Science*, v. 2, e93, 2016.

MEDVEDEVA, Masha; VOLS, Michel; WIELING, Martijn. **Using machine learning to predict decisions of the European Court of Human Rights**. *Artificial Intelligence and Law*, v. 28, n. 2, p. 237–266, 2020.

LAGE-FREITAS, André et al. **Predicting Brazilian court decisions**. *PeerJ Computer Science*, v. 8, e904, 2022.

### Técnica — NLP Jurídico / Transformers / PEFT

VASWANI, Ashish et al. **Attention Is All You Need**. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 30., 2017, Long Beach. *Proceedings...* Red Hook: Curran Associates, 2017. p. 5998–6008.

DEVLIN, Jacob et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. In: CONFERENCE OF THE NORTH AMERICAN CHAPTER OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS, 2019, Minneapolis. *Proceedings...* Stroudsburg: ACL, 2019. p. 4171–4186.

SOUZA, Fábio; NOGUEIRA, Rodrigo; LOTUFO, Roberto. **BERTimbau: Pretrained BERT Models for Brazilian Portuguese**. In: INTELLIGENT SYSTEMS — BRACIS, 9., 2020, Rio Grande. *Proceedings...* Cham: Springer, 2020. p. 403–417.

DOMINGUES, Luciano. **legal-bert-base-cased-ptbr: BERT model pre-trained on Brazilian legal corpus**. HuggingFace Hub, 2022. Disponível em: <https://huggingface.co/dominguesm/legal-bert-base-cased-ptbr>. Acesso em: 05 jun. 2026.

SUN, Chi; QIU, Xipeng; XU, Yuanbin; HUANG, Xuanjing. **How to Fine-Tune BERT for Text Classification?** In: CHINESE COMPUTATIONAL LINGUISTICS, 18., 2019, Kunming. *Proceedings...* Cham: Springer, 2019. p. 194–206.

HU, Edward J. et al. **LoRA: Low-Rank Adaptation of Large Language Models**. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 2022, online. *Proceedings...* [S.l.]: OpenReview, 2022.

LIN, Tsung-Yi et al. **Focal Loss for Dense Object Detection**. In: IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION, 2017, Veneza. *Proceedings...* [S.l.]: IEEE, 2017. p. 2980–2988.

PEDREGOSA, Fabian et al. **Scikit-learn: machine learning in Python**. *Journal of Machine Learning Research*, Cambridge, v. 12, p. 2825–2830, 2011.

RIBEIRO, Marco Tulio; SINGH, Sameer; GUESTRIN, Carlos. **"Why should I trust you?": explaining the predictions of any classifier**. In: ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 22., 2016, San Francisco. *Proceedings…* New York: ACM, 2016. p. 1135–1144.

WOLF, Thomas et al. **Transformers: state-of-the-art natural language processing**. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING: SYSTEM DEMONSTRATIONS, 2020, online. *Proceedings…* Stroudsburg: ACL, 2020. p. 38–45.

---

## 12. Ordem de Execução Recomendada

```
Semana 1:   Baixar CSVs → inspecionar → filtrar → EDA → confirmar D-05/D-06
Semana 2:   Split + limpeza → baseline K-Fold (ambos campos) → matriz ablação parcial
Semana 3:   Fine-tuning LegalBert-pt (Colab) → otimização de threshold
Semana 4:   LoRA K-Fold (Colab) → LIME → completar ablação 2×2
Semana 5:   Artigo + slides + notebook executado + repo público
```

---

*Fonte dos dados: Portal de Dados Abertos do TCU — https://sites.tcu.gov.br/dados-abertos/jurisprudencia/*
