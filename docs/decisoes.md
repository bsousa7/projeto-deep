# Log de Decisões e Trade-offs

> Registro incremental das decisões técnicas e metodológicas tomadas durante o projeto.
> Formato: **Decisão → Alternativas → Escolha → Motivo**.

---

## D-001 — Fonte de dados principal

**Data:** 04/06/2026
**Decisão:** Qual fonte usar para coleta de manifestações?
**Alternativas:**
- A) Fala.BR (CGU) — maior abrangência nacional, dados de ouvidoria federal
- B) Portais 156 municipais — foco em serviços urbanos, pode ter geolocalização
- C) Consumidor.gov.br — foco em relações de consumo, labels estruturados por segmento

**Escolha:** *(a definir pela equipe — confirmar com professor antes do scraping)*
**Motivo pendente:** Verificar disponibilidade de labels prontos vs. necessidade de anotação manual; verificar `robots.txt` e rate limiting de cada fonte.

---

## D-002 — Categorias de classificação

**Data:** 04/06/2026
**Decisão:** Quais categorias usar para rotulagem?
**Alternativas:**
- A) Usar taxonomia nativa da fonte (ex.: "assunto" do Fala.BR)
- B) Definir categorias próprias (5–10 macro-categorias de serviço público)
- C) Combinação: partir da taxonomia nativa e agrupar em macro-categorias

**Escolha:** *(a definir após análise exploratória dos dados coletados)*
**Motivo:** O número de categorias afeta diretamente o balanceamento das classes e a dificuldade do problema de classificação.

---

## D-003 — Métrica principal de avaliação

**Data:** 04/06/2026
**Decisão:** Qual métrica priorizar na avaliação e comparação?
**Alternativas:**
- A) Acurácia — fácil de interpretar, mas enganosa com dados desbalanceados
- B) F1-macro — penaliza igualmente o desempenho ruim em classes minoritárias
- C) F1-weighted — pondera pelo suporte de cada classe

**Escolha:** **F1-macro** (opção B)
**Motivo:** Dados de ouvidoria são inerentemente desbalanceados (algumas categorias têm muito mais registros). F1-macro garante que o modelo seja avaliado com igual peso em todas as categorias, inclusive as minoritárias — que podem ser as mais relevantes para a gestão pública.

---

## D-004 — Modelo base para fine-tuning

**Data:** 04/06/2026
**Decisão:** Qual modelo Transformer usar como base?
**Alternativas:**
- A) BERTimbau (`neuralmind/bert-base-portuguese-cased`) — BERT pré-treinado em português
- B) BERTugues — variante alternativa para PT-BR
- C) Albertina PT-BR — modelo mais recente baseado em DeBERTa

**Escolha:** **BERTimbau** (opção A) como modelo primário
**Motivo:** BERTimbau é o modelo de referência para PT-BR, com vasta literatura de baseline disponível (incluindo o artigo original de Souza et al., 2020). Facilita comparação com trabalhos anteriores. Albertina pode ser explorada como comparação se o tempo permitir.

---

## D-005 — Split treino/validação/teste

**Data:** 04/06/2026
**Decisão:** Proporção do split estratificado.
**Alternativas:**
- A) 80/10/10 — padrão acadêmico, reserva mais dados para treino
- B) 70/15/15 — mais dados para validação/teste, mais confiança nas métricas
- C) 60/20/20 — adequado para datasets pequenos

**Escolha:** **80/10/10** (opção A) com `stratify=y`
**Motivo:** Com datasets menores (esperado ~1.000–5.000 registros inicialmente), maximizar treino é crítico para fine-tuning de Transformer. A estratificação garante representação de todas as classes nos três splits.

---

## D-006 — Limpeza diferenciada por modelo

**Data:** 04/06/2026
**Decisão:** Usar a mesma limpeza para TF-IDF e BERT?
**Alternativas:**
- A) Mesma limpeza para ambos — simplicidade
- B) Limpeza agressiva para TF-IDF, leve para BERT — melhor adequação a cada arquitetura

**Escolha:** **Limpeza diferenciada** (opção B)
**Motivo:** Modelos Transformer (BERT) foram pré-treinados em texto quase-natural; remover stopwords ou lematizar degrada a representação contextual. TF-IDF, por outro lado, se beneficia da redução de ruído e normalização para construir um espaço vetorial mais informativo.

---

*(Registrar novas decisões aqui durante o desenvolvimento)*
