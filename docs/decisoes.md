# Log de Decisões e Trade-offs

> Registro incremental das decisões técnicas e metodológicas tomadas durante o projeto.
> Formato: **Decisão → Alternativas → Escolha → Motivo**.

---

## D-01 — Estratégia de truncação para documentos longos

**Data:** 05/06/2026
**Decisão:** Como lidar com acórdãos que excedem 512 tokens (limite do BERT)?
**Alternativas:**
- A) Truncação simples (primeiros 512 tokens) — ignora o Dispositivo final
- B) **Head+tail (128+384)** — captura início (contexto) e final (conclusão)
- C) Chunking + mean pooling — maior custo computacional
- D) Longformer — requer modelo diferente, maior complexidade

**Escolha:** **Head+tail (opção B)** — primeiros 128 + últimos 384 tokens
**Motivo:** Os primeiros tokens capturam o contexto do processo (órgão, objeto, período); os últimos tokens capturam o Dispositivo, que é a parte mais discriminativa para o desfecho. Respaldada por Sun et al. (2019), que demonstrou superioridade desta estratégia em documentos jurídicos longos.

---

## D-02 — Modelo Transformer base

**Data:** 05/06/2026
**Decisão:** Qual modelo Transformer usar como base para fine-tuning?
**Alternativas:**
- A) BERTimbau base (`neuralmind/bert-base-portuguese-cased`) — BERT pré-treinado em PT-BR geral
- B) **LegalBert-pt** (`dominguesm/legal-bert-base-cased-ptbr`) — BERTimbau re-treinado em corpus jurídico BR
- C) JurisBERT — focado em tarefas STS, não classificação

**Escolha:** **LegalBert-pt (opção B)** como modelo principal; BERTimbau como fallback
**Motivo:** LegalBert-pt foi pré-treinado em corpus jurídico brasileiro (STF, petições, decisões), o que o torna semanticamente mais próximo do domínio dos acórdãos do TCU. Maior aderência ao vocabulário jurídico reduz o número de épocas necessárias para convergência.

---

## D-03 — Métrica principal de avaliação

**Data:** 05/06/2026
**Decisão:** Qual métrica priorizar na avaliação e comparação?
**Alternativas:**
- A) Acurácia — enganosa com dados desbalanceados
- B) **F1-macro** — penaliza igualmente o desempenho ruim em classes minoritárias
- C) F1-weighted — pondera pelo suporte de cada classe

**Escolha:** **F1-macro (opção B)**
**Motivo:** Acórdãos irregulares são minoria em relação aos regulares (desbalanceamento esperado). F1-macro garante que o modelo seja avaliado com igual peso em todas as classes, penalizando falhas em "Irregular" — que é a classe de maior interesse para o radar jurimétrico.

---

## D-04 — Campo e critério do filtro temático

**Data:** 05/06/2026
**Decisão:** Qual campo e quais termos usar para o filtro temático de saúde/educação?
**Alternativas:**
- A) Filtro por órgão (UG, ministério) — pode perder casos intersetoriais
- B) Filtro por relator — sem relação direta com o tema
- C) **Filtro por texto (`sumario` ou `indexacao`)** — captura o assunto independente do órgão

**Escolha:** **Filtro textual (opção C)** com termos: "saúde", "SUS", "FNDE", "merenda",
"educação", "ministério da saúde", "secretaria de saúde", "secretaria de educação"
**Motivo:** Mais robusto — captura acórdãos de saúde/educação independente do órgão gestor ou relator. Termos cobrem tanto nível federal (SUS, FNDE) quanto estadual/municipal (secretarias).

---

## D-05 — Campo de label usado

**Data:** 05/06/2026
**Decisão:** Qual campo do CSV usar como rótulo de desfecho?
**Alternativas:**
- A) Campo `tipo` (contém apenas "Acórdão" — sem discriminação de desfecho)
- B) **Campo `situacao`** (contém "Irregular", "Regular com Ressalva", "Regular")
- C) Regex no `sumario` (fallback — cobertura parcial)

**Escolha:** **Campo `situacao` (opção B)**
**Motivo:** Inspeção do CSV (Etapa 2) confirmou que `situacao` contém o desfecho estruturado com 100% de preenchimento. O campo `tipo` contém apenas "Acórdão" (sem discriminação). O filtro `filtrar_tematico.py` prioriza `situacao`, com fallback automático para regex no `sumario`.

---

## D-06 — Campo de texto para o Transformer

**Data:** 05/06/2026
**Decisão:** Qual campo de texto usar como entrada do LegalBert-pt?
**Alternativas:**
- A) **Campo `sumario`** — disponível no CSV, cobre 100% dos registros, MVP válido
- B) Campo `voto` extraído de PDF via `pdfplumber` — mais discriminativo, maior custo operacional

**Escolha:** **Campo `sumario` (opção A) — MVP**
**Motivo:** O CSV do TCU não inclui o texto integral do Voto como coluna estruturada. Extrair o `voto` via `pdfplumber` exigiria baixar individualmente os PDFs de 2.000–4.000 acórdãos filtrados (~8–16 GB), inviável no prazo atual. O `sumario` é suficiente para o MVP. A extração de PDFs pode ser implementada como extensão futura (Estágio 2b).

---

*(Registrar novas decisões aqui durante o desenvolvimento)*
