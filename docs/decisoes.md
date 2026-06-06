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

**Data:** 05/06/2026 | **Atualizada:** 06/06/2026 (inspeção do CSV real)
**Decisão:** Qual campo do CSV usar como rótulo de desfecho?
**Alternativas:**
- A) Campo `TIPO` — contém "Acórdão" (tipo do documento, sem discriminação de desfecho)
- B) Campo `SITUACAO` — candidato; valores reais precisam ser verificados na amostra
- C) **Regex no campo `ACORDAO`** — dispositivo estruturado com texto da decisão
- D) Regex no campo `SUMARIO` — fallback com cobertura parcial

**Escolha:** **Regex no campo `ACORDAO` com fallback para `SUMARIO` (opções C/D)**
**Motivo:** Inspeção do CSV real do TCU (06/06/2026) revelou que o campo `SITUACAO` contém
o status processual (ex.: "BAIXADO", "EM TRAMITAÇÃO"), não o desfecho da auditoria.
O dispositivo da decisão ("contas irregulares", "contas regulares") consta no campo
`ACORDAO` (coluna 23) e/ou no `SUMARIO` (coluna 22). A função `_extrair_label()` em
`filtrar_tematico.py` tenta `tipo` → `situacao` → regex em `acordao` → regex em `sumario`.

---

## D-06 — Campo de texto para o Transformer

**Data:** 05/06/2026 | **Atualizada:** 06/06/2026 (inspeção do CSV real)
**Decisão:** Qual campo de texto usar como entrada do LegalBert-pt?
**Alternativas:**
- A) Campo `SUMARIO` — resumo curto, disponível no CSV, MVP válido
- B) Campo `VOTO` extraído de PDF via pdfplumber — mais discriminativo, alto custo
- C) **Campo `VOTO` direto do CSV** — disponível na coluna 29 do CSV real do TCU!

**Escolha:** **Campo `VOTO` do CSV (opção C) — sem necessidade de PDFs**
**Motivo:** A inspeção do CSV real do TCU (06/06/2026) revelou que o campo `VOTO`
(coluna 29) está disponível diretamente no arquivo CSV, contendo o texto integral do
Voto do Ministro. Isso elimina a necessidade de baixar PDFs separados e permite usar
o campo mais discriminativo para classificação sem custo adicional.
Pipeline atualizado: baseline usa `SUMARIO`; Transformer usa `VOTO`.

---

## D-07 — Estrutura real do CSV do TCU

**Data:** 06/06/2026
**Observação:** Inspeção do arquivo `acordao-completo-2023.csv` (445 MB) revelou:

| Característica | Valor |
|---|---|
| Separador | pipe (`\|`) |
| Encoding | UTF-8 (sem BOM) |
| Total de colunas | 33 |
| Nomes de colunas | MAIÚSCULO, sem acentos |
| Coluna identificador | `NUMACORDAO` (não `NUMEROACORDAO`) |
| Coluna de texto curto | `SUMARIO` (coluna 22) |
| Coluna de texto longo | `VOTO` (coluna 29) — disponível diretamente! |
| Coluna de palavras-chave | `ASSUNTO` (coluna 21) — útil para filtro temático |
| Coluna do dispositivo | `ACORDAO` (coluna 23) — fonte para extração de label |

O mapeamento correto (`NUMACORDAO` → `numeroAcordao` etc.) está documentado em
`_MAPA_COLUNAS_NORM` em `src/preprocessamento/filtrar_tematico.py`.

---

*(Registrar novas decisões aqui durante o desenvolvimento)*
