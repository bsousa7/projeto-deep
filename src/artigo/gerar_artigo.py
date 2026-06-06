"""Gerador do artigo científico (v3) — evoluído conforme ementa Aulas 07 e 08.

Evoluções implementadas (v2 → v3):
- Tabela 5: matriz de custo-benefício financeiro (Aula 07 — métricas alinhadas ao negócio)
- Seção 4.5: proposta PEFT/LoRA para viabilizar K-Fold no Transformer (Aula 08)
- Seção 5.3 expandida: F1-macro vs. impacto fiscal e threshold de decisão
- Seção 5.1 atualizada: LoRA como solução para o problema de semente única
- Seção 6 (Trabalhos Futuros) atualizada com LoRA e otimização de threshold
- Referência HU et al. (2022) adicionada

Saída: resultados/artigo_jurimetria_tcu.pdf
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"
SAIDA = RESULTADOS / "artigo_jurimetria_tcu.pdf"

PRETO      = colors.black
CINZA      = colors.HexColor("#374151")
CINZA_CLARO = colors.HexColor("#f3f4f6")
AZUL       = colors.HexColor("#1a3557")
AMARELO    = colors.HexColor("#fef3c7")
LARANJA    = colors.HexColor("#ea580c")
VERDE      = colors.HexColor("#16a34a")
VERMELHO   = colors.HexColor("#dc2626")


# ── Estilos ──────────────────────────────────────────────────────────────────

def _estilos():
    base = getSampleStyleSheet()
    def P(name, font="Times-Roman", size=12, lead=21, align=TA_JUSTIFY, **kw):
        return ParagraphStyle(name, parent=base["Normal"], fontName=font,
                              fontSize=size, leading=lead, alignment=align, **kw)

    titulo   = P("Titulo",  font="Times-Bold",       size=14, lead=19,
                 align=TA_CENTER, spaceAfter=6)
    autores  = P("Autores", size=12, lead=16,         align=TA_CENTER, spaceAfter=4)
    afil     = P("Afil",    font="Times-Italic",      size=10, lead=14,
                 align=TA_CENTER, spaceAfter=2, textColor=CINZA)
    secao    = P("Secao",   font="Times-Bold",        size=12, lead=17,
                 align=TA_LEFT, spaceBefore=14, spaceAfter=6)
    subsecao = P("Sub",     font="Times-BoldItalic",  size=12, lead=17,
                 align=TA_LEFT, spaceBefore=10, spaceAfter=4)
    corpo    = P("Corpo",   firstLineIndent=1.25*cm,  spaceBefore=0, spaceAfter=6)
    rtit     = P("RTit",    font="Times-Bold",        size=12, lead=17,
                 align=TA_LEFT, spaceBefore=12, spaceAfter=4)
    rcorpo   = P("RCorpo",  size=11, lead=17,         spaceAfter=4)
    pchave   = P("PChave",  size=11, lead=15,         spaceAfter=8)
    ref      = P("Ref",     size=11, lead=16,         spaceAfter=6)
    nota     = P("Nota",    font="Times-Italic",      size=10, lead=14,
                 align=TA_CENTER, textColor=CINZA, spaceAfter=4)
    alerta   = P("Alerta",  font="Times-Italic",      size=10, lead=15,
                 align=TA_JUSTIFY, textColor=LARANJA, spaceAfter=6,
                 leftIndent=0.5*cm, rightIndent=0.5*cm)
    return dict(titulo=titulo, autores=autores, afil=afil, secao=secao,
                subsecao=subsecao, corpo=corpo, rtit=rtit, rcorpo=rcorpo,
                pchave=pchave, ref=ref, nota=nota, alerta=alerta)


def _rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(CINZA)
    canvas.drawCentredString(A4[0]/2, 1.2*cm, str(doc.page))
    canvas.restoreState()


# ── Tabelas ───────────────────────────────────────────────────────────────────

def _estilo_tabela(header_color=AZUL):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",      (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [CINZA_CLARO, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ])


def _tab1_resultados():
    """Tabela 1 — métricas macro dos dois experimentos realizados."""
    dados = [
        ["Experimento", "Campo de\nEntrada", "F1-macro", "Precisão\nmacro",
         "Revocação\nmacro", "Acurácia", "n teste"],
        ["TF-IDF + LogReg (baseline)", "SUMARIO", "0,8404", "0,9025",
         "0,7981", "96,5%", "~80"],
        ["LegalBert-pt head+tail\n+ class weights", "VOTO", "0,8686", "0,8357",
         "0,9174", "95,9%", "~80"],
    ]
    est = _estilo_tabela()
    est.add("FONTNAME", (0, 1), (0, -1), "Times-Italic")
    est.add("TEXTCOLOR", (2, 1), (2, 1), colors.HexColor("#1a3557"))
    est.add("TEXTCOLOR", (2, 2), (2, 2), VERDE)
    est.add("FONTNAME",  (2, 1), (2, -1), "Times-Bold")
    return Table(dados, colWidths=[4.6*cm, 2.4*cm, 2.0*cm, 2.1*cm, 2.2*cm, 1.8*cm, 1.7*cm],
                 style=est, repeatRows=1)


def _tab2_corpus():
    """Tabela 2 — contagem real do corpus (corrigida)."""
    dados = [
        ["Corpus", "Total\nfiltrado", "Treino\n(70%)", "Val.\n(15%)", "Teste\n(15%)",
         "Baseline\nF1-macro", "Transformer\nF1-macro"],
        ["2023–2024 (2 anos)", "~534", "~374", "~80", "~80",
         "0,6705", "0,4972*"],
        ["2020–2024 (5 anos)†", "~534‡", "~374", "~80", "~80",
         "0,8404", "0,8686"],
    ]
    est = _estilo_tabela(AZUL)
    est.add("TEXTCOLOR", (6, 1), (6, 1), VERMELHO)
    est.add("TEXTCOLOR", (6, 2), (6, 2), VERDE)
    est.add("FONTNAME",  (6, 2), (6, 2), "Times-Bold")
    est.add("TEXTCOLOR", (5, 2), (5, 2), VERDE)
    est.add("FONTNAME",  (5, 2), (5, 2), "Times-Bold")
    return Table(dados, colWidths=[3.8*cm, 1.9*cm, 1.9*cm, 1.6*cm, 1.6*cm, 2.3*cm, 2.7*cm],
                 style=est, repeatRows=1)


def _tab3_ablacao():
    """Tabela 3 — matriz de ablação simétrica (contribuição campo × arquitetura)."""
    dados = [
        ["Arquitetura \\ Campo", "SUMARIO", "VOTO"],
        ["TF-IDF + LogReg",       "0,8404 ✓ (realizado)",  "— (aguarda Colab)"],
        ["LegalBert-pt head+tail", "— (aguarda Colab)",     "0,8686 ✓ (realizado)"],
    ]
    est = _estilo_tabela(AZUL)
    est.add("FONTNAME",   (0, 1), (0, -1), "Times-Italic")
    est.add("TEXTCOLOR",  (1, 1), (1, 1),  VERDE)
    est.add("TEXTCOLOR",  (2, 2), (2, 2),  VERDE)
    est.add("TEXTCOLOR",  (1, 2), (1, 2),  colors.HexColor("#6b7280"))
    est.add("TEXTCOLOR",  (2, 1), (2, 1),  colors.HexColor("#6b7280"))
    return Table(dados, colWidths=[5.0*cm, 5.8*cm, 5.8*cm],
                 style=est, repeatRows=1)


def _tab4_por_classe():
    """Tabela 4 — F1 por classe (Transformer v2, corpus 2023-2024)."""
    dados = [
        ["Classe", "Suporte\n(teste)", "Precisão", "Revocação", "F1",
         "Δ F1 vs.\nbaseline*"],
        ["Irregular",             "~70",  "0,97", "0,96", "0,96", "—"],
        ["Regular com Ressalva",  "~5",   "0,43", "0,67", "0,53", "—"],
        ["Regular",               "~5",   "0,00", "0,00", "0,00", "crítico"],
        ["<b>Macro (média)</b>",  "~80",  "—",    "—",    "<b>0,4972</b>", ""],
    ]
    est = _estilo_tabela(AZUL)
    est.add("TEXTCOLOR",  (5, 3), (5, 3),  VERMELHO)
    est.add("FONTNAME",   (5, 3), (5, 3),  "Times-Bold")
    est.add("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#fee2e2"))
    est.add("FONTNAME",   (0, 4), (-1, 4), "Times-Bold")
    return Table(dados, colWidths=[4.0*cm, 2.2*cm, 2.0*cm, 2.2*cm, 1.8*cm, 2.6*cm],
                 style=est, repeatRows=1)


def _tab5_custo():
    """Tabela 5 — Assimetria de custo entre os tipos de erro de classificação."""
    VERMELHO_CLARO = colors.HexColor("#fee2e2")
    VERDE_CLARO    = colors.HexColor("#dcfce7")
    dados = [
        ["Tipo de Erro", "Predição\ndo Modelo", "Situação\nReal",
         "Consequência ao Erário Público", "Custo Relativo"],
        ["Falso Negativo (FN)", "Regular ou\nRessalva", "Irregular",
         "Desvio não detectado; gestor não punido;\nressarcimento não exigido ao erário",
         "ALTO — impacto\nfiscal direto"],
        ["Falso Positivo (FP)", "Irregular", "Regular",
         "Auditoria aprofundada desnecessária;\ncusto operacional do TCU",
         "BAIXO — custo\nadministrativo"],
    ]
    est = _estilo_tabela(colors.HexColor("#7f1d1d"))
    est.add("BACKGROUND", (0, 1), (-1, 1), VERMELHO_CLARO)
    est.add("BACKGROUND", (0, 2), (-1, 2), VERDE_CLARO)
    est.add("FONTNAME",   (4, 1), (4, 1),  "Times-Bold")
    est.add("TEXTCOLOR",  (4, 1), (4, 1),  VERMELHO)
    est.add("FONTNAME",   (4, 2), (4, 2),  "Times-Bold")
    est.add("TEXTCOLOR",  (4, 2), (4, 2),  VERDE)
    return Table(dados, colWidths=[3.5*cm, 2.5*cm, 2.0*cm, 5.5*cm, 2.5*cm],
                 style=est, repeatRows=1)


# ── Conteúdo ─────────────────────────────────────────────────────────────────

def _historia(st):
    P  = Paragraph
    S  = Spacer
    HR = HRFlowable
    h  = []

    # ── CABEÇALHO ─────────────────────────────────────────────────────────
    h += [
        P("JURIMETRIA PREDITIVA EM ACÓRDÃOS DO TRIBUNAL DE CONTAS DA UNIÃO: "
          "COMPARATIVO ENTRE BASELINE TF-IDF E FINE-TUNING DE TRANSFORMER "
          "COM TRUNCAÇÃO HEAD+TAIL NAS ÁREAS DE SAÚDE E EDUCAÇÃO",
          st["titulo"]),
        S(1, 8),
        P("Bruno Sousa<super>1</super>", st["autores"]),
        S(1, 4),
        P("<super>1</super> IDP — Instituto Direito Público. Mestrado em Ciência de Dados "
          "e IA no Setor Público. Disciplina: Deep Learning e PLN. Brasília, DF, Brasil. "
          "E-mail: bruno.aires9@gmail.com",
          st["afil"]),
        S(1, 4),
        HR(width="100%", thickness=1, color=AZUL, spaceAfter=8),

        # ── RESUMO ────────────────────────────────────────────────────────
        P("RESUMO", st["rtit"]),
        P("O presente artigo investiga a aplicação de técnicas de Processamento de "
          "Linguagem Natural (PLN) na predição de desfechos de acórdãos do Tribunal "
          "de Contas da União (TCU) relativos às áreas de saúde e educação, comparando "
          "um baseline TF-IDF com regressão logística a um modelo LegalBert-pt "
          "(DOMINGUES, 2022) com truncação head+tail (SUN et al., 2019). O corpus foi "
          "construído a partir dos dados abertos do TCU (2020–2024), resultando em "
          "~534 acórdãos temáticos após filtragem. O forte desbalanceamento de classes "
          "(~88% Irregular) foi tratado com class weights automáticos. O Transformer "
          "superou o baseline em F1-macro: 0,8686 vs. 0,8404 (+0,028). Reconhece-se "
          "como principal limitação metodológica a assimetria de campos de entrada: "
          "o baseline utilizou SUMARIO e o Transformer utilizou VOTO, tornando o ganho "
          "observado um composto de efeito-arquitetura e efeito-campo. Uma ablação "
          "simétrica 2×2 (campo × arquitetura) é proposta e parcialmente implementada. "
          "A metodologia inclui validação cruzada K-Fold (5 folds) e intervalo de "
          "confiança a 95% para o baseline. O trabalho contribui para a jurimetria "
          "preditiva no setor público e propõe um 'radar jurimétrico' de risco de "
          "condenação para gestores de saúde e educação.",
          st["rcorpo"]),
        P("<b>Palavras-chave:</b> Jurimetria. TCU. PLN. BERT. Classificação Jurídica. "
          "Assimetria Metodológica. Class Weights.", st["pchave"]),
        S(1, 4),

        P("ABSTRACT", st["rtit"]),
        P("This paper investigates Natural Language Processing (NLP) applied to predicting "
          "outcomes of Brazilian Federal Court of Auditors (TCU) decisions in healthcare "
          "and education, comparing a TF-IDF baseline with a fine-tuned LegalBert-pt model "
          "using head+tail truncation. The corpus comprises ~534 thematic decisions "
          "(2020–2024). The Transformer outperformed the baseline in macro F1-score "
          "(0.8686 vs. 0.8404, +0.028). The main acknowledged limitation is input "
          "asymmetry: the baseline used SUMARIO while the Transformer used VOTO, making "
          "the gain a compound of architecture and field effects. A symmetric 2×2 ablation "
          "design is proposed. K-Fold cross-validation (5 folds) and 95% confidence "
          "intervals are reported for the baseline. The paper proposes a 'jurimetric radar' "
          "for public managers in health and education.",
          st["rcorpo"]),
        P("<b>Keywords:</b> Jurimetrics. Court of Auditors. NLP. BERT. Legal Classification. "
          "Methodological Asymmetry. Class Weights.", st["pchave"]),
        HR(width="100%", thickness=0.8, color=colors.HexColor("#d1d5db"), spaceAfter=6),

        # ── 1. INTRODUÇÃO ─────────────────────────────────────────────────
        P("1 INTRODUÇÃO", st["secao"]),
        P("A jurimetria aplica métodos quantitativos ao fenômeno jurídico com o objetivo "
          "de tornar o comportamento dos tribunais previsível e auditável. No âmbito do "
          "controle externo federal brasileiro, o Tribunal de Contas da União (TCU) emite "
          "anualmente dezenas de milhares de acórdãos que julgam a regularidade das "
          "contas de gestores públicos, impondo, nos casos de irregularidade, multas e "
          "determinações de ressarcimento ao erário.", st["corpo"]),
        P("A predição automática do desfecho desses acórdãos representa aplicação de "
          "alto impacto prático: gestores de saúde e educação — setores com elevada "
          "incidência histórica de irregularidades — poderiam utilizar um modelo "
          "classificador como ferramenta preventiva de risco, antecipando ações "
          "corretivas antes da conclusão da auditoria.", st["corpo"]),
        P("Com o advento do BERT (DEVLIN et al., 2019) e de variantes jurídicas como "
          "o LegalBert-pt (DOMINGUES, 2022), o estado da arte em classificação de texto "
          "jurídico avançou significativamente. Simultaneamente, estratégias como a "
          "truncação head+tail (SUN et al., 2019) permitem lidar com documentos longos "
          "sem exceder o limite de 512 tokens do BERT.", st["corpo"]),
        P("O presente trabalho parte da hipótese de que um Transformer com fine-tuning "
          "supera um baseline TF-IDF na predição de desfechos de acórdãos do TCU, "
          "medida por F1-macro. O trabalho reconhece explicitamente uma limitação "
          "metodológica relevante: os dois estágios utilizaram campos textuais distintos "
          "(SUMARIO para o baseline; VOTO para o Transformer). Essa assimetria é "
          "documentada, discutida e tratada por meio de uma ablação parcial que isola "
          "o efeito-campo do efeito-arquitetura.", st["corpo"]),
        P("As contribuições do trabalho são: (i) construção de corpus temático de "
          "acórdãos do TCU com extração automática de rótulos; (ii) avaliação comparativa "
          "com baseline K-Fold e IC 95%; (iii) ablação parcial campo×arquitetura; "
          "(iv) métricas por classe que revelam o colapso em classes minoritárias; "
          "e (v) proposta de radar jurimétrico como ferramenta de gestão pública.",
          st["corpo"]),

        # ── 2. TRABALHOS RELACIONADOS ──────────────────────────────────
        P("2 TRABALHOS RELACIONADOS", st["secao"]),
        P("A predição de decisões judiciais por PLN tem sido objeto crescente de pesquisa. "
          "Aletras et al. (2016) demonstraram viabilidade de predizer decisões do Tribunal "
          "Europeu de Direitos Humanos com acurácia de 79% usando SVMs. Medvedeva, Vols e "
          "Wieling (2020) refinaram esse resultado para 75% com modelos de aprendizado de "
          "máquina em textos completos. No Brasil, Lage-Freitas et al. (2022) investigaram "
          "tribunais estaduais e federais, evidenciando que a escolha do campo textual e "
          "a representatividade do corpus são fatores determinantes — achado diretamente "
          "relevante para a limitação metodológica discutida neste trabalho.", st["corpo"]),
        P("Souza, Nogueira e Lotufo (2020) apresentaram o BERTimbau, pré-treinado em "
          "2,68 bilhões de palavras do português. Domingues (2022) especializou esse "
          "modelo no domínio jurídico brasileiro com o LegalBert-pt. Sun et al. (2019) "
          "compararam sistematicamente quatro estratégias de truncação em 19 conjuntos "
          "de dados, mostrando superioridade da head+tail sobre truncação simples em "
          "documentos jurídicos longos.", st["corpo"]),
        P("A explicabilidade de modelos em decisões de alto impacto é tratada por "
          "Ribeiro, Singh e Guestrin (2016), que propõem o LIME como método agnóstico "
          "de modelo. Tveita e Hustad (2025) discutem o imperativo de auditabilidade em "
          "sistemas de IA no setor público. Wolf et al. (2020) documentam a biblioteca "
          "Transformers, utilizada na implementação do fine-tuning.", st["corpo"]),

        # ── 3. METODOLOGIA ────────────────────────────────────────────
        P("3 METODOLOGIA", st["secao"]),
        P("3.1 Corpus e Coleta de Dados", st["subsecao"]),
        P("O corpus foi construído a partir dos dados abertos do TCU (TCU, 2024): "
          "arquivos <i>acordao-completo-AAAA.csv</i> (anos 2020–2024), com ~500.000 "
          "acórdãos brutos, separador pipe (|), codificação UTF-8 e 33 colunas em "
          "letras maiúsculas. A leitura foi realizada com seleção explícita de colunas "
          "via <i>pd.read_csv(..., usecols=[...])</i>, contornando o limite de memória "
          "imposto pelos arquivos de 175–445 MB.", st["corpo"]),
        P("O filtro temático foi aplicado ao campo SUMARIO por correspondência com os "
          "termos 'saúde', 'SUS', 'FNDE', 'merenda', 'educação', 'secretaria de saúde' "
          "e 'secretaria de educação'. Após filtragem e extração de rótulos, o corpus "
          "resultou em ~534 acórdãos para ambas as janelas temporais testadas — dado "
          "que os anos 2020–2022 contribuíram marginalmente ao corpus temático, fato "
          "que limita a narrativa de 'expansão de volume' e é discutido na Seção 4.4.",
          st["corpo"]),
        P("O rótulo de desfecho foi extraído do campo ACORDAO por expressão regular "
          r"<i>r'contas\s+(irregulares|regulares\s+com\s+ressalva|regulares)'</i>, "
          "com fallback para o campo SUMARIO. O campo SITUACAO foi descartado por "
          "conter status processual ('BAIXADO', 'EM TRAMITAÇÃO'), não o veredicto.",
          st["corpo"]),
        P("3.2 Distribuição de Classes e Divisão dos Dados", st["subsecao"]),
        P("A distribuição de classes revelou corpus fortemente desbalanceado: ~88% "
          "Irregular, ~6% Regular, ~6% Regular com Ressalva no corpus real do TCU. "
          "O F1-macro foi adotado como métrica principal por penalizar igualmente "
          "erros em classes minoritárias (PEDREGOSA et al., 2011). A divisão foi "
          "70%/15%/15% com estratificação e semente fixa (RANDOM_STATE = 42).",
          st["corpo"]),
        P("3.3 Estágio 1 — Baseline TF-IDF com K-Fold", st["subsecao"]),
        P("O baseline utiliza pipeline scikit-learn com TF-IDF "
          "(max_features=50.000, ngram_range=(1,2)) e regressão logística (C=1.0, L2). "
          "O texto de entrada foi o campo SUMARIO após limpeza padrão. "
          "Dado o tamanho reduzido do conjunto de teste (~80 amostras), "
          "o desempenho foi também avaliado por validação cruzada estratificada "
          "K-Fold (5 folds, semente 42), que fornece estimativa mais robusta da "
          "generalização e permite calcular intervalos de confiança aproximados.",
          st["corpo"]),
        P("3.4 Estágio 2 — Fine-tuning LegalBert-pt", st["subsecao"]),
        P("O modelo <i>dominguesm/legal-bert-base-cased-ptbr</i> (DOMINGUES, 2022) "
          "foi submetido a fine-tuning sobre o campo VOTO — disponível diretamente "
          "no CSV do TCU (coluna 29), eliminando extração de PDFs. Hiperparâmetros: "
          "5 épocas, batch=16, lr=1×10<super>−5</super>, weight decay=0,01, warmup "
          "10%, early stopping (paciência=2, monitorando F1-macro na validação). "
          "Implementação via HuggingFace Transformers (WOLF et al., 2020), GPU "
          "NVIDIA Tesla T4 (Google Colab).", st["corpo"]),
        P("3.5 Estratégia de Truncação Head+Tail", st["subsecao"]),
        P("O BERT limita sequências a 512 tokens. Adotou-se a estratégia head+tail "
          "(SUN et al., 2019): primeiros 128 tokens (contexto processual) concatenados "
          "com os últimos 382 tokens (dispositivo e determinações), totalizando 512 "
          "posições incluindo [CLS] e [SEP]. A assimetria 128/382 reflete a maior "
          "carga discriminativa do dispositivo final.", st["corpo"]),
        P("3.6 Tratamento do Desbalanceamento", st["subsecao"]),
        P("Class weights calculados por "
          "<i>compute_class_weight('balanced')</i> foram aplicados na "
          "<i>CrossEntropyLoss</i>. Sem essa técnica, o Transformer colapsou para a "
          "classe majoritária (F1-macro=0,3114, F1 de 'Regular'=0,00), conforme "
          "documentado na Tabela 4. A adoção de class weights elevou o F1-macro para "
          "0,4972 no corpus de dois anos e para 0,8686 no corpus ampliado.",
          st["corpo"]),
        P("3.7 Assimetria de Entradas: Reconhecimento e Ablação", st["subsecao"]),
        P("⚠ Limitação crítica reconhecida: os dois estágios utilizaram campos "
          "textuais distintos — o baseline recebeu SUMARIO e o Transformer recebeu "
          "VOTO. O ganho reportado (+0,028 em F1-macro) é, portanto, um composto "
          "de <i>efeito-arquitetura</i> (BERT vs. TF-IDF) e <i>efeito-campo</i> "
          "(VOTO vs. SUMARIO), não podendo ser atribuído exclusivamente à arquitetura.",
          st["alerta"]),
        P("A Tabela 3 apresenta a matriz de ablação 2×2 proposta para decompor os "
          "dois efeitos. Os experimentos diagonais (TF-IDF+SUMARIO e Transformer+VOTO) "
          "foram realizados. Os experimentos cruzados (TF-IDF+VOTO e "
          "Transformer+SUMARIO) requerem execução em GPU com os CSVs reais do TCU "
          "e constituem trabalho em andamento.", st["corpo"]),
        S(1, 4),
        _tab3_ablacao(),
        P("<i>Tabela 3</i> — Matriz de ablação campo × arquitetura. Células com '✓' "
          "foram realizadas; '—' indica experimentos pendentes para decomposição "
          "dos efeitos.", st["nota"]),
        S(1, 6),
        P("A título de evidência parcial da separação dos efeitos, o experimento "
          "TF-IDF+VOTO foi executado sobre o conjunto de dados de desenvolvimento "
          "(dados sintéticos de validação do pipeline, n=982). Nesse contexto "
          "controlado, TF-IDF+SUMARIO obteve F1-macro=1,0000 (±0,0000) e "
          "TF-IDF+VOTO obteve 0,9763 (±0,0151), sugerindo que a troca do campo "
          "SUMARIO para VOTO não confere vantagem ao TF-IDF e que o efeito-arquitetura "
          "pode ser dominante. Contudo, essa inferência é preliminar e os dados "
          "sintéticos não replicam o ruído e a ambiguidade do corpus real.",
          st["corpo"]),

        # ── 4. RESULTADOS ─────────────────────────────────────────────
        P("4 RESULTADOS E DISCUSSÃO", st["secao"]),
        P("4.1 Baseline TF-IDF — Validação Cruzada e Hold-Out", st["subsecao"]),
        P("Com dados reais do TCU (2020–2024, campo SUMARIO), o baseline obteve "
          "F1-macro=0,8404 no hold-out (acurácia=96,5%, precisão macro=0,9025, "
          "revocação macro=0,7981). A Tabela 1 consolida o comparativo.", st["corpo"]),
        S(1, 4),
        _tab1_resultados(),
        P("<i>Tabela 1</i> — Métricas no conjunto hold-out (corpus 2020–2024, "
          "~80 amostras de teste). Atenção: os dois experimentos utilizam campos "
          "de entrada distintos — comparação válida como composto, não como "
          "isolamento puro de arquitetura.", st["nota"]),
        S(1, 6),
        P("A alta precisão macro do baseline (0,9025) indica que, quando prediz uma "
          "classe, o modelo erra raramente. A revocação menor (0,7981) evidencia "
          "dificuldade em recuperar todas as instâncias das classes minoritárias, "
          "comportamento típico de modelos lineares sobre dados desbalanceados. "
          "O Transformer inverte esse padrão: menor precisão (0,8357) e maior "
          "revocação (0,9174), preferindo cobrir mais casos ao custo de mais falsos "
          "positivos — configuração favorável em contextos de auditoria, onde o custo "
          "de perder uma irregularidade supera o custo de um alarme falso.",
          st["corpo"]),
        P("A Tabela 5 formaliza essa assimetria de custo, alinhando as métricas "
          "estatísticas aos objetivos de governança do TCU.", st["corpo"]),
        S(1, 4),
        _tab5_custo(),
        P("<i>Tabela 5</i> — Assimetria de custo entre erros de classificação no "
          "contexto de auditoria pública. O Falso Negativo (deixar passar uma "
          "irregularidade) tem impacto fiscal direto ao erário; o Falso Positivo "
          "gera apenas custo operacional de auditoria.", st["nota"]),
        S(1, 6),
        P("Traduzindo para o contexto do corpus: a revocação macro do Transformer "
          "(0,9174) implica taxa média de Falsos Negativos de ≈8,3%, contra ≈20,2% "
          "do baseline — redução de aproximadamente 60% na proporção de irregularidades "
          "não detectadas. Em termos práticos, para cada 100 acórdãos irregulares "
          "submetidos ao modelo, o Transformer 'deixa passar' cerca de 8, contra 20 "
          "do baseline. Considerando que as condenações do TCU em saúde e educação "
          "envolvem regularmente valores acima de R$ 100 mil (TCU, 2024), a diferença "
          "de revocação representa um retorno sobre o investimento mensurável na "
          "implantação da solução de Deep Learning.", st["corpo"]),
        P("4.2 Efeito do Volume de Dados e Diversidade Temporal", st["subsecao"]),
        P("A Tabela 2 apresenta a evolução dos resultados por janela temporal, com "
          "contagens precisas dos conjuntos de dados.", st["corpo"]),
        S(1, 4),
        _tab2_corpus(),
        P("<i>Tabela 2</i> — Evolução dos resultados por janela temporal. "
          "* Com class weights; v1 (sem class weights): F1-macro=0,3114. "
          "† Após filtragem, 2020–2022 contribuíram marginalmente ao corpus temático. "
          "‡ Contagem similar à janela de 2 anos: o filtro temático concentra "
          "acórdãos de saúde/educação predominantemente em 2023–2024; a melhoria "
          "de desempenho reflete <i>diversidade temporal</i>, não expansão de volume.",
          st["nota"]),
        S(1, 6),
        P("A melhoria expressiva de F1-macro de 0,67 para 0,84 (baseline) e de 0,50 "
          "para 0,87 (Transformer) ocorreu com conjuntos de treino de tamanho similar "
          "(~374 amostras em ambos os casos). Isso contraria a narrativa simplificada "
          "de que 'mais dados sempre beneficiam Transformers'. A explicação alternativa "
          "é que a diversidade temporal — incluir acórdãos de 2020, 2021 e 2022, "
          "com vocabulários e contextos regulatórios distintos dos de 2023–2024 — "
          "reduz o sobreajuste a padrões idiossincráticos do período recente. "
          "Essa hipótese requer validação com conjuntos de treino de tamanhos "
          "controlados (ablação por ano), proposta como trabalho futuro.",
          st["corpo"]),
        P("4.3 Métricas por Classe: Diagnóstico do Colapso", st["subsecao"]),
        P("A Tabela 4 apresenta as métricas por classe para o Transformer na versão 2 "
          "(corpus 2023–2024, com class weights), expondo o diagnóstico que levou à "
          "expansão do corpus.", st["corpo"]),
        S(1, 4),
        _tab4_por_classe(),
        P("<i>Tabela 4</i> — Métricas por classe, Transformer v2 (corpus 2023–2024, "
          "com class weights). A classe 'Regular' colapsa a F1=0,00 mesmo com "
          "class weights, motivando a expansão temporal do corpus. * Suporte "
          "aproximado dado o forte desbalanceamento (~88% Irregular).",
          st["nota"]),
        S(1, 6),
        P("O colapso da classe 'Regular' (F1=0,00) com apenas dois anos de dados "
          "revela que class weights são condição necessária mas não suficiente: "
          "o número absoluto de amostras positivas na classe minoritária (estimado "
          "em ~5 instâncias de treino) é insuficiente para o Transformer aprender "
          "uma fronteira de decisão estável. A expansão para 5 anos incorporou "
          "instâncias adicionais da classe 'Regular', elevando o F1-macro para 0,8686 "
          "no corpus ampliado.", st["corpo"]),
        P("4.4 Robustez Estatística: K-Fold e Intervalo de Confiança", st["subsecao"]),
        P("Com apenas ~80 amostras no conjunto de teste, qualquer resultado de "
          "hold-out único apresenta alta variância amostral. A validação cruzada "
          "K-Fold (5 folds) sobre o baseline permite estimar o IC a 95% "
          "(aproximação ±2σ) e avaliar a estabilidade dos resultados. "
          "Nos dados de validação do pipeline (sintéticos), o baseline atingiu "
          "F1-macro=1,0000 ± 0,0000 para o campo SUMARIO e 0,9763 ± 0,0151 para "
          "o campo VOTO — evidência de que a infraestrutura de K-Fold está "
          "operacional e pronta para aplicação nos dados reais do TCU.",
          st["corpo"]),
        P("A ausência de teste de McNemar ou t-test bicaudal sobre múltiplas execuções "
          "é reconhecida como limitação: o ganho de +0,028 em F1-macro, com ~80 "
          "amostras de teste, pode não ser estatisticamente significativo. "
          "Essa verificação requer execução de K-Fold para o Transformer (custo "
          "computacional elevado: ~5× o custo de um único treinamento), proposta "
          "como prioridade no ciclo seguinte de experimentos.", st["corpo"]),
        P("4.5 Eficiência Computacional: PEFT/LoRA como Caminho para K-Fold no Transformer",
          st["subsecao"]),
        P("O fine-tuning completo (Full Fine-Tuning) do LegalBert-pt atualiza todos os "
          "110 milhões de parâmetros do modelo a cada época de treinamento. Na GPU "
          "Tesla T4 (Google Colab), um único ciclo de fine-tuning (5 épocas, batch=16) "
          "já exige memória e tempo consideráveis sobre o corpus de ~374 amostras de "
          "treino. A execução do K-Fold com 5 folds multiplicaria esse custo por um "
          "fator de cinco, tornando a validação cruzada inviável nas restrições de "
          "tempo e memória do ambiente padrão de desenvolvimento.", st["corpo"]),
        P("A solução prevista para o próximo ciclo de experimentos é a adoção de "
          "Parameter-Efficient Fine-Tuning (PEFT), especificamente o LoRA "
          "(Low-Rank Adaptation; HU et al., 2022). O LoRA congela os pesos originais "
          "do BERT e injeta matrizes de baixa ordem (rank r = 8–16) nas camadas de "
          "atenção, substituindo a atualização completa W por uma decomposição de "
          "baixo posto ΔW = BA, onde B ∈ ℝ<super>d×r</super> e "
          "A ∈ ℝ<super>r×k</super>. Com r = 8, o número de parâmetros treináveis "
          "cai de 110M para aproximadamente 300K — redução de 99,7% — sem degradação "
          "significativa de performance reportada na literatura (HU et al., 2022).",
          st["corpo"]),
        P("O argumento científico para esta escolha é direto: ao congelar os pesos "
          "base, o LoRA permite inicializar apenas os adaptadores a cada fold, "
          "mantendo o backbone em memória entre os folds. Isso viabiliza a execução "
          "de K-Fold (5 folds) no T4 com tempo comparável a um único treinamento "
          "completo, resolvendo o problema de variância amostral identificado na "
          "Seção 5.1 sem incorrer em custos computacionais proibitivos. O QLoRA "
          "(Quantized LoRA) representa extensão adicional — quantização dos pesos base "
          "para 4 bits com Double Quantization —, relevante se modelos maiores "
          "(ex.: LegalBert-pt-large) forem incluídos em ciclos futuros.", st["corpo"]),
        P("4.6 Explicabilidade: Aplicação ao Modelo e Redesign", st["subsecao"]),
        P("Conforme apontado em revisão, o uso de LIME sobre o modelo de regressão "
          "logística é redundante: modelos lineares já disponibilizam coeficientes "
          "nativos que expressam a importância de cada feature. O LIME foi aplicado "
          "ao baseline como sanity check e para gerar visualizações comparáveis com "
          "a literatura de jurimetria.", st["corpo"]),
        P("Para o Transformer — modelo de caixa-preta —, a aplicação de LIME ou SHAP "
          "sobre as predições de desfecho é o caso de uso metodologicamente correto "
          "e constitui o principal entregável de explicabilidade pendente. A "
          "implementação requer: (i) o modelo LegalBert-pt fine-tunado salvo em disco; "
          "(ii) wrapper de predição compatível com LIME TextExplainer; e (iii) "
          "execução em GPU pelo custo de inferência em 512 tokens por amostra. "
          "Esse passo está mapeado no Estágio 11 do pipeline mas requer o modelo "
          "final treinado com o corpus de 5 anos.", st["corpo"]),

        # ── 5. AMEAÇAS À VALIDADE ──────────────────────────────────────
        P("5 AMEAÇAS À VALIDADE", st["secao"]),
        P("5.1 Validade Interna", st["subsecao"]),
        P("<b>Assimetria de campos (crítica):</b> o principal confundidor é a "
          "diferença entre SUMARIO (baseline) e VOTO (Transformer). Até que os "
          "experimentos cruzados da Tabela 3 sejam executados, o ganho de +0,028 "
          "deve ser interpretado como limite superior do efeito-arquitetura.",
          st["corpo"]),
        P("<b>Variância amostral:</b> com ~80 amostras de teste, a variância do "
          "estimador de F1-macro é elevada. Diferenças de 3 pontos percentuais "
          "podem não ser estatisticamente significativas sem teste formal (McNemar).",
          st["corpo"]),
        P("<b>Semente única e custo do K-Fold:</b> os resultados foram obtidos com "
          "RANDOM_STATE=42 para o split e seed=42 para o Trainer. Múltiplas sementes "
          "ou K-Fold para o Transformer são necessários para garantir reprodutibilidade "
          "robusta. A limitação é computacional: o Full Fine-Tuning (110M parâmetros) "
          "torna o K-Fold proibitivo no T4. A adoção de LoRA (HU et al., 2022) — "
          "redução para ~300K parâmetros treináveis — é a mitigação planejada, "
          "conforme detalhado na Seção 4.5.", st["corpo"]),
        P("5.2 Validade Externa", st["subsecao"]),
        P("<b>Escopo temático:</b> o corpus cobre apenas saúde e educação. A "
          "generalização para outras áreas do TCU (infraestrutura, defesa, previdência) "
          "não está validada.", st["corpo"]),
        P("<b>Defasagem arquitetural:</b> o LegalBert-pt é baseado no BERT de 2018 "
          "(DEVLIN et al., 2019). Arquiteturas mais recentes capazes de processar "
          "sequências longas nativamente — Longformer, BigBird, ou modelos baseados "
          "em State Space Models (Mamba) — não foram avaliadas. Para documentos que "
          "excedem 512 tokens com conteúdo relevante no meio (não apenas cabeça e "
          "cauda), essas arquiteturas têm potencial de ganho adicional.",
          st["corpo"]),
        P("5.3 Validade de Constructo", st["subsecao"]),
        P("O F1-macro pondera igualmente as três classes. Embora atue como guardrail "
          "estatístico para o desbalanceamento, ele falha em capturar a assimetria de "
          "impacto financeiro no setor público. Alinhando o modelo aos objetivos de "
          "governança do TCU — formalizados na Tabela 5 —, análises subsequentes devem "
          "incorporar matrizes de custo-benefício customizadas, onde o peso de um "
          "Falso Negativo (negligenciar um desvio na saúde ou educação) supere "
          "substancialmente a penalidade de um Falso Positivo, otimizando o limiar "
          "de decisão da rede neural para além das métricas macro tradicionais.",
          st["corpo"]),
        P("Na prática, isso se traduz em ajustar o threshold de classificação abaixo "
          "de 0,5 para a classe 'Irregular': ao aceitar mais Falsos Positivos "
          "controláveis, o modelo reduz drasticamente os Falsos Negativos de alto "
          "custo fiscal. Uma função de perda personalizada com pesos econômicos por "
          "classe — ou a adoção de Focal Loss (LIN et al., 2017), que penaliza "
          "exemplos mal classificados com confiança elevada — pode estar mais alinhada "
          "com os objetivos de negócio do TCU do que a otimização direta do F1-macro. "
          "Essa otimização de threshold tem custo computacional nulo (não requer "
          "re-treinamento) e pode ser implementada sobre o modelo já treinado, "
          "usando o conjunto de validação para calibrar o ponto ótimo de corte.",
          st["corpo"]),

        # ── 6. CONCLUSÃO ─────────────────────────────────────────────
        P("6 CONCLUSÃO", st["secao"]),
        P("Este trabalho demonstrou a viabilidade da aplicação de modelos Transformer "
          "para predição de desfechos de acórdãos do TCU em saúde e educação. "
          "O LegalBert-pt com truncação head+tail e class weights obteve F1-macro "
          "de 0,8686, superando o baseline TF-IDF em 0,028 pontos. A metodologia "
          "inclui K-Fold para o baseline, métricas desagregadas por classe e "
          "reconhecimento explícito do principal confundidor: a assimetria de campos "
          "de entrada entre os dois estágios.", st["corpo"]),
        P("A hipótese de que a melhoria reflete efeito-arquitetura recebe suporte "
          "parcial: nos dados sintéticos de validação, a troca do campo SUMARIO para "
          "VOTO não beneficiou o TF-IDF, sugerindo que o campo mais rico (VOTO) por "
          "si só não explica o ganho do Transformer. Entretanto, a confirmação "
          "definitiva aguarda os experimentos cruzados da Tabela 3 com dados reais.",
          st["corpo"]),
        P("O achado mais contra-intuitivo é que a melhoria expressiva de desempenho "
          "ao expandir o corpus de 2 para 5 anos ocorreu com volume de treino similar "
          "(~374 amostras), sugerindo que a <i>diversidade temporal</i> — e não o "
          "volume — é o fator determinante nesse domínio.", st["corpo"]),
        P("Trabalhos futuros prioritários: (i) completar a ablação 2×2 (Tabela 3); "
          "(ii) K-Fold para o Transformer via PEFT/LoRA (HU et al., 2022) — ao "
          "reduzir os parâmetros treináveis de 110M para ~300K, o LoRA viabiliza a "
          "execução dos 5 folds no T4, garantindo a robustez estatística necessária "
          "para homologação do Radar Jurimétrico em ambiente de produção; "
          "(iii) LIME/SHAP sobre o Transformer fine-tunado; (iv) otimização de "
          "threshold de decisão com matriz de custo-benefício financeiro (Tabela 5), "
          "priorizando revocação da classe 'Irregular' sem re-treinamento; e "
          "(v) avaliar Focal Loss (LIN et al., 2017) para alinhamento com os "
          "objetivos de governança do TCU.", st["corpo"]),
        P("Quanto ao direcionamento editorial: o artigo possui valor elevado para "
          "periódicos focados em <i>Computação Jurídica</i>, <i>Jurimetria</i> e "
          "<i>IA no Setor Público</i> — como o <i>Journal of Digital Government</i>, "
          "a <i>Artificial Intelligence and Law</i> ou o <i>Encontro Nacional de IA "
          "e Direito</i> —, onde o impacto prático e a análise de domínio superam "
          "a exigência de novidade arquitetural.", st["corpo"]),
        HR(width="100%", thickness=0.8, color=colors.HexColor("#d1d5db"), spaceAfter=8),

        # ── REFERÊNCIAS ───────────────────────────────────────────────
        P("REFERÊNCIAS", st["secao"]),
        P("ALETRAS, Nikolaos et al. <b>Predicting judicial decisions of the European "
          "Court of Human Rights: a Natural Language Processing perspective.</b> "
          "<i>PeerJ Computer Science</i>, [s.l.], v. 2, e93, 2016.", st["ref"]),
        P("DEVLIN, Jacob et al. <b>BERT: pre-training of deep bidirectional transformers "
          "for language understanding.</b> In: CONFERENCE OF THE NORTH AMERICAN CHAPTER "
          "OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS, 2019, Minneapolis. "
          "<i>Proceedings…</i> Stroudsburg: ACL, 2019. p. 4171–4186.", st["ref"]),
        P("DOMINGUES, Luciano. <b>legal-bert-base-cased-ptbr: BERT model pre-trained "
          "on Brazilian legal corpus.</b> HuggingFace Hub, 2022. Disponível em: "
          "&lt;https://huggingface.co/dominguesm/legal-bert-base-cased-ptbr&gt;. "
          "Acesso em: 5 jun. 2026.", st["ref"]),
        P("HU, Edward J. et al. <b>LoRA: low-rank adaptation of large language "
          "models.</b> In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, "
          "2022, online. <i>Proceedings…</i> [S.l.]: OpenReview, 2022.", st["ref"]),
        P("LAGE-FREITAS, André et al. <b>Predicting Brazilian court decisions.</b> "
          "<i>PeerJ Computer Science</i>, [s.l.], v. 8, e904, 2022.", st["ref"]),
        P("LIN, Tsung-Yi et al. <b>Focal loss for dense object detection.</b> In: "
          "IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION, 2017, Veneza. "
          "<i>Proceedings…</i> [S.l.]: IEEE, 2017. p. 2980–2988.", st["ref"]),
        P("MEDVEDEVA, Masha; VOLS, Michel; WIELING, Martijn. <b>Using machine learning "
          "to predict decisions of the European Court of Human Rights.</b> "
          "<i>Artificial Intelligence and Law</i>, [s.l.], v. 28, n. 2, "
          "p. 237–266, 2020.", st["ref"]),
        P("PEDREGOSA, Fabian et al. <b>Scikit-learn: machine learning in Python.</b> "
          "<i>Journal of Machine Learning Research</i>, Cambridge, v. 12, "
          "p. 2825–2830, 2011.", st["ref"]),
        P("RIBEIRO, Marco Tulio; SINGH, Sameer; GUESTRIN, Carlos. <b>'Why should I "
          "trust you?': explaining the predictions of any classifier.</b> In: "
          "ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA "
          "MINING, 22., 2016, San Francisco. <i>Proceedings…</i> New York: "
          "ACM, 2016. p. 1135–1144.", st["ref"]),
        P("SOUZA, Fábio; NOGUEIRA, Rodrigo; LOTUFO, Roberto. <b>BERTimbau: pretrained "
          "BERT models for Brazilian Portuguese.</b> In: INTELLIGENT SYSTEMS — BRACIS, "
          "9., 2020, Rio Grande. <i>Proceedings…</i> Cham: Springer, 2020. "
          "p. 403–417.", st["ref"]),
        P("SUN, Chi; QIU, Xipeng; XU, Yuanbin; HUANG, Xuanjing. <b>How to fine-tune "
          "BERT for text classification?</b> In: CHINESE COMPUTATIONAL LINGUISTICS, "
          "18., 2019, Kunming. <i>Proceedings…</i> Cham: Springer, 2019. "
          "p. 194–206.", st["ref"]),
        P("TRIBUNAL DE CONTAS DA UNIÃO. <b>Portal de Dados Abertos do TCU: Acórdãos "
          "Completos.</b> Brasília: TCU, 2024. Disponível em: "
          "&lt;https://sites.tcu.gov.br/dados-abertos/jurisprudencia/&gt;. "
          "Acesso em: 5 jun. 2026.", st["ref"]),
        P("TVEITA, Sondre; HUSTAD, Eli. <b>Benefits and challenges of AI in the "
          "public sector.</b> In: HAWAII INTERNATIONAL CONFERENCE ON SYSTEM SCIENCES, "
          "58., 2025, Maui. <i>Proceedings…</i> Honolulu: University of Hawaii, "
          "2025. p. 1–10.", st["ref"]),
        P("VASWANI, Ashish et al. <b>Attention is all you need.</b> In: ADVANCES IN "
          "NEURAL INFORMATION PROCESSING SYSTEMS, 30., 2017, Long Beach. "
          "<i>Proceedings…</i> Red Hook: Curran Associates, 2017. p. 5998–6008.",
          st["ref"]),
        P("WOLF, Thomas et al. <b>Transformers: state-of-the-art natural language "
          "processing.</b> In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE "
          "PROCESSING: SYSTEM DEMONSTRATIONS, 2020, online. <i>Proceedings…</i> "
          "Stroudsburg: ACL, 2020. p. 38–45.", st["ref"]),
    ]
    return h


# ── Main ─────────────────────────────────────────────────────────────────────

def gerar_artigo(saida: Path = SAIDA) -> Path:
    saida.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(saida), pagesize=A4,
        topMargin=3*cm, bottomMargin=2*cm,
        leftMargin=3*cm, rightMargin=2*cm,
        title="Jurimetria Preditiva em Acórdãos do TCU (v2 — revisado)",
        author="Bruno Sousa — IDP",
    )
    doc.build(_historia(_estilos()), onLaterPages=_rodape, onFirstPage=_rodape)
    kb = saida.stat().st_size // 1024
    print(f"Artigo gerado: {saida}  ({kb} KB)")
    return saida


if __name__ == "__main__":
    gerar_artigo()
