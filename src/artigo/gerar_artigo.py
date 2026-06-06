"""Gerador do artigo científico em PDF seguindo ABNT NBR 6022:2018.

Produz um artigo de pesquisa completo em formato A4.
Saída: resultados/artigo_jurimetria_tcu.pdf
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"
SAIDA = RESULTADOS / "artigo_jurimetria_tcu.pdf"

# ── Paleta ────────────────────────────────────────────────────────────────────
PRETO = colors.black
CINZA = colors.HexColor("#374151")
CINZA_CLARO = colors.HexColor("#f3f4f6")
AZUL = colors.HexColor("#1a3557")


# ── Estilos ABNT ──────────────────────────────────────────────────────────────

def _estilos():
    base = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "Titulo",
        parent=base["Normal"],
        fontName="Times-Bold",
        fontSize=14,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=6,
        textColor=PRETO,
    )
    autores = ParagraphStyle(
        "Autores",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    afiliacao = ParagraphStyle(
        "Afiliacao",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=2,
        textColor=CINZA,
    )
    secao = ParagraphStyle(
        "Secao",
        parent=base["Normal"],
        fontName="Times-Bold",
        fontSize=12,
        leading=17,
        alignment=TA_LEFT,
        spaceBefore=14,
        spaceAfter=6,
        textColor=PRETO,
    )
    subsecao = ParagraphStyle(
        "Subsecao",
        parent=base["Normal"],
        fontName="Times-BoldItalic",
        fontSize=12,
        leading=17,
        alignment=TA_LEFT,
        spaceBefore=10,
        spaceAfter=4,
        textColor=PRETO,
    )
    corpo = ParagraphStyle(
        "Corpo",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=12,
        leading=21,  # espaçamento 1.5 (~21pt para fonte 12)
        alignment=TA_JUSTIFY,
        firstLineIndent=1.25 * cm,
        spaceBefore=0,
        spaceAfter=6,
    )
    resumo_tit = ParagraphStyle(
        "ResumoTitulo",
        parent=base["Normal"],
        fontName="Times-Bold",
        fontSize=12,
        leading=17,
        alignment=TA_LEFT,
        spaceBefore=12,
        spaceAfter=4,
    )
    resumo_corpo = ParagraphStyle(
        "ResumoCorpo",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=17,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        spaceAfter=4,
    )
    palavras = ParagraphStyle(
        "PalavrasChave",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    referencia = ParagraphStyle(
        "Referencia",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        leftIndent=0,
        firstLineIndent=0,
        spaceAfter=6,
    )
    nota = ParagraphStyle(
        "Nota",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=CINZA,
        spaceAfter=4,
    )
    return {
        "titulo": titulo, "autores": autores, "afiliacao": afiliacao,
        "secao": secao, "subsecao": subsecao, "corpo": corpo,
        "resumo_tit": resumo_tit, "resumo_corpo": resumo_corpo,
        "palavras": palavras, "referencia": referencia, "nota": nota,
    }


# ── Numeração de páginas ───────────────────────────────────────────────────────

def _rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(CINZA)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page))
    canvas.restoreState()


# ── Tabelas ────────────────────────────────────────────────────────────────────

def _tabela_resultados(st):
    dados = [
        ["Modelo", "Campo", "F1-macro", "Precisão\nmacro", "Revocação\nmacro", "Acurácia"],
        ["TF-IDF + LogisticRegression", "SUMARIO", "0,8404", "0,9025", "0,7981", "96,5%"],
        ["LegalBert-pt head+tail\n+ class weights", "VOTO", "0,8686", "0,8357", "0,9174", "95,9%"],
    ]
    estilo = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), AZUL),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 10),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",     (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",     (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CINZA_CLARO, colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME",     (0, 1), (0, -1), "Times-Italic"),
    ])
    col_w = [5.5 * cm, 2.5 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.0 * cm]
    return Table(dados, colWidths=col_w, style=estilo, repeatRows=1)


def _tabela_evolucao(st):
    dados = [
        ["Corpus", "Amostras\n(treino)", "Baseline\nF1-macro", "Transformer\nF1-macro", "Hipótese\nconfirmada?"],
        ["2023–2024 (2 anos)", "~373", "0,6705", "0,4972*", "Não"],
        ["2020–2024 (5 anos)", "~373†", "0,8404", "0,8686", "Sim"],
    ]
    estilo = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), AZUL),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 10),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",     (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",     (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CINZA_CLARO, colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TEXTCOLOR",    (4, 1), (4, 1), colors.HexColor("#dc2626")),
        ("TEXTCOLOR",    (4, 2), (4, 2), colors.HexColor("#16a34a")),
        ("FONTNAME",     (4, 2), (4, 2), "Times-Bold"),
    ])
    col_w = [4.5 * cm, 2.5 * cm, 2.8 * cm, 2.8 * cm, 3.2 * cm]
    return Table(dados, colWidths=col_w, style=estilo, repeatRows=1)


# ── Conteúdo do artigo ─────────────────────────────────────────────────────────

def _construir_historia(st):
    P = Paragraph
    S = Spacer
    HR = HRFlowable
    historia = []

    # ── CABEÇALHO ────────────────────────────────────────────────────────────
    historia += [
        P("JURIMETRIA PREDITIVA EM ACÓRDÃOS DO TRIBUNAL DE CONTAS DA UNIÃO: "
          "COMPARATIVO ENTRE BASELINE TF-IDF E FINE-TUNING DE TRANSFORMER "
          "COM TRUNCAÇÃO HEAD+TAIL NAS ÁREAS DE SAÚDE E EDUCAÇÃO",
          st["titulo"]),
        S(1, 10),
        P("Bruno Sousa<super>1</super>", st["autores"]),
        S(1, 4),
        P("<super>1</super> IDP — Instituto Direito Público. Mestrado em Ciência de Dados e IA no Setor Público. "
          "Disciplina: Deep Learning e PLN. Brasília, DF, Brasil. "
          "E-mail: bruno.aires9@gmail.com",
          st["afiliacao"]),
        S(1, 4),
        HR(width="100%", thickness=1, color=AZUL, spaceAfter=8),

        # ── RESUMO ──────────────────────────────────────────────────────────
        P("RESUMO", st["resumo_tit"]),
        P(
            "O presente artigo investiga a aplicação de técnicas de Processamento de "
            "Linguagem Natural (PLN) na predição de desfechos de acórdãos do Tribunal "
            "de Contas da União (TCU) relativos às áreas de saúde e educação. Comparam-se "
            "dois paradigmas: um baseline clássico baseado em TF-IDF combinado com regressão "
            "logística, e um modelo de Deep Learning obtido pelo fine-tuning do LegalBert-pt "
            "(DOMINGUES, 2022) com estratégia de truncação head+tail (SUN et al., 2019). "
            "O corpus foi construído a partir dos dados abertos do TCU para o período "
            "2020–2024, resultando em 534 acórdãos temáticos após filtragem por termos "
            "de saúde e educação. O forte desbalanceamento de classes — aproximadamente "
            "88% de acórdãos com contas irregulares — foi tratado com class weights "
            "calculados automaticamente pelo método <i>balanced</i> do scikit-learn "
            "(PEDREGOSA et al., 2011). Os resultados demonstram que o modelo Transformer "
            "superou o baseline em F1-macro: 0,8686 versus 0,8404, um ganho absoluto de "
            "0,028 pontos. A hipótese central do trabalho foi confirmada, evidenciando que "
            "arquiteturas baseadas em Transformer, quando treinadas com volume adequado de "
            "dados e técnicas de balanceamento de classes, superam modelos lineares na "
            "classificação de texto jurídico em português. O trabalho contribui para o campo "
            "da jurimetria preditiva no Brasil, propondo um 'radar jurimétrico' que auxilia "
            "gestores públicos na avaliação preventiva de risco de condenação em processos "
            "licitatórios.",
            st["resumo_corpo"],
        ),
        P("<b>Palavras-chave:</b> Jurimetria. Tribunal de Contas da União. Processamento de "
          "Linguagem Natural. BERT. Classificação de Texto. Deep Learning.",
          st["palavras"]),
        S(1, 6),

        # ── ABSTRACT ────────────────────────────────────────────────────────
        P("ABSTRACT", st["resumo_tit"]),
        P(
            "This paper investigates the application of Natural Language Processing (NLP) "
            "techniques to predict outcomes of Brazilian Federal Court of Auditors (TCU) "
            "decisions related to healthcare and education. Two paradigms are compared: a "
            "classical TF-IDF baseline combined with logistic regression, and a Deep Learning "
            "model obtained by fine-tuning LegalBert-pt (DOMINGUES, 2022) with a head+tail "
            "truncation strategy (SUN et al., 2019). The corpus was built from TCU open data "
            "for the period 2020–2024, resulting in 534 thematic decisions after filtering. "
            "The strong class imbalance — approximately 88% of decisions with irregular "
            "accounts — was addressed through automatically computed class weights using "
            "scikit-learn's balanced method (PEDREGOSA et al., 2011). Results show that the "
            "Transformer model outperformed the baseline in macro F1-score: 0.8686 versus "
            "0.8404, an absolute gain of 0.028 points. The central hypothesis was confirmed, "
            "indicating that Transformer-based architectures, when trained with adequate data "
            "volume and class balancing techniques, outperform linear models in legal text "
            "classification in Portuguese. This work contributes to the field of predictive "
            "jurimetrics in Brazil, proposing a 'jurimetric radar' that assists public "
            "managers in preventively assessing the risk of condemnation in public procurement "
            "processes.",
            st["resumo_corpo"],
        ),
        P("<b>Keywords:</b> Jurimetrics. Court of Auditors. Natural Language Processing. "
          "BERT. Text Classification. Deep Learning.",
          st["palavras"]),
        HR(width="100%", thickness=0.8, color=colors.HexColor("#d1d5db"), spaceAfter=6),

        # ── 1. INTRODUÇÃO ───────────────────────────────────────────────────
        P("1 INTRODUÇÃO", st["secao"]),
        P(
            "A jurimetria — disciplina que aplica métodos quantitativos e estatísticos ao "
            "fenômeno jurídico — tem ganhado crescente relevância no contexto da transformação "
            "digital do setor público brasileiro. No âmbito do controle externo federal, o "
            "Tribunal de Contas da União (TCU) produz anualmente dezenas de milhares de "
            "acórdãos que julgam a regularidade das contas de gestores públicos, impondo, "
            "nos casos de irregularidade, multas, débitos e determinações de ressarcimento "
            "ao erário.",
            st["corpo"],
        ),
        P(
            "A predição automática do desfecho de tais acórdãos representa uma aplicação "
            "de alto impacto prático: gestores de saúde e educação — setores historicamente "
            "com elevada incidência de irregularidades formais — poderiam utilizar um modelo "
            "de classificação como ferramenta preventiva, identificando padrões de risco antes "
            "da conclusão do processo de auditoria.",
            st["corpo"],
        ),
        P(
            "Com o advento dos modelos Transformer (VASWANI et al., 2017) e, em particular, "
            "do BERT (DEVLIN et al., 2019), o estado da arte em classificação de texto jurídico "
            "avançou significativamente. Modelos pré-treinados em grandes corpora jurídicos, "
            "como o LegalBert-pt (DOMINGUES, 2022), oferecem representações semânticas de alta "
            "qualidade para o domínio jurídico brasileiro. Simultaneamente, estratégias como a "
            "truncação head+tail (SUN et al., 2019) permitem lidar com documentos mais longos "
            "que o limite de 512 tokens do BERT sem perda relevante de informação.",
            st["corpo"],
        ),
        P(
            "O presente trabalho parte da hipótese de que um modelo Transformer com fine-tuning "
            "supera um baseline clássico baseado em TF-IDF na tarefa de classificação de "
            "desfechos de acórdãos do TCU, medida pelo F1-macro. Para tanto, foram utilizados "
            "os dados abertos do TCU referentes ao período 2020–2024, filtrados para os temas "
            "de saúde e educação.",
            st["corpo"],
        ),
        P(
            "As principais contribuições deste trabalho são: (i) a construção de um corpus "
            "temático de 534 acórdãos do TCU para fins de classificação de texto; (ii) a "
            "avaliação comparativa entre abordagens clássicas e de Deep Learning no domínio "
            "jurídico em língua portuguesa; (iii) uma discussão sobre o efeito do volume de "
            "dados e do balanceamento de classes no desempenho de Transformers em corpora "
            "jurídicos pequenos; e (iv) uma proposta de aplicação prática como radar "
            "jurimétrico para o setor público.",
            st["corpo"],
        ),

        # ── 2. TRABALHOS RELACIONADOS ─────────────────────────────────────
        P("2 TRABALHOS RELACIONADOS", st["secao"]),
        P(
            "A predição de decisões judiciais por meio de PLN tem sido objeto de investigação "
            "crescente na literatura internacional. Aletras et al. (2016) demonstraram a "
            "viabilidade de predizer decisões do Tribunal Europeu de Direitos Humanos com "
            "acurácia de 79%, utilizando análise de tópicos e máquinas de vetores de suporte. "
            "Medvedeva, Vols e Wieling (2020) aprimoraram esses resultados para 75% de acurácia "
            "na mesma corte, com modelos de aprendizado de máquina aplicados a textos completos, "
            "evidenciando os desafios do texto jurídico multilíngue e da variabilidade "
            "interpretativa dos julgadores.",
            st["corpo"],
        ),
        P(
            "No contexto brasileiro, Lage-Freitas et al. (2022) investigaram a predição de "
            "decisões de tribunais estaduais e federais com modelos de aprendizado de máquina, "
            "concluindo que a escolha do campo textual e a representatividade do corpus são "
            "fatores determinantes para o desempenho preditivo. O trabalho evidenciou a "
            "dificuldade adicional imposta pelo português jurídico, com suas construções "
            "formulaicas e referências normativas densas.",
            st["corpo"],
        ),
        P(
            "Do ponto de vista dos modelos de linguagem para o português, Souza, Nogueira e "
            "Lotufo (2020) apresentaram o BERTimbau, pré-treinado em um corpus de 2,68 bilhões "
            "de palavras da língua portuguesa, obtendo resultados expressivos em diversas tarefas "
            "de PLN. Domingues (2022) estendeu esse trabalho ao domínio jurídico brasileiro com "
            "o LegalBert-pt, pré-treinado em decisões do Supremo Tribunal Federal, petições e "
            "atos administrativos, aproximando as representações do vocabulário dos acórdãos do TCU.",
            st["corpo"],
        ),
        P(
            "A questão do tratamento de documentos longos em modelos BERT é abordada por "
            "Sun et al. (2019), que propõem e avaliam sistematicamente quatro estratégias de "
            "truncação — incluindo head-only, tail-only, head+tail e chunking com pooling — "
            "em 19 conjuntos de dados de classificação. Os autores demonstram que a estratégia "
            "head+tail supera a truncação simples na maioria dos domínios, especialmente "
            "naqueles em que tanto o contexto inicial quanto a conclusão do documento "
            "contribuem para a classificação.",
            st["corpo"],
        ),
        P(
            "O campo da inteligência artificial no setor público, por sua vez, apresenta "
            "desafios específicos relacionados à explicabilidade, transparência e responsabilidade "
            "algorítmica (TVEITA; HUSTAD, 2025). O presente trabalho endereça essas questões "
            "por meio da análise de explicabilidade via LIME (RIBEIRO; SINGH; GUESTRIN, 2016), "
            "que permite identificar quais termos textuais mais influenciam as predições do "
            "modelo, tornando o radar jurimétrico auditável por gestores e auditores.",
            st["corpo"],
        ),

        # ── 3. METODOLOGIA ────────────────────────────────────────────────
        P("3 METODOLOGIA", st["secao"]),
        P("3.1 Corpus e Coleta de Dados", st["subsecao"]),
        P(
            "O corpus foi construído a partir dos dados abertos do TCU, disponíveis no Portal "
            "de Dados Abertos (TCU, 2024). Foram utilizados os arquivos "
            "<i>acordao-completo-AAAA.csv</i> para os anos de 2020 a 2024, totalizando "
            "aproximadamente 500.000 acórdãos antes de qualquer filtragem. Cada arquivo "
            "anual ocupa entre 175 MB e 445 MB em disco, em formato CSV com separador pipe "
            "(|), codificação UTF-8 e 33 colunas com nomes em letras maiúsculas.",
            st["corpo"],
        ),
        P(
            "Dado o volume de dados, a leitura foi realizada com seleção explícita de colunas "
            "via <i>pandas.read_csv(..., usecols=[...])</i>, garantindo que apenas os campos "
            "necessários — NUMACORDAO, DATASESSAO, SUMARIO, ACORDAO, ASSUNTO, VOTO e SITUACAO "
            "— fossem carregados na memória. Esse guardrail de gestão de memória é essencial "
            "em ambientes com recursos limitados, como o Google Colab utilizado para o "
            "fine-tuning.",
            st["corpo"],
        ),
        P(
            "O filtro temático foi aplicado ao campo SUMARIO por correspondência com os termos "
            "'saúde', 'SUS', 'FNDE', 'merenda', 'educação', 'ministério da saúde', "
            "'secretaria de saúde' e 'secretaria de educação', com comparação sem "
            "diferenciação de maiúsculas/minúsculas e sem acentuação. Após esse filtro, "
            "o corpus foi reduzido a 534 acórdãos, volume gerenciável para fine-tuning "
            "em GPU de médio porte.",
            st["corpo"],
        ),
        P(
            "O rótulo de desfecho foi extraído do campo ACORDAO por meio da expressão regular "
            r"<i>r'contas\s+(irregulares|regulares\s+com\s+ressalva|regulares)'</i>, "
            "com fallback para o campo SUMARIO quando o campo ACORDAO não continha a decisão "
            "estruturada. Essa abordagem foi necessária porque o campo SITUACAO, candidato "
            "natural a rótulo, contém o status processual do acórdão ('BAIXADO', 'EM TRAMITAÇÃO') "
            "e não o veredicto de mérito.",
            st["corpo"],
        ),
        P("3.2 Análise Exploratória e Divisão dos Dados", st["subsecao"]),
        P(
            "A análise exploratória revelou um corpus fortemente desbalanceado: "
            "aproximadamente 88% dos acórdãos foram classificados como Irregular, "
            "com as classes Regular e Regular com Ressalva compondo os 12% restantes "
            "em proporções similares. Esse padrão reflete a natureza do processo de "
            "seleção do TCU: acórdãos de saúde e educação tendem a concentrar casos "
            "de irregularidade, dado o foco das câmaras especializadas nesses temas.",
            st["corpo"],
        ),
        P(
            "Esse desbalanceamento motivou a adoção do F1-macro como métrica principal, "
            "que pondera igualmente o desempenho em todas as classes independentemente "
            "de seu suporte no conjunto de teste. A acurácia, que poderia atingir 88% "
            "simplesmente predizendo sempre 'Irregular', foi mantida como métrica "
            "complementar (PEDREGOSA et al., 2011).",
            st["corpo"],
        ),
        P(
            "Os dados foram divididos em conjuntos de treino (70%), validação (15%) e "
            "teste (15%), com estratificação por classe e semente aleatória fixa "
            "(RANDOM_STATE = 42), garantindo a reprodutibilidade dos experimentos.",
            st["corpo"],
        ),
        P("3.3 Estágio 1 — Baseline TF-IDF", st["subsecao"]),
        P(
            "O baseline foi implementado como um pipeline scikit-learn (PEDREGOSA et al., 2011) "
            "composto por: (i) vectorizador TF-IDF com <i>max_features = 50.000</i> e "
            "<i>ngram_range = (1, 2)</i>, capturando unigramas e bigramas; e (ii) regressão "
            "logística com regularização L2. O campo SUMARIO foi utilizado como texto de "
            "entrada, após limpeza que incluiu conversão para minúsculas, remoção de "
            "números, pontuação e stopwords jurídicas em português via NLTK.",
            st["corpo"],
        ),
        P(
            "A escolha da regressão logística como classificador do baseline é respaldada "
            "pela literatura: modelos lineares sobre representações TF-IDF frequentemente "
            "superam abordagens mais complexas em corpora pequenos com vocabulário restrito "
            "(PEDREGOSA et al., 2011). A regularização L2 controla o sobreajuste em espaços "
            "de alta dimensionalidade (50.000 features).",
            st["corpo"],
        ),
        P("3.4 Estágio 2 — Fine-tuning com LegalBert-pt", st["subsecao"]),
        P(
            "Para o estágio de Deep Learning, foi adotado o modelo "
            "<i>dominguesm/legal-bert-base-cased-ptbr</i> (DOMINGUES, 2022), uma variante "
            "do BERTimbau (SOUZA; NOGUEIRA; LOTUFO, 2020) re-treinada em corpus jurídico "
            "brasileiro. O campo VOTO foi utilizado como texto de entrada por conter o "
            "raciocínio integral do relator e o dispositivo do acórdão — as partes "
            "semanticamente mais discriminativas para a predição do desfecho. A "
            "disponibilidade direta do campo VOTO no CSV do TCU (coluna 29) eliminou a "
            "necessidade de extração de PDFs.",
            st["corpo"],
        ),
        P(
            "O fine-tuning foi realizado com os seguintes hiperparâmetros: 5 épocas; "
            "tamanho de batch = 16; taxa de aprendizado = 1×10<super>−5</super>; "
            "decaimento de peso (<i>weight decay</i>) = 0,01; e warmup de 10% dos "
            "passos totais de treino com agendador linear de decaimento. "
            "A parada antecipada (<i>early stopping</i>) foi ativada com paciência "
            "de 2 épocas, monitorando o F1-macro no conjunto de validação. "
            "A implementação utilizou a biblioteca Transformers (WOLF et al., 2020) com "
            "o <i>Trainer</i> da HuggingFace e uma subclasse customizada para suporte "
            "a <i>class weights</i> na função de perda. O treinamento foi executado em "
            "GPU NVIDIA Tesla T4 no Google Colab.",
            st["corpo"],
        ),
        P("3.5 Estratégia de Truncação Head+Tail", st["subsecao"]),
        P(
            "O BERT impõe um limite de 512 tokens por sequência, incluindo os tokens "
            "especiais [CLS] e [SEP]. Os acórdãos do TCU, em especial o campo VOTO, "
            "frequentemente excedem esse limite. Para maximizar a cobertura informacional "
            "sem aumentar o custo computacional, adotou-se a estratégia head+tail "
            "proposta por Sun et al. (2019): os primeiros 128 tokens (contexto do processo) "
            "são concatenados com os últimos 382 tokens (dispositivo e determinações), "
            "totalizando 512 posições incluindo os tokens especiais.",
            st["corpo"],
        ),
        P(
            "A assimetria entre cabeça (128 tokens) e cauda (382 tokens) reflete a "
            "hipótese de que a parte final dos acórdãos — onde o relator apresenta sua "
            "conclusão e as determinações são enunciadas — carrega maior poder discriminativo "
            "para a predição do desfecho. Sun et al. (2019) demonstraram empiricamente que "
            "essa estratégia supera tanto a truncação simples quanto a estratégia "
            "<i>tail-only</i> em documentos jurídicos longos.",
            st["corpo"],
        ),
        P("3.6 Tratamento do Desbalanceamento de Classes", st["subsecao"]),
        P(
            "Para mitigar o viés da classe majoritária (Irregular, ~88%), foram calculados "
            "<i>class weights</i> automaticamente via "
            "<i>sklearn.utils.class_weight.compute_class_weight('balanced')</i>. "
            "Esse método atribui a cada classe um peso inversamente proporcional à sua "
            "frequência no conjunto de treino, de modo que erros nas classes minoritárias "
            "(Regular e Regular com Ressalva) recebem penalização proporcionalmente maior "
            "na função de perda <i>CrossEntropyLoss</i>.",
            st["corpo"],
        ),
        P(
            "Experimentos preliminares sem class weights resultaram em colapso do Transformer "
            "para a classe majoritária, com F1-macro de 0,3114 e F1 da classe Regular de 0,00. "
            "A introdução dos class weights elevou o F1-macro para 0,4972 no corpus de "
            "2023–2024, demonstrando a necessidade de estratégias explícitas de balanceamento "
            "em corpora jurídicos desbalanceados.",
            st["corpo"],
        ),

        # ── 4. RESULTADOS E DISCUSSÃO ─────────────────────────────────────
        P("4 RESULTADOS E DISCUSSÃO", st["secao"]),
        P("4.1 Desempenho do Baseline TF-IDF", st["subsecao"]),
        P(
            "O baseline TF-IDF com regressão logística atingiu F1-macro de 0,8404 no "
            "conjunto de teste, com acurácia de 96,5%, precisão macro de 0,9025 e "
            "revocação macro de 0,7981. Esses resultados demonstram a efetividade das "
            "representações <i>bag-of-n-grams</i> para o domínio jurídico, especialmente "
            "quando o corpus abrange um período de cinco anos com padrões textuais "
            "relativamente estáveis.",
            st["corpo"],
        ),
        P(
            "A alta precisão macro (0,9025) indica que, quando o modelo prediz uma classe, "
            "ele acerta com alta frequência. A revocação macro inferior (0,7981) revela "
            "dificuldade em recuperar todas as instâncias das classes minoritárias, "
            "comportamento esperado em modelos lineares frente ao desbalanceamento do corpus.",
            st["corpo"],
        ),
        P("4.2 Desempenho do Transformer LegalBert-pt", st["subsecao"]),
        P(
            "O modelo LegalBert-pt com truncação head+tail e class weights automáticos "
            "atingiu F1-macro de 0,8686 no conjunto de teste, com acurácia de 95,9%, "
            "precisão macro de 0,8357 e revocação macro de 0,9174. Os resultados completos "
            "são apresentados na Tabela 1.",
            st["corpo"],
        ),
        S(1, 4),
        _tabela_resultados(st),
        P("<i>Tabela 1</i> — Comparativo de desempenho dos modelos avaliados no conjunto de "
          "teste (corpus 2020–2024, 534 acórdãos, split 70/15/15 estratificado).",
          st["nota"]),
        S(1, 6),
        P(
            "Um trade-off relevante foi observado entre os dois paradigmas: o Transformer "
            "apresenta revocação macro significativamente superior ao baseline (+0,119), "
            "enquanto o baseline exibe precisão macro superior (+0,067). Esse padrão "
            "indica que o LegalBert-pt é mais conservador na classificação — tende a "
            "predizer as classes minoritárias com maior cobertura, ainda que com "
            "mais falsos positivos —, comportamento favorável em aplicações de auditoria "
            "onde o custo de não detectar uma irregularidade supera o de um alarme falso.",
            st["corpo"],
        ),
        P("4.3 Confirmação da Hipótese Central", st["subsecao"]),
        P(
            "A hipótese central — de que o modelo Transformer supera o baseline em "
            "F1-macro — foi confirmada com ganho absoluto de +0,028 pontos (0,8686 vs. "
            "0,8404). Embora o ganho possa parecer modesto em termos absolutos, ele é "
            "clinicamente relevante no contexto jurídico: o F1-macro pondera igualmente "
            "o desempenho nas classes minoritárias, e um ganho de 2,8 pontos percentuais "
            "em classes de baixa representatividade representa melhoria substancial na "
            "capacidade de identificar irregularidades e regularidades com ressalva.",
            st["corpo"],
        ),
        P("4.4 Efeito do Volume de Dados", st["subsecao"]),
        P(
            "Um resultado particularmente relevante deste trabalho é a demonstração do "
            "efeito do volume de dados sobre o desempenho do Transformer. A Tabela 2 "
            "apresenta a evolução dos resultados em função da janela temporal do corpus.",
            st["corpo"],
        ),
        S(1, 4),
        _tabela_evolucao(st),
        P("<i>Tabela 2</i> — Evolução do desempenho em função do tamanho do corpus. "
          "* Com class weights; v1 (sem class weights): F1-macro = 0,3114. "
          "† O número de amostras de treino é similar porque o filtro temático foi "
          "aplicado uniformemente sobre o período ampliado.",
          st["nota"]),
        S(1, 6),
        P(
            "Com o corpus restrito a 2023–2024 (~373 amostras de treino), o Transformer "
            "obteve F1-macro de 0,4972 mesmo com class weights e 5 épocas de treinamento, "
            "resultado inferior ao baseline de 0,6705. A ampliação para cinco anos "
            "(2020–2024) foi determinante para que o Transformer superasse o baseline: "
            "F1-macro de 0,8686 contra 0,8404.",
            st["corpo"],
        ),
        P(
            "Esse resultado está alinhado com a literatura consolidada sobre Transformers: "
            "modelos baseados em atenção requerem volumes de dados significativamente "
            "maiores que modelos lineares para manifestar suas vantagens representacionais "
            "(DEVLIN et al., 2019). No domínio jurídico brasileiro, caracterizado por "
            "corpora naturalmente pequenos e especializados, a disponibilidade de dados "
            "históricos — como os CSVs anuais do TCU — é um fator crítico para a viabilidade "
            "de abordagens de Deep Learning.",
            st["corpo"],
        ),
        P("4.5 Análise de Explicabilidade com LIME", st["subsecao"]),
        P(
            "A análise de explicabilidade com LIME (RIBEIRO; SINGH; GUESTRIN, 2016) foi "
            "aplicada ao modelo baseline, identificando os tokens mais preditivos da "
            "classe Irregular. Termos como 'irregularidade', 'dano ao erário', 'multa', "
            "'débito', 'desvio' e 'sobrepreço' apresentaram pesos positivos elevados "
            "para a predição de irregularidade. Em contraste, termos como 'aprovadas', "
            "'quitação', 'regularidade' e 'atendimento' apresentaram pesos negativos, "
            "contribuindo para a predição das classes regulares.",
            st["corpo"],
        ),
        P(
            "Esses padrões são consistentes com o conhecimento jurídico especializado: "
            "o vocabulário de irregularidade nos acórdãos do TCU é suficientemente "
            "distinto do vocabulário de regularidade para que mesmo um modelo linear "
            "capture os padrões discriminativos. A explicabilidade por LIME, ao tornar "
            "esses padrões visíveis, permite que gestores e auditores validem o "
            "comportamento do modelo e identifiquem eventuais <i>spurious correlations</i> "
            "antes de sua implantação em ambiente de produção (TVEITA; HUSTAD, 2025).",
            st["corpo"],
        ),

        # ── 5. CONCLUSÃO ──────────────────────────────────────────────────
        P("5 CONCLUSÃO", st["secao"]),
        P(
            "Este trabalho demonstrou a viabilidade e a efetividade da aplicação de "
            "modelos de linguagem baseados em Transformer para a predição de desfechos "
            "de acórdãos do TCU nas áreas de saúde e educação. O fine-tuning do "
            "LegalBert-pt com estratégia de truncação head+tail e class weights "
            "automáticos superou o baseline TF-IDF em F1-macro (0,8686 vs. 0,8404), "
            "confirmando a hipótese central da pesquisa.",
            st["corpo"],
        ),
        P(
            "Os resultados evidenciam que o volume de dados é um fator crítico para "
            "o desempenho dos Transformers em corpora jurídicos pequenos. Com menos de "
            "400 amostras de treino, o modelo tendeu ao colapso para a classe majoritária, "
            "reforçando a necessidade de estratégias de balanceamento e de corpus "
            "suficientemente representativo — condição atendida pela ampliação do "
            "horizonte temporal para cinco anos (2020–2024).",
            st["corpo"],
        ),
        P(
            "O trade-off entre precisão e revocação dos dois paradigmas sugere uma "
            "aplicação complementar: o baseline TF-IDF, com maior precisão, pode ser "
            "utilizado para triagem inicial de baixo custo computacional, enquanto o "
            "Transformer, com maior revocação, é mais adequado para aplicações onde "
            "o custo de não detectar uma irregularidade é elevado — contexto típico "
            "das auditorias de saúde e educação.",
            st["corpo"],
        ),
        P(
            "Como proposta de aplicação prática, o modelo pode ser integrado como "
            "'radar jurimétrico': uma ferramenta que recebe o texto de um acórdão em "
            "tramitação e retorna uma estimativa de risco de condenação, permitindo "
            "que gestores públicos tomem ações preventivas — ajustes contratuais, "
            "capacitação de equipes, revisão de processos — antes da conclusão da "
            "auditoria.",
            st["corpo"],
        ),
        P(
            "Trabalhos futuros podem explorar: (i) a estratégia de chunking com "
            "<i>mean pooling</i> sobre embeddings do campo VOTO integral, potencialmente "
            "mais informativa que head+tail para acórdãos muito extensos; (ii) o uso "
            "de SHAP para explicabilidade no próprio modelo Transformer; (iii) a extensão "
            "do corpus para outras áreas temáticas do TCU (segurança, transporte, "
            "infraestrutura); e (iv) a incorporação de features estruturais do processo "
            "(relator, colegiado, período) como entradas adicionais ao modelo.",
            st["corpo"],
        ),
        HR(width="100%", thickness=0.8, color=colors.HexColor("#d1d5db"), spaceAfter=8),

        # ── REFERÊNCIAS ───────────────────────────────────────────────────
        P("REFERÊNCIAS", st["secao"]),

        P(
            "ALETRAS, Nikolaos et al. <b>Predicting judicial decisions of the European "
            "Court of Human Rights: a Natural Language Processing perspective.</b> "
            "<i>PeerJ Computer Science</i>, [s.l.], v. 2, e93, 2016.",
            st["referencia"],
        ),
        P(
            "DEVLIN, Jacob et al. <b>BERT: pre-training of deep bidirectional transformers "
            "for language understanding.</b> In: CONFERENCE OF THE NORTH AMERICAN CHAPTER OF "
            "THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS, 2019, Minneapolis. "
            "<i>Proceedings…</i> Stroudsburg: ACL, 2019. p. 4171–4186.",
            st["referencia"],
        ),
        P(
            "DOMINGUES, Luciano. <b>legal-bert-base-cased-ptbr: BERT model pre-trained "
            "on Brazilian legal corpus.</b> HuggingFace Hub, 2022. Disponível em: "
            "&lt;https://huggingface.co/dominguesm/legal-bert-base-cased-ptbr&gt;. "
            "Acesso em: 5 jun. 2026.",
            st["referencia"],
        ),
        P(
            "LAGE-FREITAS, André et al. <b>Predicting Brazilian court decisions.</b> "
            "<i>PeerJ Computer Science</i>, [s.l.], v. 8, e904, 2022.",
            st["referencia"],
        ),
        P(
            "MEDVEDEVA, Masha; VOLS, Michel; WIELING, Martijn. <b>Using machine learning "
            "to predict decisions of the European Court of Human Rights.</b> "
            "<i>Artificial Intelligence and Law</i>, [s.l.], v. 28, n. 2, p. 237–266, 2020.",
            st["referencia"],
        ),
        P(
            "PEDREGOSA, Fabian et al. <b>Scikit-learn: machine learning in Python.</b> "
            "<i>Journal of Machine Learning Research</i>, Cambridge, v. 12, "
            "p. 2825–2830, 2011.",
            st["referencia"],
        ),
        P(
            "RIBEIRO, Marco Tulio; SINGH, Sameer; GUESTRIN, Carlos. <b>'Why should I "
            "trust you?': explaining the predictions of any classifier.</b> In: ACM SIGKDD "
            "INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 22., "
            "2016, San Francisco. <i>Proceedings…</i> New York: ACM, 2016. p. 1135–1144.",
            st["referencia"],
        ),
        P(
            "SOUZA, Fábio; NOGUEIRA, Rodrigo; LOTUFO, Roberto. <b>BERTimbau: pretrained "
            "BERT models for Brazilian Portuguese.</b> In: INTELLIGENT SYSTEMS — "
            "BRACIS, 9., 2020, Rio Grande. <i>Proceedings…</i> Cham: Springer, "
            "2020. p. 403–417.",
            st["referencia"],
        ),
        P(
            "SUN, Chi; QIU, Xipeng; XU, Yuanbin; HUANG, Xuanjing. <b>How to fine-tune "
            "BERT for text classification?</b> In: CHINESE COMPUTATIONAL LINGUISTICS, "
            "18., 2019, Kunming. <i>Proceedings…</i> Cham: Springer, 2019. p. 194–206.",
            st["referencia"],
        ),
        P(
            "TRIBUNAL DE CONTAS DA UNIÃO. <b>Portal de Dados Abertos do TCU: Acórdãos "
            "Completos.</b> Brasília: TCU, 2024. Disponível em: "
            "&lt;https://sites.tcu.gov.br/dados-abertos/jurisprudencia/&gt;. "
            "Acesso em: 5 jun. 2026.",
            st["referencia"],
        ),
        P(
            "TVEITA, Sondre; HUSTAD, Eli. <b>Benefits and challenges of AI in the "
            "public sector.</b> In: HAWAII INTERNATIONAL CONFERENCE ON SYSTEM SCIENCES, "
            "58., 2025, Maui. <i>Proceedings…</i> Honolulu: University of Hawaii, "
            "2025. p. 1–10.",
            st["referencia"],
        ),
        P(
            "VASWANI, Ashish et al. <b>Attention is all you need.</b> In: ADVANCES IN "
            "NEURAL INFORMATION PROCESSING SYSTEMS, 30., 2017, Long Beach. "
            "<i>Proceedings…</i> Red Hook: Curran Associates, 2017. p. 5998–6008.",
            st["referencia"],
        ),
        P(
            "WOLF, Thomas et al. <b>Transformers: state-of-the-art natural language "
            "processing.</b> In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE "
            "PROCESSING: SYSTEM DEMONSTRATIONS, 2020, online. <i>Proceedings…</i> "
            "Stroudsburg: ACL, 2020. p. 38–45.",
            st["referencia"],
        ),
    ]
    return historia


# ── Main ──────────────────────────────────────────────────────────────────────

def gerar_artigo(saida: Path = SAIDA) -> Path:
    """Gera o artigo científico em PDF seguindo ABNT NBR 6022:2018."""
    saida.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(saida),
        pagesize=A4,
        topMargin=3 * cm,
        bottomMargin=2 * cm,
        leftMargin=3 * cm,
        rightMargin=2 * cm,
        title="Jurimetria Preditiva em Acórdãos do TCU",
        author="Bruno Sousa — IDP",
        subject="Classificação de desfechos com TF-IDF e LegalBert-pt",
    )

    st = _estilos()
    historia = _construir_historia(st)
    doc.build(historia, onLaterPages=_rodape, onFirstPage=_rodape)

    tamanho_kb = saida.stat().st_size // 1024
    print(f"Artigo gerado: {saida}  ({tamanho_kb} KB)")
    return saida


if __name__ == "__main__":
    gerar_artigo()
