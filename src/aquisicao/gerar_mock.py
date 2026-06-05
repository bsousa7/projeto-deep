"""Gerador de dataset sintético com estrutura idêntica ao CSV do TCU.

Cria dados com vocabulário jurídico sobreposto entre classes para simular
a dificuldade real de classificação em acórdãos do TCU.

NÃO usar como dado real — apenas para testes de integração do pipeline.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# ---- Vocabulário compartilhado (aparece em todas as classes) ----
_CONTEXTOS = [
    "Tomada de contas especial. Município",
    "Prestação de contas. Exercício",
    "Auditoria operacional",
    "Representação",
    "Denúncia",
    "Auditoria de conformidade",
    "Monitoramento",
]

_TEMAS_SAUDE = [
    "Secretaria Municipal de Saúde",
    "Ministério da Saúde",
    "recursos do SUS",
    "Programa Saúde da Família",
    "atenção básica",
    "aquisição de medicamentos",
    "equipamentos médicos",
    "hospital municipal",
    "UBS",
    "convênio saúde",
]

_TEMAS_EDUCACAO = [
    "FNDE",
    "Programa Nacional de Alimentação Escolar",
    "merenda escolar",
    "Secretaria Municipal de Educação",
    "obras em escolas",
    "reforma escolar",
    "PDDE",
    "Programa Dinheiro Direto na Escola",
    "secretaria de educação estadual",
    "convênio educação",
]

_IRREGULARIDADES_LEVES = [
    "Falhas formais na documentação",
    "Ausência de extratos bancários",
    "Pequenas impropriedades na execução",
    "Irregularidades sanáveis identificadas",
    "Comprovantes incompletos",
    "Atrasos na execução",
    "Falhas nos registros contábeis",
    "Controles internos deficientes",
    "Impropriedades de gestão",
    "Ausência de controles adequados",
]

_IRREGULARIDADES_GRAVES = [
    "Desvio de verbas federais",
    "Superfaturamento identificado",
    "Licitação irregular",
    "Dispensa indevida de licitação",
    "Sobrepreço comprovado",
    "Dano ao erário apurado",
    "Malversação de recursos",
    "Fraude em processo licitatório",
    "Locupletamento",
    "Cartel identificado",
]

_DESFECHOS = {
    "Irregular": [
        "Contas irregulares. Débito integral. Multa ao gestor.",
        "Julgamento irregular. Condenação solidária. Multa.",
        "Contas irregulares. Débito apurado. Inabilitação para cargo público.",
        "Irregularidade grave. Multa e débito. Comunicação ao MPF.",
        "Contas irregulares. Débito. Multa. Determinações cautelares.",
    ],
    "Regular com Ressalva": [
        "Contas regulares com ressalva. Determinações ao gestor.",
        "Julgamento regular com ressalva. Recomendações de melhoria.",
        "Contas regulares com ressalva. Alertas expedidos.",
        "Regular com ressalva. Prazo para saneamento das falhas.",
        "Contas regulares com ressalva. Ciência ao responsável.",
    ],
    "Regular": [
        "Contas regulares. Quitação plena.",
        "Julgamento regular. Quitação ao responsável.",
        "Contas regulares. Sem ressalvas.",
        "Contas regulares. Aprovação com elogios.",
        "Contas regulares. Arquivamento do processo.",
    ],
}

_COLEGIADOS = ["Plenário", "1ª Câmara", "2ª Câmara"]
_RELATORES = [
    "Ministro Augusto Nardes",
    "Ministro Bruno Dantas",
    "Ministro Vital do Rêgo",
    "Ministro Aroldo Cedraz",
    "Ministra Ana Arraes",
    "Ministro Raimundo Carreiro",
    "Ministro Jorge Oliveira",
]
_MUNICIPIOS = [
    "de São Paulo/SP", "do Rio de Janeiro/RJ", "de Manaus/AM",
    "de Fortaleza/CE", "de Cuiabá/MT", "de Belém/PA",
    "de Salvador/BA", "de Recife/PE", "de Porto Alegre/RS",
    "de Goiânia/GO", "de Natal/RN", "de Maceió/AL",
]


def _gerar_sumario(desfecho: str, incluir_veredicto: bool = True) -> str:
    """Gera sumário com vocabulário compartilhado + sinal de classe variável.

    Args:
        desfecho: Classe do acórdão.
        incluir_veredicto: Se True, inclui a frase de desfecho explícita no texto.
            Use False para simular o campo 'voto' (sem o veredicto literal).
    """
    tema_saude = random.random() > 0.5
    temas = _TEMAS_SAUDE if tema_saude else _TEMAS_EDUCACAO

    contexto = random.choice(_CONTEXTOS)
    municipio = random.choice(_MUNICIPIOS)
    tema = random.choice(temas)

    # Irregularidades: graves p/ Irregular, leves p/ Regular com Ressalva,
    # ocasionalmente com ruído realista (15–20% de overlap intencional).
    if desfecho == "Irregular":
        irr1 = random.choice(_IRREGULARIDADES_GRAVES)
        irr2 = random.choice(_IRREGULARIDADES_GRAVES + _IRREGULARIDADES_LEVES)
        corpo = f"{irr1}. {irr2}."
    elif desfecho == "Regular com Ressalva":
        irr1 = random.choice(_IRREGULARIDADES_LEVES)
        if random.random() < 0.20:
            irr2 = f"Sem evidência de {random.choice(_IRREGULARIDADES_GRAVES).lower()}."
        else:
            irr2 = random.choice(_IRREGULARIDADES_LEVES) + "."
        corpo = f"{irr1}. {irr2}"
    else:  # Regular
        if random.random() < 0.15:
            irr = random.choice(_IRREGULARIDADES_LEVES)
            corpo = f"{irr}. Sem dano ao erário. Falhas sanadas."
        else:
            corpo = "Execução regular. Documentação completa. Sem irregularidades."

    base = f"{contexto}. {municipio}. {tema}. {corpo}"
    if incluir_veredicto:
        return f"{base} {random.choice(_DESFECHOS[desfecho])}"
    return base


def gerar_mock_csv(
    ano: int = 2024,
    n_irregular: int = 300,
    n_ressalva: int = 150,
    n_regular: int = 550,
    n_outros: int = 4000,
    destino: Path = DATA_RAW,
) -> Path:
    """Gera CSV sintético com estrutura idêntica ao acordao-completo-AAAA.csv do TCU.

    Args:
        ano: Ano do arquivo simulado.
        n_irregular: Número de acórdãos irregulares temáticos.
        n_ressalva: Número de acórdãos regulares com ressalva temáticos.
        n_regular: Número de acórdãos regulares temáticos.
        n_outros: Número de acórdãos de outros temas (para testar o filtro).
        destino: Diretório de saída.

    Returns:
        Caminho do arquivo gerado.
    """
    registros = []
    numero_base = 1000 * (ano - 2000)

    pool = [
        ("Irregular", n_irregular),
        ("Regular com Ressalva", n_ressalva),
        ("Regular", n_regular),
    ]

    for desfecho, quantidade in pool:
        for _ in range(quantidade):
            numero_base += 1
            registros.append({
                "numeroAcordao": numero_base,
                "anoAcordao": ano,
                "tipo": "Acórdão",
                "situacao": desfecho,
                # sumario real inclui veredicto explícito (como no CSV do TCU)
                # texto_voto_simulado omite veredicto — simula o campo 'voto'
                "sumario": _gerar_sumario(desfecho, incluir_veredicto=True),
                "texto_voto_simulado": _gerar_sumario(desfecho, incluir_veredicto=False),
                "colegiado": random.choice(_COLEGIADOS),
                "relator": random.choice(_RELATORES),
                "dataSessao": f"{ano}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "urlArquivoPDF": (
                    f"https://pesquisa.apps.tcu.gov.br/documento/acordao-completo/{numero_base}"
                ),
            })

    _sumarios_outros = [
        "Prestação de contas. Ministério da Infraestrutura. Obras rodoviárias.",
        "Auditoria. Banco do Brasil. Operações de crédito rural.",
        "Tomada de contas. Ministério da Defesa. Aquisição de equipamentos.",
        "Representação. Petrobras. Contrato de fornecimento. Sobrepreço.",
        "Auditoria. BNDES. Concessão de crédito a grandes empresas.",
        "Prestação de contas. Ministério de Minas e Energia. Leilão.",
        "Tomada de contas especial. Contrato de consultoria. Dano.",
    ]
    for _ in range(n_outros):
        numero_base += 1
        registros.append({
            "numeroAcordao": numero_base,
            "anoAcordao": ano,
            "tipo": "Acórdão",
            "situacao": random.choice(["Regular", "Irregular", "Regular com Ressalva"]),
            "sumario": random.choice(_sumarios_outros),
            "colegiado": random.choice(_COLEGIADOS),
            "relator": random.choice(_RELATORES),
            "dataSessao": f"{ano}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "urlArquivoPDF": (
                f"https://pesquisa.apps.tcu.gov.br/documento/acordao-completo/{numero_base}"
            ),
        })

    df = pd.DataFrame(registros).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    destino.mkdir(parents=True, exist_ok=True)
    saida = destino / f"acordao-completo-{ano}.csv"
    df.to_csv(saida, index=False, encoding="utf-8")
    print(f"Mock CSV gerado: {saida} ({len(df)} registros, {saida.stat().st_size / 1e6:.1f} MB)")
    print(f"  Temáticos: {n_irregular + n_ressalva + n_regular} | Outros: {n_outros}")
    return saida


if __name__ == "__main__":
    for ano in [2023, 2024]:
        gerar_mock_csv(ano=ano)
