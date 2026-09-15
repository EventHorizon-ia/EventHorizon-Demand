import argparse
import os
import re
import sys
import unicodedata
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================
FEATURES_MODELO = [
    "store_code",
    "dia_seno",
    "dia_cos",
    "mes_seno",
    "mes_cos",
    "lag_7",
    "lag_14",
    "lag_28",
    "ewma_7",
    "ewma_28",
    "rolling_std_7",
    "fim_de_semana",
    "evento",
    "sell_price",
    "price_change",
    "snap",
]

# Mesma codificação usada no treinamento
STORE_CODE_MAP = {
    "CA_1": 0,
    "CA_2": 1,
    "CA_3": 2,
    "TX_1": 3,
    "TX_2": 4,
    "WI_1": 5,
    "WI_2": 6,
}

# ============================================================
# UTILITÁRIOS
# ============================================================
def normalizar_nome_coluna(nome: str) -> str:
    """Normaliza nome de coluna para facilitar a detecção."""
    nome = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode("utf-8")
    nome = nome.lower().strip()
    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    return nome.strip("_")


def encontrar_coluna(df: pd.DataFrame, candidatos: list) -> Optional[str]:
    """Encontra uma coluna usando vários nomes possíveis."""
    mapa = {normalizar_nome_coluna(c): c for c in df.columns}
    for cand in candidatos:
        norm = normalizar_nome_coluna(cand)
        if norm in mapa:
            return mapa[norm]
    return None


def ler_csv_robusto(caminho: str) -> pd.DataFrame:
    """Lê CSV tentando diferentes codificações."""
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if os.path.getsize(caminho) == 0:
        raise ValueError(f"Arquivo CSV vazio: {caminho}")

    encodings = [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "latin-1",
        "cp1252",
    ]

    for enc in encodings:
        try:
            df = pd.read_csv(caminho, encoding=enc)
            if not df.empty:
                return df
        except (
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ):
            continue

    return pd.read_csv(caminho)


# ============================================================
# PREPARAÇÃO DA SÉRIE DIÁRIA
# ============================================================
def para_serie_diaria(
    df: pd.DataFrame,
    col_data: str,
    col_vendas: str,
    col_preco: Optional[str] = None,
) -> pd.DataFrame:
    """
    Converte os dados para uma série diária.

    Vendas:
        - soma por dia
        - dias sem venda = 0

    Preço:
        - média diária
        - forward fill
        - backward fill
    """
    df = df.copy()

    df["_data"] = pd.to_datetime(df[col_data], errors="coerce").dt.normalize()
    df[col_vendas] = pd.to_numeric(df[col_vendas], errors="coerce")
    df = df.dropna(subset=["_data", col_vendas])

    if df.empty:
        raise ValueError("Nenhuma venda válida após conversão de data/valor.")

    agg = {col_vendas: "sum"}

    if col_preco:
        df[col_preco] = pd.to_numeric(df[col_preco], errors="coerce")
        agg[col_preco] = "mean"

    diario = df.groupby("_data").agg(agg)
    diario.index = pd.to_datetime(diario.index)
    diario = diario.sort_index()

    # Cria todos os dias do intervalo
    intervalo = pd.date_range(
        start=diario.index.min(),
        end=diario.index.max(),
        freq="D",
    )
    diario = diario.reindex(intervalo)

    # Dias sem venda = zero
    diario[col_vendas] = diario[col_vendas].fillna(0)

    # Tratamento de preço
    if col_preco:
        diario[col_preco] = diario[col_preco].ffill().bfill()
        if diario[col_preco].isna().any():
            diario[col_preco] = diario[col_preco].fillna(0)

    return diario


# ============================================================
# FEATURES
# ============================================================
def criar_features_predicao(
    serie_vendas: pd.Series,
    data_futura: pd.Timestamp,
    serie_preco: Optional[pd.Series] = None,
    store_code: int = 0,
    snap: int = 0,
    evento: int = 0,
) -> dict:
    """
    Cria exatamente as features necessárias pelo modelo.

    Não utiliza lag_365.

    A série_vendas contém:
        - histórico real
        - previsões anteriores durante previsão recursiva
    """
    if len(serie_vendas) < 28:
        raise ValueError(
            f"Série histórica insuficiente: {len(serie_vendas)} dias. "
            f"São necessários pelo menos 28 dias."
        )

    # --------------------------------------------------------
    # Calendário
    # --------------------------------------------------------
    # Python: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
    py_dow = data_futura.dayofweek

    # Conversão para o padrão M5:
    # 1=Saturday, 2=Sunday, 3=Monday, 4=Tuesday, 5=Wednesday, 6=Thursday, 7=Friday
    m5_wday = ((py_dow + 2) % 7) + 1

    # --------------------------------------------------------
    # Features cíclicas
    # --------------------------------------------------------
    dia_seno = np.sin(2 * np.pi * m5_wday / 7)
    dia_cos = np.cos(2 * np.pi * m5_wday / 7)
    mes_seno = np.sin(2 * np.pi * data_futura.month / 12)
    mes_cos = np.cos(2 * np.pi * data_futura.month / 12)

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------
    lag_7 = float(serie_vendas.iloc[-7])
    lag_14 = float(serie_vendas.iloc[-14])
    lag_28 = float(serie_vendas.iloc[-28])

    # --------------------------------------------------------
    # EWMA
    # Treinamento: vendas.shift(1).ewm(span=7, adjust=False).mean()
    # --------------------------------------------------------
    serie_anterior = serie_vendas.iloc[:-0] if False else serie_vendas
    ewma_7 = float(serie_anterior.ewm(span=7, adjust=False).mean().iloc[-1])
    ewma_28 = float(serie_anterior.ewm(span=28, adjust=False).mean().iloc[-1])

    # --------------------------------------------------------
    # VOLATILIDADE
    # Treinamento: vendas.shift(1).rolling(7).std()
    # --------------------------------------------------------
    rolling_std_7 = float(serie_vendas.iloc[-7:].std())
    if np.isnan(rolling_std_7):
        rolling_std_7 = 0.0

    # --------------------------------------------------------
    # FIM DE SEMANA
    # --------------------------------------------------------
    fim_de_semana = int(m5_wday in [1, 2])

    # --------------------------------------------------------
    # PREÇO
    # --------------------------------------------------------
    if serie_preco is not None and len(serie_preco) > 0:
        serie_preco = serie_preco.dropna()
        if len(serie_preco) > 0:
            preco_ultimo = float(serie_preco.iloc[-1])
            sell_price = preco_ultimo
            preco_futuro = preco_ultimo
            if preco_ultimo != 0:
                price_change = (preco_futuro / preco_ultimo) - 1.0
            else:
                price_change = 0.0
        else:
            sell_price = 0.0
            price_change = 0.0
    else:
        sell_price = 0.0
        price_change = 0.0

    # --------------------------------------------------------
    # FEATURES FINAIS
    # --------------------------------------------------------
    features = {
        "store_code": int(store_code),
        "dia_seno": float(dia_seno),
        "dia_cos": float(dia_cos),
        "mes_seno": float(mes_seno),
        "mes_cos": float(mes_cos),
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "ewma_7": ewma_7,
        "ewma_28": ewma_28,
        "rolling_std_7": rolling_std_7,
        "fim_de_semana": fim_de_semana,
        "evento": int(evento),
        "sell_price": float(sell_price),
        "price_change": float(price_change),
        "snap": int(snap),
    }

    return features


# ============================================================
# PREVISÃO RECURSIVA
# ============================================================
def prever_horizonte(
    modelo,
    serie_vendas: pd.Series,
    serie_preco: Optional[pd.Series],
    data_inicial: pd.Timestamp,
    horizonte: int,
    store_code: int,
    snap_d1: int,
    evento_d1: int,
) -> list:
    """
    Faz previsão recursiva.

    D+1: utiliza histórico real.
    D+2 em diante: utiliza as previsões anteriores para construir os lags seguintes.
    SNAP e evento: as flags recebidas pela CLI são aplicadas em D+1. Para os dias seguintes, permanecem 0.
    """
    vendas = serie_vendas.copy()
    precos = serie_preco.copy() if serie_preco is not None else None
    previsoes = []
    data_atual = pd.Timestamp(data_inicial)

    for passo in range(horizonte):
        if passo == 0:
            snap_atual = int(snap_d1)
            evento_atual = int(evento_d1)
        else:
            snap_atual = 0
            evento_atual = 0

        features = criar_features_predicao(
            serie_vendas=vendas,
            data_futura=data_atual,
            serie_preco=precos,
            store_code=store_code,
            snap=snap_atual,
            evento=evento_atual,
        )

        X = pd.DataFrame([features], columns=FEATURES_MODELO).astype(float)
        pred = float(modelo.predict(X)[0])
        pred = max(0.0, pred)  # Demanda não pode ser negativa

        previsoes.append((data_atual, pred))
        vendas.loc[data_atual] = pred

        if precos is not None:
            preco_atual = float(precos.iloc[-1]) if len(precos) > 0 else 0.0
            precos.loc[data_atual] = preco_atual

        data_atual += pd.Timedelta(days=1)

    return previsoes


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Previsão de demanda recursiva com LightGBM M5."
    )
    parser.add_argument("csv", help="Caminho do CSV com histórico de vendas.")
    parser.add_argument(
        "--horizonte",
        type=int,
        default=7,
        help="Quantidade de dias à frente para prever. Default: 7.",
    )
    parser.add_argument(
        "--modelo",
        default="modelo_demanda_mvp.pkl",
        help="Caminho do modelo PKL. Default: modelo_demanda_mvp.pkl.",
    )
    parser.add_argument("--store-id", default=None, help="Store ID M5, por exemplo CA_1.")
    parser.add_argument(
        "--snap", type=int, choices=[0, 1], default=0, help="Flag SNAP para D+1. Default: 0."
    )
    parser.add_argument(
        "--evento", type=int, choices=[0, 1], default=0, help="Flag evento para D+1. Default: 0."
    )
    parser.add_argument("--data-col", default=None, help="Nome da coluna de data.")
    parser.add_argument("--vendas-col", default=None, help="Nome da coluna de vendas.")
    parser.add_argument("--preco-col", default=None, help="Nome da coluna de preço.")

    args = parser.parse_args()

    # ========================================================
    # VALIDAR HORIZONTE
    # ========================================================
    if args.horizonte < 1:
        print("Erro: horizonte deve ser >= 1.", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # CARREGAR MODELO
    # ========================================================
    try:
        modelo = joblib.load(args.modelo)
    except FileNotFoundError:
        print(f"Erro: modelo não encontrado em '{args.modelo}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # CARREGAR CSV
    # ========================================================
    try:
        df = ler_csv_robusto(args.csv)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # DETECTAR COLUNAS
    # ========================================================
    col_data = args.data_col or encontrar_coluna(
        df, ["date", "data", "ds", "dia", "timestamp"]
    )
    col_vendas = args.vendas_col or encontrar_coluna(
        df, ["vendas", "sales", "venda", "demanda", "demand", "qty", "quantity", "volume", "total", "y"]
    )

    if not col_data:
        print("Erro: não foi possível identificar a coluna de data. Use --data-col.", file=sys.stderr)
        sys.exit(1)

    if not col_vendas:
        print("Erro: não foi possível identificar a coluna de vendas. Use --vendas-col.", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # NORMALIZAR DADOS
    # ========================================================
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
    df[col_vendas] = pd.to_numeric(df[col_vendas], errors="coerce")
    df = df.dropna(subset=[col_data, col_vendas])

    if df.empty:
        print("Erro: nenhuma linha válida após conversão.", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # COLUNAS OPCIONAIS
    # ========================================================
    col_preco = args.preco_col or encontrar_coluna(
        df, ["price", "sell_price", "preco", "preço", "valor", "unit_price"]
    )
    col_store = encontrar_coluna(df, ["store_id", "store", "loja", "storeid"])
    col_produto = encontrar_coluna(
        df, ["item_id", "product_id", "produto", "sku", "item", "product"]
    )

    # ========================================================
    # AVISOS
    # ========================================================
    if col_preco is None:
        print(
            "Aviso: coluna de preço não encontrada. Será usado sell_price=0 e price_change=0.",
            file=sys.stderr,
        )

    if not col_store and not args.store_id:
        print(
            "Aviso: nenhuma loja identificada. Será usado store_code=0 (CA_1).",
            file=sys.stderr,
        )

    # ========================================================
    # AGRUPAMENTO
    # ========================================================
    group_cols = []
    if col_produto:
        group_cols.append(col_produto)
    if col_store:
        group_cols.append(col_store)

    resultados_agrupados = {}
    erros = []

    # ========================================================
    # SEM GRUPAMENTO
    # ========================================================
    if not group_cols:
        try:
            diario = para_serie_diaria(df, col_data, col_vendas, col_preco)
            serie_vendas = diario[col_vendas]
            serie_preco = diario[col_preco] if col_preco else None
            data_inicial = serie_vendas.index[-1] + pd.Timedelta(days=1)
            store_code = (
                STORE_CODE_MAP.get(args.store_id, 0) if args.store_id else 0
            )

            previsoes = prever_horizonte(
                modelo=modelo,
                serie_vendas=serie_vendas,
                serie_preco=serie_preco,
                data_inicial=data_inicial,
                horizonte=args.horizonte,
                store_code=store_code,
                snap_d1=args.snap,
                evento_d1=args.evento,
            )
            resultados_agrupados["Total"] = previsoes
        except Exception as e:
            print(f"Erro ao gerar previsão: {e}", file=sys.stderr)
            sys.exit(1)

    # ========================================================
    # COM GRUPAMENTO
    # ========================================================
    else:
        for col in group_cols:
            df[col] = df[col].fillna("desconhecido")

        for chaves, grupo in df.groupby(group_cols):
            if not isinstance(chaves, tuple):
                chaves = (chaves,)

            chave_str = " | ".join(str(c) for c in chaves)

            try:
                diario = para_serie_diaria(grupo, col_data, col_vendas, col_preco)
                serie_vendas = diario[col_vendas]
                serie_preco = diario[col_preco] if col_preco else None
                data_inicial = serie_vendas.index[-1] + pd.Timedelta(days=1)

                if col_store:
                    store_id = str(grupo[col_store].iloc[0])
                    store_code = STORE_CODE_MAP.get(store_id, -1)

                    if store_code == -1:
                        print(
                            f"⚠️ Aviso: loja '{store_id}' não mapeada. Usando store_code=0 (CA_1).",
                            file=sys.stderr,
                        )
                        store_code = 0
                else:
                    store_code = (
                        STORE_CODE_MAP.get(args.store_id, 0) if args.store_id else 0
                    )

                previsoes = prever_horizonte(
                    modelo=modelo,
                    serie_vendas=serie_vendas,
                    serie_preco=serie_preco,
                    data_inicial=data_inicial,
                    horizonte=args.horizonte,
                    store_code=store_code,
                    snap_d1=args.snap,
                    evento_d1=args.evento,
                )
                resultados_agrupados[chave_str] = previsoes
            except Exception as e:
                erros.append((chave_str, str(e)))

    # ========================================================
    # VALIDAR RESULTADOS
    # ========================================================
    if not resultados_agrupados:
        print("Erro: nenhuma previsão pôde ser gerada.", file=sys.stderr)
        sys.exit(1)

    # ========================================================
    # RESULTADO
    # ========================================================
    print()
    print(f"📊 Previsão de demanda ({args.horizonte} dias)")
    print("=" * 60)

    total_horizonte = 0.0

    for chave, previsoes in resultados_agrupados.items():
        print(f"🔹 {chave}")
        soma_grupo = 0.0

        for i, (data, pred) in enumerate(previsoes):
            soma_grupo += pred
            if i == 0:
                confianca = "[D+1]"
                margem = ""
            else:
                confianca = "[RECURSIVA]"
                margem_pct = 0.05 * i
                margem = f" (±{pred * margem_pct:.1f})"

            print(
                f"   D+{i+1} ({data.strftime('%d/%m/%Y')}): "
                f"{pred:7.2f} unidades {confianca}{margem}"
            )

        print(f"   Total do Grupo no Período: {soma_grupo:.2f}")
        print("-" * 60)
        total_horizonte += soma_grupo

    # ========================================================
    # TOTAL GERAL
    # ========================================================
    if len(resultados_agrupados) > 1:
        print(
            f"🚀 TOTAL ACUMULADO GERAL ({args.horizonte} dias): "
            f"{total_horizonte:.2f} unidades"
        )
        print("=" * 60)

    # ========================================================
    # ERROS PARCIAIS
    # ========================================================
    if erros:
        print()
        print("⚠️ Alguns grupos falharam:")
        for chave, erro in erros:
            print(f"  {chave}: {erro}")


if __name__ == "__main__":
    main()