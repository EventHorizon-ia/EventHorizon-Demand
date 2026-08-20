#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mvp.py — Previsão de demanda D+1 com LightGBM treinado no M5 Walmart.

O modelo carregado (`modelo_demanda_mvp.pkl`) espera as features:
    store_code, dia_seno, dia_cos, mes_seno, mes_cos,
    lag_7, lag_14, lag_28, ewma_7, ewma_28,
    rolling_std_7, fim_de_semana, evento,
    sell_price, price_change, snap

Não usa lag_365.
lightgbm 4.5
"""

import argparse
import os
import re
import sys
import unicodedata
from typing import Optional

import joblib
import numpy as np
import pandas as pd

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

STORE_CODE_MAP = {
    "CA_1": 0,
    "CA_2": 1,
    "CA_3": 2,
    "TX_1": 3,
    "TX_2": 4,
    "WI_1": 5,
    "WI_2": 6,
}


def normalizar_nome_coluna(nome: str) -> str:
    """Normaliza nome de coluna para facilitar detecção."""
    nome = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode("utf-8")
    nome = nome.lower().strip()
    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    return nome.strip("_")


def encontrar_coluna(df: pd.DataFrame, candidatos: list) -> Optional[str]:
    """Encontra a primeira coluna do DataFrame que corresponde a um dos candidatos."""
    mapa = {normalizar_nome_coluna(c): c for c in df.columns}
    for cand in candidatos:
        norm = normalizar_nome_coluna(cand)
        if norm in mapa:
            return mapa[norm]
    return None


def ler_csv_robusto(caminho: str) -> pd.DataFrame:
    """Lê CSV tentando várias codificações e verifica se não está vazio."""
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    if os.path.getsize(caminho) == 0:
        raise ValueError(f"Arquivo CSV vazio (0 bytes): {caminho}")
    
    encodings = ["utf-8-sig", "utf-8", "utf-16", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(caminho, encoding=enc)
            if not df.empty:
                return df
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
    
    # Última tentativa sem encoding específico
    return pd.read_csv(caminho)


def para_serie_diaria(
    df: pd.DataFrame,
    col_data: str,
    col_vendas: str,
    col_preco: Optional[str] = None,
) -> pd.DataFrame:
    """
    Agrupa vendas por dia, reindexa para frequência diária e trata preço.
    """
    df = df.copy()

    # normaliza data e vendas
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

    # reindexa para todos os dias entre a primeira e a última data
    intervalo = pd.date_range(start=diario.index.min(), end=diario.index.max(), freq="D")
    diario = diario.reindex(intervalo)

    diario[col_vendas] = diario[col_vendas].fillna(0)

    if col_preco:
        diario[col_preco] = diario[col_preco].ffill().bfill()
        if diario[col_preco].isna().any():
            diario[col_preco] = diario[col_preco].fillna(0)

    return diario


def criar_features_predicao(
    serie_vendas: pd.Series,
    data_futura: pd.Timestamp,
    serie_preco: Optional[pd.Series] = None,
    store_code: int = 0,
    snap: int = 0,
    evento: int = 0,
) -> dict:
    """
    Cria as features para D+1 seguindo exatamente o pipeline do notebook.
    """
    if len(serie_vendas) < 28:
        raise ValueError(
            f"Série histórica insuficiente: {len(serie_vendas)} dias. "
            "São necessários pelo menos 28 dias de vendas para criar lag_28."
        )

    ultimo_dia = serie_vendas.index[-1]
    if pd.to_datetime(data_futura) != ultimo_dia + pd.Timedelta(days=1):
        raise ValueError(
            f"data_futura ({data_futura}) deve ser exatamente um dia após o último dado ({ultimo_dia})."
        )

    # M5 wday: 1=Sáb, 2=Dom, 3=Seg, 4=Ter, 5=Qua, 6=Qui, 7=Sex
    py_dow = data_futura.dayofweek
    m5_wday = ((py_dow + 2) % 7) + 1

    dia_seno = np.sin(2 * np.pi * m5_wday / 7)
    dia_cos = np.cos(2 * np.pi * m5_wday / 7)
    mes_seno = np.sin(2 * np.pi * data_futura.month / 12)
    mes_cos = np.cos(2 * np.pi * data_futura.month / 12)

    lag_7 = serie_vendas.iloc[-7]
    lag_14 = serie_vendas.iloc[-14]
    lag_28 = serie_vendas.iloc[-28]

    ewma_7 = serie_vendas.ewm(span=7, adjust=False).mean().iloc[-1]
    ewma_28 = serie_vendas.ewm(span=28, adjust=False).mean().iloc[-1]
    rolling_std_7 = serie_vendas.iloc[-7:].std()

    # Reproduz o comportamento do notebook:
    # fim_de_semana = wday.isin([0,6]) sobre wday M5 (1..7) -> marca apenas wday=6
    fim_de_semana = 1 if m5_wday == 6 else 0

    if serie_preco is not None and len(serie_preco) > 0 and not serie_preco.iloc[-7:].isna().all():
        preco_ultimo = serie_preco.iloc[-1]
        preco_futuro = preco_ultimo  # sem preço futuro informado, mantém o último preço
        price_change = 0.0 if preco_ultimo == 0 else (preco_futuro / preco_ultimo) - 1
        sell_price = preco_futuro
    else:
        sell_price = 0.0
        price_change = 0.0

    features = {
        "store_code": store_code,
        "dia_seno": dia_seno,
        "dia_cos": dia_cos,
        "mes_seno": mes_seno,
        "mes_cos": mes_cos,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "ewma_7": ewma_7,
        "ewma_28": ewma_28,
        "rolling_std_7": rolling_std_7,
        "fim_de_semana": fim_de_semana,
        "evento": evento,
        "sell_price": sell_price,
        "price_change": price_change,
        "snap": snap,
    }
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Previsão de demanda D+1 usando modelo LightGBM M5 (modelo_demanda_mvp.pkl)."
    )
    parser.add_argument("csv", help="Caminho do CSV com histórico de vendas.")
    parser.add_argument(
        "--modelo",
        default="modelo_demanda_mvp.pkl",
        help="Caminho do modelo pkl (default: modelo_demanda_mvp.pkl).",
    )
    parser.add_argument(
        "--store-id",
        default=None,
        help="Store ID M5 (ex.: CA_1). Usado para mapear store_code se o CSV não tiver coluna de loja.",
    )
    parser.add_argument(
        "--snap",
        type=int,
        choices=[0, 1],
        default=0,
        help="Flag SNAP para D+1 (default: 0).",
    )
    parser.add_argument(
        "--evento",
        type=int,
        choices=[0, 1],
        default=0,
        help="Flag evento para D+1 (default: 0).",
    )
    parser.add_argument("--data-col", default=None, help="Nome da coluna de data (opcional).")
    parser.add_argument("--vendas-col", default=None, help="Nome da coluna de vendas (opcional).")
    parser.add_argument("--preco-col", default=None, help="Nome da coluna de preço (opcional).")
    args = parser.parse_args()

    # carrega modelo
    try:
        modelo = joblib.load(args.modelo)
    except FileNotFoundError:
        print(f"Erro: modelo não encontrado em '{args.modelo}'.", file=sys.stderr)
        sys.exit(1)

    # carrega CSV
    try:
        df = ler_csv_robusto(args.csv)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # detecta colunas essenciais
    col_data = args.data_col or encontrar_coluna(df, ["date", "data", "ds", "dia", "timestamp"])
    col_vendas = args.vendas_col or encontrar_coluna(
        df, ["vendas", "sales", "venda", "demanda", "demand", "qty", "quantity", "volume", "total", "y"]
    )

    if not col_data:
        print("Erro: não foi possível identificar a coluna de data. Use --data-col.", file=sys.stderr)
        sys.exit(1)
    if not col_vendas:
        print("Erro: não foi possível identificar a coluna de vendas. Use --vendas-col.", file=sys.stderr)
        sys.exit(1)

    # normaliza data e vendas
    df[col_data] = pd.to_datetime(df[col_data], errors="coerce")
    df[col_vendas] = pd.to_numeric(df[col_vendas], errors="coerce")
    df = df.dropna(subset=[col_data, col_vendas])

    if df.empty:
        print("Erro: nenhuma linha válida após conversão de data/vendas.", file=sys.stderr)
        sys.exit(1)

    # colunas opcionais
    col_preco = args.preco_col or encontrar_coluna(df, ["price", "sell_price", "preco", "preço", "valor", "unit_price"])
    col_store = encontrar_coluna(df, ["store_id", "store", "loja", "storeid"])
    col_produto = encontrar_coluna(df, ["item_id", "product_id", "produto", "sku", "item", "product"])
    col_evento = encontrar_coluna(df, ["event_name_1", "evento", "feriado", "holiday", "promocao", "promotion"])

    if col_preco is None:
        print(
            "Aviso: coluna de preço não encontrada. Será usado sell_price=0 e price_change=0. "
            "A previsão pode ser menos precisa.",
            file=sys.stderr,
        )

    if not col_store and not args.store_id:
        print(
            "Aviso: usando store_code=0 (CA_1) como padrão. "
            "Se o cliente for de outra loja, use --store-id.",
            file=sys.stderr,
        )

    # define colunas de agrupamento (produto e/ou loja)
    group_cols = []
    if col_produto:
        group_cols.append(col_produto)
    if col_store:
        group_cols.append(col_store)

    resultados = []
    erros = []

    if not group_cols:
        # previsão total agregada
        diario = para_serie_diaria(df, col_data, col_vendas, col_preco)
        serie_vendas = diario[col_vendas]
        serie_preco = diario[col_preco] if col_preco else None
        data_futura = serie_vendas.index[-1] + pd.Timedelta(days=1)

        store_code = STORE_CODE_MAP.get(args.store_id, 0) if args.store_id else 0

        try:
            features = criar_features_predicao(
                serie_vendas,
                data_futura,
                serie_preco=serie_preco,
                store_code=store_code,
                snap=args.snap,
                evento=args.evento,
            )
            X = pd.DataFrame([features], columns=FEATURES_MODELO)
            pred = float(modelo.predict(X)[0])
            resultados.append(("Total", data_futura, pred))
        except Exception as e:
            print(f"Erro ao gerar previsão total: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        # previsão por grupo (produto/loja)
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
                data_futura = serie_vendas.index[-1] + pd.Timedelta(days=1)

                if col_store:
                    store_id = grupo[col_store].iloc[0]
                    store_code = STORE_CODE_MAP.get(str(store_id), -1)
                else:
                    store_code = STORE_CODE_MAP.get(args.store_id, 0) if args.store_id else 0

                features = criar_features_predicao(
                    serie_vendas,
                    data_futura,
                    serie_preco=serie_preco,
                    store_code=store_code,
                    snap=args.snap,
                    evento=args.evento,
                )
                X = pd.DataFrame([features], columns=FEATURES_MODELO)
                pred = float(modelo.predict(X)[0])
                resultados.append((chave_str, data_futura, pred))
            except Exception as e:
                erros.append((chave_str, str(e)))
                print(f"Aviso: não foi possível prever para '{chave_str}': {e}", file=sys.stderr)

    if not resultados:
        print("Erro: nenhuma previsão pôde ser gerada. Verifique se há pelo menos 28 dias de histórico.", file=sys.stderr)
        sys.exit(1)

    print("Previsão de demanda D+1")
    print("-" * 40)
    for chave, data, pred in resultados:
        print(f"{chave}: data={data.date()} | previsão={pred:.2f}")

    if len(resultados) > 1:
        total = sum(r[2] for r in resultados)
        print("-" * 40)
        print(f"Total: {total:.2f}")

    if erros:
        print("\nAlguns grupos não geraram previsão:")
        for chave, erro in erros:
            print(f"  {chave}: {erro}")


if __name__ == "__main__":
    main()