# ============================================================
# streamlit_app.py (v3 — com correção MultiIndex yfinance)
# ============================================================
#
# Corre local:
#   streamlit run streamlit_app.py
#
# Deploy Streamlit Cloud:
#   streamlit_app.py + tc_fin_sistema.py + requirements.txt
#
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import statistics
import time
from datetime import datetime, timedelta

from tc_fin_sistema import (
    TCFinSistema, construir_estado, detectar_regime,
    DefinicaoAlvo, NOMES,
)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="TC-Fin Live",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# FETCH HELPERS (com retry + cache + MultiIndex fix)
# ============================================================

def _extrair_close(df):
    """
    Extrai a série de fecho de um DataFrame, lidando com o novo
    MultiIndex do yfinance (>=0.2.48).
    """
    if df is None or df.empty:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        # Se for MultiIndex, extrai o nível 'Close'
        try:
            serie = df.xs('Close', axis=1, level=0)
            if isinstance(serie, pd.DataFrame):
                serie = serie.iloc[:, 0] # Pega a primeira coluna se houver duplicados
        except KeyError:
            return None
    else:
        if "Close" not in df.columns:
            return None
        serie = df["Close"]
        
    return serie.dropna()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(ticker, anos=20, tentativas=3):
    """Descarrega preços diários com retry e MultiIndex fix."""
    fim = datetime.now()
    inicio = fim - timedelta(days=anos * 365 + 30)

    for tent in range(tentativas):
        try:
            # CORREÇÃO: multi_level_index=False força o yfinance a devolver
            # colunas simples, facilitando a extração.
            df = yf.download(
                ticker, start=inicio, end=fim,
                interval="1d", progress=False,
                auto_adjust=True, threads=False,
                multi_level_index=False
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        if tent < tentativas - 1:
            time.sleep(2 * (tent + 1))

    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_aux_dict(tickers_list):
    """Descarrega séries auxiliares de forma resiliente."""
    fim = datetime.now()
    inicio = fim - timedelta(days=15 * 365 + 30)
    out = {}
    
    for t in tickers_list:
        for tent in range(2):
            try:
                df = yf.download(
                    t, start=inicio, end=fim,
                    interval="1d", progress=False,
                    auto_adjust=True, threads=False,
                    multi_level_index=False
                )
                serie = _extrair_close(df)
                if serie is not None:
                    out[t.lower()] = {
                        d.strftime("%Y-%m-%d"): float(v)
                        for d, v in serie.items()
                    }
                    break
            except Exception:
                pass
            if tent < 1:
                time.sleep(1)
                
    return out


# ============================================================
# TREINO + BACKTEST (com cache pesada)
# ============================================================

@st.cache_resource(show_spinner=False, max_entries=20)
def treinar_sistema(precos_tuplas, datas_tuplas, horizonte, anos_treino):
    """Treina o C3 e corre o backtest walk-forward."""
    precos = list(precos_tuplas)
    datas = list(datas_tuplas)

    janela_treino = min(int(anos_treino * 252), len(precos) - horizonte - 30)
    if janela_treino < 200:
        return None

    def_alvo = DefinicaoAlvo(horizonte, 0.0)
    est_tr, y_tr = [], []
    for t in range(janela_treino):
        s = construir_estado(precos, t)
        if s is None:
            continue
        a = def_alvo.calcular(precos, t)
        if a is None:
            continue
        est_tr.append(s)
        y_tr.append(a)

    if len(est_tr) < 100:
        return None

    sistema = TCFinSistema(custo_bps=10.0, limiar=0.55)
    sistema.treinar(est_tr, y_tr)

    p1 = sum(y_tr) / len(y_tr)

    decisoes = []
    retornos = []

    for idx in range(janela_treino, len(precos) - horizonte):
        r = sistema.decidir_direcional(precos, idx, p1=p1)
        if r is None or r.get("estado") is None:
            continue

        pt = precos[idx]
        pf = precos[idx + horizonte]
        ret_real = (pf - pt) / pt if pt > 0 else 0.0
        sinal = r["meta_signal"]

        ret_liq = 0.0
        if sinal != 0:
            ret_liq = sinal * ret_real - 0.001

        decisoes.append({
            "date": datas[idx], "idx": idx, "preco": pt,
            "sinal": sinal, "expert": r["expert"],
            "regime": r["regime"], "ret_realized": ret_real,
            "ret_liquido": ret_liq,
        })
        retornos.append(ret_liq)

    return sistema, decisoes, retornos, p1, janela_treino


def decidir_ultima_barra(sistema, precos, datas, p1, aux):
    """Corre o sistema na barra mais recente, com todos os mapas."""
    if len(precos) < 100:
        return None

    idx = len(precos) - 1
    estado = construir_estado(precos, idx)
    if estado is None:
        return None

    regime = detectar_regime(precos, idx)
    data_str = datas[idx]

    r = sistema.decidir_direcional(
        precos, idx, p1=p1,
        vix_map=aux.get("vixy", {}),
        hyg_map=aux.get("hyg", {}),
        lqd_map=aux.get("lqd", {}),
        tlt_map=aux.get("tlt", {}),
        shy_map=aux.get("shy", {}),
        data=data_str,
    )

    if r is None:
        return None

    return {
        "date": data_str, "idx": idx,
        "preco": precos[idx], "estado": estado,
        "regime": regime, "sinais": r["sinais"],
        "meta_signal": r["meta_signal"],
        "expert_escolhido": r["expert"],
        "expert_idx": NOMES.index(r["expert"]) if r["expert"] else 0,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📊 TC-Fin Live")
    st.caption("10 experts + meta-cognos")

    st.divider()
    st.subheader("Configuração")

    ticker = st.selectbox(
        "Ativo",
        ["SPY", "QQQ", "TLT", "GLD", "USO", "FXE",
         "BTC-USD", "IWM", "XLV", "XLU"],
        index=0,
    )

    horizonte = st.selectbox(
        "Horizonte (dias)", [5, 20, 60], index=1,
        help="H=20 é o mais robusto (F48–F51)",
    )

    anos_treino = st.slider(
        "Anos de treino", 3, 15, 5, 1,
    )

    st.divider()
    st.subheader("Refresh")

    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)

    if st.button("🔄 Refresh agora", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("⚠️ Yahoo Finance (15min delay). Demo — não é produção.")


# ============================================================
# UI PRINCIPAL
# ============================================================

st.title("📊 TC-Fin Live")
st.caption(
    f"**{ticker}** · H=**{horizonte}d** · "
    f"Treino=**{anos_treino}a** · "
    f"Atualizado: {datetime.now().strftime('%H:%M:%S')}"
)

# Fetch preços
with st.spinner(f"A carregar {ticker}..."):
    df = fetch_prices(ticker, anos=max(anos_treino + 3, 15))

if df is None or df.empty:
    st.error(f"❌ Sem dados para {ticker}. Yahoo rate limit? Tenta mais tarde.")
    st.stop()

# Extração segura da série
serie = _extrair_close(df)
if serie is None or len(serie) < 500:
    st.error("Dados insuficientes ou inválidos.")
    st.stop()

precos = serie.tolist()
datas = [d.strftime("%Y-%m-%d") for d in serie.index]

st.info(
    f"📈 **{len(precos)}** barras · {datas[0]} → {datas[-1]} · "
    f"Preço: **{precos[-1]:.2f}**"
)

# Fetch auxiliares
with st.spinner("A carregar séries auxiliares (VIX, crédito, curva)..."):
    aux = fetch_aux_dict(["VIXY", "HYG", "LQD", "SHY", "TLT"])

faltam_aux = [t for t in ["vixy", "hyg", "lqd", "shy", "tlt"] if t not in aux]
if faltam_aux:
    st.warning(f"⚠️ Experts exógenos inativos: {', '.join(faltam_aux)}")

# Backtest
with st.spinner("A treinar e correr backtest (pode demorar 20–40s)..."):
    result = treinar_sistema(
        tuple(precos), tuple(datas), horizonte, anos_treino,
    )

if result is None:
    st.error("Backtest falhou — dados insuficientes.")
    st.stop()

sistema, decisoes, retornos, p1, janela_treino = result
estado = decidir_ultima_barra(sistema, precos, datas, p1, aux)

# ============================================================
# TABS
# ============================================================

tab_live, tab_bt, tab_attr, tab_info = st.tabs(
    ["🎯 Decisão Atual", "📈 Backtest", "🏆 Atribuição", "ℹ️ Info"]
)

# --- TAB 1 ---
with tab_live:
    if estado is None:
        st.warning("Estado não disponível para a última barra.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Data", estado["date"])
        sinal_str = {+1: "🟢 COMPRAR", 0: "⚪ NÃO AGIR", -1: "🔴 VENDER"}[estado["meta_signal"]]
        col2.metric("Decisão", sinal_str)
        col3.metric("Preço", f"{estado['preco']:.2f}")
        col4.metric("Regime", estado["regime"] or "—")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Estado (d1, d2, vol, trend)**")
            st.code(str(estado["estado"]))
            st.markdown("**Expert escolhido**")
            st.code(estado["expert_escolhido"])
            st.markdown("**p1 (baseline freq)**")
            st.code(f"{p1:.4f}")

        with c2:
            st.markdown("**Sinais dos 10 experts**")
            rows = []
            for i, nome in enumerate(NOMES):
                s = estado["sinais"][i]
                rows.append({
                    "Expert": nome,
                    "Sinal": {+1: "+1", 0: "0", -1: "-1"}[s],
                    "Escolhido": "✅" if i == estado["expert_idx"] else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Distribuição recente (últimos 100 trades)")
        ultimos = decisoes[-100:] if len(decisoes) >= 100 else decisoes
        cont = {}
        for d in ultimos:
            cont[d["expert"]] = cont.get(d["expert"], 0) + 1
        if cont:
            df_cont = pd.DataFrame([{"Expert": k, "Escolhas": v} for k, v in sorted(cont.items(), key=lambda x: -x[1])])
            st.dataframe(df_cont, use_container_width=True, hide_index=True)

# --- TAB 2 ---
with tab_bt:
    if not decisoes:
        st.warning("Sem decisões.")
    else:
        n_trades = sum(1 for d in decisoes if d["sinal"] != 0)
        n_abst = len(decisoes) - n_trades
        acertos = sum(1 for d in decisoes if d["sinal"] != 0 and d["sinal"] * d["ret_realized"] > 0)
        hit = acertos / n_trades * 100 if n_trades > 0 else 0
        total_ret = sum(retornos)
        media = total_ret / len(retornos) if retornos else 0
        dp = statistics.stdev(retornos) if len(retornos) > 1 else 0
        sharpe = media / dp if dp > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Trades", n_trades)
        c2.metric("Abstenção", f"{n_abst/len(decisoes)*100:.0f}%")
        c3.metric("Hit rate", f"{hit:.1f}%")
        c4.metric("Retorno total", f"{total_ret:+.2f}")
        c5.metric("Sharpe/trade", f"{sharpe:+.3f}")

        st.divider()
        st.subheader("Equity curve")
        eq = []
        cum = 0
        for d, r in zip(decisoes, retornos):
            cum += r
            eq.append({"date": d["date"], "equity": cum})
        eq_df = pd.DataFrame(eq)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_df["date"], y=eq_df["equity"], mode="lines", name="TC-Fin", line=dict(color="#2E86AB", width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Data", yaxis_title="Retorno acumulado")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Preço + decisões")
        ds = [d["date"] for d in decisoes]
        ps = [d["preco"] for d in decisoes]
        ss = [d["sinal"] for d in decisoes]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ds, y=ps, mode="lines", name=ticker, line=dict(color="#333", width=1.5)))
        buys = [(d, p) for d, p, s in zip(ds, ps, ss) if s > 0]
        sells = [(d, p) for d, p, s in zip(ds, ps, ss) if s < 0]
        if buys:
            fig2.add_trace(go.Scatter(x=[b[0] for b in buys], y=[b[1] for b in buys], mode="markers", name="Comprar", marker=dict(color="green", size=6, symbol="triangle-up")))
        if sells:
            fig2.add_trace(go.Scatter(x=[s[0] for s in sells], y=[s[1] for s in sells], mode="markers", name="Vender", marker=dict(color="red", size=6, symbol="triangle-down")))
        fig2.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Data", yaxis_title="Preço")
        st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3 ---
with tab_attr:
    if not decisoes:
        st.warning("Sem dados.")
    else:
        agg = {}
        for d in decisoes:
            if d["sinal"] == 0:
                continue
            e = d["expert"]
            if e not in agg:
                agg[e] = {"n": 0, "total": 0.0, "hits": 0}
            agg[e]["n"] += 1
            agg[e]["total"] += d["ret_liquido"]
            if d["sinal"] * d["ret_realized"] > 0:
                agg[e]["hits"] += 1

        total = sum(a["n"] for a in agg.values()) or 1
        rows = []
        for exp, a in agg.items():
            hit = a["hits"] / a["n"] * 100 if a["n"] else 0
            rows.append({
                "Expert": exp,
                "Escolhas": a["n"],
                "% total": f"{a['n']/total*100:.1f}%",
                "Hit rate": f"{hit:.1f}%",
                "Retorno": f"{a['total']:+.4f}",
            })
        rows.sort(key=lambda x: -x["Escolhas"])
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        fig3 = go.Figure(go.Bar(x=[r["Expert"] for r in rows], y=[r["Escolhas"] for r in rows], marker_color="#2E86AB"))
        fig3.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=80), xaxis_tickangle=-45, xaxis_title="Expert", yaxis_title="Escolhas")
        st.plotly_chart(fig3, use_container_width=True)

# --- TAB 4 ---
with tab_info:
    st.subheader("Configuração atual")
    st.json({
        "ticker": ticker,
        "horizonte_dias": horizonte,
        "anos_treino": anos_treino,
        "barras_treino": janela_treino,
        "barras_backtest": len(decisoes),
        "p1_treino": round(p1, 4),
        "aux_disponiveis": list(aux.keys()),
        "ultima_data": datas[-1],
        "preco_atual": round(precos[-1], 2),
    })

    st.divider()
    st.subheader("Sobre o TC-Fin")
    st.markdown("""
    - **10 experts** + meta-cognos por regime (EWMA).
    - **C3**: interpretador adaptativo hierárquico.
    - **Hedge** em equities com prémio secular dominante.
    - **Alpha** em pares com dinâmicas macro distintas.
    - **Hit rate** 55–84% em dados recentes.
    """)

# --- AUTO-REFRESH ---
if auto_refresh:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
