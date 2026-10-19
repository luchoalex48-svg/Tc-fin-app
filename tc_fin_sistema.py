# ============================================================
# tc_fin_sistema.py
# ============================================================
#
# TC-Fin — Sistema consolidado (F01–F51).
#
# Arquitetura:
#   Estado informacional
#     → 10 experts paralelos
#     → Meta-cognos por regime (EWMA)
#     → Decisão (Comprar / Vender / Não Agir)
#
# Modos:
#   direcional  — sobre uma série de preços
#   pairs       — sobre o ratio de dois ativos (market-neutral)
#
# Correções aplicadas:
#   F01/F02  D distingue compra e venda
#   F03      Contrato semântico do alvo
#   F04      Direção atual afeta OCE-03
#   F05      Estabilidade modula p_queda
#   F06      alpha=0 ≠ alpha="?"
#   F07      Validade explícita
#   F08      Intensidade/confiança não repetidas
#   F09      Fatores contextuais auditáveis
#   F10      Ganhos negativos não autorizam decisão
#   F11      Instrumentação do limiar
#   F12      Bijeção categorial↔direcional
#   F51      Patch: vix_map passa com nome correto
#
# ============================================================

import math


# ============================================================
# UTILITÁRIOS
# ============================================================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def decidir(p, limiar=0.55):
    if p > limiar:
        return +1
    if p < 1.0 - limiar:
        return -1
    return 0


def retorno_liquido(sinal, p_t, p_f, custo):
    if sinal == 0 or p_t <= 0:
        return 0.0
    return sinal * ((p_f - p_t) / p_t) - custo


# ============================================================
# F03 — CONTRATO SEMÂNTICO DO ALVO
# ============================================================
#
# y = 1  ⇔  retorno(t, h) > θ
# y = 0  ⇔  retorno(t, h) <= θ  (não-subida)
#
# p_final = P(retorno > θ | estado)
# p_up    := p_final
# p_down  := 1 - p_final  [não-subida, NÃO descida]
# ============================================================

class DefinicaoAlvo:

    def __init__(self, horizonte=1, limiar=0.0):
        if horizonte < 1:
            raise ValueError("horizonte deve ser >= 1")
        if limiar < 0.0:
            raise ValueError("limiar deve ser >= 0.0")
        self.horizonte = int(horizonte)
        self.limiar = float(limiar)

    def calcular(self, precos, t):
        t_fut = t + self.horizonte
        if t_fut >= len(precos):
            return None
        if precos[t] <= 0:
            return None
        r = (precos[t_fut] - precos[t]) / precos[t]
        return 1 if r > self.limiar else 0

    def metadados(self):
        return {
            "horizonte": self.horizonte,
            "limiar": self.limiar,
            "classe_1": "retorno > limiar",
            "classe_0": "não-subida",
        }


# ============================================================
# REPRESENTAÇÃO
# ============================================================

class Representacao:

    def __init__(self, ident, estado, direcao="FLAT", contexto=None):
        self.identificador = ident
        self.estado = tuple(estado)
        if direcao not in ("UP", "DOWN", "FLAT"):
            raise ValueError(f"direcao inválida: {direcao}")
        self.direcao_atual = direcao
        self.contexto = contexto or {}


def construir_estado(precos, t, janela_curta=3, janela_longa=10):
    """
    Estado discreto (d1, d2, vol, trend) a partir de preços.
    Devolve None se não houver histórico suficiente.
    """
    if t < janela_longa:
        return None

    if precos[t - janela_curta] <= 0 or precos[t - janela_longa] <= 0:
        return None

    def _dir(r, lim):
        return "UP" if r > lim else ("DOWN" if r < -lim else "FLAT")

    r_c = (precos[t] - precos[t - janela_curta]) / precos[t - janela_curta]
    r_l = (precos[t] - precos[t - janela_longa]) / precos[t - janela_longa]

    d1 = _dir(r_c, 0.005)
    d2 = _dir(r_l, 0.010)

    rets = [(precos[i] - precos[i - 1]) / precos[i - 1]
            for i in range(t - janela_longa + 1, t + 1)
            if precos[i - 1] > 0]
    if len(rets) < 3:
        return None

    m = sum(rets) / len(rets)
    v = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
    v_bin = "HIGH" if v > 0.012 else "LOW"

    mm_c = sum(precos[t - janela_curta + 1:t + 1]) / janela_curta
    mm_l = sum(precos[t - janela_longa + 1:t + 1]) / janela_longa

    if mm_c > mm_l * 1.005:
        tr = "UP"
    elif mm_c < mm_l * 0.995:
        tr = "DOWN"
    else:
        tr = "FLAT"

    return (d1, d2, v_bin, tr)


# ============================================================
# C3 — INTERPRETADOR ADAPTATIVO HIERÁRQUICO
# ============================================================

class Node:

    def __init__(self):
        self.n = 0
        self.k = 0

    def atualizar(self, y):
        self.n += 1
        if y == 1:
            self.k += 1


class OCE02_C3:
    """
    Hierarquia: global → individuais → pares → estado completo.
    Shrinkage adaptativo por τ.
    """

    def __init__(self, alpha=1.0, beta=1.0, k0=10.0,
                 tau_min=1.0, tau_max=50.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.k0 = float(k0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.nodes = {}

    def _node(self, key):
        key = tuple(key)
        if key not in self.nodes:
            self.nodes[key] = Node()
        return self.nodes[key]

    def _ensure(self, est):
        est = tuple(est)
        self._node(())
        for x in est:
            self._node((x,))
        m = len(est)
        for i in range(m):
            for j in range(i + 1, m):
                self._node((est[i], est[j]))
        self._node(est)

    def fit(self, estados, y):
        for est, alvo in zip(estados, y):
            est = tuple(est)
            self._ensure(est)
            self._node(()).atualizar(alvo)
            for x in est:
                self._node((x,)).atualizar(alvo)
            m = len(est)
            for i in range(m):
                for j in range(i + 1, m):
                    self._node((est[i], est[j])).atualizar(alvo)
            self._node(est).atualizar(alvo)
        return self

    def _local(self, node):
        return (node.k + self.alpha) / (
            node.n + self.alpha + self.beta
        )

    def _global(self):
        node = self.nodes.get(())
        return 0.5 if node is None else self._local(node)

    def _parents(self, est):
        est = tuple(est)
        cands = []
        if len(est) == 2:
            for x in est:
                if (x,) in self.nodes:
                    cands.append((x,))
        elif len(est) == 1:
            if () in self.nodes:
                cands.append(())
        elif len(est) > 2:
            m = len(est)
            for i in range(m):
                sub = tuple(est[j] for j in range(m) if j != i)
                if sub in self.nodes:
                    cands.append(sub)
        if not cands and () in self.nodes:
            cands.append(())
        return cands

    def _parent_prob(self, est):
        pais = self._parents(est)
        if not pais:
            return self._global()
        num = den = 0.0
        for p in pais:
            node = self.nodes[p]
            w = max(1, node.n)
            num += self._local(node) * w
            den += w
        return num / den if den > 0 else self._global()

    def _deviance(self, node, p_parent):
        if node.n <= 0:
            return 0.0
        p_l = clamp(node.k / node.n, 1e-12, 1 - 1e-12)
        pp = clamp(p_parent, 1e-12, 1 - 1e-12)
        n1, n0 = node.k, node.n - node.k
        g = 0.0
        if n1 > 0:
            g += 2.0 * n1 * math.log(p_l / pp)
        if n0 > 0:
            g += 2.0 * n0 * math.log((1 - p_l) / (1 - pp))
        return max(0.0, g)

    def estimate(self, estado):
        estado = tuple(estado)
        if estado not in self.nodes:
            return {
                "p_local": None,
                "p_parent": self._global(),
                "p_final": self._global(),
                "n": 0, "k": 0, "weight": 0.0,
            }
        node = self.nodes[estado]
        p_l = self._local(node)
        p_p = self._parent_prob(estado)
        g = self._deviance(node, p_p)
        evidence = node.n / (node.n + self.k0)
        divergence = 1.0 - math.exp(-g / 2.0)
        materiality = abs(p_l - p_p)
        spec = evidence * divergence * materiality
        tau = self.tau_max - (self.tau_max - self.tau_min) * spec
        w = node.n / (node.n + tau)
        return {
            "p_local": p_l,
            "p_parent": p_p,
            "p_final": w * p_l + (1 - w) * p_p,
            "n": node.n,
            "k": node.k,
            "weight": w,
        }

    def predict_proba(self, estado):
        return self.estimate(estado)["p_final"]


# ============================================================
# 10 EXPERTS
# ============================================================

def expert_always_long(p1, limiar=0.55):
    """Baseline: long se frequência de treino > 0.55."""
    return decidir(p1, limiar)


def expert_trend_follow(close, t):
    """Momentum 20 dias."""
    if t < 20 or close[t - 20] <= 0:
        return 0
    r = (close[t] - close[t - 20]) / close[t - 20]
    if r > 0.02:
        return +1
    if r < -0.02:
        return -1
    return 0


def expert_vol_defensive(close, t):
    """Flat quando vol_curta > 1.3 × vol_longa."""
    if t < 60:
        return +1
    rets = [(close[i] - close[i - 1]) / close[i - 1]
            for i in range(t - 19, t + 1) if close[i - 1] > 0]
    rets_l = [(close[i] - close[i - 1]) / close[i - 1]
              for i in range(t - 59, t + 1) if close[i - 1] > 0]
    if len(rets) < 10 or len(rets_l) < 30:
        return +1
    m = sum(rets) / len(rets)
    v = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
    m_l = sum(rets_l) / len(rets_l)
    v_l = (sum((r - m_l) ** 2 for r in rets_l) / len(rets_l)) ** 0.5
    if v_l <= 0:
        return +1
    return 0 if v > v_l * 1.3 else +1


def expert_vix_defensive(vix, vix_media):
    """Flat quando VIX > 1.5× média 20d."""
    if vix is None or vix_media is None or vix_media <= 0:
        return +1
    return 0 if vix / vix_media > 1.5 else +1


def expert_credit_watch(hyg, lqd, hist):
    """Flat quando HYG/LQD deteriora > 2% vs média 20d."""
    if hyg is None or lqd is None or lqd <= 0:
        return +1
    hist.append(hyg / lqd)
    if len(hist) > 20:
        hist.pop(0)
    if len(hist) < 10:
        return +1
    media = sum(hist) / len(hist)
    if media <= 0:
        return +1
    return 0 if hist[-1] / media < 0.98 else +1


def expert_curve_watch(tlt, shy, hist):
    """Flat quando TLT/SHY inverte > 10% vs média 60d."""
    if tlt is None or shy is None or shy <= 0:
        return +1
    hist.append(tlt / shy)
    if len(hist) > 60:
        hist.pop(0)
    if len(hist) < 30:
        return +1
    media = sum(hist) / len(hist)
    if media <= 0:
        return +1
    return 0 if hist[-1] / media > 1.10 else +1


def expert_crypto_vol_regime(close, t):
    """Flat quando vol 30d > 4% (crypto-specific)."""
    if t < 30:
        return +1
    rets = [(close[i] - close[i - 1]) / close[i - 1]
            for i in range(t - 29, t + 1) if close[i - 1] > 0]
    if len(rets) < 20:
        return +1
    m = sum(rets) / len(rets)
    v = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
    return 0 if v > 0.04 else +1


def expert_crypto_momentum(close, t):
    """Captura parabolic moves (+10%/10d, +25%/30d) e capitulações."""
    if t < 30 or close[t - 10] <= 0 or close[t - 30] <= 0:
        return 0
    r10 = (close[t] - close[t - 10]) / close[t - 10]
    r30 = (close[t] - close[t - 30]) / close[t - 30]
    if r10 > 0.10 and r30 > 0.25:
        return +1
    if r10 < -0.10 and r30 < -0.25:
        return -1
    return 0


def expert_c3(p_c3, limiar=0.55):
    """Sinal do C3 (usado por c3_cons e c3_agg com τ diferentes)."""
    return decidir(p_c3, limiar)


# ============================================================
# DETEÇÃO DE REGIME
# ============================================================

def detectar_regime(close, t):
    """
    TREND_UP    — direção forte alta
    TREND_DOWN  — direção forte baixa
    CHOPPY_HIGH — lateral com vol alta
    CHOPPY_LOW  — lateral com vol baixa
    """
    if t < 40 or close[t - 20] <= 0:
        return None
    r20 = (close[t] - close[t - 20]) / close[t - 20]
    rets = [(close[i] - close[i - 1]) / close[i - 1]
            for i in range(t - 19, t + 1) if close[i - 1] > 0]
    if len(rets) < 10:
        return None
    m = sum(rets) / len(rets)
    v = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5
    strength = abs(r20) / (v * math.sqrt(20)) if v > 0 else 0
    vr = 1.0
    if t >= 60:
        rets_l = [(close[i] - close[i - 1]) / close[i - 1]
                  for i in range(t - 59, t + 1) if close[i - 1] > 0]
        if len(rets_l) >= 30:
            m_l = sum(rets_l) / len(rets_l)
            v_l = (sum((r - m_l) ** 2 for r in rets_l) / len(rets_l)) ** 0.5
            vr = v / v_l if v_l > 0 else 1.0
    if strength > 1.5:
        return "TREND_UP" if r20 > 0 else "TREND_DOWN"
    if vr > 1.3:
        return "CHOPPY_HIGH"
    return "CHOPPY_LOW"


# ============================================================
# META-COGNOS POR REGIME
# ============================================================

N_EXPERTS = 10
NOMES = [
    "always_long", "trend_follow", "vol_defensive",
    "vix_defensive", "credit_watch", "curve_watch",
    "crypto_vol_regime", "crypto_momentum",
    "c3_cons", "c3_agg",
]

WARMUP = 50
ALPHA_START = 0.05
ALPHA_MIN = 0.005
DECAY = 500.0
THRESHOLD = 0.0002


class MetaCognosRegime:
    """
    EWMA por (expert, regime). Aprende qual expert funciona
    em cada regime e escolhe o melhor. Default = always_long.
    """

    def __init__(self, warmup=WARMUP, threshold=THRESHOLD,
                 alpha_start=ALPHA_START, alpha_min=ALPHA_MIN,
                 decay=DECAY):
        self.warmup = warmup
        self.threshold = threshold
        self.alpha_start = alpha_start
        self.alpha_min = alpha_min
        self.decay = decay
        self.ewma = {}
        self.n_seen = 0

    def _alpha(self):
        return self.alpha_min + (
            self.alpha_start - self.alpha_min
        ) * math.exp(-self.n_seen / self.decay)

    def _get(self, regime):
        if regime not in self.ewma:
            self.ewma[regime] = [0.0] * N_EXPERTS
        return self.ewma[regime]

    def decidir(self, sinais, regime):
        if self.n_seen < self.warmup or regime is None:
            return sinais[0], 0
        ewma = self._get(regime)
        best = max(range(N_EXPERTS), key=lambda i: ewma[i])
        if best == 0:
            return sinais[0], 0
        if ewma[best] - ewma[0] > self.threshold:
            return sinais[best], best
        return sinais[0], 0

    def atualizar(self, sinais, rets, regime):
        if regime is None:
            return
        a = self._alpha()
        ewma = self._get(regime)
        for i in range(N_EXPERTS):
            r = rets[i] if sinais[i] != 0 else 0.0
            ewma[i] = (1 - a) * ewma[i] + a * r
        self.n_seen += 1


# ============================================================
# SISTEMA INTEGRADO
# ============================================================

class TCFinSistema:
    """
    Sistema consolidado TC-Fin.

    Modo direcional:
        sistema = TCFinSistema()
        sistema.treinar(estados, y)
        r = sistema.decidir_direcional(precos, idx, p1=p1,
              vix_map=..., hyg_map=..., lqd_map=...,
              tlt_map=..., shy_map=..., data=...)

    Modo pairs:
        r = sistema.decidir_pair(precos_a, precos_b, ratio, idx)
    """

    def __init__(
        self,
        tau_max_cons=50.0,
        tau_max_agg=10.0,
        custo_bps=10.0,
        limiar=0.55,
    ):
        self.c3_cons = OCE02_C3(tau_max=tau_max_cons)
        self.c3_agg = OCE02_C3(tau_max=tau_max_agg)
        self.meta = MetaCognosRegime()
        self.custo_bps = float(custo_bps)
        self.limiar = float(limiar)

        # Históricos para experts exógenos
        self.vix_hist = []
        self.credit_hist = []
        self.curve_hist = []

        # Meta-cognos específico para pairs
        self._meta_pair = None
        self._treinado = False

    # --------------------------------------------------------
    # TREINO
    # --------------------------------------------------------

    def treinar(self, estados, y):
        self.c3_cons.fit(estados, y)
        self.c3_agg.fit(estados, y)
        self._treinado = True
        return self

    # --------------------------------------------------------
    # SINAIS DOS 10 EXPERTS
    # --------------------------------------------------------

    def _sinais_10(self, precos, idx, estado,
                   vix_map=None, hyg_map=None, lqd_map=None,
                   tlt_map=None, shy_map=None, data=None,
                   p1=0.5):
        """Sinais dos 10 experts (todos ativos se mapas forem passados)."""

        # 1. Baseline
        s_base = expert_always_long(p1, self.limiar)

        # 2. Momentum
        s_trend = expert_trend_follow(precos, idx)

        # 3. Vol defensive
        s_voldef = expert_vol_defensive(precos, idx)

        # 4. VIX (exógeno)
        vix_val = (vix_map or {}).get(data) if vix_map else None
        if vix_val is not None:
            self.vix_hist.append(vix_val)
            if len(self.vix_hist) > 20:
                self.vix_hist.pop(0)
        vix_media = (
            sum(self.vix_hist) / len(self.vix_hist)
            if len(self.vix_hist) >= 10 else None
        )
        s_vix = expert_vix_defensive(vix_val, vix_media)

        # 5. Crédito
        hyg = (hyg_map or {}).get(data)
        lqd = (lqd_map or {}).get(data)
        s_credit = expert_credit_watch(hyg, lqd, self.credit_hist)

        # 6. Curva
        tlt = (tlt_map or {}).get(data)
        shy = (shy_map or {}).get(data)
        s_curve = expert_curve_watch(tlt, shy, self.curve_hist)

        # 7-8. Crypto
        s_cv = expert_crypto_vol_regime(precos, idx)
        s_cm = expert_crypto_momentum(precos, idx)

        # 9-10. C3 (dois τ)
        p_cons = self.c3_cons.predict_proba(estado)
        p_agg = self.c3_agg.predict_proba(estado)
        s_cons = expert_c3(p_cons, self.limiar)
        s_agg = expert_c3(p_agg, self.limiar)

        return [s_base, s_trend, s_voldef,
                s_vix, s_credit, s_curve,
                s_cv, s_cm, s_cons, s_agg]

    # --------------------------------------------------------
    # DECISÃO DIRECIONAL
    # --------------------------------------------------------

    def decidir_direcional(
        self,
        precos, idx, p1,
        vix_map=None, hyg_map=None, lqd_map=None,
        tlt_map=None, shy_map=None, data=None,
        # Compat: aceita o nome antigo (spy_map é VIX)
        spy_map=None,
    ):
        """
        Decisão numa única barra idx.
        `spy_map` é alias de `vix_map` (compatibilidade).
        """
        if spy_map is not None and vix_map is None:
            vix_map = spy_map

        estado = construir_estado(precos, idx)
        if estado is None:
            return {
                "meta_signal": 0, "expert": None,
                "regime": None, "estado": None, "sinais": [],
            }

        regime = detectar_regime(precos, idx)

        sinais = self._sinais_10(
            precos, idx, estado,
            vix_map=vix_map,
            hyg_map=hyg_map, lqd_map=lqd_map,
            tlt_map=tlt_map, shy_map=shy_map,
            data=data, p1=p1,
        )

        s_meta, idx_exp = self.meta.decidir(sinais, regime)

        # Atualiza meta com retorno realizado (se horizonte cobre idx)
        custo = self.custo_bps / 10000.0
        if idx + 20 < len(precos):
            p_t = precos[idx]
            p_f = precos[idx + 20]
            rets = [retorno_liquido(s, p_t, p_f, custo)
                    for s in sinais]
            self.meta.atualizar(sinais, rets, regime)

        return {
            "meta_signal": s_meta,
            "expert": NOMES[idx_exp],
            "regime": regime,
            "estado": estado,
            "sinais": sinais,
        }

    # --------------------------------------------------------
    # DECISÃO PAIRS (market-neutral)
    # --------------------------------------------------------

    def decidir_pair(self, precos_a, precos_b, ratio, idx,
                     horizonte=20):
        """
        Modo pairs. `ratio = precos_a / precos_b`.
        Sinais:
          +1 = long A / short B
          -1 = short A / long B
           0 = não agir
        """
        estado = construir_estado(ratio, idx)
        if estado is None:
            return {"meta_signal": 0, "expert_idx": None,
                    "regime": None, "z": 0.0}

        regime = detectar_regime(ratio, idx)

        # Z-score rolante
        z_scores = self._rolling_zscore(ratio, 60)
        z = z_scores[idx] if idx < len(z_scores) else 0.0

        # 3 experts mean reversion
        sinais = []
        for th in (1.0, 1.5, 2.0):
            if z > th:
                sinais.append(-1)
            elif z < -th:
                sinais.append(+1)
            else:
                sinais.append(0)

        # 4. Breakout (oposto)
        if z > 1.5:
            sinais.append(+1)
        elif z < -1.5:
            sinais.append(-1)
        else:
            sinais.append(0)

        # 5. Trend follow no ratio
        if idx >= 20 and ratio[idx - 20] > 0:
            r = ratio[idx] / ratio[idx - 20] - 1
            sinais.append(
                +1 if r > 0.02 else (-1 if r < -0.02 else 0)
            )
        else:
            sinais.append(0)

        # 6. Vol defensive no ratio
        sinais.append(expert_vol_defensive(ratio, idx))

        # 7-8. C3 no ratio
        p_cons = self.c3_cons.predict_proba(estado)
        p_agg = self.c3_agg.predict_proba(estado)
        sinais.append(
            +1 if p_cons > self.limiar
            else (-1 if p_cons < 1 - self.limiar else 0)
        )
        sinais.append(
            +1 if p_agg > self.limiar
            else (-1 if p_agg < 1 - self.limiar else 0)
        )

        s_meta, idx_exp = self._meta_pair_decidir(sinais, regime)

        return {
            "meta_signal": s_meta,
            "expert_idx": idx_exp,
            "regime": regime,
            "z": z,
            "sinais": sinais,
        }

    def _rolling_zscore(self, ratio, window):
        log_r = [math.log(r) if r > 0 else 0.0 for r in ratio]
        z = [0.0] * len(log_r)
        for i in range(window, len(log_r)):
            j = log_r[i - window + 1:i + 1]
            m = sum(j) / len(j)
            v = sum((x - m) ** 2 for x in j) / len(j)
            std = math.sqrt(v) if v > 1e-10 else 1e-6
            z[i] = (log_r[i] - m) / std
        return z

    def _meta_pair_decidir(self, sinais, regime):
        """Meta-cognos específico para pairs (8 experts do spread)."""
        if self._meta_pair is None:
            self._meta_pair = {"ewma": {}, "n": 0}

        state = self._meta_pair
        if state["n"] < WARMUP or regime is None:
            return sinais[0], 0

        if regime not in state["ewma"]:
            state["ewma"][regime] = [0.0] * len(sinais)

        ewma = state["ewma"][regime]
        best = max(range(len(sinais)), key=lambda i: ewma[i])
        if best == 0:
            return sinais[0], 0
        if ewma[best] - ewma[0] > THRESHOLD:
            return sinais[best], best
        return sinais[0], 0

    # --------------------------------------------------------
    # UTILITÁRIOS AUXILIARES
    # --------------------------------------------------------

    def resumo_c3(self):
        return {
            "c3_cons": {
                "n_nodes": len(self.c3_cons.nodes),
                "p_global": self.c3_cons._global(),
            },
            "c3_agg": {
                "n_nodes": len(self.c3_agg.nodes),
                "p_global": self.c3_agg._global(),
            },
            "treinado": self._treinado,
            "meta_n_seen": self.meta.n_seen,
        }


# ============================================================
# EXEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    print("TC-Fin Sistema Consolidado (F01–F51)")
    print("=" * 60)
    print()
    print("Experts (10):")
    for i, nome in enumerate(NOMES, 1):
        print(f"  {i:2d}. {nome}")
    print()
    print("Modos:")
    print("  - decidir_direcional(precos, idx, p1, vix_map=...)")
    print("  - decidir_pair(precos_a, precos_b, ratio, idx)")
