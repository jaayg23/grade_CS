"""
Deep Learning for Portfolio Optimization
=========================================
Replicación de Zhang, Zohren & Roberts (2020) - arXiv:2005.13665v3
Notación adaptada de Medina (revcuaeco)

Correspondencia de notación:
    Paper Zhang          Medina (revcuaeco)
    ─────────────────    ──────────────────
    w_{i,t}              w_i  (ponderador del activo i)
    r_{i,t}              r_i  (retorno del activo i)
    R_{p,t}              r_p  (retorno del portafolio)
    E(R_p)               r̄_p  (rendimiento esperado del portafolio)
    Std(R_p)             σ_p  (volatilidad del portafolio)
    L_T (Sharpe)         r̄_p / σ_p
    θ (parámetros red)   — (sin análogo clásico)
    f(θ|x_t)            — (la red neuronal como función de asignación)

Activos:
    ECOPETROL.CL, GEB.CL, PFCIBEST.CL, CIBEST.CL,
    GOOGL, AAPL, BVC.CL, NKE, IVV

Implementación: NumPy puro (sin PyTorch). LSTM con BPTT manual + Adam.
Requisitos: pip install numpy pandas yfinance matplotlib
"""
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================
# MÓDULO 0: Descarga de precios
# ============================================================

CACHE_FILE = "precios_cache.csv"

def descargar_precios(portfolio, start_date, end_date, stocks_en_usd,
                      cache_file=CACHE_FILE, max_retries=3, retry_wait=30):
    """
    Descarga precios de cierre y convierte USD a COP.

    Guarda los datos en un CSV local (cache_file) para evitar descargar
    repetidamente y sortear el rate limiting de Yahoo Finance.
    En cada ejecución carga el cache si existe; solo descarga si no existe.

    Parámetros
    ----------
    portfolio : list[str]
    start_date, end_date : str  — formato 'YYYY-MM-DD'
    stocks_en_usd : list[str]  — tickers a convertir de USD a COP
    cache_file : str           — ruta del CSV de caché
    max_retries : int          — reintentos si falla la descarga
    retry_wait : int           — segundos entre reintentos
    """
    import os

    # --- Cargar desde caché si existe ---
    if os.path.exists(cache_file):
        print(f"    [cache] Cargando precios desde '{cache_file}'...")
        stock_prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Verificar que tenga las columnas esperadas
        if all(t in stock_prices.columns for t in portfolio):
            return stock_prices
        print("    [cache] Caché incompleto, re-descargando...")

    # --- Descarga con reintentos ---
    todos = portfolio + ["COP=X"]
    raw = None

    for intento in range(1, max_retries + 1):
        print(f"    [yfinance] Descargando... (intento {intento}/{max_retries})")
        raw = yf.download(todos, start=start_date, end=end_date,
                          auto_adjust=True, progress=False)
        if not raw.empty:
            break
        if intento < max_retries:
            print(f"    [yfinance] Rate limited. Esperando {retry_wait}s...")
            time.sleep(retry_wait)

    if raw is None or raw.empty:
        raise RuntimeError(
            "No se pudieron descargar los precios después de varios intentos.\n"
            f"Espera unos minutos y vuelve a intentarlo, o crea manualmente "
            f"'{cache_file}' exportando los datos desde portfolio_m.ipynb."
        )

    # --- Extraer Close y convertir USD → COP ---
    if isinstance(raw.columns, pd.MultiIndex):
        stock_prices = raw["Close"][portfolio].copy()
        usd_cop = raw["Close"]["COP=X"]
    else:
        stock_prices = raw[["Close"]].copy()
        stock_prices.columns = portfolio
        usd_cop = None

    if usd_cop is not None:
        usd_cop = usd_cop.reindex(stock_prices.index, method='ffill')
        for ticker in stocks_en_usd:
            if ticker in stock_prices.columns:
                stock_prices[ticker] = stock_prices[ticker] * usd_cop

    stock_prices = stock_prices.dropna()

    # --- Guardar caché ---
    stock_prices.to_csv(cache_file)
    print(f"    [cache] Datos guardados en '{cache_file}'")

    return stock_prices


# ============================================================
# MÓDULO 1: Preparación de datos — Ventana deslizante (k, 2n)
# ============================================================

def crear_ventanas(precios: np.ndarray, k: int = 50):
    """
    Construye la estructura de entrada (k, 2n) del paper.

    Para cada día t, tomamos una ventana de k días previos.
    Por cada activo A_i tenemos 2 features:
        - precio de cierre normalizado: p_{i,τ} / p_{i,t}
        - retorno diario: r_{i,τ} = p_{i,τ}/p_{i,τ-1} - 1

    Parámetros
    ----------
    precios : np.ndarray, shape (T_total, n)
        Matriz de precios de cierre. Cada columna es un activo A_i.
        T_total = número total de días, n = número de activos.
    k : int
        Tamaño de la ventana lookback (default: 50 días).

    Retorna
    -------
    X : np.ndarray, shape (T, k, 2n)
        Tensor de entrada para la red.
        T = T_total - k (días utilizables).
    r_futuro : np.ndarray, shape (T, n)
        Retornos r_{i,t} del día SIGUIENTE a cada ventana.
        Estos son los retornos que se multiplican por w_i
        para obtener r_p = Σ w_i · r_i  [Ec. 5' de Medina]
    """
    T_total, n = precios.shape

    # Retornos diarios: r_{i,t} = p_{i,t}/p_{i,t-1} - 1
    retornos = precios[1:] / precios[:-1] - 1  # shape: (T_total-1, n)

    T = T_total - k - 2  # días utilizables
    X = np.zeros((T, k, 2 * n))
    r_futuro = np.zeros((T, n))

    for t in range(T):
        inicio = t + 1       # +1 porque retornos empieza un día después
        fin = inicio + k

        # Precios normalizados por el último precio de la ventana
        ventana_precios = precios[inicio:fin+1]  # k+1 precios para k retornos
        p_norm = ventana_precios[:-1] / ventana_precios[-1]  # shape: (k, n)

        # Retornos en la ventana
        ventana_retornos = retornos[inicio:fin]   # shape: (k, n)

        # Concatenar: [p_norm_1, r_1, p_norm_2, r_2, ..., p_norm_n, r_n]
        X[t] = np.column_stack([
            val for i in range(n)
            for val in (p_norm[:, i:i+1], ventana_retornos[:, i:i+1])
        ])

        # Retorno del día siguiente
        r_futuro[t] = retornos[fin]

    return X, r_futuro


# ============================================================
# MÓDULO 2: Primitivas numéricas
# ============================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================
# MÓDULO 3: Arquitectura LSTM — f(θ|x_t) → w   (NumPy puro)
# ============================================================

class LSTMPortfolio:
    """
    Red neuronal LSTM para optimización de portafolio.
    Implementada en NumPy puro con retropropagación manual (BPTT) y Adam.

    Arquitectura (Section 3.2 y 4.3 del paper):
        Input layer:  x_t ∈ ℝ^{k × 2n}
        Neural layer: LSTM con hidden_size unidades (1 capa)
        Output layer: Linear(hidden_size → n) + Softmax

    La softmax garantiza por construcción:
        (1) w_i ≥ 0        ∀i        (no ventas en corto)
        (2) Σ w_i = 1                 (presupuesto)
    """

    def __init__(self, n_activos: int, k: int = 50, hidden_size: int = 64,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        n_in = 2 * n_activos
        n_h  = hidden_size

        self.n_h       = n_h
        self.n_in      = n_in
        self.n_activos = n_activos

        # Parámetros LSTM: pesos combinados para los 4 gates (f, i, g, o)
        # Wh shape: (4*n_h, n_h + n_in)
        scale = np.sqrt(2.0 / (n_h + n_in))
        self.params = {
            'Wh': rng.normal(0, scale, (4 * n_h, n_h + n_in)).astype(np.float32),
            'bh': np.zeros(4 * n_h, dtype=np.float32),
            'Wy': rng.normal(0, scale, (n_activos, n_h)).astype(np.float32),
            'by': np.zeros(n_activos, dtype=np.float32),
        }

        # Estado Adam
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def forward(self, X: np.ndarray):
        """
        Propagación hacia adelante.

        Parámetros
        ----------
        X : np.ndarray, shape (B, k, 2n)

        Retorna
        -------
        w      : (B, n_activos)  pesos del portafolio
        h_last : (B, n_h)        último estado oculto
        cache  : list            datos intermedios para BPTT
        """
        B, seq_len, _ = X.shape
        n_h = self.n_h
        Wh, bh, Wy, by = (self.params[p] for p in ('Wh', 'bh', 'Wy', 'by'))

        h = np.zeros((B, n_h), dtype=np.float32)
        c = np.zeros((B, n_h), dtype=np.float32)
        cache = []

        for t in range(seq_len):
            x_t = X[:, t, :]                          # (B, n_in)
            z   = np.concatenate([h, x_t], axis=1)    # (B, n_h + n_in)
            G   = z @ Wh.T + bh                        # (B, 4*n_h)

            f = _sigmoid(G[:, :n_h])
            i = _sigmoid(G[:, n_h:2*n_h])
            g = np.tanh(G[:, 2*n_h:3*n_h])
            o = _sigmoid(G[:, 3*n_h:])

            c_new  = f * c + i * g
            tanh_c = np.tanh(c_new)
            h_new  = o * tanh_c

            cache.append((z, f, i, g, o, c.copy(), tanh_c))
            h, c = h_new, c_new

        # Capa de salida
        logits = h @ Wy.T + by      # (B, n_activos)
        w      = _softmax(logits)   # (B, n_activos)

        return w, h, cache

    # ----------------------------------------------------------
    # Función objetivo: Sharpe ratio negativo
    # ----------------------------------------------------------

    @staticmethod
    def sharpe_loss(w: np.ndarray, r: np.ndarray):
        """
        Calcula el NEGATIVO del Sharpe ratio (para minimizar).

            L_T = E(R_{p,t}) / sqrt(E(R²_{p,t}) - E(R_{p,t})²)   [Ec. 2]

        Parámetros
        ----------
        w : (B, n)
        r : (B, n)

        Retorna
        -------
        loss   : float   -Sharpe
        sharpe : float   +Sharpe
        """
        r_p    = (w * r).sum(axis=1)       # (B,)
        mean   = r_p.mean()
        std    = r_p.std() + 1e-8
        sharpe = mean / std
        return -sharpe, sharpe

    # ----------------------------------------------------------
    # Backward pass (BPTT) + actualización Adam
    # ----------------------------------------------------------

    def backward_and_update(self, X: np.ndarray, r: np.ndarray,
                             w: np.ndarray, h_last: np.ndarray,
                             cache: list, lr: float = 0.001) -> float:
        """
        Calcula gradientes mediante BPTT y actualiza parámetros con Adam.

        Retorna el Sharpe ratio de entrenamiento (valor positivo).
        """
        B, seq_len, _ = X.shape
        n_h = self.n_h
        Wh, bh, Wy, by = (self.params[p] for p in ('Wh', 'bh', 'Wy', 'by'))

        # ── Gradiente de la función de pérdida ──────────────────
        r_p    = (w * r).sum(axis=1)    # (B,)
        mean   = r_p.mean()
        std    = r_p.std() + 1e-8
        sharpe = mean / std

        # dL/dr_p  donde L = -sharpe
        d_rp = -(1.0 / (B * std) - mean * (r_p - mean) / (B * std**3))  # (B,)

        # dL/dw
        d_w = d_rp[:, None] * r     # (B, n_activos)

        # ── Backprop a través de softmax ─────────────────────────
        # d_logit_i = w_i * (d_w_i − Σ_j d_w_j * w_j)
        d_logits = w * (d_w - (d_w * w).sum(axis=1, keepdims=True))  # (B, n_activos)

        # ── Backprop a través de la capa lineal de salida ────────
        dWy = d_logits.T @ h_last    # (n_activos, n_h)
        dby = d_logits.sum(axis=0)   # (n_activos,)
        d_h = d_logits @ Wy          # (B, n_h)

        # ── BPTT a través del LSTM ───────────────────────────────
        dWh = np.zeros_like(Wh)
        dbh = np.zeros_like(bh)
        d_c = np.zeros((B, n_h), dtype=np.float32)

        for t in reversed(range(seq_len)):
            z, f, i, g, o, c_prev, tanh_c = cache[t]

            # h_t = o * tanh(c_t)
            d_o  = d_h * tanh_c
            d_c += d_h * o * (1.0 - tanh_c ** 2)   # total grad w.r.t. c_t

            # c_t = f * c_{t-1} + i * g
            d_f = d_c * c_prev
            d_i = d_c * g
            d_g = d_c * i
            d_c = d_c * f                            # grad w.r.t. c_{t-1}

            # A través de activaciones de los gates
            d_f_raw = d_f * f * (1.0 - f)
            d_i_raw = d_i * i * (1.0 - i)
            d_g_raw = d_g * (1.0 - g ** 2)
            d_o_raw = d_o * o * (1.0 - o)

            d_G = np.concatenate([d_f_raw, d_i_raw, d_g_raw, d_o_raw], axis=1)  # (B, 4n_h)

            dWh += d_G.T @ z                 # (4n_h, n_h+n_in)
            dbh += d_G.sum(axis=0)           # (4n_h,)

            d_z = d_G @ Wh                   # (B, n_h + n_in)
            d_h = d_z[:, :n_h]               # grad w.r.t. h_{t-1}

        grads = {'Wh': dWh, 'bh': dbh, 'Wy': dWy, 'by': dby}
        self._adam_step(grads, lr)
        return float(sharpe)

    def _adam_step(self, grads: dict, lr: float,
                   beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """Actualización Adam estándar."""
        self._t += 1
        t = self._t
        for name in self.params:
            g = grads[name]
            m = self._m[name]
            v = self._v[name]
            m[:] = beta1 * m + (1.0 - beta1) * g
            v[:] = beta2 * v + (1.0 - beta2) * g ** 2
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            self.params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inferencia sin almacenar caché."""
        w, _, _ = self.forward(X)
        return w


# ============================================================
# MÓDULO 4: Pipeline de entrenamiento
# ============================================================

def entrenar(modelo: LSTMPortfolio,
             X_train: np.ndarray, r_train: np.ndarray,
             X_val:   np.ndarray, r_val:   np.ndarray,
             epochs: int = 100, lr: float = 0.001,
             batch_size: int = 64, verbose: bool = True) -> dict:
    """
    Entrena el modelo LSTM maximizando el Sharpe ratio.

    La actualización sigue la Ec. 6 del paper:
        θ_new := θ_old + α · ∂L_T/∂θ
    (En la práctica usamos Adam.)
    """
    historial = {'train_sharpe': [], 'val_sharpe': []}
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_sharpe = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end     = min(start + batch_size, n_samples)
            idx     = indices[start:end]
            X_b     = X_train[idx]
            r_b     = r_train[idx]

            w_b, h_b, cache = modelo.forward(X_b)
            sharpe = modelo.backward_and_update(X_b, r_b, w_b, h_b, cache, lr=lr)

            epoch_sharpe += sharpe
            n_batches    += 1

        # Validación
        w_val   = modelo.predict(X_val)
        r_p_val = (w_val * r_val).sum(axis=1)
        val_s   = float(r_p_val.mean() / (r_p_val.std() + 1e-8))

        train_s = epoch_sharpe / n_batches
        historial['train_sharpe'].append(train_s)
        historial['val_sharpe'].append(val_s)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1:3d}/{epochs} | "
                  f"Sharpe train: {train_s:.4f} | "
                  f"Sharpe val: {val_s:.4f}")

    return historial


# ============================================================
# MÓDULO 5: Script principal — Comparativa k=50, 100, 250
# ============================================================

def _run_k(precios: np.ndarray, activos: list, k: int,
           hidden: int = 64, epochs: int = 100,
           batch_size: int = 64, lr: float = 0.001,
           val_split: float = 0.1) -> dict:
    """Ejecuta el pipeline completo para una ventana k dada.
    Retorna dict con historial, w_promedio, r_p_val y métricas."""
    n = precios.shape[1]

    X, r = crear_ventanas(precios, k=k)
    T       = X.shape[0]
    T_val   = max(1, int(T * val_split))
    T_train = T - T_val

    X_train = X[:T_train].astype(np.float32)
    r_train = r[:T_train].astype(np.float32)
    X_val   = X[T_train:].astype(np.float32)
    r_val   = r[T_train:].astype(np.float32)

    modelo = LSTMPortfolio(n_activos=n, k=k, hidden_size=hidden)
    historial = entrenar(modelo, X_train, r_train, X_val, r_val,
                         epochs=epochs, lr=lr, batch_size=batch_size,
                         verbose=False)

    w_final    = modelo.predict(X_val)                 # (T_val, n)
    r_p        = (w_final * r_val).sum(axis=1)         # (T_val,)

    w_promedio    = w_final.mean(axis=0)
    sharpe_val    = float(r_p.mean() / (r_p.std() + 1e-8))
    retorno_anual = float(r_p.mean() * 252)
    vol_anual     = float(r_p.std() * np.sqrt(252))

    return {
        'k':             k,
        'historial':     historial,
        'w_promedio':    w_promedio,
        'r_p_val':       r_p,
        'sharpe_val':    sharpe_val,
        'retorno_anual': retorno_anual,
        'vol_anual':     vol_anual,
        'T_train':       T_train,
        'T_val':         T_val,
    }


def main():
    """Pipeline comparativo: k = 50, 100, 250 días."""

    print("=" * 60)
    print("  Deep Learning for Portfolio Optimization")
    print("  Zhang, Zohren & Roberts (2020) — Replicación")
    print("  Comparativa ventanas: k = 50 | 100 | 250")
    print("=" * 60)

    activos       = ['ECOPETROL.CL', 'GEB.CL', 'PFCIBEST.CL', 'CIBEST.CL',
                     'GOOGL', 'AAPL', 'BVC.CL', 'NKE', 'IVV']
    stocks_en_usd = ['AAPL', 'IVV', 'GOOGL', 'NKE']
    ventanas      = [50, 100, 250]
    epochs        = 100

    print("\n[1] Descargando precios históricos (2021-01-01 → hoy)...")
    from datetime import datetime
    df_precios = descargar_precios(activos, '2021-01-01',
                                   datetime.now().strftime('%Y-%m-%d'),
                                   stocks_en_usd)
    precios = df_precios.values
    n       = precios.shape[1]
    print(f"    Precios shape: {precios.shape}  (T_total={precios.shape[0]}, n={n})")

    resultados = {}
    for k in ventanas:
        print(f"\n[k={k:3d}] Entrenando LSTM — ventana {k} días ...")
        res = _run_k(precios, activos, k=k, epochs=epochs)
        resultados[k] = res
        print(f"  → Sharpe val: {res['sharpe_val']:+.4f} | "
              f"Ret. anual: {res['retorno_anual']*100:.2f}% | "
              f"Vol. anual: {res['vol_anual']*100:.2f}%  "
              f"[{res['T_train']} train / {res['T_val']} val]")

    print("\n" + "=" * 60)
    print("  RESUMEN COMPARATIVO")
    print(f"  {'k':>5}  {'Sharpe val':>11}  {'Ret. anual':>11}  {'Vol. anual':>11}")
    print("  " + "-" * 46)
    for k, res in resultados.items():
        print(f"  {k:>5}  {res['sharpe_val']:>+11.4f}  "
              f"{res['retorno_anual']*100:>10.2f}%  "
              f"{res['vol_anual']*100:>10.2f}%")
    print("=" * 60)

    graficar_comparativa(resultados, activos)
    return resultados


# ============================================================
# MÓDULO 6: Visualización comparativa k = 50 | 100 | 250
# ============================================================

def graficar_comparativa(resultados: dict, activos: list):
    """
    Dashboard comparativo de 3 ventanas (k=50, 100, 250).

    Layout (4 filas × 3 columnas):
      Fila 0  — Sharpe train vs val por época  (una curva por k, col completa)
      Fila 1  — Sharpe final en validación     (barras comparativas)
      Fila 2  — Retorno acumulado en val       (una línea por k)
      Fila 3  — Ponderadores promedio w_i      (grouped bar, un grupo por k)
    """
    BG_FIG   = '#0d1117'
    BG_AX    = '#161b22'
    GRID_C   = '#30363d'
    TXT      = 'white'
    SUBT     = '#8b949e'

    COLORS = {50: '#58a6ff', 100: '#3fb950', 250: '#f78166'}
    DASH   = {50: '-',       100: '--',      250: '-.'}

    ventanas       = sorted(resultados.keys())
    nombres_cortos = [t.replace('.CL', '') for t in activos]
    n_activos      = len(activos)
    epocas         = range(1, len(next(iter(resultados.values()))['historial']['train_sharpe']) + 1)

    fig = plt.figure(figsize=(18, 20))
    fig.patch.set_facecolor(BG_FIG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    def _style(ax):
        ax.set_facecolor(BG_AX)
        ax.tick_params(colors=SUBT)
        ax.spines[:].set_color(GRID_C)
        ax.yaxis.grid(True, color=GRID_C, lw=0.5)
        ax.xaxis.grid(False)

    # ── Fila 0 col 0..2: Sharpe por época ────────────────────────────────
    ax_ev = fig.add_subplot(gs[0, :])
    _style(ax_ev)
    for k in ventanas:
        h = resultados[k]['historial']
        ax_ev.plot(epocas, h['train_sharpe'],
                   color=COLORS[k], lw=1.8, linestyle=DASH[k],
                   alpha=0.55, label=f'k={k} train')
        ax_ev.plot(epocas, h['val_sharpe'],
                   color=COLORS[k], lw=2.2, linestyle=DASH[k],
                   label=f'k={k} val')
        best_idx = int(np.argmax(h['val_sharpe']))
        best_val = h['val_sharpe'][best_idx]
        ax_ev.scatter(best_idx + 1, best_val, color=COLORS[k], s=60, zorder=5)
        ax_ev.annotate(f'SR={best_val:.3f}',
                       xy=(best_idx + 1, best_val),
                       xytext=(best_idx + 1 + len(epocas)*0.03, best_val + 0.03),
                       color=COLORS[k], fontsize=8,
                       arrowprops=dict(arrowstyle='->', color=COLORS[k], lw=1))
    ax_ev.axhline(0, color=GRID_C, lw=0.8, linestyle=':')
    ax_ev.set_title(
        r'Evolución del Sharpe Ratio — $L_T = \bar{r}_p\,/\,\sigma_p$'
        '\n(líneas tenues = train  ·  líneas sólidas = val)',
        color=TXT, fontsize=12, pad=10)
    ax_ev.set_xlabel('Época', color=SUBT, fontsize=10)
    ax_ev.set_ylabel('Sharpe Ratio  $L_T$', color=SUBT, fontsize=10)
    ax_ev.legend(facecolor=BG_AX, edgecolor=GRID_C,
                 labelcolor=TXT, fontsize=9, ncol=3)

    # ── Fila 1 col 0: Sharpe final (barras) ──────────────────────────────
    ax_sr = fig.add_subplot(gs[1, 0])
    _style(ax_sr)
    sharpes = [resultados[k]['sharpe_val'] for k in ventanas]
    bars    = ax_sr.bar([str(k) for k in ventanas], sharpes,
                        color=[COLORS[k] for k in ventanas],
                        edgecolor=BG_AX, linewidth=0.5)
    for bar, val in zip(bars, sharpes):
        ax_sr.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + (0.005 if val >= 0 else -0.025),
                   f'{val:+.4f}', ha='center', va='bottom',
                   color=TXT, fontsize=9)
    ax_sr.axhline(0, color=GRID_C, lw=0.8)
    ax_sr.set_title('Sharpe ratio final (validación)', color=TXT, fontsize=11, pad=8)
    ax_sr.set_xlabel('Ventana k (días)', color=SUBT, fontsize=10)
    ax_sr.set_ylabel('$L_T$', color=SUBT, fontsize=10)

    # ── Fila 1 col 1: Retorno anualizado (barras) ─────────────────────────
    ax_ret = fig.add_subplot(gs[1, 1])
    _style(ax_ret)
    rets  = [resultados[k]['retorno_anual'] * 100 for k in ventanas]
    bars2 = ax_ret.bar([str(k) for k in ventanas], rets,
                       color=[COLORS[k] for k in ventanas],
                       edgecolor=BG_AX, linewidth=0.5)
    for bar, val in zip(bars2, rets):
        ax_ret.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.2 if val >= 0 else -1.2),
                    f'{val:+.2f}%', ha='center', va='bottom',
                    color=TXT, fontsize=9)
    ax_ret.axhline(0, color=GRID_C, lw=0.8)
    ax_ret.set_title('Retorno anualizado (validación)', color=TXT, fontsize=11, pad=8)
    ax_ret.set_xlabel('Ventana k (días)', color=SUBT, fontsize=10)
    ax_ret.set_ylabel('Ret. anual (%)', color=SUBT, fontsize=10)

    # ── Fila 1 col 2: Volatilidad anualizada (barras) ─────────────────────
    ax_vol = fig.add_subplot(gs[1, 2])
    _style(ax_vol)
    vols  = [resultados[k]['vol_anual'] * 100 for k in ventanas]
    bars3 = ax_vol.bar([str(k) for k in ventanas], vols,
                       color=[COLORS[k] for k in ventanas],
                       edgecolor=BG_AX, linewidth=0.5)
    for bar, val in zip(bars3, vols):
        ax_vol.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.1,
                    f'{val:.2f}%', ha='center', va='bottom',
                    color=TXT, fontsize=9)
    ax_vol.set_title('Volatilidad anualizada (validación)', color=TXT, fontsize=11, pad=8)
    ax_vol.set_xlabel('Ventana k (días)', color=SUBT, fontsize=10)
    ax_vol.set_ylabel('Vol. anual (%)', color=SUBT, fontsize=10)

    # ── Fila 2 col 0..2: Retorno acumulado ───────────────────────────────
    ax_cum = fig.add_subplot(gs[2, :])
    _style(ax_cum)
    for k in ventanas:
        r_p = resultados[k]['r_p_val']
        cum  = (1 + r_p).cumprod()
        ax_cum.plot(cum, color=COLORS[k], lw=2,
                    linestyle=DASH[k], label=f'k={k}  (final={cum[-1]:.3f})')
        ax_cum.fill_between(range(len(cum)), 1, cum,
                            where=cum >= 1, alpha=0.08, color=COLORS[k])
        ax_cum.fill_between(range(len(cum)), 1, cum,
                            where=cum < 1,  alpha=0.08, color=COLORS[k])
    ax_cum.axhline(1, color=GRID_C, lw=0.8, linestyle=':')
    ax_cum.set_title(r'Retorno acumulado en validación  $\prod(1+r_{p,t})$',
                     color=TXT, fontsize=12, pad=8)
    ax_cum.set_xlabel('Días (validación)', color=SUBT, fontsize=10)
    ax_cum.set_ylabel('Valor del portafolio (base 1)', color=SUBT, fontsize=10)
    ax_cum.legend(facecolor=BG_AX, edgecolor=GRID_C,
                  labelcolor=TXT, fontsize=10)

    # ── Fila 3 col 0..2: Ponderadores agrupados por activo ───────────────
    ax_w  = fig.add_subplot(gs[3, :])
    _style(ax_w)
    x      = np.arange(n_activos)
    width  = 0.25
    offset = [-width, 0, width]
    for idx_k, k in enumerate(ventanas):
        w      = resultados[k]['w_promedio']
        bars_w = ax_w.bar(x + offset[idx_k], w, width,
                          color=COLORS[k], edgecolor=BG_AX,
                          linewidth=0.4, label=f'k={k}')
        for bar, val in zip(bars_w, w):
            if val > 0.04:
                ax_w.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.003,
                          f'{val:.2f}', ha='center', va='bottom',
                          color=TXT, fontsize=7)
    ax_w.axhline(1 / n_activos, color='#f0883e', lw=1.2,
                 linestyle='--', label='Equal weight')
    ax_w.set_title(r'Ponderadores promedio $\bar{w}_i$ por activo y ventana',
                   color=TXT, fontsize=12, pad=8)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels(nombres_cortos, color=TXT, fontsize=9)
    ax_w.set_ylabel(r'$w_i$', color=SUBT, fontsize=10)
    ax_w.set_ylim(0, max(
        resultados[k]['w_promedio'].max() for k in ventanas
    ) * 1.3)
    ax_w.legend(facecolor=BG_AX, edgecolor=GRID_C,
                labelcolor=TXT, fontsize=10)

    # ── Título global ─────────────────────────────────────────────────────
    fig.suptitle(
        'Deep Learning for Portfolio Optimization  —  Comparativa ventanas lookback\n'
        'Zhang, Zohren & Roberts (2020)  ·  Notación Medina (revcuaeco)',
        color=TXT, fontsize=13, y=0.995
    )

    out = 'sharpe_comparativa.png'
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  [ok] Figura guardada en '{out}'")
    plt.show()


if __name__ == "__main__":
    resultados = main()
