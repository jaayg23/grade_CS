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

Requisitos: pip install torch numpy pandas yfinance
"""
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import copy


#Descargamos los precios de las acciones

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


#Preparacion de datos - Creamos las ventanas y la variable target

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

    T = T_total - k - 2  # días utilizables (retornos tiene T_total-1 filas; fin=t+k+1 ≤ T_total-2)
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
        # Resultado: (k, 2n)
        X[t] = np.column_stack([
            val for i in range(n)
            for val in (p_norm[:, i:i+1], ventana_retornos[:, i:i+1])
        ])

        # Retorno del día siguiente: lo que multiplicamos por w para obtener r_p
        r_futuro[t] = retornos[fin]

    return X, r_futuro


#Arquitectura LSTM - Creamos la red neuronal

class LSTMPortfolio(nn.Module):
    """
    Red neuronal LSTM para optimización de portafolio.

    Arquitectura (Section 3.2 y 4.3 del paper):
        Input layer:  x_t ∈ ℝ^{k × 2n}
        Neural layer: LSTM con 64 unidades (1 capa)
        Output layer: Linear(64 → n) + Softmax

    La softmax garantiza por construcción:
        (1) w_i ≥ 0        ∀i        (no ventas en corto)
        (2) Σ w_i = 1                 (presupuesto)
    """

    def __init__(self, n_activos: int, k: int = 50, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()

        self.n_activos = n_activos
        self.input_size = 2 * n_activos

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_size, n_activos)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n.squeeze(0)
        last_hidden = self.dropout(last_hidden)
        z_raw = self.linear(last_hidden)
        w = self.softmax(z_raw)
        return w


#Funcion objetivo - Sharpe Ratio como L_T

def sharpe_loss(w, r):
    """
    Calcula el NEGATIVO del Sharpe ratio (para minimizar con el optimizador).

        L_T = E(R_{p,t}) / sqrt(E(R²_{p,t}) - E(R_{p,t})²)   [Ec. 2]
    """
    r_p = (w * r).sum(dim=1)
    mean_rp = r_p.mean()
    std_rp = r_p.std()
    sharpe = mean_rp / (std_rp + 1e-8)
    return -sharpe


#Pipeline de entrenamiento

def entrenar(modelo, X_train, r_train, X_val, r_val,
             epochs=100, lr=0.001, batch_size=64,
             patience=15, verbose=True):
    """
    Entrena el modelo LSTM maximizando el Sharpe ratio con early stopping.

    La actualización sigue la Ec. 6 del paper:
        θ_new := θ_old + α · ∂L_T/∂θ
    (En la práctica usamos Adam.)
    """
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)

    historial = {'train_sharpe': [], 'val_sharpe': []}
    n_samples = X_train.shape[0]

    best_val_sharpe = -float('inf')
    best_weights = copy.deepcopy(modelo.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        modelo.train()
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            #Condicionante para no exceder el tamaño de la muestra
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]


            X_batch = X_train[batch_idx]
            r_batch = r_train[batch_idx]

            w_batch = modelo(X_batch)
            loss = sharpe_loss(w_batch, r_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        modelo.eval()
        with torch.no_grad():
            w_train_full = modelo(X_train)
            train_loss_full = sharpe_loss(w_train_full, r_train)
            w_val = modelo(X_val)
            val_loss = sharpe_loss(w_val, r_val)

        train_sharpe = -train_loss_full.item()
        val_sharpe = -val_loss.item()

        historial['train_sharpe'].append(train_sharpe)
        historial['val_sharpe'].append(val_sharpe)

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_weights = copy.deepcopy(modelo.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1:3d}/{epochs} | "
                  f"Sharpe train: {train_sharpe:.4f} | "
                  f"Sharpe val: {val_sharpe:.4f} | "
                  f"Mejor val: {best_val_sharpe:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping en época {epoch+1}. "
                      f"Mejor Sharpe val: {best_val_sharpe:.4f}")
            break

    modelo.load_state_dict(best_weights)
    return historial

#Comparativa de ventanas

def _run_k(precios, activos, k, hidden=64, epochs=100,
           batch_size=64, lr=0.001, val_split=0.2):
    """Ejecuta el pipeline completo para una ventana k dada.
    Retorna dict con historial, w_promedio, r_p_val y métricas."""
    n = precios.shape[1]

    X, r = crear_ventanas(precios, k=k)
    T = X.shape[0]
    T_val   = max(1, int(T * val_split))
    T_train = T - T_val

    X_train = torch.FloatTensor(X[:T_train])
    r_train = torch.FloatTensor(r[:T_train])
    X_val   = torch.FloatTensor(X[T_train:])
    r_val   = torch.FloatTensor(r[T_train:])

    # Crea un tensor de pesos uniformes (1/9 para cada activo)
    w_eq_train = torch.ones_like(r_train) / n
    sharpe_eq_train = sharpe_loss(w_eq_train, r_train).item() * -1

    w_eq_val = torch.ones_like(r_val) / n
    sharpe_eq_val = sharpe_loss(w_eq_val, r_val).item() * -1

    print(f"Baseline (1/n) -> Train Sharpe: {sharpe_eq_train:.4f} | Val Sharpe: {sharpe_eq_val:.4f}")

    modelo = LSTMPortfolio(n_activos=n, k=k, hidden_size=hidden)
    historial = entrenar(modelo, X_train, r_train, X_val, r_val,
                         epochs=epochs, lr=lr, batch_size=batch_size,
                         patience=30, verbose=False)

    modelo.eval()
    with torch.no_grad():
        w_final = modelo(X_val)
        r_p = (w_final * r_val).sum(dim=1).numpy()
        

    w_promedio    = w_final.mean(dim=0).numpy()
    sharpe_val    = float(r_p.mean() / (r_p.std() + 1e-8))
    retorno_anual = r_p.mean() * 252
    vol_anual     = r_p.std() * np.sqrt(252)

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
    """Pipeline con ventana fija k=50 días."""

    print("=" * 60)
    print("  Deep Learning for Portfolio Optimization")
    print("  Zhang, Zohren & Roberts (2020) — Replicación")
    print("  Ventana lookback: k = 50 días")
    print("=" * 60)

    activos       = ['ECOPETROL.CL', 'GEB.CL', 'PFCIBEST.CL', 'CIBEST.CL',
                     'GOOGL', 'AAPL', 'BVC.CL', 'NKE', 'IVV']
    stocks_en_usd = ['AAPL', 'IVV', 'GOOGL', 'NKE']
    k             = 50
    epochs        = 100

    # --- Datos ---
    print("\n[1] Descargando precios históricos (2015-01-01 → 2026-03-16)...")
    df_precios = descargar_precios(activos, '2015-01-01', '2026-03-16', stocks_en_usd)
    precios    = df_precios.values
    n          = precios.shape[1]
    print(f"    Precios shape: {precios.shape}  (T_total={precios.shape[0]}, n={n})")

    # --- Entrenamiento ---
    print(f"\n[k={k}] Entrenando LSTM — ventana {k} días ...")
    res = _run_k(precios, activos, k=k, epochs=epochs)
    print(f"  → Sharpe val: {res['sharpe_val']:+.4f} | "
          f"Ret. anual: {res['retorno_anual']*100:.2f}% | "
          f"Vol. anual: {res['vol_anual']*100:.2f}%  "
          f"[{res['T_train']} train / {res['T_val']} val]")

    # --- Guardar pesos ---
    registro = {
        'k':             k,
        'sharpe_val':    float(res['sharpe_val']),
        'retorno_anual': float(res['retorno_anual']),
        'vol_anual':     float(res['vol_anual']),
        'pesos':         {activos[i]: float(res['w_promedio'][i])
                          for i in range(len(activos))},
    }
    import json
    pesos_file = 'zhang_best_weights.json'
    with open(pesos_file, 'w', encoding='utf-8') as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)
    print(f"\n  [✓] Pesos guardados en '{pesos_file}' "
          f"(k={k}, Sharpe={res['sharpe_val']:+.4f})")

    # --- Visualización ---
    graficar_sharpe_epocas(res)

    return res


#Visualizar Sharpe ratio por época

def graficar_sharpe_epocas(res: dict):
    """
    Gráfico único: evolución del Sharpe ratio (train y val) por época.
    """
    BG_FIG = '#0d1117'
    BG_AX  = '#161b22'
    GRID_C = '#30363d'
    TXT    = 'white'
    SUBT   = '#8b949e'
    COLOR  = '#58a6ff'

    h      = res['historial']
    k      = res['k']
    epocas = range(1, len(h['train_sharpe']) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=SUBT)
    ax.spines[:].set_color(GRID_C)
    ax.yaxis.grid(True, color=GRID_C, lw=0.5)
    ax.xaxis.grid(False)

    ax.plot(epocas, h['train_sharpe'],
            color=COLOR, lw=1.8, alpha=0.5, label='Train')
    ax.plot(epocas, h['val_sharpe'],
            color=COLOR, lw=2.2, label='Validación')

    best_idx = int(np.argmax(h['val_sharpe']))
    best_val = h['val_sharpe'][best_idx]
    ax.scatter(best_idx + 1, best_val, color=COLOR, s=80, zorder=5)
    ax.annotate(f'Mejor SR val = {best_val:.4f}',
                xy=(best_idx + 1, best_val),
                xytext=(best_idx + 1 + len(epocas) * 0.04, best_val + 0.03),
                color=COLOR, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=COLOR, lw=1))

    ax.axhline(0, color=GRID_C, lw=0.8, linestyle=':')
    ax.set_title(
        rf'Sharpe Ratio por Época — LSTM k={k}  '
        r'($L_T = \bar{{r}}_p\,/\,\sigma_p$)'
        '\n(línea tenue = train  ·  línea sólida = validación)',
        color=TXT, fontsize=12, pad=10)
    ax.set_xlabel('Época', color=SUBT, fontsize=10)
    ax.set_ylabel(r'Sharpe Ratio $L_T$', color=SUBT, fontsize=10)
    ax.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TXT, fontsize=10)

    fig.suptitle(
        'Deep Learning for Portfolio Optimization  —  k=50 días\n'
        'Zhang, Zohren & Roberts (2020)  ·  2015-01-01 → 2026-03-16',
        color=TXT, fontsize=11, y=1.02)

    out = 'sharpe_epocas.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  [✓] Figura guardada en '{out}'")
    plt.show()


if __name__ == "__main__":
    resultados = main()
