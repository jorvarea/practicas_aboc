import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
import os


def get_sp100_tickers():
    """
    Selecciona 25 activos líquidos del S&P100 para el análisis.
    Estos son algunos de los más líquidos y representativos.
    """
    tickers = [
        'AAPL',  # Apple Inc.
        'MSFT',  # Microsoft Corporation
        'GOOGL',  # Alphabet Inc.
        'AMZN',  # Amazon.com Inc.
        'TSLA',  # Tesla Inc.
        'META',  # Meta Platforms Inc.
        'NVDA',  # NVIDIA Corporation
        'JPM',   # JPMorgan Chase & Co.
        'JNJ',   # Johnson & Johnson
        'V',     # Visa Inc.
        'PG',    # Procter & Gamble Co.
        'HD',    # Home Depot Inc.
        'MA',    # Mastercard Inc.
        'UNH',   # UnitedHealth Group Inc.
        'BAC',   # Bank of America Corp.
        'DIS',   # Walt Disney Co.
        'ADBE',  # Adobe Inc.
        'CRM',   # Salesforce Inc.
        'NFLX',  # Netflix Inc.
        'XOM',   # Exxon Mobil Corporation
        'KO',    # Coca-Cola Co.
        'PEP',   # PepsiCo Inc.
        'INTC',  # Intel Corporation
        'CSCO',  # Cisco Systems Inc.
        'WMT'    # Walmart Inc.
    ]
    return tickers


def download_stock_data(tickers, start_date, end_date):
    """
    Descarga precios de cierre diarios para los tickers especificados.

    Args:
        tickers: Lista de símbolos de acciones
        start_date: Fecha de inicio (formato YYYY-MM-DD)
        end_date: Fecha de fin (formato YYYY-MM-DD)

    Returns:
        DataFrame con precios de cierre ajustados
    """
    print(f"Descargando datos para {len(tickers)} activos...")
    print(f"Período: {start_date} a {end_date}")

    # Descargar datos usando yfinance
    data = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=True)

    # Manejar diferentes estructuras de datos dependiendo del número de tickers
    try:
        if len(tickers) == 1:
            # Para un solo ticker, los datos vienen con estructura simple
            if 'Close' in data.columns:
                prices = data['Close'].to_frame()
                prices.columns = tickers
            else:
                prices = data.to_frame()
                prices.columns = tickers
        else:
            # Para múltiples tickers, verificar estructura
            if isinstance(data.columns, pd.MultiIndex):
                # Estructura multi-nivel: (precio_tipo, ticker)
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close']
                elif 'Close' in data.columns.get_level_values(0):
                    prices = data['Close']
                else:
                    print("Estructura de datos no reconocida, usando la primera columna de precio disponible")
                    price_cols = data.columns.get_level_values(0).unique()
                    prices = data[price_cols[0]]
            else:
                # Estructura simple
                prices = data

        # Eliminar filas con valores NaN
        prices = prices.dropna()

        # Verificar que tenemos datos válidos
        if prices.empty:
            raise ValueError("No se pudieron obtener datos válidos")

    except Exception as e:
        print(f"Error procesando datos: {e}")
        print("Estructura de datos recibida:")
        print(f"Columns: {data.columns}")
        print(f"Shape: {data.shape}")
        print("Intentando usar una aproximación más robusta...")

        # Aproximación más robusta: buscar cualquier columna de precios
        if isinstance(data.columns, pd.MultiIndex):
            # Buscar columnas de precio
            price_types = ['Adj Close', 'Close', 'close', 'price']
            for price_type in price_types:
                if price_type in data.columns.get_level_values(0):
                    prices = data[price_type]
                    break
            else:
                # Si no encontramos ninguna, usar la primera disponible
                prices = data[data.columns.get_level_values(0).unique()[0]]
        else:
            prices = data

        prices = prices.dropna()

    print(f"Datos descargados exitosamente. Shape: {prices.shape}")
    print(f"Período efectivo: {prices.index[0].date()} a {prices.index[-1].date()}")
    print(f"Activos con datos: {list(prices.columns)}")

    return prices


def calculate_returns(prices):
    """
    Calcula las rentabilidades simples diarias.
    r_{t,i} = (P_{t,i} - P_{t-1,i}) / P_{t-1,i}

    Args:
        prices: DataFrame con precios de cierre

    Returns:
        DataFrame con rentabilidades diarias
    """
    print("Calculando rentabilidades simples diarias...")

    # Calcular rentabilidades simples
    returns = prices.pct_change().dropna()

    print(f"Rentabilidades calculadas. Shape: {returns.shape}")
    print(f"Período de rentabilidades: {returns.index[0].date()} a {returns.index[-1].date()}")

    return returns


def calculate_statistics(returns):
    """
    Calcula el vector de rentabilidades medias (μ) y la matriz de covarianzas (Σ).

    Args:
        returns: DataFrame con rentabilidades diarias

    Returns:
        tuple: (vector_mu, matriz_sigma)
    """
    print("Calculando estadísticas (μ y Σ)...")

    # Vector de rentabilidades medias
    mu = returns.mean()

    # Matriz de covarianzas
    sigma = returns.cov()

    print(f"Estadísticas calculadas:")
    print(f"- Vector μ (rentabilidades medias): {len(mu)} activos")
    print(f"- Matriz Σ (covarianzas): {sigma.shape}")
    print(f"- Rentabilidad media diaria (promedio): {mu.mean():.6f}")
    print(f"- Volatilidad media diaria (promedio): {np.sqrt(np.diag(sigma)).mean():.6f}")

    return mu, sigma


def save_data(prices, returns, mu, sigma, data_dir='data'):
    """
    Guarda todos los datos procesados en archivos CSV.

    Args:
        prices: DataFrame con precios
        returns: DataFrame con rentabilidades
        mu: Serie con rentabilidades medias
        sigma: DataFrame con matriz de covarianzas
        data_dir: Directorio donde guardar los archivos
    """
    # Crear directorio si no existe
    os.makedirs(data_dir, exist_ok=True)

    print(f"Guardando datos en directorio '{data_dir}'...")

    # Guardar precios
    prices_file = os.path.join(data_dir, 'precios_2024.csv')
    prices.to_csv(prices_file)
    print(f"- Precios guardados en: {prices_file}")

    # Guardar rentabilidades
    returns_file = os.path.join(data_dir, 'rentabilidades_2024.csv')
    returns.to_csv(returns_file)
    print(f"- Rentabilidades guardadas en: {returns_file}")

    # Guardar vector μ
    mu_file = os.path.join(data_dir, 'mu_vector.csv')
    mu.to_csv(mu_file, header=['rentabilidad_media'])
    print(f"- Vector μ guardado en: {mu_file}")

    # Guardar matriz Σ
    sigma_file = os.path.join(data_dir, 'sigma_matrix.csv')
    sigma.to_csv(sigma_file)
    print(f"- Matriz Σ guardada en: {sigma_file}")

    # Guardar resumen estadístico
    summary = {
        'num_activos': len(mu),
        'num_observaciones': len(returns),
        'rentabilidad_media_promedio': mu.mean(),
        'rentabilidad_media_std': mu.std(),
        'volatilidad_media_promedio': np.sqrt(np.diag(sigma)).mean(),
        'volatilidad_media_std': np.sqrt(np.diag(sigma)).std(),
        'periodo_inicio': str(returns.index[0].date()),
        'periodo_fin': str(returns.index[-1].date())
    }

    summary_file = os.path.join(data_dir, 'resumen_estadistico.csv')
    pd.Series(summary).to_csv(summary_file, header=['valor'])
    print(f"- Resumen estadístico guardado en: {summary_file}")


def load_data(data_dir='data'):
    """
    Carga los datos previamente guardados.

    Args:
        data_dir: Directorio donde están los archivos

    Returns:
        tuple: (prices, returns, mu, sigma)
    """
    print(f"Cargando datos desde directorio '{data_dir}'...")

    # Cargar precios
    prices_file = os.path.join(data_dir, 'precios_2024.csv')
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)

    # Cargar rentabilidades
    returns_file = os.path.join(data_dir, 'rentabilidades_2024.csv')
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)

    # Cargar vector μ
    mu_file = os.path.join(data_dir, 'mu_vector.csv')
    mu = pd.read_csv(mu_file, index_col=0)['rentabilidad_media']

    # Cargar matriz Σ
    sigma_file = os.path.join(data_dir, 'sigma_matrix.csv')
    sigma = pd.read_csv(sigma_file, index_col=0)

    print("Datos cargados exitosamente.")
    return prices, returns, mu, sigma


def main():
    """
    Función principal que ejecuta todo el pipeline de preparación de datos.
    """
    print("="*60)
    print("PARTE I: OBTENCIÓN Y PREPARACIÓN DE DATOS")
    print("="*60)

    # Parámetros
    start_date = '2024-01-01'
    end_date = '2024-12-31'

    # Obtener tickers
    tickers = get_sp100_tickers()
    print(f"\nActivos seleccionados ({len(tickers)}): {', '.join(tickers)}")

    try:
        # Descargar datos
        prices = download_stock_data(tickers, start_date, end_date)

        # Calcular rentabilidades
        returns = calculate_returns(prices)

        # Calcular estadísticas
        mu, sigma = calculate_statistics(returns)

        # Guardar datos
        save_data(prices, returns, mu, sigma)

        print("\n" + "="*60)
        print("PREPARACIÓN DE DATOS COMPLETADA EXITOSAMENTE")
        print("="*60)

        # Mostrar información resumen
        print(f"\nRESUMEN:")
        print(f"- Número de activos: {len(tickers)}")
        print(f"- Período: {returns.index[0].date()} a {returns.index[-1].date()}")
        print(f"- Observaciones por activo: {len(returns)}")
        print(f"- Rentabilidad media diaria (promedio): {mu.mean():.6f} ({mu.mean()*252:.4f} anualizada)")
        print(f"- Volatilidad media diaria (promedio): {np.sqrt(np.diag(sigma)).mean():.6f} ({np.sqrt(np.diag(sigma)).mean()*np.sqrt(252):.4f} anualizada)")

        return prices, returns, mu, sigma

    except Exception as e:
        print(f"\nError durante la preparación de datos: {e}")
        return None, None, None, None


if __name__ == "__main__":
    main()
