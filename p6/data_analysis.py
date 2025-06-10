import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_preparation import load_data


def analyze_data():
    """
    Analiza los datos financieros procesados y muestra estadísticas clave.
    """
    print("="*60)
    print("ANÁLISIS EXPLORATORIO DE LOS DATOS FINANCIEROS")
    print("="*60)

    # Cargar datos
    prices, returns, mu, sigma = load_data()

    print(f"\n1. INFORMACIÓN GENERAL:")
    print(f"   - Número de activos: {len(mu)}")
    print(f"   - Período de datos: {returns.index[0].date()} a {returns.index[-1].date()}")
    print(f"   - Número de observaciones: {len(returns)}")

    print(f"\n2. ESTADÍSTICAS DE RENTABILIDADES:")
    print(f"   - Rentabilidad media diaria (promedio): {mu.mean():.6f}")
    print(f"   - Rentabilidad media anualizada (promedio): {mu.mean()*252:.4f}")
    print(f"   - Desviación estándar de medias: {mu.std():.6f}")

    # Top 5 y Bottom 5 activos por rentabilidad
    top_returns = mu.nlargest(5)
    bottom_returns = mu.nsmallest(5)

    print(f"\n   Top 5 activos por rentabilidad media diaria:")
    for ticker, ret in top_returns.items():
        print(f"     {ticker}: {ret:.6f} ({ret*252:.4f} anualizada)")

    print(f"\n   Bottom 5 activos por rentabilidad media diaria:")
    for ticker, ret in bottom_returns.items():
        print(f"     {ticker}: {ret:.6f} ({ret*252:.4f} anualizada)")

    print(f"\n3. ESTADÍSTICAS DE VOLATILIDAD:")
    volatilities = np.sqrt(np.diag(sigma))
    print(f"   - Volatilidad media diaria (promedio): {volatilities.mean():.6f}")
    print(f"   - Volatilidad media anualizada (promedio): {volatilities.mean()*np.sqrt(252):.4f}")
    print(f"   - Desviación estándar de volatilidades: {volatilities.std():.6f}")

    # Top 5 y Bottom 5 activos por volatilidad
    vol_series = pd.Series(volatilities, index=mu.index)
    top_vol = vol_series.nlargest(5)
    bottom_vol = vol_series.nsmallest(5)

    print(f"\n   Top 5 activos por volatilidad diaria:")
    for ticker, vol in top_vol.items():
        print(f"     {ticker}: {vol:.6f} ({vol*np.sqrt(252):.4f} anualizada)")

    print(f"\n   Bottom 5 activos por volatilidad diaria:")
    for ticker, vol in bottom_vol.items():
        print(f"     {ticker}: {vol:.6f} ({vol*np.sqrt(252):.4f} anualizada)")

    print(f"\n4. ANÁLISIS DE CORRELACIONES:")
    corr_matrix = returns.corr()

    # Estadísticas de la matriz de correlación
    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    print(f"   - Correlación media: {upper_triangle.mean():.4f}")
    print(f"   - Correlación mediana: {np.median(upper_triangle):.4f}")
    print(f"   - Correlación mínima: {upper_triangle.min():.4f}")
    print(f"   - Correlación máxima: {upper_triangle.max():.4f}")

    # Pares más correlacionados
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

    corr_pairs_df = pd.DataFrame(corr_pairs, columns=['Asset1', 'Asset2', 'Correlation'])
    top_corr = corr_pairs_df.nlargest(5, 'Correlation')
    bottom_corr = corr_pairs_df.nsmallest(5, 'Correlation')

    print(f"\n   Top 5 pares más correlacionados:")
    for _, row in top_corr.iterrows():
        print(f"     {row['Asset1']} - {row['Asset2']}: {row['Correlation']:.4f}")

    print(f"\n   Top 5 pares menos correlacionados:")
    for _, row in bottom_corr.iterrows():
        print(f"     {row['Asset1']} - {row['Asset2']}: {row['Correlation']:.4f}")

    return prices, returns, mu, sigma, volatilities, corr_matrix


def create_visualizations(prices, returns, mu, sigma, volatilities, corr_matrix):
    """
    Crea visualizaciones de los datos financieros.
    """
    print(f"\n5. CREANDO VISUALIZACIONES...")

    # Crear directorio para figuras del informe
    figuras_dir = 'figuras'
    os.makedirs(figuras_dir, exist_ok=True)

    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")

    # Crear figura con subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Evolución de precios normalizados
    ax1 = plt.subplot(2, 3, 1)
    prices_normalized = prices / prices.iloc[0] * 100
    for col in prices_normalized.columns[:10]:  # Solo primeros 10 para legibilidad
        plt.plot(prices_normalized.index, prices_normalized[col], label=col, linewidth=1)
    plt.title('Evolución de Precios Normalizados (Base 100)\n(Primeros 10 activos)', fontsize=12)
    plt.xlabel('Fecha')
    plt.ylabel('Precio Normalizado')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=45)

    # 2. Distribución de rentabilidades medias
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(mu * 252, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mu.mean() * 252, color='red', linestyle='--', label=f'Media: {mu.mean()*252:.3f}')
    plt.title('Distribución de Rentabilidades Anualizadas')
    plt.xlabel('Rentabilidad Anualizada')
    plt.ylabel('Frecuencia')
    plt.legend()

    # 3. Distribución de volatilidades
    ax3 = plt.subplot(2, 3, 3)
    vol_annualized = volatilities * np.sqrt(252)
    plt.hist(vol_annualized, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(vol_annualized.mean(), color='red', linestyle='--', label=f'Media: {vol_annualized.mean():.3f}')
    plt.title('Distribución de Volatilidades Anualizadas')
    plt.xlabel('Volatilidad Anualizada')
    plt.ylabel('Frecuencia')
    plt.legend()

    # 4. Scatter plot Rentabilidad vs Volatilidad
    ax4 = plt.subplot(2, 3, 4)
    mu_annualized = mu * 252
    plt.scatter(vol_annualized, mu_annualized, alpha=0.7, s=60)
    for i, ticker in enumerate(mu.index):
        plt.annotate(ticker, (vol_annualized[i], mu_annualized.iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Volatilidad Anualizada')
    plt.ylabel('Rentabilidad Anualizada')
    plt.title('Rentabilidad vs Volatilidad (Anualizadas)')
    plt.grid(True, alpha=0.3)

    # 5. Mapa de calor de correlaciones
    ax5 = plt.subplot(2, 3, 5)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlaciones')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 6. Serie temporal de rentabilidades acumuladas
    ax6 = plt.subplot(2, 3, 6)
    cumulative_returns = (1 + returns).cumprod()
    for col in cumulative_returns.columns[:10]:  # Solo primeros 10
        plt.plot(cumulative_returns.index, cumulative_returns[col], label=col, alpha=0.7, linewidth=1)
    plt.title('Rentabilidades Acumuladas\n(Primeros 10 activos)', fontsize=12)
    plt.xlabel('Fecha')
    plt.ylabel('Rentabilidad Acumulada')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Guardar en carpeta de figuras para el informe
    plt.savefig(os.path.join(figuras_dir, 'analisis_completo_datos.png'), dpi=300, bbox_inches='tight')
    print(f"   - Gráficos guardados en: {figuras_dir}/analisis_completo_datos.png")

    # También guardar en data por compatibilidad
    plt.savefig('data/analisis_datos_financieros.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Crear gráficos individuales para el informe
    create_individual_plots(prices, returns, mu, sigma, volatilities, corr_matrix, figuras_dir)


def create_individual_plots(prices, returns, mu, sigma, volatilities, corr_matrix, figuras_dir):
    """
    Crea gráficos individuales para incluir en el informe.
    """
    print("   - Creando gráficos individuales para el informe...")

    # 1. Gráfico de rentabilidad vs volatilidad
    plt.figure(figsize=(10, 8))
    mu_annualized = mu * 252
    vol_annualized = volatilities * np.sqrt(252)

    plt.scatter(vol_annualized, mu_annualized, alpha=0.7, s=80, c='steelblue', edgecolors='black', linewidth=0.5)
    for i, ticker in enumerate(mu.index):
        plt.annotate(ticker, (vol_annualized[i], mu_annualized.iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

    plt.xlabel('Volatilidad Anualizada', fontsize=12)
    plt.ylabel('Rentabilidad Anualizada', fontsize=12)
    plt.title('Distribución Rentabilidad-Riesgo de los Activos (2024)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Añadir líneas de referencia
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Rentabilidad = 0')
    plt.axvline(x=vol_annualized.mean(), color='orange', linestyle='--', alpha=0.5, label=f'Volatilidad Media = {vol_annualized.mean():.2f}')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, 'rentabilidad_vs_volatilidad.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Matriz de correlaciones
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f', annot_kws={'size': 8})
    plt.title('Matriz de Correlaciones entre Activos', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, 'matriz_correlaciones.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Top 10 rentabilidades
    plt.figure(figsize=(12, 8))
    top_returns = (mu * 252).sort_values(ascending=True).tail(10)
    colors = ['red' if x < 0 else 'green' for x in top_returns.values]

    plt.barh(range(len(top_returns)), top_returns.values, color=colors, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(top_returns)), top_returns.index)
    plt.xlabel('Rentabilidad Anualizada', fontsize=12)
    plt.title('Top 10 Activos por Rentabilidad Anualizada (2024)', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, 'top_rentabilidades.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Evolución de precios seleccionados
    plt.figure(figsize=(14, 8))
    prices_normalized = prices / prices.iloc[0] * 100

    # Seleccionar algunos activos representativos
    selected_assets = ['NVDA', 'TSLA', 'META', 'AAPL', 'MSFT', 'INTC', 'JNJ', 'KO']

    for asset in selected_assets:
        if asset in prices_normalized.columns:
            plt.plot(prices_normalized.index, prices_normalized[asset], label=asset, linewidth=2)

    plt.title('Evolución de Precios Normalizados - Activos Seleccionados (2024)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio Normalizado (Base 100)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, 'evolucion_precios_seleccionados.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   - Gráficos individuales guardados en la carpeta '{figuras_dir}/'")


def create_summary_table(mu, volatilities):
    """
    Crea una tabla resumen con las estadísticas de todos los activos.
    """
    print(f"\n6. TABLA RESUMEN POR ACTIVO:")

    # Crear DataFrame resumen
    summary = pd.DataFrame({
        'Ticker': mu.index,
        'Rentabilidad_Diaria': mu.values,
        'Rentabilidad_Anualizada': mu.values * 252,
        'Volatilidad_Diaria': volatilities,
        'Volatilidad_Anualizada': volatilities * np.sqrt(252),
        'Ratio_Sharpe_Aprox': (mu.values * 252) / (volatilities * np.sqrt(252))
    })

    # Ordenar por rentabilidad anualizada descendente
    summary = summary.sort_values('Rentabilidad_Anualizada', ascending=False)
    summary = summary.reset_index(drop=True)

    print(summary.to_string(index=False, float_format='%.6f'))

    # Guardar tabla
    summary.to_csv('data/resumen_por_activo.csv', index=False)
    print(f"\n   - Tabla guardada en: data/resumen_por_activo.csv")

    return summary


def main():
    """
    Función principal del análisis exploratorio.
    """
    try:
        # Análisis de datos
        prices, returns, mu, sigma, volatilities, corr_matrix = analyze_data()

        # Crear visualizaciones
        create_visualizations(prices, returns, mu, sigma, volatilities, corr_matrix)

        # Crear tabla resumen
        summary = create_summary_table(mu, volatilities)

        print("\n" + "="*60)
        print("ANÁLISIS EXPLORATORIO COMPLETADO")
        print("="*60)

        return prices, returns, mu, sigma, summary

    except Exception as e:
        print(f"Error durante el análisis: {e}")
        return None, None, None, None, None


if __name__ == "__main__":
    main()
