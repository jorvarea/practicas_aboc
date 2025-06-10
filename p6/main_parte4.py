"""
Parte IV: Validación con datos de 2025
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from data_preparation import get_sp100_tickers, load_data, download_stock_data, calculate_returns
from cma_optimization import PortfolioOptimizer


def download_validation_data(tickers, start_date='2025-01-01', end_date='2025-02-28'):
    """
    Descarga datos de validación usando la función existente de data_preparation.
    """
    print(f"\n{'='*60}")
    print("DESCARGA DE DATOS DE VALIDACIÓN (2025)")
    print(f"{'='*60}")

    prices = download_stock_data(tickers, start_date, end_date)
    return prices


def validate_portfolio(weights, returns_validation, tickers):
    """
    Valida una cartera específica usando datos de validación.
    """
    common_assets = [ticker for ticker in tickers if ticker in returns_validation.columns]
    asset_indices = [tickers.index(asset) for asset in common_assets]
    adjusted_weights = weights[asset_indices]

    if np.sum(adjusted_weights) > 0:
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
    else:
        adjusted_weights = np.ones(len(common_assets)) / len(common_assets)

    portfolio_returns = returns_validation[common_assets].dot(adjusted_weights)

    mean_return_daily = portfolio_returns.mean()
    mean_return_annual = mean_return_daily * 252

    volatility_daily = portfolio_returns.std()
    volatility_annual = volatility_daily * np.sqrt(252)

    sharpe_ratio_daily = mean_return_daily / volatility_daily if volatility_daily > 0 else 0
    sharpe_ratio_annual = sharpe_ratio_daily * np.sqrt(252)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    cumulative_return = cumulative_returns.iloc[-1] - 1

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'portfolio_returns': portfolio_returns,
        'mean_return_daily': mean_return_daily,
        'mean_return_annual': mean_return_annual,
        'volatility_daily': volatility_daily,
        'volatility_annual': volatility_annual,
        'sharpe_ratio_daily': sharpe_ratio_daily,
        'sharpe_ratio_annual': sharpe_ratio_annual,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'adjusted_weights': adjusted_weights,
        'common_assets': common_assets
    }


def validate_lambda_portfolios(returns_validation, tickers):
    """
    Valida carteras optimizadas con diferentes valores de lambda.
    """
    print(f"\n{'='*60}")
    print("VALIDACIÓN DE CARTERAS CON DIFERENTES λ")
    print(f"{'='*60}")

    _, _, mu, sigma = load_data()
    lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    validation_results = {}

    for lambda_val in lambda_values:
        print(f"\nValidando cartera con λ = {lambda_val}")

        optimizer = PortfolioOptimizer(mu.values, sigma.values, lambda_val, K=5)
        train_result = optimizer.optimize_with_projection(max_evaluations=5000, verbose=False)
        train_analysis = optimizer.analyze_portfolio(train_result['best_weights'])

        validation_metrics = validate_portfolio(
            train_result['best_weights'],
            returns_validation,
            tickers
        )

        validation_results[lambda_val] = {
            'train_result': train_result,
            'train_analysis': train_analysis,
            'validation_metrics': validation_metrics
        }

        print(f"  - Sharpe entrenamiento: {train_analysis['sharpe_ratio_annual']:.4f}")
        print(f"  - Sharpe validación: {validation_metrics['sharpe_ratio_annual']:.4f}")

    return validation_results


def analyze_validation_results(validation_results):
    """
    Analiza los resultados de validación.
    """
    print(f"\n{'='*60}")
    print("ANÁLISIS DE RESULTADOS DE VALIDACIÓN")
    print(f"{'='*60}")

    summary_data = []

    for lambda_val, data in validation_results.items():
        train = data['train_analysis']
        valid = data['validation_metrics']

        summary_data.append({
            'Lambda': lambda_val,
            'Train_Sharpe': train['sharpe_ratio_annual'],
            'Train_Return': train['expected_return_annual'],
            'Train_Vol': train['volatility_annual'],
            'Valid_Sharpe': valid['sharpe_ratio_annual'],
            'Valid_Return': valid['mean_return_annual'],
            'Valid_Vol': valid['volatility_annual'],
            'Valid_Cumulative': valid['cumulative_return'],
            'Valid_MaxDrawdown': valid['max_drawdown'],
            'Sharpe_Diff': valid['sharpe_ratio_annual'] - train['sharpe_ratio_annual'],
            'Return_Diff': valid['mean_return_annual'] - train['expected_return_annual']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Lambda').reset_index(drop=True)

    print("\nTabla resumen de validación:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

    os.makedirs('data', exist_ok=True)
    summary_df.to_csv('data/validacion_resultados.csv', index=False)
    print(f"\n✓ Resultados guardados en: data/validacion_resultados.csv")

    return summary_df


def create_validation_plots(validation_results, summary_df, save_dir='figuras'):
    """
    Crea visualizaciones de los resultados de validación.
    """
    print(f"\n{'='*50}")
    print("CREANDO VISUALIZACIONES DE VALIDACIÓN")
    print(f"{'='*50}")

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].scatter(summary_df['Train_Sharpe'], summary_df['Valid_Sharpe'],
                       c=summary_df['Lambda'], s=100, cmap='viridis', alpha=0.7)
    axes[0, 0].plot([summary_df['Train_Sharpe'].min(), summary_df['Train_Sharpe'].max()],
                    [summary_df['Train_Sharpe'].min(), summary_df['Train_Sharpe'].max()],
                    'k--', alpha=0.5, label='Línea 45°')
    axes[0, 0].set_xlabel('Sharpe Entrenamiento')
    axes[0, 0].set_ylabel('Sharpe Validación')
    axes[0, 0].set_title('Sharpe: Entrenamiento vs Validación')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(summary_df['Train_Return'], summary_df['Valid_Return'],
                       c=summary_df['Lambda'], s=100, cmap='viridis', alpha=0.7)
    axes[0, 1].plot([summary_df['Train_Return'].min(), summary_df['Train_Return'].max()],
                    [summary_df['Train_Return'].min(), summary_df['Train_Return'].max()],
                    'k--', alpha=0.5, label='Línea 45°')
    axes[0, 1].set_xlabel('Rentabilidad Entrenamiento')
    axes[0, 1].set_ylabel('Rentabilidad Validación')
    axes[0, 1].set_title('Rentabilidad: Entrenamiento vs Validación')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(summary_df['Lambda'], summary_df['Sharpe_Diff'], 'bo-')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('λ (Aversión al Riesgo)')
    axes[1, 0].set_ylabel('Diferencia Sharpe')
    axes[1, 0].set_title('Diferencia Sharpe (Validación - Entrenamiento)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(summary_df['Lambda'], summary_df['Return_Diff'], 'ro-')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('λ (Aversión al Riesgo)')
    axes[1, 1].set_ylabel('Diferencia Rentabilidad')
    axes[1, 1].set_title('Diferencia Rentabilidad (Validación - Entrenamiento)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'validacion_comparacion.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Gráficos de validación guardados en '{save_dir}/'")


def check_prerequisites():
    """Verifica que los prerrequisitos estén disponibles."""
    print("\n🔍 VERIFICANDO PRERREQUISITOS...")

    required_files = [
        'data/precios_2024.csv',
        'data/rentabilidades_2024.csv',
        'data/mu_vector.csv',
        'data/sigma_matrix.csv'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ ARCHIVOS FALTANTES:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 SOLUCIÓN: Ejecute primero 'python data_preparation.py'")
        return False
    else:
        print("✅ Todos los archivos necesarios están disponibles")
        return True


def main():
    """
    Función principal que ejecuta toda la Parte IV.
    """
    start_time = time.time()

    print("="*80)
    print("PARTE IV: VALIDACIÓN CON DATOS DE 2025")
    print("="*80)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    try:
        if not check_prerequisites():
            print("\n❌ No se pueden ejecutar los experimentos sin los datos base.")
            return False

        print("\n🚀 INICIANDO EXPERIMENTOS DE VALIDACIÓN...")

        tickers = get_sp100_tickers()
        print(f"Activos a validar: {len(tickers)} ({', '.join(tickers[:5])}...)")

        prices_validation = download_validation_data(tickers)
        returns_validation = calculate_returns(prices_validation)
        validation_results = validate_lambda_portfolios(returns_validation, tickers)
        summary_df = analyze_validation_results(validation_results)
        create_validation_plots(validation_results, summary_df)

        print(f"\n{'='*50}")
        print("GUARDANDO DATOS DE VALIDACIÓN")
        print(f"{'='*50}")

        prices_validation.to_csv('data/precios_validacion_2025.csv')
        returns_validation.to_csv('data/rentabilidades_validacion_2025.csv')

        print(f"✓ Datos de validación guardados en data/")

        total_time = time.time() - start_time

        print(f"\n⏱️  TIEMPO TOTAL DE EJECUCIÓN: {total_time:.2f} segundos")
        print("\n" + "="*80)
        print("VALIDACIÓN COMPLETADA")
        print("="*80)

        return validation_results, summary_df

    except KeyboardInterrupt:
        print("\n\n⚠️  Experimento interrumpido por el usuario")
        return False

    except Exception as e:
        print(f"\n❌ ERROR durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
