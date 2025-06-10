"""
Parte III: Modificación de CMA-ES para Gestión de Restricciones

Este script ejecuta la comparación entre los dos métodos:
1. Proyección externa (reparación externa)
2. Penalización en la función objetivo
"""

import sys
import os
import time
from datetime import datetime

from data_preparation import load_data
from cma_optimization import compare_methods


def print_header():
    """Imprime cabecera del experimento."""
    print("="*80)
    print("PARTE III: MODIFICACIÓN DE CMA-ES")
    print("="*80)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio de trabajo: {os.getcwd()}")
    print("="*80)


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


def print_summary(results):
    """Imprime resumen final del experimento."""
    print("\n" + "="*80)
    print("RESUMEN FINAL DE LA COMPARACIÓN")
    print("="*80)

    if results is None:
        print("❌ No se pudieron obtener resultados válidos")
        return

    try:
        proj_analysis = results['projection_analysis']
        pen_analysis = results['penalty_analysis']

        print("\n📊 COMPARACIÓN DE MÉTODOS:")
        print(f"\n✅ PROYECCIÓN EXTERNA:")
        print(f"   • Sharpe Ratio: {proj_analysis['sharpe_ratio_annual']:.4f}")
        print(f"   • Rentabilidad anual: {proj_analysis['expected_return_annual']:.4f}")
        print(f"   • Volatilidad anual: {proj_analysis['volatility_annual']:.4f}")
        print(f"   • Cardinalidad: {proj_analysis['cardinality']}")
        print(f"   • Suma de pesos: {proj_analysis['weight_sum']:.6f}")

        print(f"\n⚙️  PENALIZACIÓN:")
        print(f"   • Sharpe Ratio: {pen_analysis['sharpe_ratio_annual']:.4f}")
        print(f"   • Rentabilidad anual: {pen_analysis['expected_return_annual']:.4f}")
        print(f"   • Volatilidad anual: {pen_analysis['volatility_annual']:.4f}")
        print(f"   • Cardinalidad: {pen_analysis['cardinality']}")
        print(f"   • Suma de pesos: {pen_analysis['weight_sum']:.6f}")

        print(f"\n🏆 CONCLUSIONES:")
        if proj_analysis['sharpe_ratio_annual'] > pen_analysis['sharpe_ratio_annual']:
            print("   • Proyección externa obtuvo mejor Sharpe Ratio")
        else:
            print("   • Penalización obtuvo mejor Sharpe Ratio")

        print("   • Ambos métodos cumplen las restricciones tras proyección")
        print("   • Proyección externa garantiza restricciones exactas")
        print("   • Penalización ofrece más flexibilidad en la exploración")

    except Exception as e:
        print(f"❌ Error al generar resumen: {e}")


def main():
    """
    Función principal que ejecuta toda la Parte III.
    """
    start_time = time.time()

    # Imprimir cabecera
    print_header()

    try:
        # Verificar prerrequisitos
        if not check_prerequisites():
            print("\n❌ No se pueden ejecutar los experimentos sin los datos base.")
            return False

        print("\n🚀 INICIANDO COMPARACIÓN DE MÉTODOS...")

        print("Cargando datos de entrenamiento...")
        prices, returns, mu, sigma = load_data()

        print(f"Datos cargados:")
        print(f"  - Activos: {len(mu)}")
        print(f"  - Observaciones: {len(returns)}")

        # Parámetros del experimento
        lambda_risk = 1.0  # Aversión al riesgo fija para comparación
        K = 5  # Cardinalidad
        max_evaluations = 5000  # Evaluaciones por método

        print(f"\n⚙️  PARÁMETROS DEL EXPERIMENTO:")
        print(f"  - λ (aversión al riesgo): {lambda_risk}")
        print(f"  - K (cardinalidad máxima): {K}")
        print(f"  - Evaluaciones máximas: {max_evaluations}")

        # Ejecutar comparación
        results = compare_methods(
            mu=mu.values,
            sigma=sigma.values,
            lambda_risk=lambda_risk,
            K=K,
            max_evaluations=max_evaluations
        )

        if results is None:
            print("❌ Error en la comparación de métodos")
            return False

        # Imprimir resumen final
        print_summary(results)

        total_time = time.time() - start_time

        print(f"\n⏱️  TIEMPO TOTAL DE EJECUCIÓN: {total_time:.2f} segundos")

        print(f"\n📁 ARCHIVOS GENERADOS:")
        generated_files = [
            'data/comparacion_metodos.csv'
        ]

        for file in generated_files:
            if os.path.exists(file):
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (no generado)")

        print(f"\n🎯 OBJETIVO COMPLETADO:")
        print("   Se ha realizado exitosamente la comparación entre proyección")
        print("   externa y penalización para la gestión de restricciones en CMA-ES.")

        return True

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
