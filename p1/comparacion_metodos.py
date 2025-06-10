import numpy as np
import matplotlib.pyplot as plt
import time

from ADALINAGradiente import ADALINAGradiente
from ADALINANewton import ADALINANewton
from utils import load_and_preprocess_data


def compare_methods():
    """Compara ambos métodos de optimización"""

    print("=== COMPARACIÓN DE MÉTODOS ADALINA ===\n")

    try:
        X_scaled, y, scaler = load_and_preprocess_data('hou_all.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'hou_all.csv'")
        print("Por favor, asegúrate de que el archivo esté en el directorio actual")
        return

    # Entrenar con Descenso de Gradiente
    print("1. Entrenando con Descenso de Gradiente...")
    start_time = time.time()

    gradient_model = ADALINAGradiente(tolerance=1e-6, max_iterations=10000)
    gradient_model.fit(X_scaled, y)

    gradient_time = time.time() - start_time

    print(f"   ✓ Convergencia en {gradient_model.iterations} iteraciones")
    print(f"   ✓ Tiempo de entrenamiento: {gradient_time:.4f} segundos")
    print(f"   ✓ Costo final: {gradient_model.cost_history[-1]:.6f}\n")

    # Entrenar con Método de Newton
    print("2. Entrenando con Método de Newton...")
    start_time = time.time()

    newton_model = ADALINANewton(tolerance=1e-6, max_iterations=1000)
    newton_model.fit(X_scaled, y)

    newton_time = time.time() - start_time

    print(f"   ✓ Convergencia en {newton_model.iterations} iteraciones")
    print(f"   ✓ Tiempo de entrenamiento: {newton_time:.4f} segundos")
    print(f"   ✓ Costo final: {newton_model.cost_history[-1]:.6f}\n")

    # Comparación de resultados
    print("=== COMPARACIÓN DE RESULTADOS ===")
    print(f"Iteraciones - Gradiente: {gradient_model.iterations}, Newton: {newton_model.iterations}")
    print(f"Tiempo - Gradiente: {gradient_time:.4f}s, Newton: {newton_time:.4f}s")
    print(f"Eficiencia: Newton es {gradient_model.iterations/newton_model.iterations:.1f}x más rápido en iteraciones")

    # Predicciones para las primeras 5 muestras
    print(f"\n=== PREDICCIONES DE LAS PRIMERAS 5 MUESTRAS ===")
    gradient_pred = gradient_model.predict(X_scaled[:5])
    newton_pred = newton_model.predict(X_scaled[:5])

    print("Muestra | Real    | Gradiente | Newton   | Error Grad | Error Newton")
    print("-" * 70)
    for i in range(5):
        error_grad = abs(y[i] - gradient_pred[i])
        error_newton = abs(y[i] - newton_pred[i])
        print(f"{i+1:7d} | {y[i]:7.3f} | {gradient_pred[i]:9.3f} | {newton_pred[i]:8.3f} | {error_grad:10.3f} | {error_newton:12.3f}")

    # Evaluación completa
    gradient_pred_all = gradient_model.predict(X_scaled)
    newton_pred_all = newton_model.predict(X_scaled)

    mse_grad = np.mean((y - gradient_pred_all)**2)
    mse_newton = np.mean((y - newton_pred_all)**2)

    print(f"\n=== MÉTRICAS DE EVALUACIÓN ===")
    print(f"MSE Gradiente: {mse_grad:.6f}")
    print(f"MSE Newton: {mse_newton:.6f}")
    print(f"Diferencia en pesos: {np.linalg.norm(gradient_model.weights - newton_model.weights):.6f}")

    # Gráficas comparativas
    plt.figure(figsize=(15, 5))

    # Evolución del costo (escala logarítmica)
    plt.subplot(1, 3, 1)
    plt.plot(gradient_model.cost_history, label='Descenso de Gradiente', linewidth=2, alpha=0.8)
    plt.plot(newton_model.cost_history, label='Método de Newton', linewidth=2, alpha=0.8)
    plt.xlabel('Iteración')
    plt.ylabel('Costo (escala log)')
    plt.title('Evolución del Costo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Comparación de predicciones
    plt.subplot(1, 3, 2)
    # Como ambos métodos dan resultados idénticos, mostramos con un offset visual mínimo
    plt.scatter(y, gradient_pred_all, alpha=0.7, s=30, label='Gradiente', color='blue', marker='o')
    plt.scatter(y + 0.1, newton_pred_all, alpha=0.7, s=20, label='Newton', color='red', marker='x')

    # Línea de predicción perfecta (donde valores reales = valores predichos)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.8, linewidth=2,
             label='Predicción Perfecta (y=x)')

    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Valores Reales vs Predichos')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Convergencia (primeras iteraciones) - escala logarítmica
    plt.subplot(1, 3, 3)
    max_iter_plot = min(50, len(gradient_model.cost_history), len(newton_model.cost_history))
    plt.plot(gradient_model.cost_history[:max_iter_plot], label='Descenso de Gradiente',
             linewidth=2, alpha=0.8)
    plt.plot(newton_model.cost_history[:max_iter_plot], label='Método de Newton',
             linewidth=2, alpha=0.8)
    plt.xlabel('Iteración')
    plt.ylabel('Costo (escala log)')
    plt.title('Convergencia (Primeras iteraciones)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('comparacion_metodos.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfica comparativa guardada como 'comparacion_metodos.png'")


if __name__ == "__main__":
    compare_methods()
