import numpy as np
import matplotlib.pyplot as plt

from utils import load_and_preprocess_data


class ADALINAGradiente:
    def __init__(self, tolerance=1e-6, max_iterations=10000):
        """
        Inicializa el modelo ADALINA con descenso de gradiente

        Parámetros:
        tolerance: criterio de parada basado en la norma del gradiente
        max_iterations: número máximo de iteraciones
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.weights = None
        self.cost_history = []
        self.iterations = 0

    def fit(self, X, y):
        """
        Entrena el modelo ADALINA usando descenso de gradiente con paso óptimo exacto

        Parámetros:
        X: matriz de características (n_samples, n_features)
        y: vector objetivo (n_samples,)
        """
        # Añadir columna de bias (unos) a X
        X_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Inicializar pesos aleatoriamente
        np.random.seed(42)  # Para reproducibilidad
        self.weights = np.random.normal(0, 0.01, X_bias.shape[1])

        self.cost_history = []

        for iteration in range(self.max_iterations):
            # Calcular predicciones
            y_pred = X_bias @ self.weights

            # Calcular error y costo
            # L(w) = ||Xw - y||^2
            error = y_pred - y
            cost = np.sum(error**2)
            self.cost_history.append(cost)

            # Calcular gradiente
            # ∇L(w) = 2X^T(Xw - y) = 2X^T * error
            gradient = 2 * X_bias.T @ error

            # Criterio de parada: norma del gradiente
            if np.linalg.norm(gradient) <= self.tolerance:
                self.iterations = iteration + 1
                print(f"Convergencia alcanzada en {self.iterations} iteraciones")
                break

            # Calcular paso óptimo exacto para función cuadrática
            # α_k = (d_k^T d_k) / (2 d_k^T X^T X d_k)
            # donde d_k = -∇L(w_k)

            direction = -gradient
            A = X_bias.T @ X_bias

            # Calcular paso óptimo
            numerator = direction.T @ direction
            denominator = 2 * direction.T @ A @ direction

            if denominator == 0:
                print("Denominador cero en el cálculo del paso óptimo")
                break

            alpha_optimal = numerator / denominator

            # Actualizar pesos
            self.weights += alpha_optimal * direction

        else:
            self.iterations = self.max_iterations
            print(f"Máximo de iteraciones alcanzado: {self.max_iterations}")

    def predict(self, X):
        """
        Realiza predicciones con el modelo entrenado

        Parámetros:
        X: matriz de características

        Retorna:
        y_pred: predicciones
        """
        if self.weights is None:
            raise ValueError("El modelo no ha sido entrenado")

        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_bias @ self.weights

    def plot_cost_history(self):
        """
        Grafica la evolución del costo durante el entrenamiento y la guarda
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Evolución del Costo - Descenso de Gradiente')
        plt.xlabel('Iteración')
        plt.ylabel('Costo')
        plt.grid(True)
        plt.savefig('costo_gradiente.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Gráfica guardada como 'costo_gradiente.png'")


def main():
    """
    Función principal para ejecutar el entrenamiento y evaluación
    """
    # Cargar y preprocesar datos
    try:
        X, y, scaler = load_and_preprocess_data('hou_all.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'hou_all.csv'")
        print("Por favor, asegúrate de que el archivo esté en el directorio actual")
        return

    # Crear y entrenar modelo
    print("\n=== Entrenamiento con Descenso de Gradiente ===")
    model = ADALINAGradiente(tolerance=1e-6, max_iterations=10000)
    model.fit(X, y)

    # Mostrar resultados
    print(f"\nNúmero de iteraciones hasta convergencia: {model.iterations}")
    print(f"Costo final: {model.cost_history[-1]:.6f}")
    print(f"Pesos finales: {model.weights}")

    # Predicciones para las primeras 5 viviendas
    print("\n=== Comparación de Valores Reales y Predichos ===")
    y_pred_first5 = model.predict(X[:5])

    for i in range(5):
        print(f"Vivienda {i+1}: Real = {y[i]:.2f}, Predicho = {y_pred_first5[i]:.2f}, Error = {abs(y[i] - y_pred_first5[i]):.2f}")

    # Calcular métricas de evaluación
    y_pred_all = model.predict(X)
    mse = np.mean((y - y_pred_all)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred_all))

    print(f"\n=== Métricas de Evaluación ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Graficar evolución del costo
    model.plot_cost_history()


if __name__ == "__main__":
    main()
