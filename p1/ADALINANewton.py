import numpy as np
import matplotlib.pyplot as plt

from utils import load_and_preprocess_data


class ADALINANewton:
    def __init__(self, tolerance=1e-6, max_iterations=1000):
        """
        Inicializa el modelo ADALINA con método de Newton

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
        Entrena el modelo ADALINA usando el método de Newton

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

        # Calcular Hessiana (constante para función cuadrática)
        # H(w) = 2X^T X
        hessian = 2 * X_bias.T @ X_bias

        # Verificar que la Hessiana sea invertible
        try:
            hessian_inv = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Error: La matriz Hessiana no es invertible")
            print("Intentando con pseudoinversa...")
            hessian_inv = np.linalg.pinv(hessian)

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

            # Calcular dirección de Newton
            # d_k = -H^(-1) * ∇L(w_k)
            newton_direction = -hessian_inv @ gradient

            # Para función cuadrática, el método de Newton converge en un paso
            # Por tanto, no necesitamos backtracking, usamos paso completo
            alpha = 1.0

            # Actualizar pesos
            self.weights += alpha * newton_direction

        else:
            self.iterations = self.max_iterations
            print(f"Máximo de iteraciones alcanzado: {self.max_iterations}")

    def fit_direct(self, X, y):
        """
        Entrena el modelo ADALINA usando la solución directa (ecuación normal)
        Como se deriva en el informe, para funciones cuadráticas Newton converge 
        en una sola iteración a: w = (X^T X)^(-1) X^T y

        Parámetros:
        X: matriz de características (n_samples, n_features)
        y: vector objetivo (n_samples,)
        """
        # Añadir columna de bias (unos) a X
        X_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Calcular solución directa usando ecuación normal
        # w = (X^T X)^(-1) X^T y
        try:
            XtX = X_bias.T @ X_bias
            XtX_inv = np.linalg.inv(XtX)
            self.weights = XtX_inv @ X_bias.T @ y
        except np.linalg.LinAlgError:
            print("Error: Matriz singular, usando pseudoinversa")
            self.weights = np.linalg.pinv(X_bias) @ y

        # Calcular costo final
        y_pred = X_bias @ self.weights
        error = y_pred - y
        final_cost = np.sum(error**2)

        self.cost_history = [final_cost]
        self.iterations = 1

        print(f"Solución directa calculada en 1 paso")
        print(f"Costo final: {final_cost:.6f}")

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
        plt.title('Evolución del Costo - Método de Newton')
        plt.xlabel('Iteración')
        plt.ylabel('Costo')
        plt.grid(True)
        plt.savefig('costo_newton.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Gráfica guardada como 'costo_newton.png'")


def main():
    """
    Función principal para ejecutar el entrenamiento y evaluación
    """
    # Cargar y preprocesar datos
    try:
        X_scaled, y, scaler = load_and_preprocess_data('hou_all.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'hou_all.csv'")
        print("Por favor, asegúrate de que el archivo esté en el directorio actual")
        return

    # Crear y entrenar modelo
    print("\n=== Entrenamiento con Método de Newton ===")
    model = ADALINANewton(tolerance=1e-6, max_iterations=1000)
    model.fit(X_scaled, y)

    # Mostrar resultados
    print(f"\nNúmero de iteraciones hasta convergencia: {model.iterations}")
    print(f"Costo final: {model.cost_history[-1]:.6f}")
    print(f"Pesos finales: {model.weights}")

    # Demostrar la solución directa
    print("\n=== Comparación con Solución Directa (Ecuación Normal) ===")
    model_direct = ADALINANewton()
    model_direct.fit_direct(X_scaled, y)
    print(f"Pesos con solución directa: {model_direct.weights}")

    # Verificar que ambos métodos dan el mismo resultado
    weights_diff = np.linalg.norm(model.weights - model_direct.weights)
    print(f"Diferencia entre métodos: {weights_diff:.10f}")

    # Predicciones para las primeras 5 viviendas
    print("\n=== Comparación de Valores Reales y Predichos ===")
    y_pred_first5 = model.predict(X_scaled[:5])

    for i in range(5):
        print(f"Vivienda {i+1}: Real = {y[i]:.2f}, Predicho = {y_pred_first5[i]:.2f}, Error = {abs(y[i] - y_pred_first5[i]):.2f}")

    # Calcular métricas de evaluación
    y_pred_all = model.predict(X_scaled)
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
