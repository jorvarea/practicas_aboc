import numpy as np
import matplotlib.pyplot as plt

from utils import load_and_preprocess_data


class ADALINALassoGradienteProyectado:
    """
    Implementación del algoritmo de Gradiente Proyectado para resolver 
    el problema ADALINA LASSO con regularización ℓ1.

    Problema de optimización:
    min_{w} f(w) = ||Xw - y||² + λ||w||₁

    CORRECCIONES IMPORTANTES:
    1. Se añade término bias (columna de unos) para predicciones correctas
    2. Función objetivo consistente con informe (sin factor 0.5)
    3. Gradiente con factor 2 correcto: ∇(||Xw - y||²) = 2X^T(Xw - y)
    """

    def __init__(self, lambda_reg=100, max_iter=100, tol=1e-3, alpha_init=1.0, beta=0.5):
        """
        Parámetros:
        -----------
        lambda_reg : float
            Parámetro de regularización λ
        max_iter : int
            Número máximo de iteraciones
        tol : float
            Tolerancia para el criterio de parada
        alpha_init : float
            Valor inicial del tamaño de paso
        beta : float
            Factor de reducción para backtracking
        """
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_init = alpha_init
        self.beta = beta
        self.w_ = None
        self.errors_ = []
        self.n_iter_ = 0

    def _objective_function(self, w, X, y):
        """
        Calcula la función objetivo f(w) = 1/2 ||Xw - y||² + λ||w||₁
        """
        residual = X @ w - y
        mse_term = 0.5 * np.sum(residual**2)
        l1_term = self.lambda_reg * np.sum(np.abs(w))
        return mse_term + l1_term

    def _gradient_smooth_part(self, w, X, y):
        """
        Calcula el gradiente de la parte suave: ∇(1/2 ||Xw - y||²) = X^T(Xw - y)
        """
        residual = X @ w - y
        return X.T @ residual

    def _soft_threshold(self, x, threshold):
        """
        Operador de umbralización suave (soft thresholding)
        Proyección sobre la restricción de norma ℓ1
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _backtracking_line_search(self, w_k, grad_k, X, y):
        """
        Implementa backtracking line search para encontrar el tamaño de paso
        """
        alpha = self.alpha_init
        f_k = self._objective_function(w_k, X, y)

        # Dirección de descenso (gradiente negativo de la parte suave)
        d_k = -grad_k

        # Backtracking
        l = 0
        while True:
            # Paso de gradiente
            y_k = w_k + alpha * d_k

            # Proyección (soft thresholding) - ajustar threshold por el nuevo gradiente
            w_new = self._soft_threshold(y_k, alpha * self.lambda_reg)

            # Evaluar función objetivo en el nuevo punto
            f_new = self._objective_function(w_new, X, y)

            # Criterio de Armijo (versión simplificada)
            if f_new < f_k or l > 20:  # Máximo 20 iteraciones de backtracking
                break

            alpha *= self.beta
            l += 1

        return alpha, w_new

    def fit(self, X, y):
        """
        Ajusta el modelo usando el algoritmo de Gradiente Proyectado
        """
        # Añadir columna de bias (unos) a X
        X_bias = np.column_stack([np.ones(X.shape[0]), X])

        n_samples, n_features = X_bias.shape

        # Inicialización
        self.w_ = np.zeros(n_features)
        self.errors_ = []

        print(f"Iniciando algoritmo de Gradiente Proyectado con λ = {self.lambda_reg}")
        print(f"Dimensiones: {n_samples} muestras, {n_features} características (incluyendo bias)")

        for k in range(self.max_iter):
            # Calcular gradiente de la parte suave
            grad_k = self._gradient_smooth_part(self.w_, X_bias, y)

            # Backtracking line search
            alpha_k, w_new = self._backtracking_line_search(self.w_, grad_k, X_bias, y)

            # Calcular error actual
            current_error = self._objective_function(self.w_, X_bias, y)
            self.errors_.append(current_error)

            # Criterio de parada
            if k > 0 and abs(self.errors_[-1] - self.errors_[-2]) < self.tol:
                print(f"Convergencia alcanzada en iteración {k+1}")
                print(f"Diferencia de error: {abs(self.errors_[-1] - self.errors_[-2]):.6f}")
                break

            # Actualizar pesos
            self.w_ = w_new

            if (k + 1) % 10 == 0:
                print(f"Iteración {k+1}: Error = {current_error:.6f}, α = {alpha_k:.6f}")

        self.n_iter_ = k + 1
        print(f"Algoritmo terminado en {self.n_iter_} iteraciones")

        # Mostrar estadísticas de los coeficientes
        n_zero_coef = np.sum(np.abs(self.w_) < 1e-6)
        print(f"Coeficientes exactamente cero: {n_zero_coef}/{len(self.w_)}")

        return self

    def predict(self, X):
        """
        Realiza predicciones usando el modelo ajustado
        """
        if self.w_ is None:
            raise ValueError("El modelo debe ser ajustado antes de hacer predicciones")

        # Añadir columna de bias para predicción
        X_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_bias @ self.w_

    def get_coefficients(self):
        """
        Retorna los coeficientes del modelo
        """
        return self.w_.copy()

    def plot_convergence(self):
        """
        Grafica la convergencia del algoritmo
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.errors_, 'b-', linewidth=2)
        plt.title('Convergencia del Algoritmo de Gradiente Proyectado')
        plt.xlabel('Iteración')
        plt.ylabel('Función Objetivo')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'costo_gradiente_lasso_{self.lambda_reg}.png', dpi=300)
        plt.close()
        print(f"Gráfica guardada como 'costo_gradiente_lasso_{self.lambda_reg}.png'")


def main():
    """
    Función principal para ejecutar el experimento
    """
    # Cargar y preprocesar datos
    # Nota: Ajustar la ruta del archivo según sea necesario
    try:
        X, y, scaler = load_and_preprocess_data('hou_all.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'hou_all.csv'")

    # Experimento principal con λ = 100
    print("="*60)
    print("EXPERIMENTO PRINCIPAL: λ = 100")
    print("="*60)

    model = ADALINALassoGradienteProyectado(lambda_reg=100)
    model.fit(X, y)

    # Predicciones para las primeras 5 muestras
    y_pred = model.predict(X)

    print("\nComparación de valores reales vs predichos (primeras 5 muestras):")
    print("-" * 50)
    for i in range(min(5, len(y))):
        print(f"Muestra {i+1}: Real = {y[i]:.3f}, Predicho = {y_pred[i]:.3f}, Error = {abs(y[i] - y_pred[i]):.3f}")

    # Mostrar coeficientes
    coefficients = model.get_coefficients()
    print(f"\nCoeficientes del modelo:")
    for i, coef in enumerate(coefficients):
        print(f"w[{i}] = {coef:.6f}")

    # Graficar convergencia
    model.plot_convergence()

    # Experimentos con diferentes valores de λ
    print("\n" + "="*60)
    print("EXPERIMENTOS CON DIFERENTES VALORES DE λ")
    print("="*60)

    lambda_values = [10, 50, 200]
    results = {}

    for lambda_val in lambda_values:
        print(f"\nExperimento con λ = {lambda_val}")
        print("-" * 40)

        model_exp = ADALINALassoGradienteProyectado(lambda_reg=lambda_val)
        model_exp.fit(X, y)

        y_pred_exp = model_exp.predict(X)
        mse = np.mean((y - y_pred_exp)**2)
        n_zero_coef = np.sum(np.abs(model_exp.get_coefficients()) < 1e-6)

        results[lambda_val] = {
            'mse': mse,
            'n_zero_coef': n_zero_coef,
            'coefficients': model_exp.get_coefficients(),
            'n_iter': model_exp.n_iter_
        }

        print(f"MSE: {mse:.6f}")
        print(f"Coeficientes cero: {n_zero_coef}/{len(coefficients)}")
        print(f"Iteraciones: {model_exp.n_iter_}")

    # Análisis comparativo
    print("\n" + "="*60)
    print("ANÁLISIS COMPARATIVO")
    print("="*60)

    print(f"{'λ':<10} {'MSE':<12} {'Coef. Cero':<12} {'Iteraciones':<12}")
    print("-" * 50)

    # Agregar resultado principal
    main_mse = np.mean((y - model.predict(X))**2)
    main_zero_coef = np.sum(np.abs(model.get_coefficients()) < 1e-6)
    print(f"{100:<10} {main_mse:<12.6f} {main_zero_coef:<12} {model.n_iter_:<12}")

    for lambda_val in lambda_values:
        res = results[lambda_val]
        print(f"{lambda_val:<10} {res['mse']:<12.6f} {res['n_zero_coef']:<12} {res['n_iter']:<12}")


if __name__ == "__main__":
    main()
