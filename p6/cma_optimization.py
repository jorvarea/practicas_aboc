import numpy as np
import pandas as pd
import cma
import time
import os
from projection import project_weights


class PortfolioOptimizer:
    """
    Optimizador de carteras usando CMA-ES con restricciones de cardinalidad.
    """

    def __init__(self, mu, sigma, lambda_risk, K):
        """
        Inicializa el optimizador.

        Args:
            mu (np.array): Vector de rentabilidades medias
            sigma (np.array): Matriz de covarianzas
            lambda_risk (float): Parámetro de aversión al riesgo
            K (int): Cardinalidad máxima (número de activos)
        """
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.lambda_risk = lambda_risk
        self.K = K
        self.S = len(mu)  # Número de activos

        print(f"Optimizador inicializado:")
        print(f"  - Número de activos (S): {self.S}")
        print(f"  - Cardinalidad máxima (K): {self.K}")
        print(f"  - Parámetro de aversión al riesgo (λ): {self.lambda_risk}")

    def markowitz_objective(self, weights):
        """
        Función objetivo de Markowitz: f(w) = 0.5 * w^T * Σ * w - λ * w^T * μ

        Args:
            weights (np.array): Vector de pesos de la cartera

        Returns:
            float: Valor de la función objetivo
        """
        weights = np.array(weights)
        risk_term = 0.5 * weights.T @ self.sigma @ weights
        return_term = self.lambda_risk * weights.T @ self.mu
        return risk_term - return_term

    def penalty_objective(self, weights):
        """
        Función objetivo con penalización para restricciones.

        Args:
            weights (np.array): Vector de pesos

        Returns:
            float: Valor de función objetivo penalizada
        """
        weights = np.array(weights)

        # Función objetivo base
        base_objective = self.markowitz_objective(weights)

        # Penalizaciones
        beta = 1000.0  # Penalización por cardinalidad
        gamma = 1000.0  # Penalización por presupuesto

        # Penalización por exceso de cardinalidad
        cardinality = np.sum(np.abs(weights) > 1e-6)
        cardinality_penalty = beta * max(0, cardinality - self.K)

        # Penalización por violación de presupuesto
        budget_penalty = gamma * abs(np.sum(weights) - 1.0)

        return base_objective + cardinality_penalty + budget_penalty

    def optimize_with_projection(self, max_evaluations=5000, verbose=True):
        """
        Optimiza usando proyección externa (reparación externa).

        Args:
            max_evaluations (int): Número máximo de evaluaciones
            verbose (bool): Mostrar información de progreso

        Returns:
            dict: Resultados de optimización
        """
        if verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZACIÓN CON PROYECCIÓN EXTERNA")
            print(f"{'='*60}")

        def objective_with_projection(weights):
            # Proyectar pesos para cumplir restricciones
            projected_weights = project_weights(weights, self.K)
            # Evaluar función objetivo en pesos proyectados
            return self.markowitz_objective(projected_weights)

        # Configurar CMA-ES
        initial_guess = np.random.random(self.S)
        initial_sigma = 0.5

        options = {
            'maxfevals': max_evaluations,
            'verbose': -1 if not verbose else -9,
            'seed': 42
        }

        start_time = time.time()

        # Ejecutar optimización
        es = cma.CMAEvolutionStrategy(initial_guess, initial_sigma, options)

        while not es.stop():
            solutions = es.ask()
            fitness_values = [objective_with_projection(x) for x in solutions]
            es.tell(solutions, fitness_values)

        optimization_time = time.time() - start_time

        # Obtener mejor solución y proyectarla
        best_raw = es.result.xbest
        best_weights = project_weights(best_raw, self.K)
        best_fitness = self.markowitz_objective(best_weights)

        if verbose:
            print(f"\nOptimización completada:")
            print(f"  - Generaciones: {es.result.iterations}")
            print(f"  - Evaluaciones: {es.result.evaluations}")
            print(f"  - Tiempo: {optimization_time:.2f} segundos")
            print(f"  - Mejor fitness: {best_fitness:.6f}")

        return {
            'method': 'projection',
            'best_weights': best_weights,
            'best_fitness': best_fitness,
            'generations': es.result.iterations,
            'evaluations': es.result.evaluations,
            'time': optimization_time,
            'cma_result': es.result
        }

    def optimize_with_penalty(self, max_evaluations=5000, verbose=True):
        """
        Optimiza usando penalización en la función objetivo.

        Args:
            max_evaluations (int): Número máximo de evaluaciones
            verbose (bool): Mostrar información de progreso

        Returns:
            dict: Resultados de optimización
        """
        if verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZACIÓN CON PENALIZACIÓN")
            print(f"{'='*60}")

        # Configurar CMA-ES
        initial_guess = np.random.random(self.S)
        initial_sigma = 0.5

        options = {
            'maxfevals': max_evaluations,
            'verbose': -1 if not verbose else -9,
            'seed': 42
        }

        start_time = time.time()

        # Ejecutar optimización
        es = cma.CMAEvolutionStrategy(initial_guess, initial_sigma, options)

        while not es.stop():
            solutions = es.ask()
            fitness_values = [self.penalty_objective(x) for x in solutions]
            es.tell(solutions, fitness_values)

        optimization_time = time.time() - start_time

        # Obtener mejor solución
        best_raw = es.result.xbest
        best_fitness_penalty = self.penalty_objective(best_raw)

        # Proyectar para obtener solución factible
        best_weights = project_weights(best_raw, self.K)
        best_fitness = self.markowitz_objective(best_weights)

        if verbose:
            print(f"\nOptimización completada:")
            print(f"  - Generaciones: {es.result.iterations}")
            print(f"  - Evaluaciones: {es.result.evaluations}")
            print(f"  - Tiempo: {optimization_time:.2f} segundos")
            print(f"  - Mejor fitness (penalizado): {best_fitness_penalty:.6f}")
            print(f"  - Mejor fitness (proyectado): {best_fitness:.6f}")

        return {
            'method': 'penalty',
            'best_weights': best_weights,
            'best_fitness': best_fitness,
            'best_fitness_penalty': best_fitness_penalty,
            'best_raw': best_raw,
            'generations': es.result.iterations,
            'evaluations': es.result.evaluations,
            'time': optimization_time,
            'cma_result': es.result
        }

    def analyze_portfolio(self, weights, asset_names=None):
        """
        Analiza una cartera dada.

        Args:
            weights (np.array): Vector de pesos
            asset_names (list): Nombres de activos (opcional)

        Returns:
            dict: Métricas de la cartera
        """
        weights = np.array(weights)

        # Métricas básicas
        expected_return = weights.T @ self.mu
        variance = weights.T @ self.sigma @ weights
        volatility = np.sqrt(variance)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0

        # Restricciones
        cardinality = np.sum(np.abs(weights) > 1e-6)
        weight_sum = np.sum(weights)

        # Anualizadas (252 días laborables)
        expected_return_annual = expected_return * 252
        volatility_annual = volatility * np.sqrt(252)
        sharpe_ratio_annual = sharpe_ratio * np.sqrt(252)

        analysis = {
            'expected_return_daily': expected_return,
            'expected_return_annual': expected_return_annual,
            'volatility_daily': volatility,
            'volatility_annual': volatility_annual,
            'variance': variance,
            'sharpe_ratio_daily': sharpe_ratio,
            'sharpe_ratio_annual': sharpe_ratio_annual,
            'cardinality': cardinality,
            'weight_sum': weight_sum,
            'weights': weights
        }

        # Activos seleccionados
        if asset_names is not None:
            selected_assets = []
            for i, (name, weight) in enumerate(zip(asset_names, weights)):
                if abs(weight) > 1e-6:
                    selected_assets.append((name, weight))
            analysis['selected_assets'] = selected_assets

        return analysis


def compare_methods(mu, sigma, lambda_risk=1.0, K=5, max_evaluations=5000):
    """
    Compara los métodos de proyección y penalización.

    Args:
        mu: Vector de rentabilidades medias
        sigma: Matriz de covarianzas
        lambda_risk: Parámetro de aversión al riesgo
        K: Cardinalidad máxima
        max_evaluations: Evaluaciones máximas por método

    Returns:
        dict: Resultados de comparación
    """
    print(f"\n{'='*60}")
    print("COMPARACIÓN DE MÉTODOS: PROYECCIÓN vs PENALIZACIÓN")
    print(f"{'='*60}")
    print(f"Parámetros:")
    print(f"  - λ (aversión al riesgo): {lambda_risk}")
    print(f"  - K (cardinalidad): {K}")
    print(f"  - Evaluaciones máximas: {max_evaluations}")

    # Crear optimizador
    optimizer = PortfolioOptimizer(mu, sigma, lambda_risk, K)

    # Optimizar con proyección
    result_projection = optimizer.optimize_with_projection(max_evaluations)

    # Optimizar con penalización
    result_penalty = optimizer.optimize_with_penalty(max_evaluations)

    # Analizar carteras resultantes
    analysis_projection = optimizer.analyze_portfolio(result_projection['best_weights'])
    analysis_penalty = optimizer.analyze_portfolio(result_penalty['best_weights'])

    # Crear tabla de comparación
    print(f"\n{'='*60}")
    print("RESULTADOS DE COMPARACIÓN")
    print(f"{'='*60}")

    comparison_data = {
        'Métrica': [
            'Fitness',
            'Sharpe Ratio (anual)',
            'Rentabilidad (anual)',
            'Volatilidad (anual)',
            'Cardinalidad',
            'Suma de pesos',
            'Tiempo (s)',
            'Evaluaciones'
        ],
        'Proyección': [
            f"{result_projection['best_fitness']:.6f}",
            f"{analysis_projection['sharpe_ratio_annual']:.4f}",
            f"{analysis_projection['expected_return_annual']:.4f}",
            f"{analysis_projection['volatility_annual']:.4f}",
            f"{analysis_projection['cardinality']:.0f}",
            f"{analysis_projection['weight_sum']:.6f}",
            f"{result_projection['time']:.2f}",
            f"{result_projection['evaluations']}"
        ],
        'Penalización': [
            f"{result_penalty['best_fitness']:.6f}",
            f"{analysis_penalty['sharpe_ratio_annual']:.4f}",
            f"{analysis_penalty['expected_return_annual']:.4f}",
            f"{analysis_penalty['volatility_annual']:.4f}",
            f"{analysis_penalty['cardinality']:.0f}",
            f"{analysis_penalty['weight_sum']:.6f}",
            f"{result_penalty['time']:.2f}",
            f"{result_penalty['evaluations']}"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Guardar resultados
    os.makedirs('data', exist_ok=True)
    comparison_df.to_csv('data/comparacion_metodos.csv', index=False)

    return {
        'optimizer': optimizer,
        'projection_result': result_projection,
        'penalty_result': result_penalty,
        'projection_analysis': analysis_projection,
        'penalty_analysis': analysis_penalty,
        'comparison_df': comparison_df
    }
