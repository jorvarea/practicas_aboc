import numpy as np
import matplotlib.pyplot as plt
import random


class HopfieldNTorres:
    def __init__(self, N, mu1=1.0, mu2=1.0):
        """
        Inicializa la Red de Hopfield para el problema de N-Torres

        Args:
            N: Tamaño del tablero (N x N)
            mu1: Parámetro de penalización para restricciones de filas
            mu2: Parámetro de penalización para restricciones de columnas
        """
        self.N = N
        self.mu1 = mu1
        self.mu2 = mu2

        # Matriz de estados (tablero aplanado)
        self.num_neurons = N * N
        self.states = np.zeros(self.num_neurons)

        # Matrices de pesos y umbrales
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.thresholds = np.zeros(self.num_neurons)

        self._configure_network()

    def _index_to_position(self, idx):
        """Convierte índice lineal a posición (i,j) en el tablero"""
        return idx // self.N, idx % self.N

    def _position_to_index(self, i, j):
        """Convierte posición (i,j) a índice lineal"""
        return i * self.N + j

    def _configure_network(self):
        """Configura los pesos y umbrales de la red según el problema"""
        # Inicializar pesos
        for idx1 in range(self.num_neurons):
            i1, j1 = self._index_to_position(idx1)

            for idx2 in range(self.num_neurons):
                if idx1 == idx2:
                    self.weights[idx1, idx2] = 0  # Sin auto-conexión
                else:
                    i2, j2 = self._index_to_position(idx2)

                    # Conexiones dentro de la misma fila
                    if i1 == i2 and j1 != j2:
                        self.weights[idx1, idx2] = -2 * self.mu1
                    # Conexiones dentro de la misma columna
                    elif j1 == j2 and i1 != i2:
                        self.weights[idx1, idx2] = -2 * self.mu2
                    else:
                        self.weights[idx1, idx2] = 0

        # Configurar umbrales
        for idx in range(self.num_neurons):
            i, j = self._index_to_position(idx)
            self.thresholds[idx] = -(self.mu1 + self.mu2)

    def calculate_energy(self):
        """Calcula la energía actual de la configuración"""
        energy = 0

        # Término cuadrático de la función de energía
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                energy -= 0.5 * self.weights[i, j] * self.states[i] * self.states[j]

        # Término lineal (umbrales)
        for i in range(self.num_neurons):
            energy += self.thresholds[i] * self.states[i]

        return energy

    def calculate_violations(self):
        """Calcula el número de violaciones de restricciones"""
        violations = 0

        # Violaciones en filas (más o menos de 1 torre por fila)
        for i in range(self.N):
            row_sum = 0
            for j in range(self.N):
                idx = self._position_to_index(i, j)
                row_sum += self.states[idx]
            violations += abs(row_sum - 1)

        # Violaciones en columnas (más o menos de 1 torre por columna)
        for j in range(self.N):
            col_sum = 0
            for i in range(self.N):
                idx = self._position_to_index(i, j)
                col_sum += self.states[idx]
            violations += abs(col_sum - 1)

        return violations

    def initialize_random(self):
        """Inicializa el estado de la red aleatoriamente"""
        self.states = np.random.choice([0, 1], size=self.num_neurons)

    def initialize_heuristic(self):
        """Inicialización heurística: una torre por fila en posición aleatoria"""
        self.states = np.zeros(self.num_neurons)
        for i in range(self.N):
            j = random.randint(0, self.N - 1)
            idx = self._position_to_index(i, j)
            self.states[idx] = 1

    def update_neuron(self, neuron_idx):
        """Actualiza el estado de una neurona específica"""
        # Calcular el potencial sináptico
        potential = 0
        for j in range(self.num_neurons):
            potential += self.weights[neuron_idx, j] * self.states[j]

        # Aplicar función de activación
        if potential >= self.thresholds[neuron_idx]:
            new_state = 1
        else:
            new_state = 0

        return new_state

    def evolve(self, max_iterations=1000, convergence_check=10):
        """
        Evoluciona la red hasta convergencia

        Args:
            max_iterations: Número máximo de iteraciones
            convergence_check: Número de iteraciones sin cambio para considerar convergencia

        Returns:
            Tupla (converged, iterations, energy_history, violations_history)
        """
        energy_history = []
        violations_history = []
        no_change_count = 0

        for iteration in range(max_iterations):
            old_states = self.states.copy()

            # Actualizar todas las neuronas en orden aleatorio
            neuron_order = list(range(self.num_neurons))
            random.shuffle(neuron_order)

            for neuron_idx in neuron_order:
                self.states[neuron_idx] = self.update_neuron(neuron_idx)

            # Calcular métricas
            energy = self.calculate_energy()
            violations = self.calculate_violations()

            energy_history.append(energy)
            violations_history.append(violations)

            # Verificar convergencia
            if np.array_equal(old_states, self.states):
                no_change_count += 1
                if no_change_count >= convergence_check:
                    return True, iteration + 1, energy_history, violations_history
            else:
                no_change_count = 0

        return False, max_iterations, energy_history, violations_history

    def get_board_representation(self):
        """Devuelve la representación del tablero como matriz N×N"""
        board = np.zeros((self.N, self.N))
        for idx in range(self.num_neurons):
            i, j = self._index_to_position(idx)
            board[i, j] = self.states[idx]
        return board

    def is_valid_solution(self):
        """Verifica si la configuración actual es una solución válida"""
        return self.calculate_violations() == 0

    def visualize_solution(self, title="Solución del Problema de N-Torres", save_path=None):
        """Visualiza la solución en el tablero y la guarda como imagen"""
        board = self.get_board_representation()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Crear tablero de ajedrez
        for i in range(self.N):
            for j in range(self.N):
                color = 'lightgray' if (i + j) % 2 == 0 else 'white'
                ax.add_patch(plt.Rectangle((j, self.N-1-i), 1, 1,
                                           facecolor=color, edgecolor='black'))

                # Colocar torres
                if board[i, j] == 1:
                    ax.text(j + 0.5, self.N-1-i + 0.5, '♜',
                            fontsize=24, ha='center', va='center', color='red')

        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.N + 1))
        ax.set_yticks(range(self.N + 1))
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Información adicional
        violations = self.calculate_violations()
        energy = self.calculate_energy()
        valid = "Sí" if self.is_valid_solution() else "No"

        plt.figtext(0.02, 0.02, f'Violaciones: {int(violations)} | Energía: {energy:.2f} | Válida: {valid}',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

        plt.tight_layout()

        # Guardar imagen si se especifica la ruta
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Imagen guardada: {save_path}")

        plt.close()  # Cerrar la figura para liberar memoria


def solve_n_torres(N, mu1=1.0, mu2=1.0, max_attempts=10, max_iterations=1000):
    """
    Resuelve el problema de N-Torres usando Red de Hopfield

    Args:
        N: Tamaño del tablero
        mu1, mu2: Parámetros de penalización
        max_attempts: Número máximo de intentos
        max_iterations: Iteraciones máximas por intento

    Returns:
        Mejor solución encontrada
    """
    best_solution = None
    best_violations = float('inf')
    best_energy = float('inf')

    print(f"Resolviendo problema de {N}-Torres...")
    print(f"Parámetros: μ₁={mu1}, μ₂={mu2}")
    print("-" * 50)

    for attempt in range(max_attempts):
        # Crear nueva instancia
        hopfield = HopfieldNTorres(N, mu1, mu2)

        # Probar diferentes inicializaciones
        if attempt % 2 == 0:
            hopfield.initialize_heuristic()
            init_type = "Heurística"
        else:
            hopfield.initialize_random()
            init_type = "Aleatoria"

        # Evolucionar la red
        converged, iterations, energy_hist, violations_hist = hopfield.evolve(max_iterations)

        # Evaluar solución
        final_violations = hopfield.calculate_violations()
        final_energy = hopfield.calculate_energy()

        print(f"Intento {attempt + 1:2d}: {init_type:10s} | "
              f"Iter: {iterations:3d} | Conv: {'Sí' if converged else 'No':2s} | "
              f"Violaciones: {int(final_violations):2d} | Energía: {final_energy:6.2f}")

        # Guardar mejor solución
        if (final_violations < best_violations or
                (final_violations == best_violations and final_energy < best_energy)):
            best_solution = hopfield
            best_violations = final_violations
            best_energy = final_energy

    print("-" * 50)
    print(f"Mejor solución: {int(best_violations)} violaciones, energía {best_energy:.2f}")

    if best_solution.is_valid_solution():
        print("¡Solución válida encontrada! ✓")
    else:
        print("No se encontró solución completamente válida ✗")

    return best_solution


# Ejemplo de uso
if __name__ == "__main__":
    # Resolver para diferentes tamaños
    for N in [4, 5, 6]:
        print(f"\n{'='*60}")
        print(f"PROBLEMA DE {N}-TORRES")
        print(f"{'='*60}")

        # Resolver el problema
        solution = solve_n_torres(N, mu1=2.0, mu2=2.0, max_attempts=15)

        # Generar nombre de archivo para la imagen
        image_filename = f"hopfield_{N}_torres_solucion.png"

        # Visualizar y guardar la mejor solución
        solution.visualize_solution(
            title=f"Mejor Solución para {N}-Torres (Red de Hopfield)",
            save_path=image_filename
        )

        # Mostrar tablero en texto
        board = solution.get_board_representation()
        print(f"\nTablero {N}×{N}:")
        for i in range(N):
            row = ""
            for j in range(N):
                if board[i, j] == 1:
                    row += "T "
                else:
                    row += ". "
            print(row)

        print(f"Imagen del tablero guardada como: {image_filename}")
