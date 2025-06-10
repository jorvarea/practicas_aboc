import numpy as np


def project_weights(w_tilde, K):
    """
    Proyecta un vector de pesos arbitrario para cumplir las restricciones.

    Pasos:
    1. Eliminar pesos negativos (sustituir por 0)
    2. Seleccionar los K componentes de mayor valor
    3. Normalizar esos K pesos para que sumen 1 (restricción de presupuesto)
    4. Fijar a 0 el resto

    Args:
        w_tilde (np.array): Vector de pesos arbitrario de dimensión S
        K (int): Número máximo de activos con peso > 0 (cardinalidad)

    Returns:
        np.array: Vector de pesos proyectado que cumple las restricciones
    """
    # Verificar inputs
    if K <= 0 or K > len(w_tilde):
        raise ValueError(f"K debe estar entre 1 y {len(w_tilde)}, pero se recibió {K}")

    # Paso 1: Eliminar pesos negativos (sustituir por 0)
    w_positive = np.maximum(w_tilde, 0)

    # Paso 2: Seleccionar los K componentes de mayor valor
    # Obtener los índices de los K mayores valores
    k_largest_indices = np.argpartition(w_positive, -K)[-K:]

    # Crear vector proyectado inicializado en ceros
    w_projected = np.zeros_like(w_tilde)

    # Paso 3: Normalizar los K pesos seleccionados para que sumen 1
    k_largest_weights = w_positive[k_largest_indices]

    # Verificar que al menos uno de los K pesos es positivo
    if np.sum(k_largest_weights) == 0:
        # Si todos los pesos son cero, asignar pesos uniformes a los K activos
        w_projected[k_largest_indices] = 1.0 / K
    else:
        # Normalizar para que sumen 1
        w_projected[k_largest_indices] = k_largest_weights / np.sum(k_largest_weights)

    # Paso 4: El resto ya está fijado a 0 por inicialización

    return w_projected
