# Práctica 5: Resolución del Problema de N-Torres con Redes de Hopfield

## 1. Introducción

El problema de las N-Torres es un clásico problema de optimización combinatoria que consiste en colocar N torres en un tablero de ajedrez de tamaño N×N de tal manera que ninguna torre pueda atacar a otra. En el ajedrez, una torre puede moverse cualquier número de casillas a lo largo de una fila o columna, por lo que la restricción fundamental es que no puede haber más de una torre en la misma fila o columna.

Este informe presenta la implementación de una Red de Hopfield para resolver este problema, siguiendo el enfoque de optimización basado en minimización de energía.

## 2. Modelado del Problema

### 2.1 Variables de Decisión

El problema se modela utilizando variables binarias:

$$s_{ij} \in \{0, 1\} \quad \text{para } i, j \in \{1, 2, \ldots, N\}$$

donde:

- $s_{ij} = 1$ si hay una torre en la posición $(i,j)$ del tablero
- $s_{ij} = 0$ si la posición $(i,j)$ está vacía

### 2.2 Restricciones

El problema presenta dos tipos de restricciones:

**Restricción 1 - Filas:** Exactamente una torre por fila

$$\sum_{j=1}^{N} s_{ij} = 1, \quad \forall i \in \{1, 2, \ldots, N\}$$

**Restricción 2 - Columnas:** Exactamente una torre por columna

$$\sum_{i=1}^{N} s_{ij} = 1, \quad \forall j \in \{1, 2, \ldots, N\}$$

### 2.3 Función de Energía

Siguiendo la metodología de Redes de Hopfield, se construye una función de energía que penaliza las violaciones de restricciones:

$$E = \sum_{i=1}^{N} \mu_{1i}\left(\sum_{j=1}^{N} s_{ij} - 1\right)^2 + \sum_{j=1}^{N} \mu_{2j}\left(\sum_{i=1}^{N} s_{ij} - 1\right)^2$$

Desarrollando algebraicamente esta expresión y organizándola en la forma estándar de Hopfield:

$$E = -\frac{1}{2}\sum_{k}\sum_{j} w_{kj} s_k s_j + \sum_{k} \theta_k s_k$$

### 2.4 Configuración de la Red

Los parámetros de la red se configuran como:

**Pesos sinápticos:**

- $w_{ij,ik} = -2\mu_1$ cuando $k \neq j$ (misma fila)
- $w_{ij,rj} = -2\mu_2$ cuando $r \neq i$ (misma columna)
- $w_{ij,ij} = 0$ (sin auto-conexión)
- $w_{ij,rk} = 0$ para otros casos

**Umbrales:**

$$\theta_{ij} = -(\mu_1 + \mu_2)$$

## 3. Implementación

### 3.1 Arquitectura de la Red

- **Neuronas:** $N^2$ neuronas (una por cada casilla del tablero)
- **Conectividad:** Cada neurona se conecta con todas las neuronas de su misma fila y columna
- **Función de activación:** Función escalón binaria
- **Dinámica:** Actualización asíncrona con orden aleatorio

### 3.2 Algoritmo de Evolución

1. **Inicialización:** Se probaron dos estrategias:

   - Aleatoria: Estados binarios aleatorios
   - Heurística: Una torre por fila en posición aleatoria

2. **Evolución:** Actualización iterativa siguiendo la regla:

   $$
   s_k(t+1) = \begin{cases}
   1 & \text{si } \sum_{j} w_{kj} s_j(t) \geq \theta_k \\
   0 & \text{en caso contrario}
   \end{cases}
   $$

3. **Convergencia:** La red converge cuando no hay cambios de estado durante varias iteraciones consecutivas

## 4. Resultados Experimentales

Se realizaron experimentos con tableros de diferentes tamaños, utilizando parámetros $\mu_1 = \mu_2 = 2.0$ y 15 intentos por cada tamaño.

### 4.1 Tablero 4×4

| Métrica                        | Valor   |
| ------------------------------ | ------- |
| Intentos totales               | 15      |
| Soluciones válidas encontradas | 9 (60%) |
| Iteraciones promedio           | 11.2    |
| Convergencia                   | 100%    |
| Mejor energía                  | -16.00  |

**Solución encontrada:**

![Solución 4-Torres](hopfield_4_torres_solucion.png)

### 4.2 Tablero 5×5

| Métrica                        | Valor   |
| ------------------------------ | ------- |
| Intentos totales               | 15      |
| Soluciones válidas encontradas | 7 (47%) |
| Iteraciones promedio           | 11.4    |
| Convergencia                   | 100%    |
| Mejor energía                  | -20.00  |

**Solución encontrada:**

![Solución 5-Torres](hopfield_5_torres_solucion.png)

### 4.3 Tablero 6×6

| Métrica                        | Valor   |
| ------------------------------ | ------- |
| Intentos totales               | 15      |
| Soluciones válidas encontradas | 5 (33%) |
| Iteraciones promedio           | 11.3    |
| Convergencia                   | 100%    |
| Mejor energía                  | -24.00  |

**Solución encontrada:**

![Solución 6-Torres](hopfield_6_torres_solucion.png)

## 5. Análisis de Resultados

### 5.1 Rendimiento por Tamaño

Los resultados muestran una clara tendencia: conforme aumenta el tamaño del tablero, disminuye la probabilidad de encontrar soluciones válidas:

- **N=4:** 60% de éxito
- **N=5:** 47% de éxito
- **N=6:** 33% de éxito

Esto es esperado debido al aumento exponencial del espacio de búsqueda ($2^{N^2}$ configuraciones posibles).

### 5.2 Convergencia y Estabilidad

- **Convergencia:** Todos los intentos convergieron, demostrando la estabilidad del algoritmo
- **Iteraciones:** El número de iteraciones se mantiene consistente (~11-12) independientemente del tamaño
- **Energía:** Las soluciones válidas presentan energías más negativas, confirmando la correcta formulación

### 5.3 Estrategias de Inicialización

Ambas estrategias (aleatoria y heurística) mostraron capacidad para encontrar soluciones válidas, aunque no se observa una diferencia significativa en el rendimiento entre ambas.

### 5.4 Análisis Energético

Para una solución válida de tamaño $N$, la energía teórica mínima es:

$$E_{\text{min}} = -N \cdot (\mu_1 + \mu_2) = -N \cdot 4 = -4N$$

Los resultados experimentales confirman esta predicción:

- $N=4$: $E = -16 = -4 \times 4$ ✓
- $N=5$: $E = -20 = -4 \times 5$ ✓
- $N=6$: $E = -24 = -4 \times 6$ ✓
