# Práctica 4: Resolución de la Rejilla Mágica mediante Programación de Enteros

## 1. Descripción del Problema

La **Rejilla Mágica** es un puzzle similar al Sudoku donde se debe rellenar una rejilla N×N con números enteros no negativos, cumpliendo la siguiente regla:

> Todas las subrejillas de tamaño M×L y L×M deben sumar exactamente un valor constante K.

### Parámetros del Problema

- **N**: Dimensión de la rejilla principal (N×N)
- **M, L**: Dimensiones de las subrejillas (M×L y L×M)
- **K**: Suma objetivo constante para todas las subrejillas

## 2. Modelado Matemático

### 2.1 Variables de Decisión

Se define una variable entera para cada celda de la rejilla:

$$x_{i,j} \in \mathbb{Z}_{\geq 0}, \quad \text{para } i,j = 0,1,\ldots,N-1$$

Donde $x_{i,j}$ representa el valor en la posición $(i,j)$ de la rejilla.

### 2.2 Función Objetivo

Para encontrar cualquier solución factible, se minimiza la suma total:

$$\min \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} x_{i,j}$$

### 2.3 Restricciones

**Restricciones para subrejillas M×L:**

$$\sum_{di=0}^{M-1} \sum_{dj=0}^{L-1} x_{i+di, j+dj} = K$$

Para todas las posiciones válidas: $i = 0,1,\ldots,N-M$ y $j = 0,1,\ldots,N-L$

**Restricciones para subrejillas L×M:**

$$\sum_{di=0}^{L-1} \sum_{dj=0}^{M-1} x_{i+di, j+dj} = K$$

Para todas las posiciones válidas: $i = 0,1,\ldots,N-L$ y $j = 0,1,\ldots,N-M$

## 3. Algoritmo de resolución

El solver CBC utiliza el método **Branch-and-Bound**:

1. **Relajación lineal**: Se resuelve primero sin restricciones de integridad
2. **Ramificación**: Si la solución no es entera, se divide en subproblemas
3. **Poda**: Se eliminan ramas que no pueden mejorar la solución actual
4. **Convergencia**: Se encuentra la solución óptima entera

## 4. Caso de Estudio: Ejemplo de Demostración

### 4.1 Parámetros

- **N = 6**: Rejilla 6×6
- **M = 3, L = 2**: Subrejillas 3×2 y 2×3
- **K = 7**: Suma objetivo

### 4.2 Complejidad del Problema

- **Variables**: $N^2 = 36$ (una por cada celda)
- **Restricciones**: 
$$R = (N-M+1)(N-L+1) + (N-L+1)(N-M+1) = 40$$
  - $(N-M+1)(N-L+1) = 4 \times 5 = 20$ restricciones para subrejillas 3×2
  - $(N-L+1)(N-M+1) = 5 \times 4 = 20$ restricciones para subrejillas 2×3

### 4.3 Solución Encontrada

```
┌─────────────────┐
│1 2 1 0 3 0│
│2 1 0 3 0 1│
│1 0 3 0 1 2│
│0 3 0 1 2 1│
│3 0 1 2 1 0│
│0 1 2 1 0 3│
└─────────────────┘
```

## 5. Análisis de Resultados

### 5.1 Eficiencia del Solver

- **Tiempo de resolución**: 0.01 segundos
- **Nodos explorados**: 0 (solución encontrada inmediatamente)
- **Iteraciones**: 0
- **Valor objetivo**: $\sum_{i,j} x_{i,j} = 42$

### 5.2 Verificación de Restricciones

✅ **Todas las 40 restricciones cumplidas:**

- 20 subrejillas 3×2 suman exactamente $K = 7$
- 20 subrejillas 2×3 suman exactamente $K = 7$

### 5.3 Estadísticas de la Solución

- **Suma total**: $\sum_{i,j} x_{i,j} = 42$
- **Valor mínimo**: $\min_{i,j} x_{i,j} = 0$
- **Valor máximo**: $\max_{i,j} x_{i,j} = 3$
- **Promedio por celda**: $\bar{x} = \frac{42}{36} = 1.17$
