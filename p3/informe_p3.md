# Informe Práctica 3: Programación de Enteros

## El Problema del Viajante de Comercio (TSP)

## 1. Introducción

El Problema del Viajante de Comercio (TSP por sus siglas en inglés, Travelling Salesman Problem) es uno de los problemas más famosos en el campo de la optimización combinatorial.

### 1.1 Definición del Problema

Dada una lista de ciudades (siendo el total de ciudades N) y las distancias entre cada par de ellas, **¿cuál es la ruta más corta posible que visita cada ciudad exactamente una vez y al finalizar regresa a la ciudad origen?**

## 2. Modelado del Problema

### 2.1 Variables de Decisión

Para cada arista posible entre ciudades, se define una variable binaria:

$$x_i = \begin{cases}
1 & \text{si la arista } i \text{ se incluye en la ruta óptima} \\
0 & \text{en caso contrario}
\end{cases}$$

### 2.2 Función Objetivo

Minimizar la distancia total recorrida:

$$\min \sum_{i} d_i \cdot x_i$$

donde $d_i$ es la distancia euclidiana de la arista $i$.

### 2.3 Restricciones

**Restricción de grado:** Cada ciudad debe tener exactamente grado 2:
$$\sum_{i \in \delta(j)} x_i = 2, \quad \forall j$$

**Restricción de número de aristas:** Total de aristas igual al número de ciudades:
$$\sum_{i} x_i = N$$

**Eliminación de subtours:** Se agregan restricciones dinámicamente para evitar ciclos parciales.

### 2.4 Cálculo de Distancias

La distancia euclidiana entre ciudades se calcula como:

$$d_{ij} = \sqrt{(\text{lon}_i - \text{lon}_j)^2 + (\text{lat}_i - \text{lat}_j)^2}$$

## 3. Implementación

### 3.1 Algoritmo de Resolución

1. **Generación de puntos**: Crear puntos aleatorios dentro del territorio de Estados Unidos
2. **Cálculo de distancias**: Matriz de distancias euclidianas entre todas las ciudades
3. **Modelado**: Definir variables, función objetivo y restricciones en PuLP
4. **Resolución iterativa**: Eliminar subtours hasta obtener una solución válida
5. **Visualización**: Mostrar la ruta óptima en el mapa

### 3.2 Detección de Subtours

Se utiliza búsqueda en profundidad (DFS) para identificar componentes conexas y eliminar subtours mediante restricciones adicionales.

## 4. Resultados

### 4.1 Configuración del Experimento

- **Número de ciudades**: 5
- **Número de aristas posibles**: 10
- **Semilla aleatoria**: 3 (para reproducibilidad)
- **Área**: Territorio continental de Estados Unidos

### 4.2 Solución Obtenida

```
Número de ciudades: 5
Número de aristas: 10
Iteración 1: Encontrados 1 subtours
¡Solución encontrada sin subtours!
Distancia total óptima: 2.9785
Solución encontrada en 0 iteraciones
```

**Ruta óptima:**
$$0 \rightarrow 1 \rightarrow 3 \rightarrow 4 \rightarrow 2 \rightarrow 0$$

**Distancia total:**
$$d^* = 2.9785 \text{ unidades}$$

## 5. Visualización

La solución óptima se muestra en el siguiente mapa:

![Solución óptima](mapa_solucion.png)
