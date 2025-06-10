# Práctica 1: Métodos de optimización para ADALINA

## 1. Desarrollo de las ecuaciones

El objetivo del modelo ADALINA es encontrar el vector de pesos `w` que minimiza la función de error cuadrático. Esta función de coste, también llamada `L(w)`, se define como la norma euclidiana al cuadrado del error:

$$L(w) = ||Xw - y||^2$$

Donde:

- `X` es la matriz de características (datos de entrada).
- `w` es el vector de pesos que queremos optimizar.
- `y` es el vector de valores reales (salidas deseadas).
- `Xw` da como resultado `ŷ`, el vector de predicciones del modelo.

### 1.1. Descenso de gradiente

El método de descenso de gradiente es un algoritmo iterativo que actualiza los pesos en cada paso para minimizar esta función de coste. La fórmula general de actualización es:

$$w_{k+1} = w_k + \alpha_k d_k$$

Para el descenso de gradiente, la dirección de búsqueda `dₖ` se elige en el sentido opuesto al gradiente de la función de coste, ya que el gradiente apunta en la dirección de máximo crecimiento. Por lo tanto:

$$d_k = -\nabla L(w_k)$$

Para implementar el algoritmo, necesitamos dos componentes clave:

1.  El gradiente de la función de coste, `∇L(w)`.
2.  El tamaño de paso óptimo, `αₖ`, que para esta práctica se debe calcular de forma exacta.

---

#### 1.1.1. Cálculo del Gradiente (∇L(w))

Primero, expandimos la función de coste `L(w)` para poder derivarla más fácilmente. La norma al cuadrado `||v||²` es igual a `vᵀv`.

$$L(w) = (Xw - y)^T (Xw - y)$$

Aplicando las propiedades de la transpuesta `(A-B)ᵀ = Aᵀ - Bᵀ` y `(AB)ᵀ = BᵀAᵀ`:

$$L(w) = ((Xw)^T - y^T) (Xw - y)$$

$$L(w) = (w^T X^T - y^T) (Xw - y)$$

Expandiendo los términos:

$$L(w) = w^T X^T Xw - w^T X^T y - y^T Xw + y^T y$$

Dado que `yᵀXw` es un escalar, es igual a su transpuesta: `(yᵀXw)ᵀ = wᵀXᵀy`. Por lo tanto, los dos términos centrales son iguales y podemos combinarlos:

$$L(w) = w^T X^T Xw - 2y^T Xw + y^T y$$

Ahora, calculamos el gradiente (la derivada con respecto al vector `w`). Usando las reglas de derivación matricial:

- `∇(wᵀAw) = 2Aw` (para `A` simétrica, y `XᵀX` lo es)
- `∇(bᵀw) = b`
- `∇(constante) = 0`

Derivamos `L(w)` con respecto a `w`:

$$\nabla L(w) = \nabla_w (w^T (X^T X) w - 2y^T Xw + y^T y)$$
$$\nabla L(w) = 2(X^T X)w - 2(X^T)^T y + 0$$
$$\nabla L(w) = 2X^T Xw - 2X^T y$$

Finalmente, el gradiente de la función de coste es:

$$\boxed{\nabla L(w) = 2(X^T Xw - X^T y)}$$

Este es el gradiente que se utilizará para determinar la dirección de descenso `dₖ = -∇L(wₖ)`.

---

#### 1.1.2. Desarrollo del Tamaño de Paso Óptimo (αₖ)

En cada paso `k`, debemos encontrar el `α` que minimiza la función a lo largo de la dirección de descenso `dₖ`. Formalmente:

$$\alpha_k = \arg\min_{\alpha \ge 0} L(w_k + \alpha d_k)$$

La función de coste de ADALINA es una función cuadrática, similar al ejemplo desarrollado en los apuntes. Podemos seguir un procedimiento análogo para encontrar `αₖ` de forma analítica.

La función objetivo `L(w)` se puede expresar como `f(x) = bᵀx + (1/2)xᵀAx` si identificamos `x` con `w`, `A` con `2XᵀX` y `b` con `-2Xᵀy`. Para este tipo de funciones, la teoría nos da una fórmula directa para el `αₖ` óptimo.

$$\alpha_k = \frac{d_k^T d_k}{d_k^T A d_k}$$

Sustituyendo en esta fórmula:

- `A = 2XᵀX` (el término que multiplica a `wᵀ(...)w` en `L(w)`, ajustado por el factor 1/2 de la fórmula genérica).

Obtenemos la expresión para `αₖ`:

$$\boxed{\alpha_k = \frac{d_k^T d_k}{2 d_k^T X^T X d_k}}$$

### 1.2. Método de Newton

El método de Newton es un algoritmo de optimización de segundo orden que utiliza la curvatura de la función para converger más rápidamente hacia el mínimo. En lugar de simplemente seguir la dirección opuesta al gradiente, Newton escala esta dirección utilizando la inversa de la matriz Hessiana (la matriz de las segundas derivadas).

La idea es aproximar la función de coste $L(w)$ en la vecindad del punto actual $w_k$ mediante su expansión de Taylor de segundo orden, que es una función cuadrática. Luego, se calcula el mínimo exacto de esta aproximación cuadrática y se toma ese punto como la siguiente iteración, $w_{k+1}$.

La fórmula de actualización general del método de Newton es:

$$w_{k+1} = w_k - [H(w_k)]^{-1} \nabla L(w_k)$$

donde $H(w_k)$ es la matriz Hessiana de la función de coste evaluada en $w_k$. Para implementar este método, necesitamos calcular el gradiente (ya obtenido) y la matriz Hessiana.

---

#### 1.2.1. Cálculo de la Matriz Hessiana (H(w))

La matriz Hessiana, $H(w)$ o $\nabla^2 L(w)$, se obtiene derivando el gradiente $\nabla L(w)$ con respecto al vector $w$.

Partimos del gradiente que calculamos anteriormente:

$$\nabla L(w) = 2X^T Xw - 2X^T y$$

Ahora, derivamos esta expresión con respecto a $w$:

$$H(w) = \nabla_w (2X^T Xw - 2X^T y)$$

Aplicando las reglas de derivación matricial, la derivada del término $2X^T Xw$ con respecto a $w$ es $2X^T X$. El segundo término, $-2X^T y$, no depende de $w$, por lo que su derivada es cero.

Por lo tanto, la matriz Hessiana es:

$$\boxed{H(w) = 2X^T X}$$

Una observación clave es que la matriz Hessiana **es constante** y no depende de $w$. Esto se debe a que nuestra función de coste original $L(w)$ es una función cuadrática.

---

#### 1.2.2. Ecuación de Actualización de Newton

Ahora sustituimos el gradiente y la matriz Hessiana en la fórmula de actualización de Newton:

$$w_{k+1} = w_k - (2X^T X)^{-1} (2X^T Xw_k - 2X^T y)$$

Podemos sacar el factor constante 2 de ambos términos:

$$w_{k+1} = w_k - (2X^T X)^{-1} 2(X^T Xw_k - X^T y)$$

$$w_{k+1} = w_k - \left( \frac{1}{2} (X^T X)^{-1} \right) 2(X^T Xw_k - X^T y)$$

$$w_{k+1} = w_k - (X^T X)^{-1} (X^T Xw_k - X^T y)$$

Expandiendo la expresión:

$$w_{k+1} = w_k - (X^T X)^{-1} (X^T X) w_k + (X^T X)^{-1} X^T y$$

Dado que $(X^T X)^{-1} (X^T X)$ es la matriz identidad $I$:

$$w_{k+1} = w_k - I \cdot w_k + (X^T X)^{-1} X^T y$$

$$w_{k+1} = w_k - w_k + (X^T X)^{-1} X^T y$$

Esto nos lleva a la solución final:

$$\boxed{w = (X^T X)^{-1} X^T y}$$

Esta es la famosa **ecuación normal**, la solución analítica para el problema de mínimos cuadrados lineales.

Como la función de coste de ADALINA es cuadrática, la aproximación de segundo orden de Newton es exacta. Esto significa que el método de Newton encuentra el mínimo global de la función **en una sola iteración**, independientemente del punto de partida $w_0$. Por esta razón, el método no requiere un bucle iterativo ni el cálculo de un tamaño de paso, ya que salta directamente a la solución óptima.

## 2. Reporte de Resultados

A continuación, se presentan los resultados obtenidos al aplicar ambos métodos al problema de regresión lineal con el dataset de Boston Housing.

### 2.1. Comparación de Métodos ADALINA

#### Comparación de Resultados

- **Iteraciones:** Gradiente: 795, Newton: 2
- **Tiempo:** Gradiente: 0.0686s, Newton: 0.0010s
- **Eficiencia:** Newton es 397.5x más rápido en iteraciones

#### Explicación de la diferencia en iteraciones

El método de Newton requiere significativamente menos iteraciones (2 vs. 795) que el Descenso de Gradiente debido a su naturaleza de segundo orden. Mientras que el Descenso de Gradiente utiliza solo la información de la primera derivada (el gradiente) para determinar la dirección de descenso, el Método de Newton incorpora la información de la segunda derivada a través de la matriz Hessiana. La Hessiana proporciona información sobre la curvatura de la función, lo que permite al método de Newton dar pasos más grandes y en la dirección correcta, saltando directamente al mínimo para funciones cuadráticas. Como la función de coste de ADALINA es cuadrática, la aproximación de segundo orden de Newton es exacta, lo que le permite alcanzar la solución óptima en un número mínimo de iteraciones.

#### Predicciones de las Primeras 5 Muestras

| Muestra | Real   | Gradiente | Newton | Error Grad | Error Newton |
| :------ | :----- | :-------- | :----- | :--------- | :----------- |
| 1       | 21.600 | 25.053    | 25.053 | 3.453      | 3.453        |
| 2       | 34.700 | 30.591    | 30.591 | 4.109      | 4.109        |
| 3       | 33.400 | 28.640    | 28.640 | 4.760      | 4.760        |
| 4       | 36.200 | 27.971    | 27.971 | 8.229      | 8.229        |
| 5       | 28.700 | 25.297    | 25.297 | 3.403      | 3.403        |

#### Métricas de Evaluación

- **MSE Gradiente:** 21.865580
- **MSE Newton:** 21.865580
- **Diferencia en pesos:** 0.000000

Los valores de MSE son idénticos para ambos métodos, lo que indica que ambos algoritmos convergieron a la misma solución óptima (o muy cercana a ella), como se esperaba dado que la función de coste es convexa. La diferencia en pesos de 0.000000 confirma que ambos métodos encontraron el mismo conjunto de pesos óptimos.

#### Conclusión sobre la Eficiencia de Ambos Métodos

Los resultados demuestran claramente que el **Método de Newton es significativamente más eficiente** que el Descenso de Gradiente para este problema específico de regresión lineal con la función de coste cuadrática de ADALINA.

Aunque ambos métodos convergen a la misma solución y obtienen el mismo costo final (MSE), el Método de Newton lo logra en solo 2 iteraciones y un tiempo de entrenamiento de 0.0010 segundos, en contraste con las 795 iteraciones y 0.0686 segundos del Descenso de Gradiente. Esto lo convierte en un método 397.5 veces más rápido en términos de iteraciones.

La superioridad del Método de Newton en este caso particular se debe a que la función de coste es cuadrática, lo que permite que la aproximación de segundo orden de Newton sea exacta y, por lo tanto, converge al mínimo global en muy pocas iteraciones. En contraste, el Descenso de Gradiente, al ser un método de primer orden, requiere muchos más pasos pequeños para navegar por el espacio de búsqueda hasta alcanzar el mínimo.

En problemas con funciones de coste no cuadráticas, el Método de Newton podría requerir un número mayor de iteraciones y ser computacionalmente más costoso por la necesidad de calcular y invertir la matriz Hessiana en cada paso. Sin embargo, para problemas cuadráticos como el de ADALINA, es el método de elección por su rapidez y convergencia directa.

#### Gráfica comparativa de la convergencia

A continuación, se muestra una gráfica que ilustra la convergencia de ambos métodos a lo largo de las iteraciones. Esta gráfica permite visualizar cómo el Método de Newton alcanza el mínimo de la función de coste de forma casi instantánea, mientras que el Descenso de Gradiente lo hace de manera más gradual.

![Gráfica comparativa de la convergencia](comparacion_metodos.png)
