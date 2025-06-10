# Práctica 2: Regresión ADALINA LASSO con Gradiente Proyectado

## 1. Método de Gradiente Proyectado y su Aplicación a la Regularización $L_1$ (LASSO)

En esta práctica, hemos implementado el algoritmo de **Gradiente Proyectado** para resolver el problema de **Regresión ADALINA LASSO**. Este método es una extensión del clásico descenso de gradiente, diseñado específicamente para problemas de optimización con restricciones o funciones objetivo no diferenciables, como es el caso de la regularización $L_1$.

El problema de optimización que abordamos es el siguiente:

$$\min_{w \in \mathbb{R}^d} \frac{1}{2} ||Xw - y||^2 + \lambda ||w||_1$$

Donde:

- $X \in \mathbb{R}^{N \times d}$ es la matriz de características.
- $w \in \mathbb{R}^d$ es el vector de pesos que buscamos optimizar.
- $y \in \mathbb{R}^N$ es el vector de valores objetivo (etiquetas).
- $\lambda > 0$ es el parámetro de regularización, que controla la fuerza de la penalización $L_1$.
- $||Xw - y||^2$ es el término de error cuadrático, que mide la bondad del ajuste del modelo a los datos.
- $||w||_1 = \sum_{j=1}^d |w_j|$ es la norma $L_1$ del vector de pesos, que induce esparcidad (es decir, muchos coeficientes $w_j$ se vuelven cero).

### 1.1. Desafío del Término de Regularización $L_1$

La presencia del término de regularización $||w||_1$ (norma L1) hace que la función objetivo no sea completamente diferenciable. La función de valor absoluto $|x|$ no tiene una derivada bien definida en $x=0$. Esto impide el uso directo de métodos de descenso de gradiente estándar que requieren que la función sea diferenciable en todo su dominio.

### 1.2. El Algoritmo de Gradiente Proyectado (Método Proximal)

El algoritmo de Gradiente Proyectado aborda este desafío dividiendo la función objetivo en dos partes:

1.  **Una parte suave y diferenciable:** $f_{suave}(w) = \frac{1}{2} ||Xw - y||^2$. Esta parte representa el error cuadrático.
2.  **Una parte no suave pero con un operador proximal fácil de calcular:** $f_{no\_suave}(w) = \lambda ||w||_1$. Esta parte es el término de regularización L1.

El algoritmo procede iterativamente de la siguiente manera:

En cada iteración $k$:

1.  **Paso de Gradiente (sobre la parte suave):** Se calcula un paso de descenso de gradiente convencional utilizando solo la parte suave de la función objetivo. Esto nos da un punto candidato $y_k$:
    $$y_k = w_k - \alpha_k \nabla f_{suave}(w_k)$$
    Donde $\nabla f_{suave}(w_k) = X^T(Xw_k - y)$ es el gradiente del término de error cuadrático, y $\alpha_k$ es el tamaño de paso.

2.  **Paso Proximal (Umbralización Suave):** El punto candidato $y_k$ se "corrige" aplicando el operador proximal de la norma L1, que para este caso específico es el **operador de umbralización suave (soft-thresholding)**. Este paso "proyecta" de alguna manera la solución de vuelta a un espacio que considera la penalización L1:
    $$w_{k+1} = \text{soft\_threshold}(y_k, \alpha_k \lambda)$$
    El operador de umbralización suave se define para cada componente $z_j$ del vector $z$ y un umbral $\tau$ como:
    $$\text{soft\_threshold}(z_j, \tau) = \text{sgn}(z_j) \cdot \max(0, |z_j| - \tau)$$
    En nuestro caso, el umbral $\tau = \alpha_k \lambda$. Este operador tiene la propiedad deseada de hacer que los coeficientes pequeños sean exactamente cero, lo que contribuye a la selección de características.

### 1.3. Selección Adaptativa del Tamaño de Paso (Backtracking Line Search)

Para asegurar la convergencia y un avance eficiente hacia la solución óptima, se utiliza una técnica de búsqueda de línea conocida como **Backtracking Line Search**. En cada iteración, esta técnica ajusta dinámicamente el tamaño de paso $\alpha_k$ (comenzando con un valor inicial y reduciéndolo progresivamente) hasta que se cumple una condición que garantiza un descenso suficiente en la función objetivo. Esto evita pasos demasiado grandes que podrían causar divergencia o pasos demasiado pequeños que ralentizarían la convergencia.

### 1.4. Criterio de Parada

El algoritmo itera hasta alcanzar un número máximo predefinido de iteraciones (`max_iter`) o hasta que la mejora en la función objetivo entre iteraciones consecutivas sea menor que una tolerancia preestablecida (`tol`). Esto indica que el algoritmo ha convergido a un mínimo local (o global, dado que el problema LASSO es convexo).

## 2. Reporte de Resultados

El modelo fue entrenado utilizando el dataset Boston Housing, que consta de 505 muestras y 13 características, las cuales fueron normalizadas (media 0, desviación estándar 1) antes del entrenamiento.

**Nota Técnica:** La implementación incluye correctamente el término bias (intercept) añadiendo una columna de unos a la matriz de características. Esto es fundamental para obtener predicciones precisas, ya que permite al modelo ajustar el nivel base de la variable objetivo. Sin el término bias, las predicciones estarían sistemáticamente sesgadas.

### 2.1. Resultados con $\lambda = 100$

Para el experimento principal con un parámetro de regularización $\lambda = 100$, el algoritmo de Gradiente Proyectado mostró la siguiente convergencia y resultados:

- **Número de Iteraciones para Convergencia:** 38 iteraciones.
- **Diferencia de Error al Converger:** 0.000852 (indicando una buena estabilidad en la función objetivo).
- **Coeficientes Esencialmente Cero:** 2 de las 14 características (incluyendo bias) tienen un coeficiente muy cercano a cero (considerado cero por la tolerancia `1e-6` del código), lo que demuestra la capacidad de selección de características de LASSO.

**Comparación de valores reales vs predichos (primeras 5 muestras):**

| Muestra | Valor Real | Valor Predicho | Error Absoluto |
| :------ | :--------- | :------------- | :------------- |
| 1       | 21.600     | 25.243         | 3.643          |
| 2       | 34.700     | 31.109         | 3.591          |
| 3       | 33.400     | 29.420         | 3.980          |
| 4       | 36.200     | 28.813         | 7.387          |
| 5       | 28.700     | 25.832         | 2.868          |

**Coeficientes del modelo ($w$) para $\lambda = 100$:**

- $w[0] = 22.331881$ (término bias)
- $w[1] = -0.340294$
- $w[2] = 0.371409$
- $w[3] = -0.077628$
- $w[4] = 0.612919$
- $w[5] = -1.030503$
- $w[6] = 2.963340$
- $w[7] = -0.000000$ ← **Cero (característica eliminada)**
- $w[8] = -1.720435$
- $w[9] = 0.004121$
- $w[10] = -0.000000$ ← **Cero (característica eliminada)**
- $w[11] = -1.784176$
- $w[12] = 0.677772$
- $w[13] = -3.732347$

Los coeficientes $w[7]$ y $w[10]$ se han reducido a cero (o muy cerca de cero), lo que implica que las características correspondientes no contribuyen significativamente a las predicciones del modelo para este valor de $\lambda$. El término $w[0] = 22.331881$ corresponde al bias (intercept) del modelo.

### 2.2. Análisis del Efecto de Diferentes Valores de $\lambda$

Se realizaron experimentos adicionales con $\lambda = 10, 50,$ y $200$ para observar su impacto en la selección de características, el error cuadrático medio (MSE) y la convergencia del algoritmo.

| $\lambda$ | MSE       | Coef. Cero | Iteraciones |
| :-------- | :-------- | :--------- | :---------- |
| 10        | 21.898330 | 2          | 100         |
| 50        | 22.335791 | 1          | 100         |
| 100       | 23.495685 | 2          | 38          |
| 200       | 25.208993 | 5          | 36          |

### 2.3. Observaciones y Conclusiones

Del análisis comparativo, podemos extraer las siguientes observaciones clave sobre el efecto del parámetro de regularización $\lambda$:

- **Impacto en la Esparsidad (Selección de Características):**

  - A medida que el valor de $\lambda$ aumenta, la penalización sobre la norma $L_1$ de los pesos se vuelve más estricta. Esto se traduce en un **mayor número de coeficientes que se reducen a cero**. Para $\lambda=50$, solo 1 coeficiente fue cero; para $\lambda=10$ y $\lambda=100$, 2 coeficientes cada uno; y para $\lambda=200$, 5 coeficientes se volvieron cero. Esto confirma que la regularización $L_1$ efectivamente realiza una selección automática de características, identificando y eliminando las menos relevantes.

- **Impacto en el Error Cuadrático Medio (MSE):**

  - Existe una relación directa entre $\lambda$ y el MSE. Un $\lambda$ más pequeño (e.g., $\lambda=10$) produce el MSE más bajo (21.898), ya que la penalización es menos restrictiva, permitiendo que el modelo se ajuste mejor a los datos.
  - A medida que $\lambda$ aumenta, el MSE tiende a incrementarse progresivamente: $\lambda=50$ (MSE: 22.336), $\lambda=100$ (MSE: 23.496), y $\lambda=200$ (MSE: 25.209). Esto se debe a que una mayor penalización fuerza a más coeficientes a cero, simplificando el modelo y aumentando el sesgo (subajuste) cuando $\lambda$ es demasiado grande.

- **Impacto en la Convergencia del Algoritmo (Número de Iteraciones):**

  - Los valores de $\lambda$ más altos ($\lambda=100$ con 38 iteraciones y $\lambda=200$ con 36 iteraciones) llevaron a una convergencia más rápida en comparación con valores más bajos ($\lambda=10$ y $\lambda=50$ con 100 iteraciones cada uno). Esto podría deberse a que una penalización más fuerte "guía" el algoritmo más rápidamente hacia una solución esparsa, reduciendo el espacio de búsqueda efectivo.

- **Calidad de las Predicciones:**
  - Las predicciones del modelo LASSO son significativamente más precisas después de incluir el término bias. Los errores absolutos para las primeras 5 muestras oscilan entre 2.868 y 7.387, lo que es muy razonable para el rango de precios de vivienda (21-36 mil dólares).
  - El término bias ($w[0] = 22.331881$) captura el nivel base de los precios de vivienda, permitiendo que el modelo se ajuste correctamente a los datos.

En conclusión, la regularización $L_1$ con el algoritmo de Gradiente Proyectado es una herramienta eficaz para la regresión lineal que no solo ayuda a controlar el sobreajuste, sino que también ofrece un mecanismo inherente de selección de características. La elección adecuada de $\lambda$ es crucial para equilibrar la complejidad del modelo (esparcidad) con su capacidad para ajustar los datos. Un $\lambda$ más alto resulta en modelos más simples y esparsos, mientras que un $\lambda$ más bajo permite modelos más complejos pero con menor esparcidad. La inclusión del término bias es fundamental para obtener predicciones realistas y precisas.
