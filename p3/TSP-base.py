from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pulp

# Cargar el archivo .mat
data = loadmat('usborder.mat')

# Acceder a la estructura 'usbborder'
usbborder = data['usbborder']

# Extraer las variables x, y, xx, yy desde la estructura
x = usbborder['x'][0][0].flatten()
y = usbborder['y'][0][0].flatten()
xx = usbborder['xx'][0][0].flatten()
yy = usbborder['yy'][0][0].flatten()

# Crear un polígono a partir de los puntos x e y
polygon = np.column_stack((x, y))  # Combinar x e y en una matriz de puntos
path = Path(polygon)  # Crear un objeto Path para el polígono

# Configuración inicial
np.random.seed(3)
nStops = 5
stopsLon = np.zeros(nStops)
stopsLat = np.zeros(nStops)
n = 0

# Generar puntos aleatorios dentro del borde de Estados Unidos
while n < nStops:
    xp = np.random.rand() * 1.5
    yp = np.random.rand()
    if path.contains_point((xp, yp)):  # Verificar si el punto está dentro del polígono
        stopsLon[n] = xp
        stopsLat[n] = yp
        n += 1

# Dibujar el borde de Estados Unidos y los puntos de parada
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='red', label='Borde de Estados Unidos')
plt.scatter(stopsLon, stopsLat, color='blue', marker='*', s=100, label='Puntos de parada')
plt.title('Puntos de parada del viajante de comercio')
plt.legend()
plt.savefig('mapa.png')

####### CÓDIGO DE OPTIMIZACIÓN DEL TSP #######

# Calcular matriz de distancias
def calcular_distancia(lon1, lat1, lon2, lat2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)


# Crear matriz de distancias
distancias = np.zeros((nStops, nStops))
for i in range(nStops):
    for j in range(nStops):
        if i != j:
            distancias[i][j] = calcular_distancia(stopsLon[i], stopsLat[i],
                                                  stopsLon[j], stopsLat[j])

# Crear todas las posibles conexiones (aristas)
P = []  # Lista de pares de ciudades
for i in range(nStops):
    for j in range(i + 1, nStops):
        P.append([i, j])

P = np.array(P)
num_edges = len(P)

print(f"Número de ciudades: {nStops}")
print(f"Número de aristas: {num_edges}")

# Crear el problema de optimización
prob = pulp.LpProblem("TSP", pulp.LpMinimize)

# Variables de decisión: decision_vars[i] = 1 si la arista i se incluye en la ruta, 0 si no
decision_vars = pulp.LpVariable.dicts("x", range(num_edges), cat='Binary')

# Función objetivo: minimizar la suma de distancias
prob += pulp.lpSum([distancias[P[i][0]][P[i][1]] * decision_vars[i] for i in range(num_edges)])

# Restricción 1: Cada ciudad debe tener exactamente grado 2 (una entrada y una salida)
for ciudad in range(nStops):
    # Encontrar todas las aristas que conectan con esta ciudad
    aristas_ciudad = []
    for i in range(num_edges):
        if P[i][0] == ciudad or P[i][1] == ciudad:
            aristas_ciudad.append(i)

    # La suma de aristas conectadas a cada ciudad debe ser exactamente 2
    prob += pulp.lpSum([decision_vars[i] for i in aristas_ciudad]) == 2

# Restricción 2: Número total de aristas debe ser igual al número de ciudades
prob += pulp.lpSum([decision_vars[i] for i in range(num_edges)]) == nStops

# Función para encontrar subtours en una solución
def encontrar_subtours(solucion):
    """Encuentra todos los subtours en una solución dada"""
    # Crear lista de adyacencia
    adj = [[] for _ in range(nStops)]

    for i in range(num_edges):
        if solucion[i] == 1:
            adj[P[i][0]].append(P[i][1])
            adj[P[i][1]].append(P[i][0])

    visitado = [False] * nStops
    subtours = []

    for start in range(nStops):
        if not visitado[start]:
            # DFS para encontrar el componente conectado
            tour = []
            stack = [start]

            while stack:
                node = stack.pop()
                if not visitado[node]:
                    visitado[node] = True
                    tour.append(node)
                    for neighbor in adj[node]:
                        if not visitado[neighbor]:
                            stack.append(neighbor)

            if tour:
                subtours.append(tour)

    return subtours


# Resolver iterativamente eliminando subtours
max_iteraciones = 50
iteracion = 0

while iteracion < max_iteraciones:
    # Resolver el problema
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.LpStatusOptimal:
        print("No se encontró solución óptima")
        break

    # Obtener la solución
    solucion = [decision_vars[i].varValue for i in range(num_edges)]

    # Encontrar subtours
    subtours = encontrar_subtours(solucion)

    print(f"Iteración {iteracion + 1}: Encontrados {len(subtours)} subtours")

    if len(subtours) == 1:
        print("¡Solución encontrada sin subtours!")
        break

    # Agregar restricciones para eliminar subtours
    for subtour in subtours:
        if len(subtour) > 1 and len(subtour) < nStops:
            # Restricción de eliminación de subtour
            aristas_subtour = []
            for i in range(num_edges):
                if P[i][0] in subtour and P[i][1] in subtour:
                    aristas_subtour.append(i)

            if aristas_subtour:
                prob += pulp.lpSum([decision_vars[i] for i in aristas_subtour]) <= len(subtour) - 1

    iteracion += 1

# Crear x_tsp_sol como array de la solución
x_tsp_sol = np.array([decision_vars[i].varValue for i in range(num_edges)])

print(f"Distancia total óptima: {pulp.value(prob.objective):.4f}")
print(f"Solución encontrada en {iteracion} iteraciones")

####### VISUALIZACIÓN DE LA SOLUCIÓN #######

# Encontrar los segmentos de la ruta óptima
segments = np.where(x_tsp_sol == 1)[0]

# Dibujar la ruta óptima
plt.figure(figsize=(12, 8))
plt.plot(x, y, color='red', label='Borde de Estados Unidos', alpha=0.7)
plt.scatter(stopsLon, stopsLat, color='blue', marker='*', s=150, label='Puntos de parada', zorder=5)

# Agregar números a las ciudades
for i in range(nStops):
    plt.annotate(str(i), (stopsLon[i], stopsLat[i]), xytext=(5, 5),
                 textcoords='offset points', fontsize=12, fontweight='bold')

# Dibujar las conexiones de la ruta óptima
for seg in segments:
    plt.plot([stopsLon[P[seg, 0]], stopsLon[P[seg, 1]]],
             [stopsLat[P[seg, 0]], stopsLat[P[seg, 1]]],
             color='green', linewidth=3, alpha=0.8, zorder=3)

plt.title('Solución Óptima del Problema del Viajante de Comercio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('mapa_solucion.png')

# Mostrar información adicional
print("\nRuta óptima:")
# Reconstruir la ruta para mostrarla de forma ordenada
adj = [[] for _ in range(nStops)]
for seg in segments:
    adj[P[seg][0]].append(P[seg][1])
    adj[P[seg][1]].append(P[seg][0])

# Construir la ruta comenzando desde la ciudad 0
ruta = [0]
actual = 0
visitado = {0}

while len(ruta) < nStops:
    for vecino in adj[actual]:
        if vecino not in visitado:
            ruta.append(vecino)
            visitado.add(vecino)
            actual = vecino
            break

ruta.append(0)  # Volver al inicio
print(" -> ".join(map(str, ruta)))
print(f"Distancia total: {pulp.value(prob.objective):.4f}")
