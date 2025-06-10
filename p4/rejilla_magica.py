import pulp
import numpy as np

def crear_rejilla_magica(N, M, L, K):
    """
    Resuelve el problema de la rejilla mágica usando programación de enteros.
    
    Parámetros:
    - N: Dimensión de la rejilla (N x N)
    - M, L: Dimensiones de las subrejillas (M x L y L x M)
    - K: Suma constante que deben tener todas las subrejillas
    
    Retorna:
    - matriz: La rejilla mágica como array de NumPy
    - status: Estado de la solución
    """
    
    # Crear el problema de optimización
    problema = pulp.LpProblem("Rejilla_Magica", pulp.LpMinimize)
    
    # Variables de decisión: x[i][j] representa el valor en la celda (i,j)
    x = {}
    for i in range(N):
        for j in range(N):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Integer')
    
    # Función objetivo: minimizar la suma total (para encontrar cualquier solución válida)
    problema += pulp.lpSum([x[i, j] for i in range(N) for j in range(N)])
    
    # Restricciones para subrejillas M x L
    print(f"Agregando restricciones para subrejillas {M} x {L}...")
    for i in range(N - M + 1):
        for j in range(N - L + 1):
            suma_subrejilla = pulp.lpSum([x[i + di, j + dj] 
                                        for di in range(M) 
                                        for dj in range(L)])
            problema += suma_subrejilla == K, f"Restriccion_MxL_{i}_{j}"
    
    # Restricciones para subrejillas L x M
    print(f"Agregando restricciones para subrejillas {L} x {M}...")
    for i in range(N - L + 1):
        for j in range(N - M + 1):
            suma_subrejilla = pulp.lpSum([x[i + di, j + dj] 
                                        for di in range(L) 
                                        for dj in range(M)])
            problema += suma_subrejilla == K, f"Restriccion_LxM_{i}_{j}"
    
    # Resolver el problema
    print("Resolviendo el problema...")
    problema.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Verificar el estado de la solución
    status = pulp.LpStatus[problema.status]
    print(f"Estado de la solución: {status}")
    
    if problema.status == pulp.LpStatusOptimal:
        # Extraer la solución
        matriz = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                matriz[i, j] = int(x[i, j].varValue)
        
        return matriz, status
    else:
        return None, status

def verificar_solucion(matriz, N, M, L, K):
    """
    Verifica que la solución cumple con todas las restricciones.
    """
    print("\n=== VERIFICACIÓN DE LA SOLUCIÓN ===")
    
    # Verificar subrejillas M x L
    print(f"\nVerificando subrejillas {M} x {L}:")
    todas_correctas_ML = True
    for i in range(N - M + 1):
        for j in range(N - L + 1):
            suma = np.sum(matriz[i:i+M, j:j+L])
            print(f"Subrejilla [{i}:{i+M}, {j}:{j+L}] = {suma}", end="")
            if suma == K:
                print(" ✓")
            else:
                print(" ✗")
                todas_correctas_ML = False
    
    # Verificar subrejillas L x M
    print(f"\nVerificando subrejillas {L} x {M}:")
    todas_correctas_LM = True
    for i in range(N - L + 1):
        for j in range(N - M + 1):
            suma = np.sum(matriz[i:i+L, j:j+M])
            print(f"Subrejilla [{i}:{i+L}, {j}:{j+M}] = {suma}", end="")
            if suma == K:
                print(" ✓")
            else:
                print(" ✗")
                todas_correctas_LM = False
    
    if todas_correctas_ML and todas_correctas_LM:
        print("\n🎉 ¡Todas las restricciones se cumplen correctamente!")
    else:
        print("\n❌ Algunas restricciones no se cumplen.")
    
    return todas_correctas_ML and todas_correctas_LM

def mostrar_rejilla(matriz, titulo="REJILLA MÁGICA"):
    """
    Muestra la rejilla de forma elegante.
    """
    print(f"\n=== {titulo} ===")
    N = matriz.shape[0]
    
    # Calcular el ancho máximo para formato
    max_val = np.max(matriz)
    ancho = len(str(max_val)) + 1
    
    # Línea superior
    print("┌" + "─" * (ancho * N + N - 1) + "┐")
    
    # Filas de la matriz
    for i in range(N):
        print("│", end="")
        for j in range(N):
            if j > 0:
                print(" ", end="")
            print(f"{matriz[i, j]:>{ancho-1}}", end="")
        print("│")
    
    # Línea inferior
    print("└" + "─" * (ancho * N + N - 1) + "┘")

def main():
    """
    Función principal que ejecuta el programa.
    """
    print("🧩 GENERADOR DE REJILLA MÁGICA 🧩")
    print("=" * 50)
    
    # Entrada de datos
    print("\nIntroduce los parámetros del problema:")
    try:
        N = int(input("Dimensión de la rejilla (N): "))
        M = int(input("Primera dimensión de subrejilla (M): "))
        L = int(input("Segunda dimensión de subrejilla (L): "))
        K = int(input("Suma constante (K): "))
    except ValueError:
        print("❌ Error: Por favor introduce valores enteros válidos.")
        return
    
    # Validación básica
    if N <= 0 or M <= 0 or L <= 0 or K < 0:
        print("❌ Error: Todos los valores deben ser positivos (K ≥ 0).")
        return
    
    if M > N or L > N:
        print("❌ Error: Las dimensiones de subrejilla no pueden exceder N.")
        return
    
    print(f"\n📊 Configuración del problema:")
    print(f"   • Rejilla: {N} × {N}")
    print(f"   • Subrejillas: {M} × {L} y {L} × {M}")
    print(f"   • Suma objetivo: {K}")
    
    # Calcular número de restricciones
    num_restricciones_ML = (N - M + 1) * (N - L + 1)
    num_restricciones_LM = (N - L + 1) * (N - M + 1)
    total_restricciones = num_restricciones_ML + num_restricciones_LM
    
    print(f"   • Variables: {N * N}")
    print(f"   • Restricciones: {total_restricciones}")
    
    # Resolver el problema
    print(f"\n🔄 Iniciando resolución...")
    matriz, status = crear_rejilla_magica(N, M, L, K)
    
    if matriz is not None:
        print(f"\n✅ ¡Solución encontrada!")
        mostrar_rejilla(matriz)
        
        # Verificar la solución
        verificar_solucion(matriz, N, M, L, K)
        
        # Estadísticas adicionales
        print(f"\n📈 Estadísticas de la solución:")
        print(f"   • Suma total: {np.sum(matriz)}")
        print(f"   • Valor mínimo: {np.min(matriz)}")
        print(f"   • Valor máximo: {np.max(matriz)}")
        print(f"   • Promedio: {np.mean(matriz):.2f}")
        
    else:
        print(f"\n❌ No se pudo encontrar una solución.")
        print(f"   Estado: {status}")
        print("   Posibles causas:")
        print("   • Los parámetros dados no permiten una solución válida")
        print("   • El problema es demasiado restrictivo")
        print("   • Se necesita más tiempo de cálculo")

def ejemplo_demo():
    """
    Ejecuta el ejemplo dado en el enunciado.
    """
    print("🎯 EJECUTANDO EJEMPLO DE DEMOSTRACIÓN")
    print("=" * 50)
    print("Parámetros: N=6, M=3, L=2, K=7")
    
    matriz, status = crear_rejilla_magica(6, 3, 2, 7)
    
    if matriz is not None:
        mostrar_rejilla(matriz, "EJEMPLO RESUELTO")
        verificar_solucion(matriz, 6, 3, 2, 7)
    else:
        print(f"❌ No se pudo resolver el ejemplo. Estado: {status}")

if __name__ == "__main__":
    # Preguntar si quiere ejecutar el ejemplo o introducir parámetros propios
    print("¿Qué deseas hacer?")
    print("1. Ejecutar ejemplo de demostración (N=6, M=3, L=2, K=7)")
    print("2. Introducir parámetros personalizados")
    
    try:
        opcion = input("\nElige una opción (1 o 2): ").strip()
        
        if opcion == "1":
            ejemplo_demo()
        elif opcion == "2":
            main()
        else:
            print("❌ Opción no válida. Ejecutando ejemplo por defecto...")
            ejemplo_demo()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")