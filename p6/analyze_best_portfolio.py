import pandas as pd
from data_preparation import load_data, get_sp100_tickers
from cma_optimization import PortfolioOptimizer
import numpy as np

# Cargar datos
_, _, mu, sigma = load_data()
tickers = get_sp100_tickers()

print('🏆 ANÁLISIS DE LA CARTERA λ=0.1 (MEJOR EN VALIDACIÓN):')
print('='*60)

# Analizar composición de cartera λ=0.1 (la mejor en validación)
optimizer = PortfolioOptimizer(mu.values, sigma.values, 0.1, K=5)
result = optimizer.optimize_with_projection(max_evaluations=2000, verbose=False)
weights = result['best_weights']

# Mostrar composición
print('\n📊 COMPOSICIÓN DE LA CARTERA:')
print('-'*40)
active_weights = [(tickers[i], w) for i, w in enumerate(weights) if w > 0.001]
active_weights.sort(key=lambda x: x[1], reverse=True)

for ticker, weight in active_weights:
    print(f'{ticker}: {weight:.3f} ({weight*100:.1f}%)')

print(f'\nCardinalidad efectiva: {len(active_weights)}')
print(f'Suma total: {sum(w for _, w in active_weights):.6f}')

# Analizar características de los activos seleccionados
print('\n📈 CARACTERÍSTICAS DE ACTIVOS SELECCIONADOS:')
print('-'*50)
for ticker, weight in active_weights:
    idx = tickers.index(ticker)
    ret_annual = mu.iloc[idx] * 252
    vol_annual = np.sqrt(sigma.iloc[idx, idx]) * np.sqrt(252)
    print(f'{ticker}: Ret={ret_annual:.3f} ({ret_annual*100:.1f}%), Vol={vol_annual:.3f} ({vol_annual*100:.1f}%), Peso={weight*100:.1f}%')

# Comparar con otras estrategias
print('\n⚖️  COMPARACIÓN CON OTRAS ESTRATEGIAS:')
print('-'*50)
lambdas = [0.5, 1.0, 2.0, 5.0]
for lam in lambdas:
    optimizer = PortfolioOptimizer(mu.values, sigma.values, lam, K=5)
    result = optimizer.optimize_with_projection(max_evaluations=1000, verbose=False)
    weights_other = result['best_weights']
    active_other = [(tickers[i], w) for i, w in enumerate(weights_other) if w > 0.001]
    active_other.sort(key=lambda x: x[1], reverse=True)

    print(f'λ={lam}: {", ".join([f"{t}({w*100:.1f}%)" for t, w in active_other[:3]])}...')

print('\n🎯 CONCLUSIÓN:')
print('='*60)
print('λ=0.1 selecciona activos con alta rentabilidad esperada')
print('y está dispuesta a asumir más volatilidad individual')
print('para obtener mejor rendimiento conjunto.')
