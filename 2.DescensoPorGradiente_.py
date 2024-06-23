import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

# Generar una matriz aleatoria A de 5x100 y un vector b en R^5
np.random.seed(0)
A = np.random.randn(5, 100)
b = np.random.randn(5)

# Definir la función de costo F(x)
def F(x):
    return np.transpose(A @ x - b) @ (A @ x - b)

# Definir la función de costo F2(x) con regularización L2
sigma_max = np.linalg.svd(A, compute_uv=False)[0]
delta2 = 10**(-2) * sigma_max

def F2(x, delta2):
    return F(x) + delta2 * np.linalg.norm(x)**2

# Calcular la matriz Hessiana de F(x)
H = 2 * np.transpose(A) @ A

# Calcular el autovalor más grande del Hessiano
eigvals = np.linalg.eigvals(H)
eigvals = eigvals[np.isreal(eigvals)]  # Filtrar valores complejos
lambda_max = np.real(eigvals).max()    # Obtener el máximo valor real

#Definir un step la mitad del original
#step = (1/lambda_max) * 0.5

# Definir el paso de gradiente descendente
step = 1 / lambda_max
#0.004126283950896481

#Definir un step 100% mas grande que el original
#step =(1/lambda_max)*2
#0.008252567901792962

# Función de gradiente descendente para F(x)
def grad_F(x):
    return 2 * np.transpose(A) @ (A @ x - b)

# Función de gradiente descendente para F2(x)
def gradient_descent(initial_x, step, iterations):
    x = initial_x
    trajectory = [x]
    for _ in range(iterations):
        x = x - step * grad_F(x)
        trajectory.append(x)
    return np.array(trajectory)

def gradient_descent_F2(initial_x, step, iterations, delta2):
    x = initial_x
    trajectory = [x]
    for _ in range(iterations):
        gradient = grad_F(x) + 2 * delta2 * x  # Gradiente de F2(x)
        x = x - step * gradient
        trajectory.append(x)
    return np.array(trajectory)

# Condición inicial y número de iteraciones
initial_x = np.random.randn(100)
iterations = 1000

# Ejecutar gradiente descendente para F(x)
trajectory = gradient_descent(initial_x, step, iterations)

# Ejecutar gradiente descendente para F2(x)
trajectory_F2 = gradient_descent_F2(initial_x, step, iterations, delta2)

# Evaluar F(x) y F2(x) a lo largo de las trayectorias
F_values = np.array([F(x) for x in trajectory])
F2_values = np.array([F2(x, delta2) for x in trajectory_F2])

# Solución del problema Ax = b usando SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)
x_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

# Evaluar F(x) y F2(x) en la solución de SVD
F_svd = F(x_svd)
F2_svd = F2(x_svd, delta2)

# Graficar F(x) y F2(x)
plt.figure(figsize=(12, 6))

plt.plot(F_values, label='F(x)')
plt.plot(F2_values, label='F2(x)')
# plt.axhline(y=F_svd, color='r', linestyle='--', label='Solución SVD en F(x)')
# plt.axhline(y=F2_svd, color='g', linestyle='--', label='Solución SVD en F2(x)')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función de costo')
plt.title('Descenso por gradiente de F(x) y F2(x)')
plt.legend()
plt.grid(True)
plt.show()

final_solution = trajectory[-1]
final_solution_F2 = trajectory_F2[-1]

# Diferencias contra la solución SVD
diff_without_reg = np.linalg.norm(final_solution - x_svd)
diff_with_reg = np.linalg.norm(final_solution_F2 - x_svd)

print("Diferencia entre la solución sin regularización y la solución SVD:", diff_without_reg)
print("Diferencia entre la solución con regularización y la solución SVD:", diff_with_reg)

# Diferencia analítica

numerical_diff_without_reg =  np.linalg.norm(A@final_solution-b)
numerical_diff_with_reg =  np.linalg.norm(A@final_solution_F2-b)

print("Diferencia entre la solución sin regularización y la ground truth:", numerical_diff_without_reg)
print("Diferencia entre la solución con regularización y la ground truth:", numerical_diff_with_reg)

# Diferentes valores de delta2
deltas = [10**(-5) * sigma_max, 10**(-2) * sigma_max, 10**(-1) * sigma_max]
print(deltas)
results = []

for delta2 in deltas:
    trajectory = gradient_descent_F2(initial_x, step, iterations, delta2)
    F2_values = np.array([F2(x, delta2) for x in trajectory])
    
    # Evaluar F(x) y F2(x) en la solución de SVD
    F_svd = F(x_svd)
    F2_svd = F2(x_svd, delta2)
    
    final_solution_F2 = trajectory[-1]
    
    # Diferencias contra la solución SVD
    diff_with_reg = np.linalg.norm(final_solution_F2 - x_svd)
    numerical_diff_with_reg =  np.linalg.norm(A@final_solution_F2 - b)
    
    results.append({
        "delta2": delta2,
        "F2_values": F2_values,
        "F2_svd": F2_svd,
        "diff_with_reg": diff_with_reg,
        "numerical_diff_with_reg": numerical_diff_with_reg
    })

# Graficar F2(x) para diferentes valores de delta2
plt.figure(figsize=(12, 6))

for result in results:
    plt.plot(result["F2_values"], label=f'F2(x) con delta2={result["delta2"]}')

plt.yscale('log')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función de costo')
plt.title('Descenso por gradiente de F2(x) con diferentes valores de delta2')
plt.legend()
plt.grid(True)
plt.show()

table_data = [
    [f"{result['delta2']}", result["diff_with_reg"], result["numerical_diff_with_reg"]]
    for result in results
]

print(table_data)
