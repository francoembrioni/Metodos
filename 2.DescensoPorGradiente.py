import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
A = np.random.randn(5, 100)
b = np.random.randn(5)

def F(x):
    return np.transpose(A @ x - b) @ (A @ x - b)

sigma_max = np.linalg.svd(A, compute_uv=False)[0]
delta2 = 10**(-2) * sigma_max

def F2(x):
    return F(x) + delta2 * np.linalg.norm(x)**2

H = 2 * np.transpose(A) @ A

eigvals = np.linalg.eigvals(H)
eigvals = eigvals[np.isreal(eigvals)]  
lambda_max = np.real(eigvals).max()    

#Definir un step la mitad del original
#step = (1/lambda_max) * 0.5


# Definir el paso de gradiente descendente
step = 1 / lambda_max
#0.004126283950896481

#Definir un step 100% mas grande que el original
#step =(1/lambda_max)*2
#0.008252567901792962

def grad_F(x):
    return 2 * np.transpose(A) @ (A @ x - b)

def gradient_descent(initial_x, step, iterations):
    x = initial_x
    trajectory = [x]
    for _ in range(iterations):
        x = x - step * grad_F(x)
        trajectory.append(x)
    return np.array(trajectory)

def gradient_descent_F2(initial_x, step, iterations):
    x = initial_x
    trajectory = [x]
    for _ in range(iterations):
        gradient = grad_F(x) + 2 * delta2 * x  
        x = x - step * gradient
        trajectory.append(x)
    return np.array(trajectory)

initial_x = np.random.randn(100)
iterations = 1000

trajectory = gradient_descent(initial_x, step, iterations)

trajectory_F2 = gradient_descent_F2(initial_x, step, iterations)

F_values = np.array([F(x) for x in trajectory])
F2_values = np.array([F2(x) for x in trajectory_F2])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
x_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

F_svd = F(x_svd)
F2_svd = F2(x_svd)

plt.figure(figsize=(12, 6))

plt.plot(F_values, label='F(x)')
plt.plot(F2_values, label='F2(x)')
plt.axhline(y=F_svd, color='r', linestyle='--', label='Solución SVD en F(x)')
plt.axhline(y=F2_svd, color='g', linestyle='--', label='Solución SVD en F2(x)')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la función de costo')
plt.title('Descenso por gradiente de F(x) y F2(x)')
plt.legend()
plt.grid(True)
plt.show()
