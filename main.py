# Модель: Метод хорд та метод Якобі (5 семестр)
# Автор: Кузьменко Костянтин, група АІ-233

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
#   ЗАВДАННЯ 1 МЕТОД ХОРД
# -------------------------------

def f(x):
    return x**4 - 18*x**2 + 6


coefs = [1, 0, -18, 0, 6]
roots = np.roots(coefs)
real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]

print("Реальні корені рівняння x^4 – 18x^2 + 6 = 0:")
for r in real_roots:
    print(f"{r:.6f}")

xs = np.linspace(-10, 10, 4001)
sign_intervals = []
for i in range(len(xs)-1):
    if f(xs[i]) * f(xs[i+1]) < 0:
        sign_intervals.append((xs[i], xs[i+1]))

print("\nІнтервали, де є корені:")
for a, b in sign_intervals:
    print(f"{a:.3f}  {b:.3f}")

def secant(func, x0, x1, eps=0.01, max_iter=100):
    for k in range(max_iter):
        f0, f1 = func(x0), func(x1)
        if abs(f1 - f0) < 1e-12:
            break
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        if abs(x2 - x1) < eps:
            return x2, k+1
        x0, x1 = x1, x2
    return x1, max_iter

a, b = sign_intervals[0]
root_s, iters_s = secant(f, a, b, eps=0.01)

print(f"\nУточнений корінь методом хорд: {root_s:.6f} (ітерацій: {iters_s})")

plt.figure(figsize=(8,4))
plt.axhline(0)
plt.plot(xs, f(xs), label="f(x)")
for r in real_roots:
    plt.plot(r, 0, 'ro')
plt.title("Графік функції x^4 – 18x^2 + 6")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
#   ЗАВДАННЯ 2 — МЕТОД ЯКОБІ
# -------------------------------

A = np.array([
    [3, 1, 0, 0],
    [1, 4, -1, 0],
    [0, -1, 5, 1],
    [0, 0, 1, 2]
], float)

b = np.array([5, 3, 12, 6], float)

def jacobi(A, b, eps=0.01, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x, np.inf) < eps:
            return x_new, k+1
        x = x_new
    return x, max_iter

xj, it_j = jacobi(A, b, eps=0.01)

print("\nРозв’язок методом Якобі:")
for i in range(4):
    print(f"x{i+1} = {xj[i]:.6f}")
print(f"Ітерацій: {it_j}")

x_exact = np.linalg.solve(A, b)
print("\nТочне рішення:")
print(x_exact)

print("\nПохибка:")
print(np.abs(x_exact - xj))
