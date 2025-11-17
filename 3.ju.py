# %% [md]
# # Лабораторная работа № 3

# %%
import numpy as np
import pandas as pd
import sympy as sp
import math
from typing import List

# %% [md]
# ## 1
# Используя таблицу значений $Y_1$   функции y = f(x) ,
# вычисленных в точках $X_i$
# построить интерполяционные многочлены Лагранжа и Ньютона,
# проходящие через точки {$X_i$, $Y_i$}.
# Вычислить значение погрешности интерполяции в точке $X^*$ .

# $y=e^x$

# $X_i = -2, -1, 0, 1$

# $X_i = -2, -1, 0.2, 1$

# $X^* = -0.5$


# %%
def lagrange_newton(xs: List[float], point: float, func):
    ys = [func(x) for x in xs]
    ws = [1 for _ in xs]
    for i in range(len(xs)):
        for j, x in enumerate(xs):
            if j == i:
                continue
            ws[i] *= (xs[i] - x)
    x = sp.Symbol("x")
    fws = [ys[i]/ws[i] for i in range(4)]
    lx = (fws[0]*(x-xs[1])*(x-xs[2])*(x-xs[3]) +
          fws[1]*(x-xs[0])*(x-xs[2])*(x-xs[3]) +
          fws[2]*(x-xs[0])*(x-xs[1])*(x-xs[3]) +
          fws[3]*(x-xs[0])*(x-xs[1])*(x-xs[2]))

    lprecise = sp.lambdify(x, lx)(point)
    print("Лагранж: ")
    print(f"Интерполяционное: {lprecise}, Точное: {
          func(point)}, погрешность: {abs(func(point) - lprecise)}")
    print("\t", str(lx))

    ys2 = [(ys[i]-ys[i+1])/(xs[i]-xs[i+1]) for i in range(len(ys) - 1)]
    ys3 = [(ys2[i]-ys2[i+1])/(xs[i]-xs[i+2]) for i in range(len(ys2) - 1)]
    ys4 = [(ys3[i]-ys3[i+1])/(xs[i]-xs[i+3]) for i in range(len(ys3) - 1)]

    nx = (ys[0] +
          (x - xs[0]) * ys2[0] +
          (x-xs[0])*(x-xs[1])*ys3[0] +
          (x-xs[0])*(x-xs[1])*(x-xs[2])*ys4[0])
    nprecise = sp.lambdify(x, nx)(point)
    print("Ньютон: ")
    print(f"Интерполяционное: {nprecise}, Точное: {
        func(point)}, погрешность: {abs(func(point) - nprecise)}")
    print("\t", str(nx))
    print()


lagrange_newton([-2, -1, 0, 1], -0.5, lambda x: math.exp(x))
lagrange_newton([-2, -1, 0.2, 1], -0.5, lambda x: math.exp(x))
