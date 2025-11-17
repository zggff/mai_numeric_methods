# %% [md]
# # Лабораторная работа № 3

# %%
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib import pyplot as plt
import math
from typing import List
import matrix as m

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
    lx = sp.UnevaluatedExpr((fws[0]*(x-xs[1])*(x-xs[2])*(x-xs[3]) +
                             fws[1]*(x-xs[0])*(x-xs[2])*(x-xs[3]) +
                             fws[2]*(x-xs[0])*(x-xs[1])*(x-xs[3]) +
                             fws[3]*(x-xs[0])*(x-xs[1])*(x-xs[2])))

    lprecise = sp.lambdify(x, lx)(point)
    print("Лагранж: ")
    print(f"Интерполяционное: {lprecise}, Точное: {
          func(point)}, погрешность: {abs(func(point) - lprecise)}")
    print("\t", str(lx))

    ys2 = [(ys[i]-ys[i+1])/(xs[i]-xs[i+1]) for i in range(len(ys) - 1)]
    ys3 = [(ys2[i]-ys2[i+1])/(xs[i]-xs[i+2]) for i in range(len(ys2) - 1)]
    ys4 = [(ys3[i]-ys3[i+1])/(xs[i]-xs[i+3]) for i in range(len(ys3) - 1)]

    nx = sp.UnevaluatedExpr((ys[0] +
                             (x - xs[0]) * ys2[0] +
                             (x-xs[0])*(x-xs[1])*ys3[0] +
                             (x-xs[0])*(x-xs[1])*(x-xs[2])*ys4[0]))
    nprecise = sp.lambdify(x, nx)(point)
    print("Ньютон: ")
    print(f"Интерполяционное: {nprecise}, Точное: {
        func(point)}, погрешность: {abs(func(point) - nprecise)}")
    print("\t", str(nx))
    print()


lagrange_newton([-2, -1, 0, 1], -0.5, lambda x: math.exp(x))
lagrange_newton([-2, -1, 0.2, 1], -0.5, lambda x: math.exp(x))

# %% [md]
# ## 2
# Построить кубический сплайн для функции, заданной в узлах интерполяции,
# предполагая, что сплайн имеет нулевую кривизну при $x=x_0$  и
# $x=x_4$. Вычислить значение функции в точке $x=X^*$

# $X^* = -0.5$

# $x_i = -2.0, -1.0, 0.0, 1.0, 2.0$

# $f_i=0.13534, 0.36788, 1.0, 2.7183, 7.3891$


# %% rerun
def solve_rerun(left: m.Matrix, d: List[float] | List[int]) -> List[float]:
    assert left.rows == left.cols
    for i in range(left.rows):
        for j in range(left.cols):
            # three diagonals
            assert abs(j - i) <= 1 or left[i][j] == 0, \
                "must be a 3 diagonal matrix"

    a = [0 if i == 0 else left[i][i - 1] for i in range(left.rows)]
    b = [left[i][i] for i in range(left.rows)]
    c = [0 if i == left.rows - 1
         else left[i][i + 1] for i in range(left.rows)]
    p = [0.0 for _ in left]
    q = [0.0 for _ in left]
    for i in range(left.rows):
        for j in range(-1, 1):
            p[i] = -c[i] / (b[i] + a[i] * p[i - 1])
            q[i] = (d[i] - a[i]*q[i - 1]) / (b[i] + a[i] * p[i - 1])
    x = [0.0 for _ in left]
    x[-1] = q[-1]
    for i in range(left.rows - 1)[::-1]:
        x[i] = x[i + 1] * p[i] + q[i]
    return x


# %%
def spline(xs: List[float], fs: List[float], point: float):
    hs = [xs[i]-xs[i-1] for i in range(1, len(xs))]

    size = len(hs) - 1

    cs = m.Matrix.identity(size)
    for i in range(size):
        if i - 1 >= 0:
            cs[i][i-1] = hs[i]
        cs[i][i] = 2 * (hs[i] + hs[i+1])
        if i + 1 < len(hs)-1:
            cs[i][i+1] = hs[i+1]
    right = [3 * ((fs[i+2] - fs[i+1])/hs[i+1] - (fs[i+1]-fs[i])/hs[i])
             for i in range(size)]
    c_s = [0] + solve_rerun(cs, right)
    a_s = fs[:size+1]
    b_s = [(fs[i+1] - fs[i]) / hs[i] - 1/3*hs[i]*(c_s[i+1] + 2*c_s[i])
           for i in range(size)]
    b_s.append((fs[size+1] - fs[size]) / hs[size] -
               2/3*hs[size]*(c_s[size]))
    d_s = [(c_s[i+1] - c_s[i]) / (3 * hs[i]) for i in range(size)]
    d_s.append(-c_s[size]/(3 * hs[size]))

    inte = [(xs[i], xs[i+1]) for i in range(len(hs))]
    print(f"intervals: {[inte]}")
    print(f"a_i:       {[float(f'{x:0.4f}') for x in a_s]}")
    print(f"b_i:       {[float(f'{x:0.4f}') for x in b_s]}")
    print(f"c_i:       {[float(f'{x:0.4f}') for x in c_s]}")
    print(f"d_i:       {[float(f'{x:0.4f}') for x in d_s]}")

    i = -1
    for j, a in enumerate(inte):
        l, h = a
        if point >= l and point <= h:
            i = j
            break

    x = sp.Symbol("x")
    l, h = inte[i]
    f = sp.UnevaluatedExpr(a_s[i] + b_s[i] * (x - l) + c_s[i] *
                           (x - l) ** 2 + d_s[i] * (x - l) ** 3)
    print(f"f({point}) = {f}")
    print(f"f({point}) = {sp.lambdify(x, f)(point)}")


spline([-2.0, -1.0, 0.0, 1.0, 2.0],
       [0.13534, 0.36788, 1.0, 2.7183, 7.3891], -0.5)
# spline([0.0, 1.0, 2.0, 3.0, 4.0], [0, 1.8415, 2.9093, 3.1411, 3.2432], 1.5)

# %% [md]
# ## 3
# Для таблично заданной функции путем решения нормальной системы
# МНК найти приближающие многочлены a) 1-ой  и б) 2-ой степени.
# Для каждого из приближающих многочленов вычислить сумму квадратов
# ошибок. Построить графики приближаемой функции и приближающих многочленов.

# $x = -3.0, -2.0, -1.0, 0.0, 1.0, 2.0$

# $y = 0.04979, 0.13534, 0.36788, 1.0, 2.7183, 7.3891$


# %%
def solve3(xs: List[float], ys: List[float]):
    s = len(xs)
    x_sums = [sum([x**i for x in xs]) for i in range(s)]
    y_ress = [sum([ys[j] * xs[j] ** i for j in range(s)])
              for i in range(s)]

    first = m.Matrix([
        [x_sums[0], x_sums[1], y_ress[0]],
        [x_sums[1], x_sums[2], y_ress[1]]
    ])
    second = m.Matrix([
        [x_sums[0], x_sums[1], x_sums[2], y_ress[0]],
        [x_sums[1], x_sums[2], x_sums[3], y_ress[1]],
        [x_sums[2], x_sums[3], x_sums[4], y_ress[2]]
    ])

    print(first.str(2))
    print(second.str(2))

    solved1 = first.gauss_transform().col(-1).vec()
    solved2 = second.gauss_transform().col(-1).vec()

    x = sp.Symbol("x")
    appr1 = sp.UnevaluatedExpr(solved1[0] + solved1[1] * x)
    appr2 = sp.UnevaluatedExpr(
        solved2[0] + solved2[1] * x + solved2[2] * x * x)
    f1 = sp.lambdify(x, appr1)
    f2 = sp.lambdify(x, appr2)
    print(f"Первое приближение: {appr1}")
    print(f"ошибка: {sum([(f1(xs[i]) - ys[i]) ** 2 for i in range(s)])}")
    print(f"Второе приближение: {appr2}")
    print(f"ошибка: {sum([(f2(xs[i]) - ys[i]) ** 2 for i in range(s)])}")

    plt.plot(xs, [f1(x) for x in xs], 'r', label="первое")
    plt.plot(xs, [f2(x) for x in xs], 'g', label="второе")
    plt.plot(xs, ys, "bo", label="исходное")
    plt.legend()
    plt.show()


solve3([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
       [0.04979, 0.13534, 0.36788, 1.0, 2.7183, 7.3891])

# solve3([0.0, 1.7, 3.4, 5.1, 6.8, 8.5],
#        [0.0, 1.3038, 1.8439, 2.2583, 2.6077, 2.9155])
