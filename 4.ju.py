# %% [md]
# # Лабораторная работа № 4

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from typing import List, Callable
from matrix import Matrix

# %% [md]
# ## 1
# Реализовать методы Эйлера, Рунге-Кутты и Адамса 4-го порядка
# в виде программ, задавая в качестве входных данных шаг сетки  .
# С использованием разработанного программного обеспечения решить
# задачу Коши для ОДУ 2-го порядка на указанном отрезке.
# Оценить погрешность численного решения с использованием метода
# Рунге – Ромберга и путем сравнения с точным решением.
# Явный Эйлер, ЭйЛер-Коши, первый улучшенный Эйлер, рунге-кутты 4 пор

# __Задача Коши__
# $$y''+4xy'+(4x^2+2)y=0$$
# $$y(0)=1$$
# $$y'(0)=2$$
# $$x \in [0,1], h =0.1$$
# __Точное решение__
# $$ y = (1+x)e^{-x^2}$$

# %% [md]
# __преобразуем__
# $$y''= -4xy'-(4x^2+2)y$$
# __заменим__ $y_1=y, y_2=y'$

# $$y_1'=y_2$$
# $$y_2'= -4xy_2-(4x^2+2)y_1$$
# $$y_1(0) = 1 $$
# $$y_2(0) = 2 $$


# %%
def f1(x, y):
    y1, y2 = y
    return np.array([y2, -4*x*y2 - (4*x**2 + 2)*y1])


def precise(h: float = 0.1) -> np.array:
    @np.vectorize
    def y(x: float) -> float:
        return (1 + x) * math.exp(-x * x)
    return y(np.arange(0, 1 + h, h))


def euler_apparent(h: float = 0.1, y0=[1, 1], x0=0,
                   x1=1, f=f1) -> np.array:
    x = np.arange(x0, x1 + h, h)
    y = np.zeros((len(x), 2))
    y[0] = y0
    for k in range(len(x) - 1):
        y[k+1] = y[k] + h * f(x[k], y[k])
    return y[:, 0]


def euler_koshi(h: float = 0.1, y0=[1, 1], x0=0,
                x1=1, f=f1) -> np.array:
    x = np.arange(x0, x1 + h, h)
    y = np.zeros((len(x), 2))
    y[0] = y0
    for k in range(len(x)-1):
        yk = y[k] + h * f(x[k], y[k])
        y[k+1] = y[k] + h * (f(x[k], y[k]) + f(x[k+1], yk)) / 2
    return y[:, 0]


def euler_better(h: float = 0.1, y0=[1, 1], x0=0,
                 x1=1,  f=f1) -> np.array:
    x = np.arange(x0, x1 + h, h)
    y = np.zeros((len(x), 2))
    y[0] = y0
    for k in range(len(x)-1):
        yk = y[k] + h/2 * f(x[k], y[k])
        y[k+1] = y[k] + h * f(x[k] + h/2, yk)
    return y[:, 0]


def runge_kutte(h: float = 0.1, y0=[1, 1], x0=0,
                x1=1, f=f1) -> np.array:
    x = np.arange(x0, x1 + h, h)
    y = np.zeros((len(x), 2))
    y[0] = y0
    for k in range(len(x)-1):
        k1 = h * f(x[k], y[k])
        k2 = h * f(x[k] + h/2, y[k] + (1/2)*k1)
        k3 = h * f(x[k] + h/2, y[k] + (1/2)*k2)
        k4 = h * f(x[k] + h, y[k] + k3)
        y[k+1] = y[k] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y[:, 0]


def runge_romberg(method: Callable[[float], np.array],
                  h: float,
                  p: int,
                  y0=[1, 1],
                  x0=0,
                  x1=1,
                  f=f1):
    yh, yh2 = method(h, y0, x0, x1, f), method(h/2, y0, x0, x1, f)[::2]
    runge = (yh - yh2) / (math.pow(2, p) - 1)
    return np.abs(runge)


h = 0.1
pd.set_option("display.precision", 3)
df = pd.DataFrame({"x": np.arange(0, 1 + h, h)})
df["prec"] = precise(h)
df["euler"] = euler_apparent(h)
df["euler_abs"] = np.abs(df["prec"] - df["euler"])
df["euler_rr"] = runge_romberg(euler_apparent, h, 1)

df["koshi"] = euler_koshi(h)
df["koshi_abs"] = np.abs(df["prec"] - df["koshi"])
df["koshi_rr"] = runge_romberg(euler_koshi, h, 2)

df["better"] = euler_better(h)
df["better_abs"] = np.abs(df["prec"] - df["better"])
df["better_rr"] = runge_romberg(euler_better, h, 2)

df["runge"] = runge_kutte(h)
df["runge_abs"] = np.abs(df["prec"] - df["runge"])
df["runge_rr"] = runge_romberg(runge_kutte, h, 4)

df

# %%
plt.subplot()
plt.plot(df["x"], df["prec"], label="точное")
plt.plot(df["x"], df["euler"], label="Эйлер")
plt.plot(df["x"], df["koshi"], label="Эйлер-Коши")
plt.plot(df["x"], df["better"], label="первый улучшенный Эйлер")
plt.plot(df["x"], df["runge"], label="Рунге-Кутте")
plt.legend()

# %% [md]
# ## 2
# Реализовать метод стрельбы и конечно-разностный метод решения
# краевой задачи для ОДУ в виде программ. С использованием
# разработанного программного обеспечения решить краевую задачу
# для обыкновенного дифференциального уравнения 2-го порядка на
# указанном отрезке. Оценить погрешность численного решения с
# использованием метода Рунге – Ромберга и путем сравнения с
# точным решением.

# __Краевая задача__
# $$ y''-2(1+tg(x)^2)y=0$$
# $$ y(0) = 0$$
# $$ y(\frac{\pi}{6})=-\frac{\sqrt{3}}{3}$$
# __Точное решение__
# $$ y(x) = -tg(x) $$

# %% [md]
# __преобразуем__
# $$ y'' = 2(1 + tg^2(x))y$$
# __заменим__ $y_1 = y, y_2 = y'$
# $$ y_1' = y_2 $$
# $$ y_2' = 2(1 + tg^2(x))y_1 $$


# %% [md]
# так как конец - $\frac{\pi}{6}$, добавим методы для решения не
# через шаг, а через linspace

# %%
def runge_kutte_lp(n: int = 10, y0=[1, 1], x0=0,
                   x1=1, f=f1) -> np.array:
    x = np.linspace(x0, x1, n)
    y = np.zeros((len(x), 2))
    y[0] = y0
    for k in range(len(x)-1):
        h = x[k+1]-x[k]
        k1 = f(x[k], y[k])
        k2 = f(x[k] + h/2, y[k] + (h/2)*k1)
        k3 = f(x[k] + h/2, y[k] + (h/2)*k2)
        k4 = f(x[k] + h, y[k] + h*k3)
        y[k+1] = y[k] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return y[:, 0]


def runge_romberg_lp(method: Callable[[float], np.array],
                     n: int,
                     p: int
                     ):
    yh, yh2 = method(n), method(n * 2)[::2]
    runge = (yh - yh2) / (math.pow(2, p) - 1)
    return np.abs(runge)


# %% [md]
# #### Метод стрельбы

# %%
N = 40
h = 0.1
targ = - np.sqrt(3) / 3
end = math.pi / 6


def f2(x, y):
    y1, y2 = y
    return np.array([y2, 2 * y1 * (1 + np.pow(np.tan(x), 2))])


@np.vectorize
def f_precise(x):
    return -np.tan(x)


def shoot(cnt=10, e: float = 0.0001):
    n = [1, 0.8]
    solutions = [
        runge_kutte_lp(n=cnt, y0=[0, 1], x0=0, x1=end, f=f2),
        runge_kutte_lp(n=cnt, y0=[0, 0.8], x0=0, x1=end, f=f2),
    ]
    phi = [
        solutions[-2][-2] - targ,
        solutions[-1][-1] - targ]
    for _ in range(100):
        n.append(
            n[-1] - (n[-1] - n[-2]) / (phi[-1] - phi[-2]) * phi[-1])
        solutions.append(
            runge_kutte_lp(n=cnt, y0=[0, n[-1]], x0=0, x1=end, f=f2))
        phi.append(solutions[-1][-1] - targ)
        if abs(phi[-1]) <= e:
            break
    return solutions[-1]


def ends(cnt=10, e: float = 0.0001):
    pass


df = pd.DataFrame({"x": np.linspace(0, end, N)})
df["y"] = f_precise(df["x"])
df["shoot"] = shoot(cnt=N)
df["shoot_abs"] = np.abs(df["y"] - df["shoot"])
df["shoot_rr"] = runge_romberg_lp(shoot, N, 4)
df

# %% [md]
# #### Метод конечно-разностный


# %%
def solve_rerun(left: Matrix, d: List[float] | List[int]) -> List[float]:
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


# %% [md]
# $$ y''-2(1+tg(x)^2)y=0$$
# $$ y'' = \frac{y_{k+1} - 2y_k + y_{k-1}}{h^2} $$
# $$ \frac{y_{k+1} - 2y_k + y_{k-1}}{h^2} = 2(1 + tg^2(x))y_k $$
# $$ y_{k+1} - 2y_k + y_{k-1} = 2(1 + tg^2(x))y_k*h^2 $$
# $$ y_{k+1} - 2y_k(1 + (1 + tg^2(x))*h^2) + y_{k-1} = 0 $$
# __трех диагональная__
# $$ - 2y_1(1 + (1 + tg^2(x))*h^2) + y_2 = -y_a $$
# $$ y_{k+1} - 2y_k(1 + (1 + tg^2(x))*h^2) + y_{k-1} = 0 $$
# $$ y_{n-1} - 2y_n(1 + (1 + tg^2(x))*h^2) = -y_b $$


# %%
x = np.linspace(0, end, N)
h = x[1] - x[0]
mat = Matrix.identity(N)
for i in range(0, N):
    if i > 0:
        mat[i][i-1] = 1
    mat[i][i] = -2*(1 + (1 + math.tan(x[i]) ** 2) * h * h)
    if i < N-1:
        mat[i][i+1] = 1
d = [0] + [0 for _ in range(N - 2)] + [-targ]

df["end"] = solve_rerun(mat, d)
df["end_abs"] = np.abs(df["y"] - df["end"])
df["end_rr"] = runge_romberg_lp(shoot, N, 2)
df


# %%
plt.subplot()
plt.plot(df["x"], df["y"], label="точное")
plt.plot(df["x"], df["end"], label="конечное")
plt.plot(df["x"], df["shoot"], label="стрельбы")
plt.legend()
