# %% [md]
# # Лабораторная работа № 2


# %% import some module
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import math

# %% [md]
# ### Задание № 1
# Реализовать методы простой итерации и Ньютона решения нелинейных
# уравнений в виде программ, задавая в качестве входных данных
# точность вычислений. С использованием разработанного программного
# обеспечения найти положительный корень нелинейного уравнения
# (начальное приближение определить графически).
# Проанализировать зависимость погрешности вычислений
# от количества итераций.
# $$ f(x) = e^x - 2x - 2 = 0 $$
# $$ x = g(x) = ln(2x + 2) $$


# %%
x = sp.symbols("x")
y = math.e ** x - 2 * x - 2
y_func = sp.lambdify(x, y)


# %%
xs = np.arange(1, 2, 0.1)
plt.plot(xs, y_func(xs), 'r', label='f(x)')
plt.plot(xs, xs * 0, "k--")

start = 1.6
end = 1.7

plt.vlines([start, end], -1.5, 1, linestyles='dashed', colors='blue',
           label="границы")
plt.xticks(xs)
plt.legend()
plt.show()


# %% [md]
# #### Простая итерация


# %% simple iter
x_x = sp.log(2*x + 2)
x_diff = x_x.diff()
x_x_func = sp.lambdify(x, x_x)
x_x_diff_func = sp.lambdify(x, x_diff)

q = x_x_diff_func(1.6)

plt.plot(xs, x_x_diff_func(xs), 'r', label="g'(x)")
plt.plot(xs, xs - xs + q, "k--")
plt.vlines([start, end], 0, 0.6, linestyles='dashed', colors='blue',
           label="границы")
plt.xticks(xs)
plt.legend()
plt.show()


# %% [md]
# q = {eval}`q`


# %%
epsilon = 0.0001

x0 = x_x_func((start + end) / 2)
x1 = x0
coef = q / (1 - q)
epsilon_adjusted = epsilon / coef
for i in range(10000):
    x1 = x_x_func(x0)
    if abs(x1 - x0) <= epsilon_adjusted:
        break
    x0 = x1

print(x1)


# %% [md]
# #### метод ньютона


# %%
y_x = y.diff()
y_xx = y_x.diff()
y_x_func = sp.lambdify(x, y_x)
y_xx_func = sp.lambdify(x, y_xx)

plt.plot(xs, y_x_func(xs), 'r', label="f '(x)")
plt.plot(xs, y_xx_func(xs), 'b', label="f ''(x)")
plt.vlines([start, end], 0, 7, linestyles='dashed', colors='blue',
           label="границы")
plt.xticks(xs)
plt.legend()
plt.show()


# %% [md]
# так как обе производные на отрезке [{eval}`start`; {eval}`end`]
# положительные, то в качестве $x_0$ можно взять правую
# границу: $x_0=$ {eval}`end`


# %%
epsilon = 0.0001

x0 = end
x1 = end
for i in range(100000):
    x1 = x0 - y_func(x0) / y_x_func(x0)
    if abs(x1 - x0) < epsilon:
        break
    x0 = x1

print(x1)


# %% [markdown]
# ### Задание № 2

# Реализовать методы простой итерации и Ньютона решения систем
# нелинейных уравнений в виде программного кода, задавая в качестве
# входных данных точность вычислений. С использованием разработанного
# программного обеспечения решить систему нелинейных уравнений
# (при наличии нескольких решений найти то из них, в котором значения
# неизвестных являются положительными); начальное приближение
# определить графически. Проанализировать зависимость погрешности
# вычислений от количества итераций.
# Реализовать методы простой итерации и Ньютона решения систем
# \begin{align}
# & a = 3 \\
# & x_1 - cos(x_2) = 1 \\
# & x_2 - lg(x_1 + 1) = a
# \end{align}
