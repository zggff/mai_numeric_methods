# %% [md]
"""
# Лабораторная работа № 1
"""


# %% import modules
from matrix import Matrix
import solutions as s


# %% [md]
"""
### Задание № 1
"""


# %%
a = Matrix([
                [1, 2, -1, -7],
                [8,	0, -9, -3],
                [2, -3, 7, 1],
                [1, -5, -6, 8]])

b = [-23, 39, -7, 30]

print(f"solution: \n{s.solve_gauss(a, b).str()}")

print(f"determinant = \n{a.det()}")
inv = a.inverse()
print(f"inverse matrix: \n{inv.str()}")
print(f"proof   matrix: \n{(inv * a).str()}")


# %% [md]
"""
### Задание № 2
"""


# %%
a = Matrix([
                [6, -5, 0, 0, 0],
                [-6, 16, 9, 0, 0],
                [0, 9, -17, -3, 0],
                [0, 0, 8, 22, -8],
                [0, 0, 0, 6, -13]
             ])

b = [-58., 161, -114, -90, -55]

print(f"solution: \n{s.solve_rerun(a, b).str()}")


# %% [md]
"""
### Задание № 3
"""


# %%
a = Matrix([
    [23, -6, -5, 9],
    [8, 22, -2, 5],
    [7, -6, 18, -1],
    [3, 5, 5, -19]
])  # 6

b = [232, -82, 202, -57.0]  # 6

print(s.solve_simple_iter(a, b, 0.01))
print(s.solve_zeidel(a, b, 0.01))


# %% [md]
"""
### Задание № 4
"""


# %%
a = Matrix([
    [9, 2, -7],
    [2, -4, -1],
    [-7, -1, 1]
])
vals, vectors, n = a.eigen_rotation(0.03)
print(f"{n} {vals} \n{vectors.str()}")


# %% [md]
"""
### Задание № 5
"""


# %%
a = Matrix([
    [8, -1, -3],
    [-5, 9, -8],
    [4, -5, 7]
])

q, r = a.qr()
print(f"q = \n{q.str()}")
print(f"r = \n{r.str()}")
a = a.eigen_qr(0.01)
print(a)
