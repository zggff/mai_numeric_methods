# %% [md]
"""
# Лабораторная работа № 1
"""


# %% import modules
from matrix import Matrix
from typing import List, Tuple


# %% [md]
"""
### Задание № 1
"""


# %% solve gauss
def solve_gauss(left: Matrix, right: List[float] | List[int]) -> Matrix:
    left = left.append_right(right)
    return left.gauss_transform()


# %% 1.1 gauss
a = Matrix([
                [1, 2, -1, -7],
                [8,	0, -9, -3],
                [2, -3, 7, 1],
                [1, -5, -6, 8]])

b = [-23, 39, -7, 30]

print(f"solution: \n{solve_gauss(a, b).str()}")

print(f"determinant = \n{a.det()}")
inv = a.inverse()
print(f"inverse matrix: \n{inv.str()}")
print(f"proof   matrix: \n{(inv * a).str()}")


# %% [md]
"""
### Задание № 2
"""


# %% rerun
def solve_rerun(left, d: List[float] | List[int]) -> Matrix:
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
    return Matrix.from_col(x)


# %% 1.2 rerun
a = Matrix([
                [6, -5, 0, 0, 0],
                [-6, 16, 9, 0, 0],
                [0, 9, -17, -3, 0],
                [0, 0, 8, 22, -8],
                [0, 0, 0, 6, -13]
             ])
b = [-58., 161, -114, -90, -55]

print(f"solution: \n{solve_rerun(a, b).str()}")


# %% [md]
"""
### Задание № 3
"""


# %% solve simple iter
def solve_simple_iter(left, right: List[float] | List[int],
                      e: float, tries: int = 1000) -> Tuple[List[float], int]:
    assert left.cols == left.rows
    for i in range(left.rows):
        if left[i][i] == 0:
            for j in range(left.rows):
                if left[j][i] != 0 and left[i][j] != 0:
                    left[j][i], left[i][j] = left[i][j], left[j][i]
        assert left[i][i] != 0

    b = Matrix([[right[i] / left[i][i]] for i in range(left.rows)])
    a = Matrix(size=(left.rows, left.rows))
    for i in range(left.rows):
        for j in range(left.rows):
            if i == j:
                continue
            a[i][j] = - left[i][j] / left[i][i]

    a_abs = a.normc()
    if a_abs >= 1:
        coef = 1
    else:
        coef = a_abs / (1 - a_abs)

    x = b
    i = 0
    while i != tries:
        i += 1
        x2 = b + a * x
        diff = coef * (x2 - x).normc()
        # print(f"{i}: {diff}")
        # x2.transpose().print()
        # print()
        x = x2
        if diff <= e:
            break
    return (x.vec(), i)


# %% solve zeidel
def solve_zeidel(left, right: List[float] | List[int],
                 e: float, tries: int = 1000) -> Tuple[List[float], int]:
    assert left.cols == left.rows
    for i in range(left.rows):
        if left[i][i] == 0:
            for j in range(left.rows):
                if left[j][i] != 0 and left[i][j] != 0:
                    left[j][i], left[i][j] = left[i][j], left[j][i]
        assert left[i][i] != 0

    b = Matrix([[right[i] / left[i][i]] for i in range(left.rows)])
    a = Matrix(size=(left.rows, left.rows))
    for i in range(left.rows):
        for j in range(left.rows):
            if i == j:
                continue
            a[i][j] = - left[i][j] / left[i][i]

    c = a.upper().normc()
    a_abs = a.normc()
    if a_abs >= 1:
        coef = 1
    else:
        coef = c / (1 - a_abs)

    # coef = c / (1 - a_abs)
    # if a_abs >= 1:
    #     coef = 1

    x = b
    n = 0
    while n != tries:
        n += 1
        x2 = x.copy()
        for i in range(x2.rows):
            subsum = sum([a[i][j] * x2[j][0] for j in range(a.cols)])
            x2[i][0] = b[i][0] + subsum
        diff = coef * (x2 - x).normc()
        x = x2
        if diff <= e:
            break
    return (x.vec(), n)


# %% 1.3 simple iter and zeidel
a = Matrix([
    [23, -6, -5, 9],
    [8, 22, -2, 5],
    [7, -6, 18, -1],
    [3, 5, 5, -19]
])  # 6

b = [232, -82, 202, -57.0]  # 6

print(solve_simple_iter(a, b, 0.01))
print(solve_zeidel(a, b, 0.01))


# %% [md]
"""
### Задание № 4
"""


# %% 1.4 rotation
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


# %% 1.5 qr
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
