from typing import List, Tuple
from matrix import Matrix


def solve_gauss(left: Matrix, right: List[float] | List[int]) -> Matrix:
    left = left.append_right(right)
    return left.gauss_transform()


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


def solve_simple_iter(left, right: List[float] | List[int],
                      e: float) -> Tuple[List[float], int]:
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
    coef = a_abs / (1 - a_abs)
    if a_abs >= 1:
        coef = 1

    x = b
    i = 0
    while True:
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


def solve_zeidel(left, right: List[float] | List[int],
                 e: float) -> Tuple[List[float], int]:
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
    coef = c / (1 - a_abs)
    if a_abs >= 1:
        coef = 1

    x = b
    n = 0
    while True:
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
