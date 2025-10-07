from __future__ import annotations
from typing import List, Tuple
import math as m
import cmath as cm


class Matrix:
    def __init__(self, mat: List[List[float]] | None = None,
                 size: Tuple[int, int] = (0, 0)) -> None:
        if mat is None:
            self.rows, self.cols = size
            self.mat = [
                [0.0 for _ in range(self.cols)] for _ in range(self.rows)]
            return
        self.rows = len(mat)
        assert self.rows > 0
        self.cols = len(mat[0])
        assert self.cols > 0
        assert all([len(row) == self.cols for row in mat])
        self.mat = mat

    @classmethod
    def identity(cls, n: int) -> Matrix:
        res = cls(size=(n, n))
        for i in range(n):
            res[i][i] = 1
        return res

    @classmethod
    def from_col(cls, col: List[float] | List[int]) -> Matrix:
        res = cls(size=(len(col), 1))
        for i in range(len(col)):
            res[i][0] = col[i]
        return res

    def append_right(self, other: Matrix | List[float] | List[int]) -> Matrix:
        if isinstance(other, Matrix):
            assert self.rows == other.rows
            res = Matrix(size=(self.rows, self.cols + other.cols))
            for i in range(res.rows):
                for j in range(self.cols):
                    res[i][j] = self[i][j]
                for j in range(other.cols):
                    res[i][j + self.cols] = other[i][j]
        else:
            res = Matrix(size=(self.rows, self.cols + 1))
            for i in range(res.rows):
                for j in range(self.cols):
                    res[i][j] = self[i][j]
                res[i][-1] = other[i]
        return res

    def without(self, rowi: int = -1, coli: int = -1) -> Matrix:
        assert rowi <= self.rows and coli <= self.cols
        mat = [[val for j, val in enumerate(row) if j != coli]
               for i, row in enumerate(self.mat) if i != rowi]
        res = Matrix(None)
        res.rows = self.rows - (0 if rowi == -1 else 1)
        res.cols = self.cols - (0 if coli == -1 else 1)
        res.mat = mat
        return res

    def copy(self) -> Matrix:
        return self.without(-1, -1)

    def mult_by_number(self, other: float):
        for i in range(self.rows):
            for j in range(self.cols):
                self.mat[i][j] *= other

    def mult_by_matrix(self, other: Matrix) -> Matrix:
        assert self.cols == other.rows
        res = Matrix(size=(self.rows, other.cols))
        for i in range(res.rows):
            for j in range(res.cols):
                subres = 0
                for k in range(self.cols):
                    subres += self[i][k] * other[k][j]
                res[i][j] = subres
        return res

    def __add__(self, other: Matrix) -> Matrix:
        assert self.rows == other.rows and self.cols == other.cols
        res = self.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                res[i][j] += other[i][j]
        return res

    def __sub__(self, other: Matrix) -> Matrix:
        assert self.rows == other.rows and self.cols == other.cols
        res = self.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                res[i][j] -= other[i][j]
        return res

    def __mul__(self, other: Matrix | float | int) -> Matrix:
        if isinstance(other, Matrix):
            return self.mult_by_matrix(other)
        res = self.copy()
        res.mult_by_number(other)
        return res

    def __imul__(self, other: float | int) -> Matrix:
        self.mult_by_number(other)
        return self

    def __getitem__(self, index: int) -> List[float]:
        return self.mat[index]

    def __setitem__(self, index: int, item: List[float]) -> None:
        self.mat[index] = item

    def transpose_self(self) -> None:
        self = self.transpose()

    def transpose(self) -> Matrix:
        res = Matrix(size=(self.cols, self.rows))
        for i in range(self.rows):
            for j in range(self.cols):
                res[j][i] = self[i][j]
        return res

    def str(self, precision: int = 3) -> str:
        res = ""
        for row in self.mat:
            res += "|\t"
            for col in row:
                res += f"{col:.{precision}f}\t"
            res += "|\n"
        return res

    def inverse_basic(self) -> Matrix:
        det = self.det()
        assert det > 0
        res = self.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                minor1 = self.without(i, j)
                res.mat[i][j] = minor1.det()
                if (i + j) % 2 == 1:
                    res.mat[i][j] *= -1
        res *= (1/det)
        return res.transpose()

    def lu(self) -> Tuple[Matrix, Matrix]:
        upp = self.copy()
        low = Matrix(size=(self.rows, self.cols))
        for row in range(upp.rows):
            if upp[row][row] == 0:
                for row_other in range(row + 1, upp.rows):
                    if upp[row_other][row] != 0:
                        upp[row], upp[row_other] = upp[row_other], upp[row]
                        break
            if upp[row][row] == 0 and upp[row][-1] == 0:
                break
            low[row][row] = 1
            for row_next in range(row + 1, upp.rows):
                mult = upp[row_next][row] / upp[row][row]
                low[row_next][row] = mult
                for col in range(upp.cols):
                    upp[row_next][col] -= upp[row][col] * mult
        return (low, upp)

    def det(self) -> float:
        assert self.rows == self.cols
        _, upper = self.lu()
        res = 1
        for i in range(self.cols):
            res *= upper[i][i]
        return res

    def gauss_transform(self) -> Matrix:
        _, u = self.lu()
        s = self.rows
        assert u[-1][-1] != 0
        res = Matrix(size=(self.cols - self.rows, self.rows))
        for i in range(res.cols)[::-1]:
            for j in range(res.rows):
                sub = 0
                for k in range(i, res.cols):
                    sub += res[j][k] * u[i][k]

                res[j][i] = (u[i][s + j] - sub) / u[i][i]
        return res.transpose()

    def inverse(self) -> Matrix:
        assert self.rows == self.cols
        appended = self.append_right(Matrix.identity(self.rows))
        return appended.gauss_transform()

    def col(self, j: int) -> Matrix:
        res = Matrix(size=(self.rows, 1))
        res.mat = [[self[i][j]] for i in range(self.rows)]
        return res

    def norm(self) -> float:
        return max([sum(
                [abs(self[i][j]) for i in range(self.rows)])
                for j in range(self.cols)])

    def norm2(self) -> float:
        return sum(val ** 2 for row in self.mat for val in row) ** (1/2)

    def normc(self) -> float:
        return max([sum(
                [abs(self[i][j]) for j in range(self.cols)])
                for i in range(self.rows)])

    def vec(self) -> List[float]:
        if self.rows == 1:
            return self[0]
        if self.cols == 1:
            return [row[0] for row in self]
        return []

    def upper(self) -> Matrix:
        res = self.copy()
        for i in range(self.rows):
            for j in range(i):
                res[i][j] = 0
        return res

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i][j] != other[i][j]:
                    return False
        return True

    def eigen_rotation(self, precision: float, tries: int = 1000
                       ) -> Tuple[List[float], Matrix, float]:
        assert self == self.transpose(), "matrix must be symmetric"
        assert self.rows > 1

        a = self.copy()
        itern = 0
        u0 = Matrix.identity(self.rows)
        while itern != tries:
            itern += 1
            i = 0
            j = 1
            for i0 in range(a.rows):
                for j0 in range(a.rows):
                    if i0 != j0 and abs(a[i0][j0]) > abs(a[i][j]):
                        i = i0
                        j = j0
            phi = m.pi / 4
            if abs(a[i][i] - a[j][j]) > precision * 0.1:
                phi = 1/2 * m.atan(2 * a[i][j] / (a[i][i] - a[j][j]))
            u = Matrix.identity(self.rows)
            u[i][i] = m.cos(phi)
            u[i][j] = -m.sin(phi)
            u[j][i] = m.sin(phi)
            u[j][j] = m.cos(phi)

            a = u.transpose() * a * u
            u0 = u0 * u
            t = sum(
                [a[i0][j0] ** 2 for j0 in range(a.rows) for i0 in range(j0)]
            ) ** (1/2)
            # print(f"{itern-1} [{t}]:{u.str(4)}{a.str(4)}\n")
            if t <= precision:
                break
        return ([a[i][i] for i in range(a.rows)], u0, itern)

    def qr(self) -> Tuple[Matrix, Matrix]:
        assert self.rows == self.cols, "must be a square matrix"
        r = self.copy()
        q = Matrix.identity(r.rows)
        for j in range(r.cols):
            v = r.col(j)
            for i in range(j):
                v[i][0] = 0
            sign = 1 if r[j][j] >= 0 else -1
            evclid = sum([r[i][j] ** 2 for i in range(j, r.rows)]) ** (1/2)
            v[j][0] += sign * evclid
            vt = v.transpose()
            coef = 2.0 / (vt * v)[0][0]
            h = Matrix.identity(r.rows) - (v * vt) * coef
            q = q * h
            r = h * r
        return (q, r)

    def eigen_qr(self, precision: float, tries: int = 1000
                 ) -> List[complex]:
        a = self.copy()
        iters = 0
        ress = []
        while iters < tries:
            res = []
            iters += 1
            q, r = a.qr()
            a = r * q
            j = 0
            while j < a.cols:
                sub = sum([a[i][j] ** 2
                          for i in range(j + 1, a.rows)]) ** (1/2)
                if sub <= precision:
                    res.append(a[j][j])
                    j += 1
                else:
                    b = -(a[j][j] + a[j+1][j+1])
                    c = a[j][j] * a[j+1][j+1] - a[j][j+1] * a[j+1][j]
                    d = b * b - 4 * c
                    dsqrt = cm.sqrt(d)
                    res.append((-b + dsqrt) / 2)
                    res.append((-b - dsqrt) / 2)
                    j += 2
            ress.append(res)
            if len(ress) > 3:
                ress.pop(0)
            done = len(ress) >= 3
            if done:
                for i in range(self.rows):
                    if isinstance(ress[-1][i], complex):
                        if ress[-1][i].imag == 0:
                            done = False
                            break
                        if abs(ress[-1][i] - ress[-2][i]) > precision:
                            done = False
                            break
                    else:
                        if (abs(ress[-1][i] - ress[-2][i]) >
                           abs(ress[-2][i] - ress[-3][i])):
                            done = False
                            break
            if done:
                break
        # print(f"{iters}: \n{a.str()}")
        return ress[-1]
