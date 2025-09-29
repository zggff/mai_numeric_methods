from __future__ import annotations
from typing import List, Tuple


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
        res = "\n"
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

    def get_col(self, j: int) -> Matrix:
        res = Matrix(size=(1, self.cols))
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
