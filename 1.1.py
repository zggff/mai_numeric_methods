from matrix import Matrix
import solutions as s


a = Matrix([
                [1, 2, -1, -7],
                [8,	0, -9, -3],
                [2, -3, 7, 1],
                [1, -5, -6, 8]])

b = [-23, 39, -7, 30]

print(f"solution: {s.solve_gauss(a, b).str()}")

print(f"determinant = {a.det()}")
print(f"inverse matrix: {a.inverse().str()}")
