from matrix import Matrix
import solutions as s


# a = Matrix([
#     [10, 1, 1],
#     [2, 10, 1],
#     [2, 2, 10]
# ])
# b = [12.0, 13, 14]

a = Matrix([
    [23, -6, -5, 9],
    [8, 22, -2, 5],
    [7, -6, 18, -1],
    [3, 5, 5, -19]
])

b = [232, -82, 202, -57.0]

print(s.solve_simple_iter(a, b, 0.01))
print(s.solve_zeidel(a, b, 0.01))
