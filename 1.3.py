from matrix import Matrix
import solutions as s


# a = Matrix([
#     [10, 1, 1],
#     [2, 10, 1],
#     [2, 2, 10]
# ])  # given
# b = [12.0, 13, 14]  # given

a = Matrix([
    [23, -6, -5, 9],
    [8, 22, -2, 5],
    [7, -6, 18, -1],
    [3, 5, 5, -19]
])  # 6

# a = Matrix([
#     [19, -4, -9, -1],
#     [-2, 20, -2, -7],
#     [6, -5, -25, 9],
#     [0, -3, -9, 12]
# ])  # 1


b = [232, -82, 202, -57.0]  # 6

# b = [100, -5, 34, 69]  # 1

print(s.solve_simple_iter(a, b, 0.01))
print(s.solve_zeidel(a, b, 0.01))
