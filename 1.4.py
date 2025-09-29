from matrix import Matrix

# a = Matrix([
#     [4, 2, 1],
#     [2, 5, 3],
#     [1, 3, 6]
# ])

a = Matrix([
    [9, 2, -7],
    [2, -4, -1],
    [-7, -1, 1]
])

vals, vectors, n = a.own_rotation(0.03)

print(f"{n} {vals} {vectors.str()}")
