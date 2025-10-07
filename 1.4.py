from matrix import Matrix

a = Matrix([
    [4, 2, 1],
    [2, 5, 3],
    [1, 3, 6]
])

a = Matrix([
    [9, 2, -7],
    [2, -4, -1],
    [-7, -1, 1]
])

# every column is a vector
vals, vectors, n = a.eigen_rotation(0.03)

print(f"{n} {vals} \n{vectors.str()}")
