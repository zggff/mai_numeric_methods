from matrix import Matrix

# a = Matrix([
#     [1, 3, 1],
#     [1, 1, 4],
#     [4, 3, 1]
# ])

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
