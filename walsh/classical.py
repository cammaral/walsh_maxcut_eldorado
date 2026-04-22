
import pennylane.numpy as np

def generate_coef(_fk, _w):
    _aux = []
    for i in range(len(_w)):
        _aux.append(np.dot(_fk, _w[i]) / len(_w))
    return _aux

def generate_fk(f, n):
    _N = 2**n
    _x = np.array(range(_N), dtype=np.float32) / _N
    _fk = f(_x)
    return np.array(_fk, dtype=np.float32), np.array(_x, dtype=np.float32)

def binary_expansion(j, n):
    return [int(bit) for bit in bin(j)[2:].zfill(n)]

def dyadic_expansion(x, n):
    if x < 1 and x >= 0:
        expansion = []
        for _ in range(n):
            bit = int(x * 2)
            expansion.append(bit)
            x = x * 2 - bit
        return expansion

def walsh_functions(j, x, n):
    bj = binary_expansion(j, n)  # ji -> n
    bj = bj[::-1]
    dx = dyadic_expansion(x, n)  # xk -> n
    walsh = (-1) ** (np.dot(bj, dx))
    return walsh

def generate_series(n):
    _series = []
    _N = 2**n
    _j = np.arange(0, _N)  # j = 0, ..., N-1
    _x = np.array(range(_N), dtype=np.float32) / _N  # xk = 0/N, 1/N ..., N-1/N
    for j in _j:
        _aux = []
        for x in _x:
            _aux.append(walsh_functions(j, x, n))  # wj -> n
        _series.append(_aux)
    return np.array(_series, dtype=np.float32)