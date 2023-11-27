from dataclasses import dataclass
import numpy as np
from typing import List

# Variantas: 10

@dataclass
class Lygtis:
    koeficientai: list[float]
    rezultatas: List[float]

@dataclass
class LygciuSistema:
    lygtys: list[Lygtis]

def swap_lygtis(lygciu_sistema: LygciuSistema, i, j):
    tmp = lygciu_sistema.lygtys[i]
    lygciu_sistema.lygtys[i] = lygciu_sistema.lygtys[j]
    lygciu_sistema.lygtys[j] = tmp

def switch_leading(lygciu_sistema: LygciuSistema):
    for i in range(len(lygciu_sistema.lygtys)):
        if lygciu_sistema.lygtys[i].koeficientai[i] != 0: continue

        for j in range(len(lygciu_sistema.lygtys)):
            if lygciu_sistema.lygtys[j].koeficientai[i] == 0: continue

            swap_lygtis(lygciu_sistema, i, j)

def gauti_sklaida(lygciu_sistema: LygciuSistema, x, rezultato_idx=0):
    n = lygciu_sistema.lygtys[0].koeficientai
    sklaida = 0
    for lygtis in lygciu_sistema.lygtys:
        issistatyta = sum(x[i]*lygtis.koeficientai[i] for i in range(len(n)))
        sklaida += abs(issistatyta - lygtis.rezultatas[rezultato_idx])
    return sklaida

def gauso_atispindzio_metodas(lygciu_sistema: LygciuSistema, epsilon=1e-7):
    koeficientai = list(lygtis.koeficientai for lygtis in lygciu_sistema.lygtys)
    A = np.matrix(koeficientai).astype(float)
    rezultatai = list(lygtis.rezultatas for lygtis in lygciu_sistema.lygtys)
    b = np.matrix(rezultatai).astype(float)

    n = (np.shape(A))[0]
    nb = (np.shape(b))[1]
    A1 = np.hstack((A, b))

    # tiesioginis etapas(atspindziai):
    for i in range(0, n - 1):
        z = A1[i:n, i]
        zp = np.zeros(np.shape(z))
        zp[0] = np.linalg.norm(z)
        omega = z - zp
        omega = omega / np.linalg.norm(omega)
        Q = np.identity(n - i) - 2 * omega * omega.transpose()
        A1[i:n, :] = Q.dot(A1[i:n, :])

    if np.sum(np.abs(A1[n-1, 0:n-nb+1])) < epsilon:
        if abs(A1[n-1, n]) < epsilon:
            print("Be galo daug sprendiniu")
        else:
            print("Nera sprendiniu")
        return

    # atgalinis etapas:
    x = np.zeros(shape=(n, 1))
    for i in range(n - 1, -1, -1):
        x[i, :] = (A1[i, n:n + nb] - A1[i, i + 1:n] * x[i + 1:n, :]) / A1[i, i]

    print("Sprendinys:")
    for i in range(0, n):
        print(f"x{i} = {x[i, 0]:.5f}")

    print("Sklaida:", gauti_sklaida(lygciu_sistema, list(x.flat)))

def paprastu_iteraciju_metodas(lygciu_sistema: LygciuSistema, epsilon=1e-12, alpha=1, iteraciju_kiekis=1000):
    koeficientai = list(lygtis.koeficientai for lygtis in lygciu_sistema.lygtys)
    A = np.matrix(koeficientai).astype(float)
    rezultatai = list(lygtis.rezultatas for lygtis in lygciu_sistema.lygtys)
    b = np.matrix(rezultatai).astype(float)

    n = np.shape(A)[0]
    alpha_mat = np.array([alpha, alpha, alpha, alpha]) # laisvai parinkti metodo parametrai
    Atld = np.diag(1.0/np.diag(A)).dot(A) - np.diag(alpha_mat)
    btld = np.diag(1.0/np.diag(A)).dot(b)

    x = np.zeros(shape = (n,1))
    x1 = np.zeros(shape = (n,1))
    for iteracija in range(iteraciju_kiekis):
        x1 = ((btld - Atld.dot(x)).transpose()/alpha_mat).transpose()
        prec = np.linalg.norm(x1 - x)/(np.linalg.norm(x) + np.linalg.norm(x1))
        if prec < epsilon: break
        x[:]=x1[:]

    print("Reikiamas iteraciju skaicius", iteracija)
    print("Sprendinys:")
    for i in range(0, n):
        print(f"x{i} = {x[i, 0]:.5f}")

    print("Sklaida:", gauti_sklaida(lygciu_sistema, list(x.flat)))

def lu_sklaidos_metodas(lygciu_sistema: LygciuSistema):
    koeficientai = list(lygtis.koeficientai for lygtis in lygciu_sistema.lygtys)
    A = np.matrix(koeficientai).astype(float)
    rezultatai = list(lygtis.rezultatas for lygtis in lygciu_sistema.lygtys)
    b = np.matrix(rezultatai).astype(float)

    n = np.shape(A)[0]
    P = np.arange(0, n)

    # tiesioginis etapas:
    for i in range(n-1):
        iii = abs(A[i:n,i]).argmax()
        A[[i,i+iii],:] = A[[i+iii,i],:] # sukeiciamos eilutes
        P[[i,i+iii]] = P[[i+iii,i]] # sukeiciami eiluciu numeriai

        for j in range (i+1,n):
            r = A[j,i]/A[i,i]
            A[j,i:n+1] = A[j,i:n+1] - A[i,i:n+1]*r;
            A[j,i] = r
    b = b[P]

    # 1-as atgalinis etapas, sprendziama Ly=b, y->b
    for i in range(1, n):
        b[i] = b[i] - A[i,0:i]*b[0:i]

    # 2-as atgalinis etapas , sprendziama Ux=b, x->b
    for i in range (n-1,-1,-1) :
        b[i] = (b[i] - A[i,i+1:n]*b[i+1:n])/A[i,i]

    print("Sprediniai:")
    for i, column in enumerate(b.transpose()):
        print(f"{i+1}.")
        if not np.isfinite(column).all():
            print("  Nera arba be galo daug sprendiniu")
            continue

        for j in range(0, n):
            print(f"  x{j} = {column[0, j]:.5f}")

        print(f"  Sklaida = {gauti_sklaida(lygciu_sistema, list(column.flat), i)}")

def a_dalis(lygciu_sistemos: list[LygciuSistema]):
    for lygciu_sistema in lygciu_sistemos:
        switch_leading(lygciu_sistema)

    print("=========== Atspindzio metodas ==========")
    for i, lygciu_sistema in enumerate(lygciu_sistemos):
        print("-------- Lygciu sistema ", i+1)
        gauso_atispindzio_metodas(lygciu_sistema)

    print("========== Paprastu iteraciju ===========")
    paprastu_iteraciju_metodas(lygciu_sistemos[0])

def b_dalis(lygciu_sistema: LygciuSistema):
    print("============= LU sklaidos ===============")
    lu_sklaidos_metodas(lygciu_sistema)

a_dalis([
    # 8
    LygciuSistema([
        Lygtis([ 4,  3, -1,  1], [ 12]),
        Lygtis([ 3,  9, -2, -2], [ 10]),
        Lygtis([-1, -2, 11, -1], [-28]),
        Lygtis([ 1, -2, -1,  5], [ 16]),
    ]),
    # 13
    LygciuSistema([
        Lygtis([1, -2,  3, 4], [11]),
        Lygtis([0, -7,  3, 1], [ 2]),
        Lygtis([1,  0, -1, 1], [-4]),
        Lygtis([2, -2,  2, 5], [ 7]),
    ]),
    # 20
    LygciuSistema([
        Lygtis([ 2, 4,  6, -2], [2]),
        Lygtis([ 1, 3,  1, -3], [1]),
        Lygtis([ 1, 1,  5,  1], [7]),
        Lygtis([ 2, 3, -3, -2], [2]),
    ])
])

b_dalis(
    # 10
    LygciuSistema([
        Lygtis([ 6,  1, 3, -2], [ 8,  67,  -5.25]),
        Lygtis([ 6,  8, 1, -1], [14,  77,   0   ]),
        Lygtis([12, -2, 4, -1], [13, 126, -13.5 ]),
        Lygtis([ 8,  1, 1,  5], [15,  95,  -7.25]),
    ])
)