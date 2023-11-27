from math import cos, sin, pi
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np

@dataclass
class Lygtis:
    koeficientai: list[float]
    rezultatas: list[float]

@dataclass
class LygciuSistema:
    lygtys: list[Lygtis]

def lerp(x, min_x, max_x):
    return min_x + x * (max_x - min_x)

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
        # if abs(A1[n-1, n]) < epsilon:
        #     print("Be galo daug sprendiniu")
        # else:
        #     print("Nera sprendiniu")
        return None

    # atgalinis etapas:
    x = np.zeros(shape=(n, 1))
    for i in range(n - 1, -1, -1):
        x[i, :] = (A1[i, n:n + nb] - A1[i, i + 1:n] * x[i + 1:n, :]) / A1[i, i]

    # print("Sprendinys:")
    # for i in range(0, n):
    #     print(f"x{i} = {x[i, 0]:.5f}")

    # print("Sklaida:", gauti_sklaida(lygciu_sistema, list(x.flat)))

    return list(x.flat)

def gauti_x_taskus(from_x, to_x, density) -> list[float]:
    x_s = []
    for i in range(density):
        x_s.append(lerp(float(i) / (density-1), from_x, to_x))

    return x_s

def gauti_xy_taskus(F, from_x, to_x, density) -> tuple[list[float], list[float]]:
    x_s = gauti_x_taskus(from_x, to_x, density)
    y_s = []
    for x in x_s:
        y_s.append(F(x))
    return (x_s, y_s)

def plot_function(F, from_x, to_x, density, **kvargs):
    x_s, y_s = gauti_xy_taskus(F, from_x, to_x, density)

    plt.plot(x_s, y_s, **kvargs)

def gauti_vienanario_sprendinius(x_mazgai: list[float], y_mazgai: list[float]) -> list[float]:
    lygtys = []

    for i, y in enumerate(y_mazgai):
        koeficientai = []
        x = x_mazgai[i]
        for j in range(len(x_mazgai)):
            koeficientai.append(x ** j)
        lygtys.append(Lygtis(koeficientai, [y]))

    return gauso_atispindzio_metodas(LygciuSistema(lygtys))

def gauti_y_is_vienanariu(vienanariai: list[float], x: float):
    y = 0
    for i, a in enumerate(vienanariai):
        y += a * x ** i
    return y

def gauti_ciobysevo_abscises(n: int) -> list[float]:
    x_s = []
    for i in range(n):
        x = cos(pi*(2*i + 1) / (2*n))
        x_s.append(x)
    return x_s

def konvertuoti_is_ciobysevo(abscises: list[float], a: float, b: float) -> list[float]:
    X_s = []
    for x in abscises:
        X = (a+b)/2 + (b-a)/2*x
        X_s.append(X)
    return X_s

def gauti_vienanariu_paklaidas(F, x_s: list[float], vienanariai):
    paklaidos = []
    for x in x_s:
        interpoliuotas_y = gauti_y_is_vienanariu(vienanariai, x)
        tikras_y = F(x)
        paklaidos.append(abs(interpoliuotas_y - tikras_y))
    return paklaidos

def plot_vienanariu_palyginima(F, x_s: list[float], vienanariai):
    plt.plot(x_s, list(F(x) for x in x_s), color="blue", label="Tikrasis")

    y_interpoliuotas = list(gauti_y_is_vienanariu(vienanariai, x) for x in x_s)
    plt.plot(x_s, y_interpoliuotas, color="red", label="Interpoliuotas")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpoliuotas vs originalus")

def plot_vienanariu_paklaida(F, x_s: list[float], vienanariai):
    y_paklaida = gauti_vienanariu_paklaidas(F, x_s, vienanariai)
    plt.plot(x_s, y_paklaida, color="red")
    plt.xlabel("x")
    plt.ylabel("paklaida")
    plt.title("Paklaida")

def main(F, x_range, mazgu_kiekis, tarpiniai_taskai):
    from_x, to_x = x_range

    if True:
        mazgai = gauti_xy_taskus(F, from_x, to_x, mazgu_kiekis)
        vienanariai = gauti_vienanario_sprendinius(mazgai[0], mazgai[1])
        x_s = gauti_x_taskus(from_x, to_x, tarpiniai_taskai)

        plot_vienanariu_palyginima(F, x_s, vienanariai)
        #plot_vienanariu_paklaida(F, x_s, vienanariai)
        print("Paklaida:", sum(gauti_vienanariu_paklaidas(F, x_s, vienanariai)))

    if False:
        ciobysevo_abscises = gauti_ciobysevo_abscises(mazgu_kiekis)
        x_mazgai = konvertuoti_is_ciobysevo(ciobysevo_abscises, to_x, from_x)
        y_mazgai = list(F(x) for x in x_mazgai)
        vienanariai = gauti_vienanario_sprendinius(x_mazgai, y_mazgai)

        x_s = gauti_x_taskus(from_x, to_x, tarpiniai_taskai)
        plot_vienanariu_palyginima(F, x_s, vienanariai)
        #plot_vienanariu_paklaida(F, x_s, vienanariai)
        print("Paklaida:", sum(gauti_vienanariu_paklaidas(F, x_s, vienanariai)))

    plt.show()

main(
    F = lambda x: cos(2*x) * (sin(3*x) + 1.5) - cos(x/5),
    x_range = (-2, 3),
    mazgu_kiekis = 10,
    tarpiniai_taskai = 30
)