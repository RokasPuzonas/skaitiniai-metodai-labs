
from typing import Optional
import shapefile
from shapely import geometry
import numpy as np
import matplotlib.pyplot as plt

shape = shapefile.Reader("ne_10m_admin_0_countries.shp")
shape_records = shape.shapeRecords()

def find_country_id(name: str) -> Optional[int]:
    for i in range(len(shape_records)):
        feature = shape_records[i]
        if feature.record.NAME_EN == name:
            return i

def get_country_points(country_id: int):
    feature = shape_records[country_id]
    geo_interface = feature.shape.__geo_interface__
    coordinates = geo_interface['coordinates']

    if geo_interface['type'] == 'MultiPolygon':
        largest_area_idx = 0
        area = 0
        for idx in range(len(coordinates)):
            points = coordinates[idx][0]
            polygon_area = geometry.Polygon(points).area
            if polygon_area > area:
                area = polygon_area
                largest_area_idx = idx

        return coordinates[largest_area_idx][0]
    else:
        return coordinates[0]

def approximate(t, s, NL, m):
    sqrt_2 = np.sqrt(2)
    a, b = min(t), max(t)
    smooth = (b - a) * s * 2 ** (-NL / 2)

    details = []
    for _ in range(m):
        smooth_evens = smooth[0::2]
        smooth_odds  = smooth[1::2]

        details.append((smooth_evens - smooth_odds) / sqrt_2)
        smooth = (smooth_evens + smooth_odds) / sqrt_2

    return smooth, details

def haro_scaling(x, j, k, a, b):
    eps = 1e-9
    xtld = (x - a) / (b - a)
    xx = 2 ** j * xtld - k
    h = 2 ** (j / 2) * (np.sign(xx + eps) - np.sign(xx - 1 - eps)) / (2 * (b - a))
    return h

def haro_wavelet(x, j, k, a, b):
    eps = 1e-9
    xtld = (x - a) / (b - a)
    xx = 2 ** j * xtld - k
    h = 2 ** (j / 2) * (np.sign(xx + eps) - 2 * np.sign(xx - 0.5) + np.sign(xx - 1 - eps)) / (2 * (b - a))
    return h

def get_accuracy(new_x, new_y, original_x, original_y):
    x_err = 1/2*sum((new_x - original_x)**2)
    y_err = 1/2*sum((new_y - original_y)**2)
    return x_err + y_err

def main(country: str, levels: list[int], show_accuracy_plot):
    country_id = find_country_id(country)
    country_points = get_country_points(country_id)

    X, Y = [], []
    for (x, y) in country_points:
        X.append(x)
        Y.append(y)

    N = max(levels)
    nN = 2 ** N

    t = np.zeros(len(X))
    for i in range(1, len(X)):
        dx = X[i] - X[i-1]
        dy = Y[i] - Y[i-1]
        t[i] = t[i-1] + np.linalg.norm((dx, dy))

    t_min = t[0]
    t_max = t[-1]
    t_interp = np.linspace(t_min, t_max, nN)
    X_interp = np.interp(t_interp, t, X)
    Y_interp = np.interp(t_interp, t, Y)

    m = N

    smooth_x, det_x = approximate(t_interp, X_interp, N, m)
    smooth_y, det_y = approximate(t_interp, Y_interp, N, m)

    hx = np.zeros(nN)
    hy = np.zeros(nN)
    for k in range(2 ** (N - m)):
        scalar = haro_scaling(t_interp, N - m, k, t_min, t_max)
        hx += smooth_x[k] * scalar
        hy += smooth_y[k] * scalar

    accuracies = []
    for i in range(m):
        show_plot = (i+1 in levels)

        if show_plot: plt.plot(X, Y, 'r', label="Originalus")
        h1x = np.zeros(nN)
        h1y = np.zeros(nN)

        for k in range(2 ** (N - m + i)):
            wavelet = haro_wavelet(t_interp, N - m + i, k, t_min, t_max)
            h1x += det_x[m-i-1][k] * wavelet
            h1y += det_y[m-i-1][k] * wavelet

        hx += h1x
        hy += h1y

        accuracy = get_accuracy(hx, hy, X_interp, Y_interp)
        accuracies.append(accuracy)

        if show_plot:
            plt.plot(hx, hy, 'b')
            plt.plot((hx[0], hx[-1]), (hy[0], hy[-1]), 'b', label="Aproksimuotas")
            plt.title(f"{i+1}/{N} detalumo lygis")
            plt.legend()
            plt.show()

    if show_accuracy_plot:
        plt.plot(range(1,m+1), accuracies)
        plt.xlabel("Detalumo lygis")
        plt.ylabel("Paklaida")
        plt.title("Paklaidos priklausomybė nuo detalumo lygio")
        plt.show()

main(
    country = "Zambia",
    levels = [1, 4, 8, 10],
    show_accuracy_plot = False,
)