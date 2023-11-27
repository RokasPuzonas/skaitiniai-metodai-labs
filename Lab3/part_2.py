import csv
import matplotlib.pyplot as plt
import numpy as np

def lerp(x, min_x, max_x):
    return min_x + x * (max_x - min_x)

def remove_empty_years(emissions):
    empty_emissions = []
    for (year, emission) in emissions:
        if emission == None:
            empty_emissions.append((year, emission))

    for empty_entry in empty_emissions:
        emissions.remove(empty_entry)

def get_country_emissions(country: str):
    emissions_path = "API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_5995567.csv"

    with open(emissions_path, "r", newline="", encoding="utf-8") as f:
        # Skip first 4 lines
        for _ in range(4):
            f.readline()

        reader = csv.reader(f, delimiter=",", quotechar='"')
        header_row = next(reader)
        years = list(int(header) for header in header_row[4:] if header != '')

        for row in reader:
            if row[0] == country:
                emissions_str = row[4:4+len(years)]
                emissions = list(float(emission) if emission != "" else None for emission in emissions_str)
                emission_points = list(zip(years, emissions))
                remove_empty_years(emission_points)

                years     = list(row[0] for row in emission_points)
                emissions = list(row[1] for row in emission_points)
                return years, emissions

def lagrange_dx_2d(
        x,
        x_prev, y_prev,
        x_curr, y_curr,
        x_next, y_next):
    return (((x - x_curr) + (x - x_next)) / ((x_prev - x_curr) * (x_prev - x_next))) * y_prev + \
           (((x - x_prev) + (x - x_next)) / ((x_curr - x_prev) * (x_curr - x_next))) * y_curr + \
           (((x - x_prev) + (x - x_curr)) / ((x_next - x_prev) * (x_next - x_curr))) * y_next

def akima_derivatives(Xs: list[float], Ys: list[float]):
    assert len(Xs) == len(Ys)

    result = []
    N = len(Xs)
    for i in range(N):
        pivot = i
        if pivot == 0:
            pivot += 1
        elif pivot == N-1:
            pivot -= 1

        x = Xs[i]

        x_prev = Xs[pivot-1]
        y_prev = Ys[pivot-1]
        x_curr = Xs[pivot  ]
        y_curr = Ys[pivot  ]
        x_next = Xs[pivot+1]
        y_next = Ys[pivot+1]

        result.append(lagrange_dx_2d(x, x_prev, y_prev, x_curr, y_curr, x_next, y_next))
    return result

def draw_hermite_curve(X: list[float], Y: list[float], dY: list[float], scalar = 1):
    assert len(X) == len(Y) == len(dY)
    N = len(X)

    plt.plot(X[0], Y[0], 'ro')
    for i in range(N - 1):
        plot_x = np.linspace(X[i], X[i+1], scalar)
        plot_y = []

        plt.plot(X[i+1], Y[i+1], 'ro')
        d = X[i+1] - X[i]
        for k in range(scalar):
            s = plot_x[k] - X[i]
            U1 = 1 - 3 * (s**2 / d**2) + 2 * (s**3 / d**3)
            V1 = s - 2 * (s**2 / d) + (s**3 / d**2)
            U2 = 3 * (s ** 2 / d **2) - 2 * (s**3 / d ** 3)
            V2 = -1 * (s ** 2 / d) + (s ** 3 / d ** 2)

            f  = U1*Y[i  ] + V1*dY[i  ]
            f += U2*Y[i+1] + V2*dY[i+1]
            plot_y.append(f)
        plt.plot(plot_x, plot_y, 'b')
        plt.draw()
    plt.show()

def main(country, interpolation):
    years, emissions = get_country_emissions(country)

    X = np.array(years)
    Y = emissions
    dY = akima_derivatives(years, emissions)

    draw_hermite_curve(X, Y, dY, 2 + interpolation)

main(
    country = "Zambia",
    interpolation = 5
)