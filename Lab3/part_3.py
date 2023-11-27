import csv
import numpy as np
import matplotlib.pyplot as plt

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

def approximate(X, Y, degree, scalar = 1):
    degree += 1

    G = np.zeros((len(X), degree), dtype=float)
    for i in range(degree):
        G[:, i] = np.power(X, i)

    coefficients = np.linalg.solve(
        np.dot(np.transpose(G), G),
        np.dot(np.transpose(G), Y)
    )

    approx_x = np.linspace(X[0], X[-1], len(X) * scalar);
    approx_y = np.zeros(approx_x.size, dtype=float)
    for i in range(degree):
        approx_y += np.power(approx_x, i) * coefficients[i];

    return approx_x, approx_y

def main(country, degrees, scale):
    years, emissions = get_country_emissions(country)

    X = np.array(years, dtype=float)
    Y = np.array(emissions)

    for degree in degrees:
        approx_x, approx_y = approximate(X, Y, degree, scale)
        plt.plot(approx_x, approx_y, 'b')

        for i in range(len(X)):
            plt.plot(X[i], Y[i], 'ro')

        plt.title(f"{degree} laipsnio")
        plt.show()

main(
    country = "Zambia",
    degrees = [1, 2, 3, 5],
    scale = 10
)