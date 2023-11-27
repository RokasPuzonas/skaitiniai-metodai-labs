from dataclasses import dataclass
from random import Random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

@dataclass
class Point:
    x: float
    y: float

    @staticmethod
    def rand(rand: Random, from_x: float, to_x: float, from_y: float, to_y: float):
        return Point(
            rand.uniform(from_x, to_x),
            rand.uniform(from_y, to_y)
        )

def gen_points(
        rand: Random,
        n: int,
        x_range = (-10, 10),
        y_range = (-10, 10)
    ):
    points = []
    while len(points) < n:
        point = Point.rand(rand, x_range[0], x_range[1], y_range[0], y_range[1])
        if point not in points:
            points.append(point)
    return points

def get_total_price(
        shops: list[Point],
        new_shops: list[Point],
        C,
        Cp
    ):
    total_price = 0
    for i, point in enumerate(new_shops):
        total_price += Cp(point.x, point.y)
        for other_point in shops:
            total_price += C(point.x, point.y, other_point.x, other_point.y)

        for j, other_point in enumerate(new_shops):
            if i == j: continue
            total_price += C(point.x, point.y, other_point.x, other_point.y)
    return total_price

def get_price_gradient(
        shops: list[Point],
        new_shops: list[Point],
        C,
        Cp,
        step = 0.001
    ):
    grad = []
    current_price = get_total_price(shops, new_shops, C, Cp)
    for i in range(len(new_shops)):
        new_shops[i].x += step
        new_price_x = get_total_price(shops, new_shops, C, Cp)
        new_shops[i].x -= step

        new_shops[i].y += step
        new_price_y = get_total_price(shops, new_shops, C, Cp)
        new_shops[i].y -= step

        grad.append(Point(
            new_price_x - current_price,
            new_price_y - current_price
        ))

    L = 0
    for grad_point in grad:
        L += grad_point.x**2 + grad_point.y**2
    L = L**0.5

    for grad_point in grad:
        grad_point.x /= L
        grad_point.y /= L

    return grad

def apply_gradient(points: list[Point], gradient: list[Point], step_size: float):
    for i, point in enumerate(points):
        point.x += step_size * gradient[i].x
        point.y += step_size * gradient[i].y

def gradient_descent(
        shops: list[Point],
        new_shops: list[Point],
        C,
        Cp,
        max_iterations = 100,
        step_size = 0.5,
        epsilon = 1e-6
    ):

    # iterations = []
    # price_over_iterations = []

    total_price = 1e10
    price_gradient = get_price_gradient(shops, new_shops, C, Cp)
    for iteration_idx in range(max_iterations):
        apply_gradient(new_shops, price_gradient, -step_size)

        new_total_price = get_total_price(shops, new_shops, C, Cp)
        # iterations.append(iteration_idx+1)
        # price_over_iterations.append(new_total_price)
        if abs(total_price - new_total_price) < epsilon:
            # plt.plot(iterations, price_over_iterations, linestyle='-', color='blue', label='Tikslo funkcija')
            # plt.legend()
            # plt.ylabel("Kaina")
            # plt.xlabel("Iteracija")

            # plt.show()
            # plt.waitforbuttonpress()
            return iteration_idx+1

        if total_price > new_total_price:
            total_price = new_total_price
        else:
            apply_gradient(new_shops, price_gradient, +step_size)
            price_gradient = get_price_gradient(shops, new_shops, C, Cp)
            step_size *= 0.9

    return -1

def main(
        N: list[Point],
        m: int,
        C,
        Cp,
        rand,
        max_iterations,
        step_size,
    ):
    shops = N
    new_shops = gen_points(rand, m)

    print("Starting price: ", get_total_price(shops, new_shops, C, Cp))

    iterations_used = gradient_descent(shops, new_shops, C, Cp, max_iterations, step_size)
    if iterations_used == -1:
        print("ERROR: Failed to reach minimum, not enough iterations")
        return

    print("Iterations: ", iterations_used)
    print("Minumum price: ", get_total_price(shops, new_shops, C, Cp))

    #plt.add_patch(patches.Rectangle((-10, -10), 20, 20, linestyle='--', facecolor='none', edgecolor='black'))
    plt.scatter(list(p.x for p in shops), list(p.y for p in shops), c="r", label="Esami")
    plt.scatter(list(p.x for p in new_shops), list(p.y for p in new_shops), c="b", label="Nauji")
    plt.legend()

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

# Variantas: 10
rand = Random(3)
main(
    rand = rand,
    N = gen_points(rand, 20),
    m = 20,
    C = lambda x1, y1, x2, y2: np.exp(-0.3 * ((x1 - x2)**2 + (y1 - y2)**2)),
    Cp = lambda x1, y1: (x1**4 + y1**4)/1000 + (np.sin(x1) + np.cos(y1))/5 + 0.4,

    max_iterations = 1000,
    step_size=0.5
)