import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import Iterable
import tkinter as tk

EPSILON = 1e-12

def ScrollTextBox(w,h):
    """
    sukuria TextBox su scrolais pagal w ir pagal h

    w ir h _ plotis ir aukstis

    Grazina textBoxa T

    Programoje textBox sukuriamas:     T=ScrollTextBox(140,20)

    Irasoma komandomis :   T.insert(END,str+"\n");  T.yview(END)
    """
    root = tk.Tk(); root.title("Scroll Text Box")
    frame1=tk.Frame(root); frame1.pack()
    #T = ScrolledText(root, height=h, width=w,wrap=WORD)  # jeigu reikia scrolinti tik pagal h
    scrollbarY = tk.Scrollbar(frame1, orient='vertical'); scrollbarY.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbarX = tk.Scrollbar(frame1, orient='horizontal'); scrollbarX.pack(side=tk.BOTTOM, fill=tk.X)
    T = tk.Text(frame1, width = w, height = h, wrap = tk.NONE,yscrollcommand = scrollbarY.set,xscrollcommand = scrollbarX.set)
    T.pack();
    scrollbarX.config(command = T.xview); scrollbarY.config(command = T.yview)
    return T

def SpausdintiMatrica(*args):
    """
    A - matrica

    T - TextBox

    str - eilute pradzioje, str=, neprivaloma
    """
    A=args[0]; T=args[1]; str="";
    if len(args) == 3: str=args[2];
    siz=np.shape(A)
    T.insert(tk.END,"\n"+str+"=")
    if len(siz) > 0:
        for i in range (0,siz[0]):
            T.insert(tk.END,"\n")
            if len(siz) > 1:
               for j in range (0,siz[1]):  T.insert(tk.END,"%12g   "%A[i,j]);
            else: T.insert(tk.END,"%12g   "%A[i])
    else : T.insert(tk.END,"%12g   "%A);
    T.yview(tk.END)
    T.update()

def Pavirsius(X, Y,LFF):
    """
    X,Y - meshgrid masyvai

    LF - dvieju kintamuju vektorines funkcijos vardas, argumentas paduodamas vektoriumi, isejimas vektorius ilgio 2

    rezultatas - dvigubas meshgrid masyvas Z[:][:][0:1]
    """
    siz=np.shape(X)
    Z=np.zeros(shape=(siz[0],siz[1],2))
    for i in range (0,siz[0]):
        for j in range (0,siz[1]):  Z[i,j,:]=LFF([X[i][j],Y[i][j]]).transpose();
    return Z

@dataclass
class BroidenoRezultatas:
    x1: float
    x2: float
    y1: float
    y2: float
    tikslumas: float
    iteracija: int

@dataclass
class Tinklelis:
    from_x1: float
    to_x1: float
    from_x2: float
    to_x2: float
    density: int

def gauti_bendra_lygti(Z1, Z2):
    def LF(x):  # grazina reiksmiu stulpeli
        s = np.array([
            Z1(x[0], x[1]),
            Z2(x[0], x[1]),
        ])
        s.shape=(2,1)
        return np.matrix(s)

    return LF

def gauti_tiksluma(x1, x2, f1, f2):
    x_abs_diff = np.abs(x1-x2)
    x_abs_sum = x1+x2
    f1_abs = np.abs(f1)
    f2_abs = np.abs(f2)
    if not np.isscalar(x1):
        x_abs_diff = sum(x_abs_diff)
        x_abs_sum = sum(x_abs_sum)
        f1_abs = sum(f1_abs)
        f2_abs = sum(f2_abs)

    if x_abs_sum > EPSILON:
        return x_abs_diff/(x_abs_sum + f1_abs + f2_abs)
    else:
        return x_abs_diff + f1_abs + f2_abs

def broideno_metodas_iter(
        init_x1: float,
        init_x2: float,
        Z1,
        Z2,
    ) -> Iterable[BroidenoRezultatas]:

    MAX_ITERATIONS = 80 # didziausias leistinas iteraciju skaicius
    JACOB_DX = 0.1 # dx pradiniam Jakobio matricos iverciui

    if init_x1 == 0 and init_x2 == 0:
        return

    n = 2 # lygciu skaicius
    x = np.matrix(np.zeros(shape=(n,1)))
    x[0] = init_x1
    x[1] = init_x2

    LF = gauti_bendra_lygti(Z1, Z2)

    A = np.matrix(np.zeros(shape=(n,n)))
    x1 = np.zeros(shape=(n,1))
    for i in range(0,n):
        x1 = np.matrix(x)
        x1[i] += JACOB_DX
        A[:,i] = (LF(x1) - LF(x))/JACOB_DX

    ff = LF(x)

    for i in range(1, MAX_ITERATIONS+1):
        deltax = -np.linalg.solve(A,ff)
        x1 = np.matrix(x + deltax)
        ff1 = LF(x1)
        A += (ff1 - ff - A*deltax)*deltax.transpose()/(deltax.transpose() * deltax)
        tiksl = gauti_tiksluma(x, x1, ff, ff1)
        ff = ff1
        x = x1

        yield BroidenoRezultatas(
            x1[0, 0],
            x1[1, 0],
            ff1[0, 0],
            ff1[1, 0],
            tiksl[0, 0],
            i
        )

def broideno_metodas(
        init_x1: float,
        init_x2: float,
        Z1,
        Z2,
    ):
    for rez in broideno_metodas_iter(init_x1, init_x2, Z1, Z2):
        if rez.tikslumas < EPSILON:
            return rez

def viz_broideno_metodas(
        init_x1: float,
        init_x2: float,
        Z1,
        Z2,
        view_x1 = (-8, 8),
        view_x2 = (-8, 8)
    ):

    T = ScrollTextBox(100,20) # sukurti teksto isvedimo langa
    T.insert(tk.END,"Broideno metodas")

    #------ Grafika: funkciju LF pavirsiai -----------------------------------------------------------------------
    fig1 = plt.figure(1, figsize=plt.figaspect(0.5))

    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('Z')

    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    plt.draw()  #plt.pause(1);
    xx = np.linspace(view_x1[0], view_x1[1], 20)
    yy = np.linspace(view_x2[0], view_x2[1], 20)
    X, Y = np.meshgrid(xx, yy)
    Z = Pavirsius(X, Y, gauti_bendra_lygti(Z1, Z2))

    # surf1 = ax1.plot_surface(X, Y, Z[:,:,0], color='blue', alpha=0.4, linewidth=0.1, antialiased=True)
    wire1 = ax1.plot_wireframe(X, Y, Z[:,:,0], color='black', alpha=1, linewidth=1, antialiased=True)
    surf2 = ax1.plot_surface(X, Y, Z[:,:,1], color='purple', alpha=0.4, linewidth=0.1, antialiased=True)

    CS11  = ax1.contour(X, Y, Z[:,:,0],[0], colors='b')
    CS12  = ax1.contour(X, Y, Z[:,:,1],[0], colors='g')
    CS1   = ax2.contour(X, Y, Z[:,:,0],[0], colors='b')
    CS2   = ax2.contour(X, Y, Z[:,:,1],[0], colors='g')

    XX = np.linspace(-5,5,2)
    YY = XX
    XX, YY = np.meshgrid(XX, YY)
    ZZ = XX*0
    zeroplane = ax2.plot_surface(XX, YY, ZZ, color='gray', alpha=0.4, linewidth=0, antialiased=True)
    #---------------------------------------------------------------------------------------------------------------

    init_y1 = Z1(init_x1, init_x2)
    init_y2 = Z2(init_x1, init_x2)
    ax1.plot3D([init_x1, init_x1], [init_x2, init_x2], [0, init_y1], "m*-")
    plt.draw()
    plt.pause(1)

    prev_x = (init_x1, init_x2)
    prev_y = (init_y1, init_y2)
    final_rez = None
    for rez in broideno_metodas_iter(init_x1, init_x2, Z1, Z2):
        SpausdintiMatrica(rez.tikslumas, T, "tiksl")
        if rez.tikslumas < EPSILON:
            final_rez = rez
            break

        SpausdintiMatrica(np.matrix([[rez.x1], [rez.x2]]), T, "x1")

        #------ Grafika:  -----------------------------------------------------------------------------------------------
        ax1.plot3D([rez.x1,rez.x1], [rez.x2,rez.x2], [0     ,0     ], "ro-")  # reikia prideti antra indeksa, kadangi x yra matrica
        ax1.plot3D([rez.x1,rez.x1], [rez.x2,rez.x2], [rez.y1,rez.y1], "c-.")
        ax1.plot3D([rez.x1,rez.x1], [rez.x2,rez.x2], [0     ,rez.y1], "m*-")
        ax2.plot3D([rez.x1,rez.x1], [rez.x2,rez.x2], [0     ,0     ], "ro-")
        plt.draw()
        plt.pause(2)
        #---------------------------------------------------------------------------------------------------------------
        prev_x = (rez.x1, rez.x2)
        prev_y = (rez.y1, rez.y2)

    #------ Grafika:  ---------------------------------------
    ax1.plot3D([prev_x[0],prev_x[0]], [prev_y[1],prev_y[1]], [0,0], "ks")
    ax2.plot3D([prev_x[0],prev_x[0]], [prev_y[1],prev_y[1]], [0,0], "ks")
    plt.draw()
    plt.pause(1)
    #--------------------------------------------------------

    assert final_rez
    SpausdintiMatrica(np.matrix([[final_rez.x1], [final_rez.x2]]), T, "Sprendinys")
    SpausdintiMatrica(final_rez.tikslumas, T, "Galutinis tikslumas")

    # print("Plotting pre-finished")
    # plt.show()
    # print("Plotting finished")

def iter_tinklelis(tinklelis: Tinklelis):
    for x1_idx in range(tinklelis.density):
        x1 = lerp(tinklelis.from_x1, tinklelis.to_x1, (x1_idx + 0.5)/tinklelis.density)
        for x2_idx in range(tinklelis.density):
            x2 = lerp(tinklelis.from_x2, tinklelis.to_x2, (x2_idx + 0.5)/tinklelis.density)

            yield (x1, x2)

def viz_broideno_tinklelis(
        Z1,
        Z2,
        tinklelis: Tinklelis,
        precision = 5,
        circle_colors = [
            (0.1, 0.1, 0.1),
            (0.1, 0.8, 0.8),
            (0.1, 0.8, 0.1),
            (0.1, 0.1, 0.8),
            (0.8, 0.8, 0.1),
            (0.8, 0.1, 0.8),
            (0.4, 0.4, 0.4)
        ]
    ):

    tinklelio_reiksmes = {}
    sprendiniai = []
    for x1, x2 in iter_tinklelis(tinklelis):
        broideno_rez = broideno_metodas(x1, x2, Z1, Z2)
        if broideno_rez:
            rez_x1 = round(broideno_rez.x1, precision)
            rez_x2 = round(broideno_rez.x2, precision)

            tinklelio_reiksmes[(x1, x2)] = (rez_x1, rez_x2)
            if (rez_x1, rez_x2) not in sprendiniai:
                sprendiniai.append((rez_x1, rez_x2))
        else:
            tinklelio_reiksmes[(x1, x2)] = None

    print(sprendiniai)

    fig1 = plt.figure(1)

    ax = fig1.add_subplot(1, 1, 1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(tinklelis.from_x1, tinklelis.to_x1)
    ax.set_ylim(tinklelis.from_x2, tinklelis.to_x2)
    plt.draw()

    xx = np.linspace(tinklelis.from_x1, tinklelis.to_x1, tinklelis.density)
    yy = np.linspace(tinklelis.from_x2, tinklelis.to_x2, tinklelis.density)
    X, Y = np.meshgrid(xx, yy)
    Z = Pavirsius(X, Y, gauti_bendra_lygti(Z1, Z2))

    ax.contour(X, Y, Z[:,:,0],[0], colors=[(0.8, 0.4, 0.1, 1)], linewidths=4.0)
    ax.contour(X, Y, Z[:,:,1],[0], colors=[(0.8, 0.1, 0.4, 1)], linewidths=4.0)

    assert (len(sprendiniai)+1) <= len(circle_colors)

    circle_width = (tinklelis.to_x1 - tinklelis.from_x1) / tinklelis.density / 2
    circle_height = (tinklelis.to_x2 - tinklelis.from_x2) / tinklelis.density / 2
    circle_radius = min(circle_width, circle_height) * 0.85

    for i in range(len(sprendiniai)):
        x1, x2 = sprendiniai[i]
        pos = np.array([x1, x2])
        star_vertices = np.array([
            [0, 1], [0.3, 0.3], [1, 0.3], [0.45, 0], [0.6, -0.7],
            [0, -0.35], [-0.6, -0.7], [-0.45, 0], [-1, 0.3], [-0.3, 0.3]
        ]) * (circle_radius*1.5)
        star = patches.Polygon(star_vertices + pos, closed=True, edgecolor='black', facecolor=circle_colors[i+1], zorder=5)
        ax.add_patch(star)

    for x1, x2 in iter_tinklelis(tinklelis):
        sprendinys_idx = 0
        if tinklelio_reiksmes[(x1, x2)] is not None:
            sprendinys_idx = sprendiniai.index(tinklelio_reiksmes[(x1, x2)])+1

        circle_color = circle_colors[sprendinys_idx]

        ax.add_patch(patches.Circle((x1, x2), circle_radius, edgecolor=circle_color, facecolor=circle_color))

    plt.show()

def lerp(min_x, max_x, percent):
    return min_x + (max_x - min_x) * percent

def main(Z1, Z2, tinklelis: Tinklelis):
    viz_broideno_metodas(2.5, 0.3, Z1, Z2)

    viz_broideno_tinklelis(Z1, Z2, tinklelis)

    print(broideno_metodas(-4, 0, Z1, Z2))
    print(broideno_metodas( 4, 0, Z1, Z2))
    print(broideno_metodas(-2, 4, Z1, Z2))
    print(broideno_metodas( 2, 4, Z1, Z2))


# Variantas: 10
main(
    lambda x1, x2: x1**2 + 2*(x2 - np.cos(x1))**2 - 20,
    lambda x1, x2: x1**2 * x2 - 2,
    tinklelis = Tinklelis(
        from_x1=-8, to_x1=8,
        from_x2=-8, to_x2=8,
        density=60
    )
)