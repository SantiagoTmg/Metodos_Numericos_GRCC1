import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import lambdify, Symbol
import sympy as sym
from IPython.display import display


# ###################################################################
def cubic_spline_clamped(
    xs: list[float], ys: list[float], B0: float, B1: float
) -> list[sym.Symbol]:
    """
    Cubic spline interpolation ``S``. Every two points are interpolated by a cubic polynomial
    ``S_j`` of the form ``S_j(x) = a_j + b_j(x - x_j) + c_j(x - x_j)^2 + d_j(x - x_j)^3.``

    xs must be different  but not necessarily ordered nor equally spaced.

    ## Parameters
    - xs, ys: points to be interpolated
    - B0, B1: derivatives at the first and last points

    ## Return
    - List of symbolic expressions for the cubic spline interpolation.
    """

    points = sorted(zip(xs, ys), key=lambda x: x[0])  # sort points by x
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    n = len(points) - 1  # number of splines
    h = [xs[i + 1] - xs[i] for i in range(n)]  # distances between  contiguous xs

    alpha = [0] * (n + 1)  # prealloc
    alpha[0] = 3 / h[0] * (ys[1] - ys[0]) - 3 * B0
    alpha[-1] = 3 * B1 - 3 / h[n - 1] * (ys[n] - ys[n - 1])

    for i in range(1, n):
        alpha[i] = 3 / h[i] * (ys[i + 1] - ys[i]) - 3 / h[i - 1] * (ys[i] - ys[i - 1])

    l = [2 * h[0]]
    u = [0.5]
    z = [alpha[0] / l[0]]

    for i in range(1, n):
        l += [2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * u[i - 1]]
        u += [h[i] / l[i]]
        z += [(alpha[i] - h[i - 1] * z[i - 1]) / l[i]]

    l.append(h[n - 1] * (2 - u[n - 1]))
    z.append((alpha[n] - h[n - 1] * z[n - 1]) / l[n])
    c = [0] * (n + 1)  # prealloc
    c[-1] = z[-1]

    x = sym.Symbol("x")
    splines = []
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - u[j] * c[j + 1]
        b = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d = (c[j + 1] - c[j]) / (3 * h[j])
        a = ys[j]
        print(j, a, b, c[j], d)
        S = a + b * (x - xs[j]) + c[j] * (x - xs[j]) ** 2 + d * (x - xs[j]) ** 3

        splines.append(S)
    splines.reverse()
    return splines



def animate_cubic_spline_B1_variation(xs, ys, B0, B1_values, interval=100):
    """
    Crea una animación mostrando cómo varían los splines cúbicos al cambiar B1.
    
    Args:
        xs: Lista de coordenadas x de los puntos
        ys: Lista de coordenadas y de los puntos
        B0: Valor fijo de la derivada en el primer punto
        B1_values: Rango de valores para B1 (derivada en el último punto)
        interval: Intervalo entre frames en milisegundos
    """
    # Ordenar los puntos
    sorted_points = sorted(zip(xs, ys), key=lambda x: x[0])
    sorted_xs = [x for x, _ in sorted_points]
    sorted_ys = [y for _, y in sorted_points]
    
    # Configurar la figura
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(min(sorted_xs)-0.5, max(sorted_xs)+0.5)
    ax.set_ylim(min(sorted_ys)-2, max(sorted_ys)+2)
    ax.grid(True)
    ax.set_title('Variación de splines cúbicos al cambiar B1')
    ax.set_xlabel('x')
    ax.set_ylabel('S(x)')
    
    # Puntos de interpolación
    scat = ax.scatter(sorted_xs, sorted_ys, color='red', s=100, zorder=3, label='Puntos de interpolación')
    
    # Líneas para los splines
    lines = []
    x_sym = Symbol('x')
    n_intervals = len(sorted_xs) - 1
    for _ in range(n_intervals):
        line, = ax.plot([], [], lw=2)
        lines.append(line)
    
    # Texto para mostrar el valor de B1
    b1_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.8))
    
    # Función de inicialización
    def init():
        for line in lines:
            line.set_data([], [])
        b1_text.set_text('')
        return lines + [b1_text, scat]
    
    # Función de animación
    def update(frame):
        B1 = B1_values[frame]
        splines = cubic_spline_clamped(xs=sorted_xs, ys=sorted_ys, B0=B0, B1=B1)
        
        for j, (S, (x_start, x_end)) in enumerate(zip(splines, zip(sorted_xs[:-1], sorted_xs[1:]))):
            x_range = np.linspace(x_start, x_end, 100)
            S_func = lambdify(x_sym, S, 'numpy')
            y_vals = S_func(x_range)
            lines[j].set_data(x_range, y_vals)
            lines[j].set_label(f'S_{j}(x), B1={B1:.1f}')
        
        b1_text.set_text(f'B1 = {B1:.2f}')
        ax.legend(loc='upper left')
        return lines + [b1_text, scat]
    
    # Crear la animación
    ani = FuncAnimation(fig, update, frames=len(B1_values),
                        init_func=init, blit=True, interval=interval)
    
    plt.close()
    return ani

# Parámetros de ejemplo
xs = [0, 1, 2]
ys = [1, 5, 3]
B0 = 1
B1_values = np.linspace(-20, 20, 40)  # Rango de valores para B1

# Generar la animación
ani = animate_cubic_spline_B1_variation(xs, ys, B0, B1_values, interval=200)

# Mostrar la animación (en Jupyter Notebook)
from IPython.display import HTML
HTML(ani.to_jshtml())

ani.save('Variacion_de_b1.gif', writer='pillow', fps=10)