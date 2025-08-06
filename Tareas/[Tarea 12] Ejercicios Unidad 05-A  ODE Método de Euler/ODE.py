
from typing import Callable


def ODE_euler(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:
    """Resuelve (numéricamente) una EDO de la forma
        dy/dt = f(t, y)
            y(t_0) = y_t0, a <= t_0 <= b
    utilizando el método de Euler para los N+1 puntos del intervalo temporal [a, b].

    Genera N+1 puntos de malla con:
        t_i = a + i*h, h = (a - b) / N,
    donde h es el tamaño del paso.


    ## Parámetros
    ``a``: tiempo inicial
    ``b``: tiempo final
    ``f``: función de dos variables ``t`` y ``y``
    ``y_t0``: condición inicial
    ``N``: número de puntos de malla

    ## Devuelve
    ``ys``: lista de los N+1 valores aproximados de y
    ``ts``: lista de los N+1 puntos de malla
    ``h``: el tamaño de paso h

    """
    h = (b - a) / N
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        y += h * f(t, y)
        ys.append(y)

        t += h
        ts.append(t)
    return ys, ts, h



from math import factorial


def ODE_euler_nth(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    f_derivatives: list[Callable[[float, float], float]],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:
    """Resuelve (numéricamente) una EDO de la forma
        dy/dt = f(t, y)
            y(t_0) = y_t0, a <= t_0 <= b
    utilizando el método de Taylor con (m - 1)ª derivada para los N+1 puntos en el intervalo de tiempo [a, b].

    Genera N+1 puntos de malla con:
        t_i = a + i*h, h = (a - b) / N,
    donde h es el tamaño del paso.


    ## Parámetros
    ``a``: tiempo inicial
    ``b``: tiempo final
    ``f``: función de dos variables ``t`` y ``y``
    f_derivadas``: lista de (m - 1)th derivadas de f
    y_t0``: condición inicial
    ``N``: número de puntos de la malla

    ## Devuelve
    ``ys``: lista de los N+1 valores aproximados de y
    ``ts``: lista de los N+1 puntos de malla
    ``h``: el tamaño de paso h

    """
    h = (b - a) / N
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        T = f(t, y)
        ders = [
            h / factorial(m + 2) * mth_derivative(t, y)
            for m, mth_derivative in enumerate(f_derivatives)
        ]
        T += sum(ders)
        y += h * T
        ys.append(y)

        t += h
        ts.append(t)
    return ys, ts, h


import numpy as np

def interpolacion_lineal(y_1, y_2, x_val):
    """
    Realiza la interpolación lineal para un valor x_val dado los puntos (x, y).

    :param x: Lista o array de valores x conocidos.
    :param y: Lista o array de valores y conocidos.
    :param x_val: El valor x para el cual se desea interpolar el valor y.
    :return: El valor interpolado y correspondiente a x_val.
    """
    # Convertir listas a arrays numpy
    y_1 = np.array(y_1)
    y_2 = np.array(y_2)

    # Usar la función interp1d de scipy para la interpolación lineal
    from scipy.interpolate import interp1d

    # Crear la función de interpolación
    f = interp1d(y_1, y_2, kind='linear', fill_value='extrapolate')

    # Obtener el valor interpolado
    y_val = f(x_val)

    return y_val







