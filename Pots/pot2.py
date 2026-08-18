"""
РЕШЕНИЕ ЗАДАЧИ ОБ ОСТЫВАНИИ ВТОРОЙ КАСТРЮЛИ (ЗАДАЧА 2)
"""
import matplotlib.pyplot as plt
import numpy as np 
from scipy.integrate import odeint
from scipy.optimize import fsolve

# === ПАРАМЕТРЫ ВТОРОЙ КАСТРЮЛИ ===
V = 3e-3
m = 2.0
c = 3800.0
T0 = 80.0
T1 = 30.0
T_env = 20.0

T_delta = 273.15

# === ГЕОМЕТРИЯ ПЕРВОЙ КАСТРЮЛИ ===
R = (V/(2*np.pi))**(1/3)
S = 5*np.pi*(R**2)


t_slice = np.linspace(0, 5000, 1000)

lambda_ = 0.1459983465350712

# === АНАЛИТИЧЕСКОЕ РЕШЕНИЕ ===
def T_analytical(t):
    """Аналитическое решение ОДУ с переменным коэффициентом k(T) = lambda*(T + T_delta)"""
    # Константа начальных условий
    X = (T0 - T_env) / (T0 + T_delta)
    
    # Эффективный показатель затухания
    mu = lambda_ * S * (T_env + T_delta) / (m * c)
    
    # Явная формула температуры
    exp_term = np.exp(-mu * t)
    T = (T_env + T_delta * X * exp_term) / (1 - X * exp_term)
    
    return T


def dT_dt(T,t):
    """Возвращает правую часть дифференциального уравнения"""
    term = -lambda_ * S / (m * c) * (T + T_delta) * (T - T_env)

    return term

def T_numeric(t=t_slice, func=dT_dt):
    """Численное решение дифференциального уравнения"""
    term = odeint(func=func, y0=T0, t=t_slice)

    return term


def find_T_numeric():
    """Поиск времени остывания до 30 градусов численно"""
    idx = np.where(T_numeric(t_slice) <= T1)[0]
    t_minute = t_slice[idx[0]]/60

    return t_minute

def find_time(t):
    """Возвращает невязку"""
    return T_analytical(t) - T1

def t_solution(func=find_time):
    """Поиск времени остывания до 30 градусов аналитически"""
    t_minute = fsolve(func, 1000)
    t_minute = t_minute[0]/60

    return t_minute


def main():
    T_numeric_target = find_T_numeric()
    print(f"Численно время: {T_numeric}")
    T_analytic_target = t_solution()
    print(f"Аналитически время: {T_analytic_target}")

    plt.figure(figsize=(10, 5))
    plt.plot(t_slice/60, T_numeric(t_slice), "g-", label="Численное решение", linewidth=4)
    plt.plot(t_slice/60, T_analytical(t_slice), "r--", label="Аналитическое решение", linewidth=4)
    plt.axhline(y=T1, color="gray", linewidth=2, linestyle="--", label="Цель 30°C")
    plt.axvline(x=t_solution(), color="green", linewidth=2, linestyle='--', label=f"время = {t_solution()}")
    plt.legend()

    plt.xlabel("Время (t)")
    plt.ylabel("Температура °C")

    plt.title("Сравнение аналитического и численного решений")
    plt.grid()
    plt.show()


main()