"""
РЕШЕНИЕ ЗАДАЧИ ОБ ОСТЫВАНИИ ПЕРВОЙ КАСТРЮЛИ (ЗАДАЧА 2)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# === ПАРАМЕТРЫ ПЕРВОЙ КАСТРЮЛИ ===

T_delta = 273.15

V = 5e-3
m = 4 
c = 4200
T0 = 90
delta_t = 2400
T1 = 50
T_env = 25

# === ГЕОМЕТРИЯ ПЕРВОЙ КАСТРЮЛИ ===
R = (V/(2*np.pi))**(1/3)
S = 5*np.pi*(R**2)

# === ИДЕНТИФИКАЦИЯ λ ===
# k = ( S*λ(T_env_1 + T_delta) )/ mc

log_term = (T1 - T_env)/(T1 + T_delta) * (T0 + T_delta)/(T0 - T_env)
T_env_K_1 = T_env + T_delta

lambda_ = (-np.log(log_term) * m * c) / (T_env_K_1 * S * delta_t)

# === ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ ===

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


def T_numeric(t_slice, func=dT_dt):
    """Численное решение дифференциального уравнения"""
    term = odeint(func=func,y0=T0, t=t_slice)

    return term


print(lambda_)

def main():
    t_slice = np.linspace(0, 5000, 500)
    # Строим график для аналитического решения
    plt.figure(figsize=(10, 5))
    plt.plot(t_slice/60, T_analytical(t_slice), 'b', label="Аналитическое решение", linewidth=2)
    plt.plot(t_slice/60, T_numeric(t_slice), 'r--', label="Численное решение", linewidth=2)

    plt.scatter([40], [50], color='green',zorder=2, s=100,label='Условие (40 мин, 50°C)')

    plt.xlabel("Время(мин.)")
    plt.ylabel("Температура (°C)")
    plt.grid()
    plt.legend()
    plt.title("Сравнение аналитического и численного решений")
    plt.show()

if __name__ == "__main__":
    main()