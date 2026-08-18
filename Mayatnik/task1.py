# 1. Задать известные параметры (l, g, α0, k)
# 2. Решить ОДУ численно (odeint)
# 3. Найти период и амплитуды из полученного решения
# 4. Сравнить с аналитической оценкой для малых углов (T = 2π√(l/g))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. Задать известные параметры (l, g, α0, k)
l = 1
g = 9.8
a0 = 0.3
a_diff_0 = 0 # производная от а(0)

INF = 2*np.pi*np.sqrt(l/g) # период математического маятника (оценка снизу)
SUP = np.sqrt(np.pi/2) * INF # период физического маятника (оценка сверху)


def analyticSolution(t):
    a_solution = a0 * np.cos(np.sqrt(g/l)*t) 
    return a_solution

def getDiffTerm(y, t):
    """
    Возвращает правую часть дифференциального уравнения
    для передачи в функцию numericSolution
    """
    # a" = -g/l * sin(a) =>
    # => a' = w;
    # w' = -g/l * sin(a)

    # начальные условия: alpha'(0) = 0, (omega = 0); alpha0 = 0.3 (alpha = 0.3)
    alpha, omega = y[0], y[1] # вектор начальных условий 
    dalpha_dt = omega
    domega_dt = (-g/l) * np.sin(alpha)

    return [dalpha_dt, domega_dt]

# 2. Решить ОДУ численно (odeint, EilerSolution)
def numericSolution(t_span, func=getDiffTerm):
    solValues = odeint(func, [a0, a_diff_0], t_span)

    return solValues

def EilerSolution(t_span):
    """
    Метод Эйлера-Кромера:
    возвращает массив углов
    """
    tau = t_span[1] - t_span[0]
    t_max = 20.0
    N = len(t_span)

    alpha = 0.3 # угол
    omega = 0.0 # (скорость) производная от alpha

    # массивы для сохранения истории
    t_vals = np.linspace(0, t_max, N)
    alpha_hist = np.zeros(N)
    omega_hist = np.zeros(N)

    for i in range(N):
        alpha_hist[i] = alpha
        
        dalpha_dt = omega
        domega_dt = -(g/l)*np.sin(alpha)

        omega = omega + tau*domega_dt
        alpha = alpha + tau*omega

    return alpha_hist

# === ПОИСК ПИКОВ С ПАРАБОЛИЧЕСКИМ УТОЧНЕНИЕМ ===
def findPeaksWithParabola(t_vals, alpha_hist):
    peaks_time = []
    amplitudes = []
    N = len(alpha_hist)

    for i in range(1, N - 1):
        if alpha_hist[i - 1]< alpha_hist[i] > alpha_hist[i + 1]:
            t1, t2, t3 = t_vals[i - 1], t_vals[i], t_vals[i + 1]
            a1, a2, a3 = alpha_hist[i - 1], alpha_hist[i], alpha_hist[i + 1]

            # составляющие формулы (для визуального упрощения вычислений)
            t_i = t2
            dt = t2 - t1
            denom = a3 - 2*a2 + a1 # знаменатель формулы

            # формула вершины параболы через три точки 
            t_peak = t_i - (dt/2.0) * (a3 - a1)/denom

            # формула для нахождения амплитуды
            denom = 8*(a1 - 2*a2 + a3)
            A_i = a2 - ((a3 - a1)**2) / denom

            amplitudes.append(A_i)
            peaks_time.append(t_peak)
    
    return np.array(peaks_time), np.array(amplitudes)

def showGraph(
        t_span, 
        analyticValues, 
        numericValues,  
        AlphaEilerValues,
        diffNumericMethods,
        A_physic, 
        T_physic
        ):
    """
    Строит график всех решений
    (Analytic, Odeint, Eiler)
    """
    fugure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.plot(t_span, analyticValues, "r--", linewidth=2, label= "Аналитическое решение (Математический маятник)")

    # ax1.plot(t_span, numericValues,"b-", linewidth=2, label= "Численное решение (scipy.inegrate.odeint)")
    
    ax1.plot(t_span, AlphaEilerValues, "g-", linewidth=2, label= "Эйлер-Кромер")
    ax1.set_xlabel("Время (с)")
    ax1.set_ylabel("a (угол отклонения (рад.)")
    ax1.set_title(f"Модель колебаний (Средняя амплитуда = {A_physic:.3f} рад)\nПериод: {T_physic:.4f}\nОценка периода физического маятника:\n{INF:.4f} < {T_physic:.4f} < {SUP:.4f}")
    ax1.legend()


    # Не обязательно, но ради интереса
    ax2.plot(t_span, diffNumericMethods, "r-", linewidth=2)
    ax2.set_xlabel("Время(с.)")
    ax2.set_ylabel("Погрешность (а рад)")
    ax2.set_title("\nРазность численных методов (odeint - EilerSolution)")

    plt.tight_layout(pad=0.3)
    plt.grid()
    plt.show()

def main():
    t_span = np.linspace(0, 20, 1000)
    analyticValues = analyticSolution(t_span)
    solValues = numericSolution(t_span)
    numericValues = solValues[:, 0] # извлекаем угол alpha, массив точек времени указан в showGraph() 

    # Эйлер
    AlphaEilerValues = EilerSolution(t_span)
    # разность методов
    diffNumericMethods = numericValues - AlphaEilerValues

    # параболическое уточнение
    peaks_time, amplitudes = findPeaksWithParabola(t_span, AlphaEilerValues)

    # 3. Найти период и амплитуды из полученного решения
    period = np.diff(peaks_time)
    T_physic = np.mean(period)

    A_physic = np.mean(amplitudes)

    # 4. Сравнить с аналитической оценкой для малых углов (T = 2π√(l/g))
    # если неверно код упадет (AssertionError)
    assert INF < T_physic < SUP

    showGraph(
        t_span, 
        analyticValues, 
        numericValues, 
        AlphaEilerValues, 
        diffNumericMethods,
        A_physic, 
        T_physic
        )

main()


