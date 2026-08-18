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

def EilerSolution():
    """
    Метод Эйлера-Кромера:
    возвращает массив углов
    """
    tau = 0.1
    t_max = 20.0
    N = 200

    alpha = 0.3 # угол
    omega = 0.0 # (скорость) производная от alpha

    # массивы для сохранения истории
    t_vals = np.linspace(0, t_max, N)
    alpha_hist = np.zeros(N)
    omega_hist = np.zeros(N)

    for i in range(N):
        t_vals[i] = i * tau # массив времени
        
        dalpha_dt = omega
        domega_dt = -(g/l)*np.sin(alpha)

        omega = omega + tau*domega_dt
        alpha = alpha + tau*omega

        alpha_hist[i] = alpha

    return [t_vals, alpha_hist]

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

def showGraph(t_span, analyticValues, numericValues, TimeEilerValues, AlphaEilerValues):
    plt.figure(figsize=(10, 5))
    plt.plot(t_span, analyticValues, "r-", linewidth=2, label= "Аналитическое решение (Математический маятник)")
    plt.plot(t_span, numericValues, "b-", linewidth=2, label= "Численное решение")
    plt.plot(TimeEilerValues, AlphaEilerValues, "g-", linewidth=2, label= "Эйлер-Кромер (Физический маятник)")
    plt.xlabel("Мин.(Время)")
    plt.ylabel("a (угол отклонения (рад.)")
    plt.title("Модель колебаний физического и математического маятников")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    t_span = np.linspace(0, 20, 1000)
    analyticValues = analyticSolution(t_span)
    solValues = numericSolution(t_span)
    numericValues = solValues[:, 0] # извлекаем угол alpha, массив точек времени указан в showGraph() 

    # Эйлер
    TimeEilerValues, AlphaEilerValues = EilerSolution()
    # параболическое уточнение
    peaks_time, amplitudes = findPeaksWithParabola(TimeEilerValues, AlphaEilerValues)
    # оценка периода физ. маятника
    period = np.diff(peaks_time)
    T_physic = np.mean(period)

    print(f"Оценка периода физического маятника:\n")
    print(f"{INF:.4f} < {T_physic:.4f} < {SUP:.4f}\n")

    A_physic = np.mean(amplitudes)

    print(f"Амплитуда: {A_physic:.4f} ")

    showGraph(t_span, analyticValues, numericValues, TimeEilerValues, AlphaEilerValues)

main()


