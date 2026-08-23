"""
ЧИСЛЕННОЕ РЕШЕНИЕ ЗАДАЧИ ГАЗЕТЧИКА
"""

import numpy as np 
import matplotlib.pyplot as plt

# определяем параметры 
alpha = 10 # используем равномерное распределение U~[alpha, betta]
betta = 100 

a_l = 2.0 # прибыль с одной газеты
b_l = 1.0  # убыток с одной газеты 

def monteKarlo(k_values, N) -> list[float]:
    """
    Возвращает средний доход для каждого k
    (Метод Монте-Карло)
    """
    # генерируем случайные числа спроса для каждого k (матрица)
    demand_matrix = np.random.uniform(alpha, betta, (len(k_values), N))
    
    k_more = k_values[:, np.newaxis] # размноженный k для операций с demand_matrix

    sold = np.minimum(k_more, demand_matrix) # продали
    unsold = np.maximum(0, k_more - demand_matrix) # не продали

    profit_matrix = a_l*sold - b_l*unsold

    M_stat = np.mean(profit_matrix, axis=1)

    return M_stat

def getParabolaCoeffs(k_values, M_stat):
    """Возвращает коэффициенты аппроксимирующей параболы"""
    column_stack = np.column_stack([k_values**2, k_values, np.ones_like(k_values)])
    coeffs, _, _, _ = np.linalg.lstsq(column_stack, M_stat)

    return coeffs

def minSquareMethod(coeffs, k_values):
    """
    Возвращает аппроксимированный параболой M_stat,
    используя метод наименьших квадратов
    """
    A, B, C = coeffs

    M_approx = A*k_values**2 + B*k_values + C

    return M_approx

def k_optApprox(coeffs):
    """Возвращает оптимальное количество газет для approxParabola"""
    A, B, _ = coeffs

    k_opt = -B/ (2*A)

    return k_opt

def M_optApprox(coeffs, k_opt):
    """Возвращает максимальную прибыль"""
    A, B, C = coeffs
    M_opt = A*k_opt**2 + B*k_opt + C

    return M_opt 

def showGraph(
        k_values, 
        M_stat,
        M_approx,
        k_opt_approx,
        M_opt_approx
        ):
    """Строит графики Монте-Карло и аппроксимирующей параболы"""
    plt.figure(figsize=(10, 6))

    plt.plot(k_values, M_stat, "b.", linewidth=1, label="Метод Монте-Карло")
    plt.plot(k_values, M_approx, "r-", linewidth=2, label="Аппроксимация параболой")

    # прерывистые линии для точки пика параболы (макс. прибыль M_opt)
    plt.axvline(
        x=k_opt_approx, 
        color="green", 
        linestyle="--", 
        label=f"Оптимальное кол-во газет(k_opt): {k_opt_approx}"
        )
    plt.axhline(y=M_opt_approx, color="green", linestyle="--")

    # рисуем точку для M_opt
    plt.scatter(
        x=k_opt_approx, 
        y=M_opt_approx, 
        zorder=5, 
        color="green", 
        label=f"Максимальная прибыль(M_opt): {M_opt_approx}", 
        linewidths=5
        )

    plt.xlabel("Объем закупки")
    plt.ylabel("M(k) - ожидаемый доход")
    
    plt.legend()

    plt.grid()

    plt.show()


def main():
    """Функция-оркестратор"""
    N: int = 100_000 # кол-во испытаний для метода Монте-Карло
    k_values: list = np.linspace(alpha, betta, 200)

    M_stat: list = monteKarlo(k_values, N)

    coeffs: list = getParabolaCoeffs(k_values, M_stat)

    M_approx: list = minSquareMethod(coeffs, k_values)
    k_opt_approx: float = k_optApprox(coeffs)
    M_opt_approx: float = M_optApprox(coeffs, k_opt_approx)

    showGraph(
        k_values, 
        M_stat,
        M_approx, 
        k_opt_approx,
        M_opt_approx
        )

main()