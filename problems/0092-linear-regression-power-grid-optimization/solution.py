import math

PI = 3.14159

def power_grid_forecast(consumption_data):
    n = len(consumption_data)
    x = list(range(1, n + 1))
    
    # 1) Убираем дневную флуктуацию
    detrended = []
    for i in range(n):
        fluct = 10 * math.sin(2 * PI * (i + 1) / 10)
        detrended.append(consumption_data[i] - fluct)
    
    # 2) Линейная регрессия
    sum_x = sum(x)
    sum_y = sum(detrended)
    sum_xy = sum(x[i] * detrended[i] for i in range(n))
    sum_x2 = sum(i ** 2 for i in x)
    
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    
    # 3) Предсказываем базовое потребление на день 15
    base_15 = m * 15 + b
    
    # 4) Добавляем флуктуацию для дня 15
    fluct_15 = 10 * math.sin(2 * PI * 15 / 10)
    pred_15 = base_15 + fluct_15
    
    # 5) Добавляем 5% запас и округляем вверх
    final = math.ceil(1.05 * round(pred_15))
    return final
