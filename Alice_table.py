import pandas as pd
import numpy as np
from numpy import pi, cos, sin, sqrt, mean
from math import atan2, floor
from numpy.linalg import norm


def adc(voltage_value: float, voltage_range=2, resolution=12) -> str:
    """АЦП Алисы. Возвращает битовую строку длины resolution, которая соответсвует оцифрованному входному значению
    При границах (-voltage_range / 2, voltage_range / 2)
    Входное значение: значение амплитуды
    Выходное значение: битовая строка"""
    grid = np.linspace(-voltage_range / 2, voltage_range / 2, 2 ** resolution - 1)
    left = 0
    right = len(grid)
    if voltage_value < grid[0]:
        return '0' * resolution
    while right > left + 1:
        center = floor((left + right) / 2)
        if voltage_value >= grid[center]:
            left = center
        else:
            right = center
    return '0' * (resolution - len(bin(right)[2:])) + bin(right)[2:]


def rotate_matrix(bob_ref_q, bob_ref_p, alice_ref_1=1, alice_ref_2=0):
    """Функция создает матрицу поворота состояний Алисы, чтобы получить состояния Боба
    В референсе два импульса: Алиса передает состояния (1,0) Боб первый импульс проецирует на Q квадратуру,
    второй импульс - проецирует на квадратуру P
    bob_ref_q - значение первого импульса, измеренного Бобом, фаза Алисы - 0, фаза Боба - 0
    bob_ref_p - значение второго импульса, измеренного Бобом, фаза Алисы - 0, фаза Боба - pi/2"""
    k = sqrt(bob_ref_q ** 2 + bob_ref_p ** 2)  # нормировочный коэффициент

    X = bob_ref_p * alice_ref_1 - bob_ref_q * alice_ref_2
    Y = bob_ref_q * alice_ref_1 + bob_ref_p * alice_ref_2
    teta = atan2(X, Y)
    matrix = np.array([[cos(teta), -sin(teta)],
                       [sin(teta), cos(teta)]])
    # print(matrix)
    # print(k)
    return matrix, k


def rotate_alice_signals(rotate_matr, norm_coeff, q_a, p_a):
    """Поворот состояния в фазовой плоскости
    Входные значения:
    - rotate_matr: матрица поворота (класс numpy.ndarray)
    - norm_coeff: нормировочный коэффициент (float)
    - q_a, q_b: значения квадратур Алисы (float)"""
    # return sqrt(norm_coeff) * np.dot(rotate_matr, np.array([[q_a], [p_a]]))
    return norm_coeff * np.dot(rotate_matr, np.array([[q_a], [p_a]]))


Alice_table = {
    'Ref to signal ratio': [],  # отношение количества опорных импульсов к сигнальным в рамках цикла
    'Pulse type': [],  # тип конкретного импульса (опорный или сигнальный)
    'Amplitude': [],  # амплитуда импульса
    'Phase': [],  # фаза импульса
    'Q': [],  # Значение Q квадратуры
    'P': [],  # Значение P квадратуры
    'Bob\'s quadrature': [],  # Измеренная квадратура Боба (Q или P квадратура)
    'Bob\'s quadrature value': [],  # Измеренное значение квадратуры Боба
    'alpha': [],  # Угол доворота состояния Алисы
    'T_eff': [],  # Коэффициент пропускания канала
    'Rotated Q': [],  # Значение Q квадратуры Алисы после фазовой постройки
    'Rotated P': [],  # Значение P квадратуры Алисы после фазовой постройки
    'Ultimate state': [],  # Конечное значение состояния Алисы
    'Alice bits': [],  # Оцифрованное конечное значения состояния Алисы
    'Linear difference': [],  # Разница между значениями квадратур Алисы и Боба
}
Bob_table = pd.read_csv('Bob table.csv')  # загрузка данных о сигналах, зарегистрированных Получателем
# print(Bob_table)

# print(Alice_table)
list_for_diff = []
signal_to_ref_ratio = 2
# __________________________________________________________________________________________________________________
count_ref_signal = (2, 2 * signal_to_ref_ratio)
# __________________________________________________________________________________________________________________

# __________________________________________________________________________________________________________________
# number_of_cycles = int(len(Bob_table['Quadrature']) / sum(count_ref_signal))
# __________________________________________________________________________________________________________________
index_step = sum(count_ref_signal)
# p = 0
# for p in range(number_of_cycles):
teta_first_cycle = atan2(Bob_table['Value'][1], Bob_table['Value'][0])
alpha_first_cycle = Bob_table['Value'][0] / cos(teta_first_cycle)
teta_am = atan2(Bob_table['Value'][3], Bob_table['Value'][2]) - teta_first_cycle
amplitude_mod_att = Bob_table['Value'][2] / (alpha_first_cycle * cos(teta_am + teta_first_cycle))
teta_am = teta_am + pi if amplitude_mod_att < 0 else teta_am
amplitude_mod_att = abs(amplitude_mod_att)
# print(teta_first_cycle)
# print(teta_am)
# print(amplitude_mod_att)
# print(alpha_first_cycle)
cycle_number = 55
for p in range(cycle_number):
    Alice_table['Bob\'s quadrature'] += Bob_table['Quadrature'][p * index_step:(p + 1) * index_step].tolist()
    # Alice_table['Bob\'s quadrature'] += ['Q', 'P', 'Q', 'P', 'Q', 'P']
    Alice_table['Bob\'s quadrature value'] += [item for item in
                                               Bob_table['Value'][p * index_step:(p + 1) * index_step].tolist()]
    Alice_table['Ref to signal ratio'] = Alice_table['Ref to signal ratio'] + \
                                         [count_ref_signal[0], count_ref_signal[1]] + \
                                         ['-' for i in range(signal_to_ref_ratio * 2)]

    # ______________________________________________________________________________________________________________
    Alice_table['Phase'] += [0, 0]
    AM_phase = teta_am
    for summ in range(signal_to_ref_ratio):
        Alice_table['Phase'] += [0 + AM_phase, 3 * pi / 2 + AM_phase]
    # ______________________________________________________________________________________________________________

    # задание значений амплитуд и фаз состояний Алисы

    # ______________________________________________________________________________________________________________
    ALice_amplitude_ref = 1
    # ______________________________________________________________________________________________________________

    Alice_table['Amplitude'] += [ALice_amplitude_ref for l in range(count_ref_signal[0])]

    # starting_index = p * sum(Alice_table['Ref to signal ratio'][:2])
    # рассчет значений квадратур

    Alice_table['Pulse type'] += ['ref' for i in range(Alice_table['Ref to signal ratio'][0])] + \
                                 ['signal' for i in range(Alice_table['Ref to signal ratio'][1])]

    """
    if p == 1:
        with open('Alice table.json', 'w') as file:
            json.dump(Alice_table, file, indent=4)
        exit()
    """
    # рассчет угла поворота и пропускания канала

    Y = float(Alice_table['Bob\'s quadrature value'][1 + p * index_step]) \
        #    * Alice_table['Q'][p * index_step]  # - \
    # float(Alice_table['Bob\'s quadrature value'][p * index_step]) \
    # * Alice_table['P'][1 + p * index_step]

    X = float(Alice_table['Bob\'s quadrature value'][p * index_step]) \
        #   * Alice_table['Q'][p * index_step]  # + \
    # float(Alice_table['Bob\'s quadrature value'][1 + p * index_step]) \
    # * Alice_table['P'][1 + p * index_step]

    alpha = atan2(Y, X)
    T_eff = norm(
        np.array([[Alice_table['Bob\'s quadrature value'][p * index_step]],
                  [Alice_table['Bob\'s quadrature value'][1 + p * index_step]]]))

    for i in range(index_step):
        Alice_table['alpha'].append(alpha)
        Alice_table['T_eff'].append(T_eff)
    matrix, norm_coeff = rotate_matrix(float(Alice_table['Bob\'s quadrature value'][p * index_step]),
                                       float(Alice_table['Bob\'s quadrature value'][1 + p * index_step]))
    # print(T_eff, norm_coeff)

    A_sig = amplitude_mod_att * ALice_amplitude_ref  # * 0.8899125613297986
    for i in range(count_ref_signal[1]):
        Alice_table['Amplitude'].append(A_sig)

    for i, item in enumerate(Alice_table['Amplitude'][p * index_step:(p + 1) * index_step]):
        Alice_table['Q'].append(item * cos(Alice_table['Phase'][i]))
        Alice_table['P'].append(item * sin(Alice_table['Phase'][i]))
    for l in range(count_ref_signal[0]):
        Alice_table['Ultimate state'].append('-')
        Alice_table['Linear difference'].append('-')
        Alice_table['Alice bits'].append('-')

    for i in range(p * index_step, (p + 1) * index_step):
        rotated_state = rotate_alice_signals(matrix, norm_coeff, Alice_table['Q'][i], Alice_table['P'][i])
        # print(rotated_state)
        Alice_table['Rotated Q'].append(rotated_state[0][0])
        Alice_table['Rotated P'].append(rotated_state[1][0])

        if Alice_table['Bob\'s quadrature'][i] == 'Q' and i > p * index_step + 1:
            Alice_table['Ultimate state'].append(rotated_state[0][0])
            Alice_table['Linear difference'].append(float(
                abs((rotated_state[0][0] - Alice_table['Bob\'s quadrature value'][i]) / (
                    Alice_table['Bob\'s quadrature value'][i]))))
            list_for_diff.append(
                abs((rotated_state[0][0] - Alice_table['Bob\'s quadrature value'][i]) / rotated_state[0][0]))
            Alice_table['Alice bits'].append(adc(rotated_state[0][0]))

        elif i > p * index_step + 1:
            Alice_table['Ultimate state'].append(rotated_state[1][0])
            Alice_table['Alice bits'].append(adc(rotated_state[1][0]))
print("Средная относительная разница значений Алисы и Боба: {}".format((mean(list_for_diff))))
# print(json.dumps(table, indent=4))
"""for item in Alice_table:
    print(item, len(Alice_table[item]))
print(Alice_table)"""

df = pd.DataFrame(Alice_table)
ref_indexes = {i: 'ref' for i in range(Alice_table['Ref to signal ratio'][0])}
sig_indexes = {i + Alice_table['Ref to signal ratio'][0]: 'signal' for i in
               range(Alice_table['Ref to signal ratio'][1])}
indexes = {**ref_indexes, **sig_indexes}
# print(indexes)
df.rename({0: 'ref', 1: 'ref', 2: 'signal', 3: 'signal', 4: 'signal'}, axis='index')
df.rename(index={0: "ref"})
# print(df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    # print(df)
    pass
df.to_csv('CSV files/Alice table new.csv')
