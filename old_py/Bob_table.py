from math import floor  # , ceil
from numpy import mean, std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter


def transform(old_filename, new_filename, sample_count=200000):
    """Трансформация файла: удаление инфомации о параметрах осциллографа, ограничение количетсва точек"""
    df = pd.read_csv(old_filename)  # , dtype={"Time": float, "Amplitude": float})
    (n, m) = df.shape
    body = df.tail(n - 6 - 4675).iloc[:, 3:].astype('float')  # отсечение первых информационных строк
    # body = body.head(sample_count) # ограничение колличества точек
    body.columns = ['Time', 'Amplitude']  # переименование названий колонок
    body.to_csv(new_filename, index=False)  # импортирование данных в csv файл


def bob_adc(voltage_value, voltage_range=2, resolution=12):
    """Оцифровка: возвращение битовой последовательности"""
    # step = voltage_range / (2 ** resolution)
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


def ref_pulse_definition(samples, max_std, start=0, pulses_in_ref=2):
    """Определение референсных импульсов
    Вывод: массив вида """
    indexes_for_ref = [[] for i in range(pulses_in_ref + 2)]  # возвращаемый список
    i_pulse_start = start

    ref_plots = []
    # ref_plots.append(plt.plot(samples[0][i_pulse_start], samples[1][i_pulse_start], '*', color='blue'))
    for j in range(pulses_in_ref):
        # цикл поиска референсных импульсов
        # indexes_for_ref[j].append(i_pulse_start)
        for i in range(i_pulse_start + 20, max(np.shape(samples))):
            checking_std = max(max_std, std(samples[1][i_pulse_start:i - 1]))
            # print(i_pulse_start, i, 1)
            if abs(samples[1][i] - mean(samples[1][i_pulse_start:i])) > checking_std * 5:
                # проверка выхода за границы пяти СКО от среднего значения по значениям пмплитуд
                break

        pulse_std = std(samples[1][i_pulse_start:i])  # СКО полученного импульса
        i -= 1
        while abs(samples[1][i_pulse_start - 1] - mean(samples[1][i_pulse_start:i + 1])) < checking_std * 5:
            if i_pulse_start >= 1:
                i_pulse_start -= 1
            else:
                break
            # print(i_pulse_start - 1, i + 1, 2)
            checking_std = max(max_std, std(samples[1][i_pulse_start:i - 1]))
        if j == 0:
            indexes_for_ref[-1].append(i_pulse_start)
        indexes_for_ref[j].append(i_pulse_start)
        indexes_for_ref[j].append(int((i + i_pulse_start) / 2))  # добавление центральной точки
        # в список индексов импульса
        indexes_for_ref[j].append(i)  # добавление конечной точки импульса
        indexes_for_ref[-2].append(pulse_std)  # добавление в список СКО импульсов СКО нынешнего импульса
        # print(i_pulse_start, i, 3)
        indexes_for_ref[j].append(mean(samples[1][i_pulse_start:i]))  # добавление среднего значения амплитуды импульса
        # [[0, 7, 14, 0.16371428185714285] - (левая граница импульса, середина импульса, правая граница импульса),
        # 0.008264986175124557 - максимальное СКО среди импульсов в референсе, [0, 62]] - границы референса

        for index_next_pulse in range(i + 1, max(np.shape(samples)) - 1):
            # цикл по точкам после найденного импульса для определения точки следующего импульса
            if abs(samples[1][index_next_pulse] - samples[1][index_next_pulse + 5]) < pulse_std * 5:
                # условие, что разница амплитуд двух последовательных точек будет меньше 5 СКО предыдущего импульса
                break
        index_next_pulse += 4
        i_pulse_start = index_next_pulse + 4
    indexes_for_ref[-1].append(index_next_pulse)
    indexes_for_ref[-2] = max(
        indexes_for_ref[-2])  # замена списко СКО всех импульсов на максимальный СКО по имульсам

    pulses = indexes_for_ref[:-2]  # список информации об имплуьсах: их границы, центр и среднее значение
    # ref_plots = []
    for i, item in enumerate(pulses):
        # if i == 0:
        ref_plots.append(plt.plot(samples[0][item[1]], samples[1][item[1]], '*', color='purple'))
        # ref_plots.append(plt.plot(samples[0][item[0]], samples[1][item[0]], '*', color='black'))
        # ref_plots.append(plt.plot(samples[0][item[2]], samples[1][item[2]], '*', color='red'))
    ref_plots.append(
        plt.plot(samples[0][indexes_for_ref[-1][1]], samples[1][indexes_for_ref[-1][1]], '*', color='pink'))

    return indexes_for_ref, ref_plots


def signal_pulse_definition(samples, ref_info, signal_number=0):
    index_shift = signal_number * (ref_info[-1][1] - ref_info[-1][0])
    pulse_centers = [item[1] - ref_info[-1][0] + ref_info[-1][1] + index_shift for item in ref_info[:-2]]
    pulse_width = [floor((item[2] - item[0]) / 2) for item in ref_info[:-2]]
    # for count, center in enumerate(pulse_centers):
    #    print(center - floor(pulse_width[count] / 2), center + floor(pulse_width[count] / 2) , 4)
    # pulse_values = [abs(mean(samples[1][center - floor(pulse_width[count] / 2): center + floor(pulse_width[count] / 2)]))
    #                 for count, center in enumerate(pulse_centers)]
    pulse_values = [
        mean(samples[1][center - floor(pulse_width[count] / 2): center + floor(pulse_width[count] / 2)])
        for count, center in enumerate(pulse_centers)]
    left_dots = [
        plt.plot(samples[0][pulse_centers[i] - floor(pulse_width[i] / 2)],
                 samples[1][pulse_centers[i] - floor(pulse_width[i] / 2)],
                 'r*')
        for i in range(len(pulse_centers))]
    center_dots = [plt.plot(samples[0][pulse_centers[i]], samples[1][pulse_centers[i]], '*', color='blue')
                   for i in range(len(pulse_centers))]
    right_dots = [
        plt.plot(samples[0][pulse_centers[i] + floor(pulse_width[i] / 2)],
                 samples[1][pulse_centers[i] + floor(pulse_width[i] / 2)],
                 '*', color='black')
        for i in range(len(pulse_centers))]
    plotting = [left_dots, center_dots, right_dots]
    return pulse_values, plotting


def make_cycle(samples, cycle_start, cycle_number, signal_to_ref_ratio=2):
    global MAXIMUM_STD, state_table, index_start_cycle, mean_ref, mean_sig
    ref_indexes, ref_plots = ref_pulse_definition(samples, max_std=MAXIMUM_STD, start=cycle_start)
    print(ref_indexes)
    ref_ampl = sum([(value[3]) ** 2 for value in ref_indexes[:2]])
    a = (2 * cycle_number) / (cycle_number + 1)
    b = 2 / (cycle_number + 1)
    mean_ref = (a * mean_ref + b * ref_ampl) / 2
    for i in range(2):
        state_table['Pulse type'].append('ref')
        state_table['Value'].append(ref_indexes[i][3])
        state_table['Bits'].append(bob_adc(ref_indexes[i][3]))
    state_table['Quadrature'].append('Q')
    state_table['Quadrature'].append('P')
    # MAXIMUM_STD = max(MAXIMUM_STD, ref_indexes[-2])
    for i in range(signal_to_ref_ratio):
        signal_values, signal_plotting = signal_pulse_definition(samples, ref_indexes, signal_number=i)
        # print(signal_values)
        sig_ampl = sum([(value) ** 2 for value in signal_values])
        n = cycle_number * 2 - 1 + i
        a = (2 * n) / (n + 1)
        b = 2 / (n + 1)
        mean_sig = (a * mean_sig + b * sig_ampl) / 2
        state_table['Pulse type'] += ['signal', 'signal']
        state_table['Value'] += signal_values
        state_table['Bits'] += [bob_adc(value) for value in signal_values]
        state_table['Quadrature'] += ['Q', 'Q']
    next_cycle = int((ref_indexes[-1][-1] - ref_indexes[-1][0]) * signal_to_ref_ratio * 1.1) + ref_indexes[-1][-1]
    print((ref_indexes[-1][-1] - ref_indexes[-1][0]) * signal_to_ref_ratio * 0.01)
    return next_cycle


def bob_main(filename_transformed='Ampl modul/amp and phase o-pi2 transformed.csv'):
    global MAXIMUM_STD, state_table, index_start_cycle, mean_ref, mean_sig
    mean_ref, mean_sig = 0, 0
    statistics = [
        [],
        [],

    ]
    state_table = {
        'Pulse type': [],
        'Quadrature': [],
        'Value': [],
        'Bits': [],
    }

    index_start_cycle = 0
    cycles = 55
    end = (cycles + 1) * (55032 - 49035)

    MAXIMUM_STD = 0.001
    data = pd.read_csv(filename_transformed, dtype={"Time": float, "Amplitude": float})
    samples = np.array([np.array(data['Time']), np.array(data['Amplitude'])])

    samples = np.array([[(item - samples[0][0]) / (samples[0][2] - samples[0][1]) for item in samples[0]], samples[1]])
    plt.plot(samples[0][:end], samples[1][:end])
    for i in range(cycles):
        try:
            index_start_cycle = make_cycle(samples, cycle_start=index_start_cycle, cycle_number=i + 1)
        except IndexError:
            break
        print(index_start_cycle)
    public_table = state_table
    df = pd.DataFrame(state_table)
    df.to_csv('Bob table.csv')
    for i, item in enumerate(public_table['Pulse type']):
        if item == 'signal':
            public_table['Value'][i] = '-'
            public_table['Bits'][i] = '-'

    df_public = pd.DataFrame(public_table)
    df_public.to_csv('Bob public table.csv')
    print(f'mean_ref = {mean_ref}')
    print(f'mean_sig = {mean_sig}')
    print(f'sig to ref ratio = {np.sqrt(mean_sig / mean_ref)}')
    plt.show()


if __name__ == '__main__':
    filename_to_transform = 'Ampl modul/amp and phase o-pi2.csv'  # изначальный файл
    filename_transformed = 'Ampl modul/amp and phase o-pi2 transformed.csv'  # трансформированный файл
    # transform(filename_to_transform, filename_transformed)
    start = perf_counter()
    bob_main()
