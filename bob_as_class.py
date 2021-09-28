import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import json
import matplotlib.colors as mcolors
import random


def file_transorm(old_filename: str, new_filename: str, start_index=4675, sample_count=200000) -> None:
    """Трансформация файла: удаление инфомации о параметрах осциллографа, ограничение количетсва точек"""
    df = pd.read_csv(old_filename)  # , dtype={"Time": float, "Amplitude": float})
    (n, m) = df.shape
    body = df.tail(n - 6 - start_index).iloc[:, 3:].astype('float')  # отсечение первых информационных строк
    # (n, m) = df.shape
    # body = body.head(sample_count) # ограничение колличества точек
    body.columns = ['Time', 'Amplitude']  # переименование названий колонок
    body.to_csv(new_filename, index=False)  # импортирование данных в csv файл):


class Bob:
    # TODO: Добавить описание
    # TODO: написание тестов для существующих данных
    # TODO: Использование статистики
    # TODO: Доработка ПО для работы с импульсным режимом
    """
    Класс боб для реализации оцифровки сигнала
    Необходимо задавать: data_filename, global_index, total_cycle_number, min_indexes_in_pulse, max_indexes_in_pulse
    """

    def __init__(self, data_filename: str,
                 bob_table_filename='Bob_table_from_class.csv',
                 global_index=0,
                 statistics_ref_1_pulse=(0, 0),
                 statistics_ref_2_pulse=(0, 0),
                 max_std=0.001,
                 total_cycle_number=1,
                 adc_voltage_range=2,
                 adc_resolution=12,
                 ref_sig_quantity=(2, 4),
                 min_indexes_in_pulse=200,
                 max_indexes_in_pulse=2000):
        self.global_index = global_index  # индекс начала текущего цикла
        self.state_table = {'Pulse type': [], 'Quadrature': [], 'Value': [], 'Bits': []}  # итоговая таблица
        self.statistics_ref_1_pulse = statistics_ref_1_pulse  # статистика первого ипульса в референсе:
        # Первое значение - индекс-середина импульса, ширина в индекса импульса
        self.statistics_ref_2_pulse = statistics_ref_2_pulse  # статистика второго ипульса в референсе:
        # Первое значение - индекс-середина импульса, ширина в индекса импульса
        self.max_std = max_std  # максимальное значение СКО по импульсам
        self.total_cycle_number = total_cycle_number  # общее число циклов, которое нужно выполнить, если хватит
        self.adc_voltage_range = adc_voltage_range  # диапазон АЦП V p-p
        self.adc_resolution = adc_resolution  # битность АЦП
        self.ref_sig_quantity = ref_sig_quantity
        self.ref_mid_indexes = []  # индексы середин референсных импульсов:
        # каждый элемент self.ref_mid_indexes является массивом середин референсных импульсов,
        # соответствующих одному циклу
        self.ref_left_indexes = []  # индексы левых границ референсных импульсов:
        # каждый элемент self.ref_left_indexes является массивом левых границ референсных импульсов,
        # соответствующих одному циклу
        self.ref_right_indexes = []  # индексы правых границ референсных импульсов:
        # каждый элемент self.ref_left_indexes является массивом правых границ референсных импульсов,
        # соответствующих одному циклу
        self.sig_mid_indexes = []  # индексы середин сигнальных импульсов:
        # каждый элемент self.sig_mid_indexes является массивом середин сигнальных импульсов,
        # соответствующих одному циклу
        self.sig_left_indexes = []  # индексы левых границ сигнальных импульсов:
        # каждый элемент self.sig_left_indexes является массивом левых границ сигнальных импульсов,
        # соответствующих одному циклу
        self.sig_right_indexes = []  # индексы правых границ сигнальных импульсов:
        # каждый элемент self.sig_left_indexes является массивом правых границ сигнальных импульсов,
        # соответствующих одному циклу
        self.samples = np.array([
            np.array(pd.read_csv(data_filename, dtype={"Time": float, "Amplitude": float})['Time']),
            np.array(pd.read_csv(data_filename, dtype={"Time": float, "Amplitude": float})['Amplitude'])
        ])
        self.min_indexes_in_pulse = min_indexes_in_pulse
        self.max_indexes_in_pulse = max_indexes_in_pulse

    def bob_adc(self, voltage_value: float) -> str:
        """Оцифровка: возвращение битовой последовательности"""
        # step = voltage_range / (2 ** resolution)
        voltage_range = self.adc_resolution
        resolution = self.adc_resolution
        grid = np.linspace(-voltage_range / 2, voltage_range / 2, 2 ** resolution - 1)
        left = 0
        right = len(grid)
        if voltage_value < grid[0]:
            return '0' * resolution
        while right > left + 1:
            center = mt.floor((left + right) / 2)
            if voltage_value >= grid[center]:
                left = center
            else:
                right = center
        return '0' * (resolution - len(bin(right)[2:])) + bin(right)[2:]

    def ref_pulse_definition(self) -> list:
        """Определение референсных импульсов
        Вывод: массив вида [
                            [l1, c1, r1, a1], индексы левой границы, центра, правой границы и значение амплитуды 1-го реф импульса
                            [l2, c2, r2, a2], индексы левой границы, центра, правой границы и значение амплитуды 2-го реф импульса
                            max_std, максимольное значение СКО импульсов по всем референсным импульсам
                            [start, end] начало и конец интервала реф импульсов
                            ]"""
        pulses_in_ref = self.ref_sig_quantity[0]
        indexes_for_ref = [[] for i in range(pulses_in_ref + 2)]  # возвращаемый список
        i_pulse_start = self.global_index

        for j in range(pulses_in_ref):
            checking_std = self.max_std
            """"Цикл поиска референсных импульсов"""
            # indexes_for_ref[j].append(i_pulse_start)
            i = i_pulse_start + self.min_indexes_in_pulse
            checking_std = max(self.max_std, float(np.std(self.samples[1][i_pulse_start:i - 1])))
            while abs(self.samples[1][i] - np.mean(self.samples[1][i_pulse_start:i])) <= checking_std * 5:
                # проверка выхода за границы пяти СКО от среднего значения по значениям пмплитуд
                checking_std = max(self.max_std, float(np.std(self.samples[1][i_pulse_start:i - 1])))
                i += 1

            pulse_std = np.std(self.samples[1][i_pulse_start:i])  # СКО полученного импульса
            i -= 1
            # TODO: Проверить первое условие выхода из цикла при сдвиге левой границы влево
            i_left = i_pulse_start
            while abs(self.samples[1][i_pulse_start - 1] -
                      np.mean(self.samples[1][i_pulse_start:i + 1])) < checking_std * 5 \
                    and i_pulse_start >= 1 \
                    and abs(i_pulse_start - i_left) <= self.max_indexes_in_pulse:
                """Сдвиг левой границы"""
                i_pulse_start -= 1
                checking_std = max(self.max_std, float(np.std(self.samples[1][i_pulse_start:i - 1])))
            if abs(i_pulse_start - i_left) >= self.max_indexes_in_pulse:
                i_pulse_start = i_left
            if j == 0:
                indexes_for_ref[-1].append(i_pulse_start)
            indexes_for_ref[j].append(i_pulse_start)
            self.ref_left_indexes.append(i_pulse_start)
            # print(i_pulse_start)
            indexes_for_ref[j].append(int((i + i_pulse_start) / 2))  # добавление центральной точки
            # в список индексов импульса
            self.ref_mid_indexes.append(int((i + i_pulse_start) / 2))
            indexes_for_ref[j].append(i)  # добавление конечной точки импульса
            self.ref_right_indexes.append(i)
            indexes_for_ref[-2].append(pulse_std)  # добавление в список СКО импульсов СКО нынешнего импульса
            indexes_for_ref[j].append(
                np.mean(self.samples[1][i_pulse_start:i]))  # добавление среднего значения амплитуды импульса
            index_next_pulse = i + 1

            for index_next_pulse in range(i + 1, self.max_indexes_in_pulse):
                # цикл по точкам после найденного импульса для определения точки следующего импульса
                if abs(self.samples[1][index_next_pulse] - self.samples[1][index_next_pulse + 5]) < pulse_std * 5:
                    # условие, что разница амплитуд двух последовательных точек будет меньше 5 СКО предыдущего импульса
                    break
            index_next_pulse += 4
            # print(index_next_pulse)
            i_pulse_start = index_next_pulse + 4
        try:
            indexes_for_ref[-1].append(index_next_pulse)
        except NameError:
            print('Number of ref pulses in cycle is equal to 0, set the right value.')
            exit()
        indexes_for_ref[-2] = max(
            indexes_for_ref[-2])  # замена списко СКО всех импульсов на максимальный СКО по имульсам

        return indexes_for_ref

    def signal_pulse_definition(self, ref_info: list) -> list:
        samples = self.samples
        pulse_values = []
        for signal_number in range(round(self.ref_sig_quantity[1] / self.ref_sig_quantity[0])):
            index_shift = signal_number * (ref_info[-1][1] - ref_info[-1][0])
            pulse_centers = [item[1] - ref_info[-1][0] + ref_info[-1][1] + index_shift for item in ref_info[:-2]]
            pulse_width = [mt.floor((item[2] - item[0]) / 2) for item in ref_info[:-2]]
            # for count, center in enumerate(pulse_centers):
            #    print(center - floor(pulse_width[count] / 2), center + floor(pulse_width[count] / 2) , 4)
            pulse_values += [
                np.mean(
                    samples[1][center - mt.floor(pulse_width[count] / 2): center + mt.floor(pulse_width[count] / 2)])
                for count, center in enumerate(pulse_centers)]
            self.sig_left_indexes += [(pulse_centers[i] - mt.floor(pulse_width[i] / 2))
                                      for i in range(len(pulse_centers))]
            self.sig_right_indexes += [(pulse_centers[i] + mt.floor(pulse_width[i] / 2))
                                       for i in range(len(pulse_centers))]
            self.sig_mid_indexes += [pulse_center for pulse_center in pulse_centers]
        return pulse_values

    def make_cycle(self) -> None:
        ref_indexes = self.ref_pulse_definition()
        # a = (2 * cycle_number) / (cycle_number + 1)
        # b = 2 / (cycle_number + 1)
        for i in range(2):
            self.state_table['Pulse type'].append('ref')
            self.state_table['Value'].append(ref_indexes[i][3])
            self.state_table['Bits'].append(self.bob_adc(ref_indexes[i][3]))
        self.state_table['Quadrature'].append('Q')
        self.state_table['Quadrature'].append('P')
        # MAXIMUM_STD = max(MAXIMUM_STD, ref_indexes[-2])
        signal_values = self.signal_pulse_definition(ref_indexes)
        self.state_table['Value'] += signal_values
        self.state_table['Bits'] += [self.bob_adc(value) for value in signal_values]
        # print(signal_values)
        # n = cycle_number * 2 - 1 + i
        # a = (2 * n) / (n + 1)
        # b = 2 / (n + 1)
        for i in range(int(self.ref_sig_quantity[1] / self.ref_sig_quantity[0])):
            self.state_table['Pulse type'] += ['signal', 'signal']
            self.state_table['Quadrature'] += ['Q', 'Q']
        next_cycle = int((ref_indexes[-1][-1] - ref_indexes[-1][0]) * int(
            self.ref_sig_quantity[1] / self.ref_sig_quantity[0]) * 1.1) + ref_indexes[-1][-1]
        self.global_index = next_cycle


class PlottingSignal:
    def __init__(self, start, stop, data_filename,
                 *indexes_for_dot_plotting, ):

        self.samples = np.array([
            np.array(pd.read_csv(data_filename, dtype={"Time": float, "Amplitude": float})['Time']),
            np.array(pd.read_csv(data_filename, dtype={"Time": float, "Amplitude": float})['Amplitude'])
        ])
        self.samples = np.array([[(item - self.samples[0][0]) / (self.samples[0][2] - self.samples[0][1])
                                  for item in self.samples[0]], self.samples[1]])
        plt.plot(self.samples[0][start:stop], self.samples[1][start:stop])
        for item, color in indexes_for_dot_plotting:
            # colors = [color for color in mcolors.CSS4_COLORS.keys()]
            # color = random.choice(colors)
            for j, dot in enumerate(item):
                plt.plot(self.samples[0][dot], self.samples[1][dot], '*', color=color)

        plt.show()


if __name__ == '__main__':
    bob = Bob(data_filename='Ampl modul/amp and phase o-pi2 transformed.csv')
    # ref1 = bob.ref_pulse_definition()
    for i in range(150):
        bob.make_cycle()
    # sig1 = bob.signal_pulse_definition(ref1)

    with open('bob_table_class.json', 'w') as json_file:
        json.dump(bob.state_table, json_file, indent=4)

    df = pd.DataFrame(bob.state_table)
    df.to_csv('Bob class table.csv')

    plot = PlottingSignal(0, max(bob.ref_right_indexes[-1], bob.sig_right_indexes[-1]) + 500,
                          'Ampl modul/amp and phase o-pi2 transformed.csv',
                          (bob.sig_right_indexes, 'red'),
                          (bob.sig_mid_indexes, 'green'),
                          (bob.sig_left_indexes, 'blue'),
                          (bob.ref_mid_indexes, 'coral'))