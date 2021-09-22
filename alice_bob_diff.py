import pandas as pd
import numpy as np

from Bob_table import *
from Alice_table import *

ratio_signal_to_ref = 2
mean_distance = []
# for ratio_signal_to_ref in range(2, 10):
ratio_signal_to_ref = 2

# bob_main()

# alice_main(signal_to_ref_ratio=ratio_signal_to_ref)

Alice_table = pd.read_csv('Alice table.csv', dtype={'Alice bits': 'string'})
Bob_table = pd.read_csv('Bob table.csv', dtype={"Bits": 'string'})

(n, m) = Alice_table.shape
results = []
num_results = []

for i in range(n):
    if Alice_table['Pulse type'][i] == 'ref':
        results.append('-')
    else:
        results.append(sum(1 for (a, b) in zip(str(Bob_table['Bits'][i]), Alice_table['Alice bits'][i]) if a != b))
        num_results.append(
            sum(1 for (a, b) in zip(str(Bob_table['Bits'][i]), Alice_table['Alice bits'][i]) if a != b))

Res_dict = {
    'Bob\'s values': Bob_table['Value'],
    'Alice\'s values': Alice_table['Ultimate state'],
    'Bob\'s bits': Bob_table['Bits'],
    'Alice\'s bits': Alice_table['Alice bits'],
    'Hamming distance': np.array(results),
}
# Alice_table.loc[:, 'Hamming distance'] = pd.Series(np.array(results), index=Alice_table.index)
results = pd.DataFrame(Res_dict)
results.to_csv('Result.csv')
mean_distance.append(np.mean(num_results))

print(mean_distance)
