# this script generates the dictionary of costs
import pandas as pd
import pickle
from utils import parse_cut_id
###
### problem generation parameters
number_machine = 1
number_row = 1
number_stockpile = 4
### load csv file
df_cost = pd.read_csv('SyntheticData6/Cost.csv',index_col=0) # load cost csv file
D_cost = {}
for row in df_cost.itertuples():
    node_machine_1 , node_row_1, node_stockpile_1 , node_bench_1, node_cut_1 = parse_cut_id(row.Index)
    node_machine_2 , node_row_2, node_stockpile_2 , node_bench_2, node_cut_2 = parse_cut_id(row.Index)
    # current purpose is to generate problem for one machine and one row
    if node_machine_1 <= number_machine and node_machine_2 <= number_machine and node_row_1 <= number_row and \
        node_row_2 <= number_row and node_stockpile_1 <= number_stockpile and node_stockpile_2 <= number_stockpile and \
            row.Job_1 =='SN' and row.Job_2 =='SN':
        D_cost[(row.Index, row.Job_Step_2, row.Job_1, row.Job_2)] = row.Cost

# str_dic = 'eka_testproblem/D_cost_{}_{}.pickle'.format(number_machine, number_row)
str_dic = 'eka_testproblem/D_cost_{}_SN_SN.pickle'.format(number_stockpile)

with open(str_dic,'wb') as dic_file:
    pickle.dump(D_cost,dic_file)