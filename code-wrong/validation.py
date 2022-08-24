import pandas as pd
from utils import pickle, np, nx

limits_upper_solution = np.array([2.3886, 100, 0.2191, 0.106, 0.0311, 4.0246])
limits_lower_solution = np.array([0, 61.165, 0, 0, 0, 0])
limits_upper_window   = np.array([2.4677, 100, 0.2686, 0.109, 0.03615, 4.1777])
limits_lower_window   = np.array([0, 60.9636, 0, 0, 0, 0])

def calc_violation(array_average,case):
    if case == 1:
        min_limit = limits_lower_solution
        max_limit = limits_upper_solution
    else:
        min_limit = limits_lower_window
        max_limit = limits_upper_window
    # we see only Fe has a lower bound therefore the lower bound violation is calculated only considering Fe
    # :: ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
    # we calculate the violation of bounds by bracket operator
    viol_lower = np.abs(min((array_average[1]-min_limit[1])/min_limit[1],0))
    viol_upper = (max_limit-array_average)/max_limit
    viol_upper[viol_upper > 0] = 0
    viol_upper = np.sum(abs(viol_upper))
    viol = np.round(viol_lower + viol_upper,6)
    return viol

# D = pickle.load(open('eka_testproblem/D_nodes_{}.pickle'.format(4),'rb')) # only one stockpile
df = pd.DataFrame()
df_nodes = pd.read_csv('SyntheticData6/Nodes.csv',index_col=0)# load nodes.csv file
number_stockpile = 4

seq = []
for k in np.arange(number_stockpile)+1:
    for j in np.arange(10)+1:
        for i in np.arange(4)+1:
            if j < 10:
                number_col = '0' + str(j)
            else:
                number_col = j
            seq.append('01-01-0{}-0{}-{}'.format(k,i,number_col))

# S = seq[:27] #S1
# S = seq[27:27+28]
# S=  seq[27+28:27+28+26]

# with open('S1_DGA.pickle','rb') as pickle_str:
#     S1 = pickle.load(pickle_str)
#
# with open('S2_DGA.pickle','rb') as pickle_str:
#     S2 = pickle.load(pickle_str)
#
# with open('S3_DGA.pickle','rb') as pickle_str:
#     S3 = pickle.load(pickle_str)
#
# with open('S4_DGA.pickle','rb') as pickle_str:
#     S4 = pickle.load(pickle_str)
with open('MMAS_best.pickle','rb') as pickle_str:
    path = pickle.load(pickle_str)

with open('eka_testproblem/D_cost_4_SN_SN.pickle','rb') as pickle_str:
    D_cost = pickle.load(pickle_str)



S = [x[1] for x in path]

df = pd.DataFrame()
for node in S:
    df = df.append(df_nodes.loc[node])

tonnage_path = df['Cut_Tonnage'].to_numpy()
cost_path = df['Cost'].to_numpy() # reclamation cost
cost_moving=np.array([D_cost[k] for k in path])
df = df.drop(['CaO','Cost','Cut_Volume','MgO','Cut_Tonnage','Product_Description','TiO2'],axis=1)
average_chemical = df.mean().values

#
# cost=)
# cost += cost_path


utility = np.sum((cost_path+cost_moving)/tonnage_path) + 1.33
print(np.sum(utility)) # 1.33 is the utility cost for entry reclamation

index = 0
penalty_parcel = []
for e in df.itertuples():
    if index >= 3:
        df_current = pd.DataFrame()
        node_current = df.iloc[index]
        node_1 =  df.iloc[index-1]
        node_2 =  df.iloc[index-2]
        df_current = df_current.append(node_current)
        df_current = df_current.append(node_1)
        df_current = df_current.append(node_2)
        penalty_parcel.append(calc_violation(df_current.mean().to_numpy(),2))
    index += 1

print(calc_violation(average_chemical,1))
print(np.sum(penalty_parcel))
x