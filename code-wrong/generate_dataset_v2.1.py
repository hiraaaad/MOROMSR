# this script generates the dictionary of nodes and graphs
import pandas as pd
import pickle
import networkx as nx
from utils import Node, np, parse_cut_id, calc_pos_node
import matplotlib.pyplot as plt

### problem generation parameters
number_machine = 1
number_row = 1
number_stockpile = 4
### load synthetic data set
df_nodes = pd.read_csv('SyntheticData6/Nodes.csv',index_col=0) # load nodes.csv file
# df_nodes = df_nodes.reset_index(drop=True)
df_prec = pd.read_csv('SyntheticData6/Precedence.csv',index_col=0) # load precedence.csv file | to perform a job_step_2, all required job_step_1 should be done wrt. direction
df_prec_SN = df_prec.loc[df_prec['Job']=='SN']
df_prec_NS = df_prec.loc[df_prec['Job']=='NS']

### setup dictionary of nodes and bidrectional graphs
G_SN = {k: nx.DiGraph() for k in np.arange(4)+1} # graph for SN reclamation
G_NS = {k: nx.DiGraph() for k in np.arange(4)+1} # graph for NS reclamation
D_nodes={}

### Main section
for row in df_nodes.itertuples():
    Index = row.Index
    node_row : np.uint8
    node_stockpile : np.uint8
    node_bench : np.uint8
    node_cut : np.uint8
    node_machine : np.uint8
    node_machine , node_row, node_stockpile , node_bench, node_cut = parse_cut_id(row.Index)
    # current purpose is to generate problem for one machine and one row
    if node_machine <= number_machine and node_row <= number_row and node_stockpile <= number_stockpile:
        ## update my node dictionary
        node_prec_SN = list(df_prec_SN.loc[df_prec_SN['Job_Step_2']==Index].index) # determine the list of nodes must be reclaimed before accessing this node
        node_prec_NS = list(df_prec_NS.loc[df_prec_NS['Job_Step_2']==Index].index)

        node_prec_1_SN = list(df_prec_SN.loc[df_prec_SN.index == Index]['Job_Step_2']) # determine the list of nodes can be reclaimed after accessing this node
        node_prec_1_NS = list(df_prec_NS.loc[df_prec_NS.index == Index]['Job_Step_2'])

        D_nodes[Index] = Node(row, node_prec_SN, node_prec_NS, node_prec_1_SN, node_prec_1_NS,node_machine , node_row, node_stockpile , node_bench, node_cut)

        # ## update the graph
        # ## nodes
        # # SN
        # nodes_list = node_prec_1_SN.copy()
        # nodes_list.append(Index)
        # nodes_list = set(nodes_list)
        # for i,node_name in enumerate(nodes_list):
        #     if node_name not in G_SN[node_stockpile].nodes():
        #         node_machine , node_row, node_stockpile , node_bench, node_cut = parse_cut_id(node_name)
        #         G_SN[node_stockpile].add_node(node_name)
        #
        # # SN
        # nodes_list = node_prec_1_NS.copy()
        # nodes_list.append(Index)
        # nodes_list = set(nodes_list)
        # for i,node_name in enumerate(nodes_list):
        #     if node_name not in G_NS[node_stockpile].nodes():
        #         node_machine , node_row, node_stockpile , node_bench, node_cut = parse_cut_id(node_name)
        #         pos_node = calc_pos_node(node_stockpile, node_bench, node_cut)
        #     G_NS[node_stockpile].add_node(node_name)
        #
        # ## edges
        # # SN
        # for i,e in enumerate(node_prec_1_SN):
        #     if (Index, e) not in G_SN[node_stockpile].edges():
        #         G_SN[node_stockpile].add_edge(Index, e)
        #
        # # NS
        # for i,e in enumerate(node_prec_1_NS):
        #     if (Index, e) not in G_NS[node_stockpile].edges():
        #         G_NS[node_stockpile].add_edge(Index, e)

# str_dic = 'eka_testproblem/D_nodes_{}_{}.pickle'.format(number_machine, number_row)
str_dic = 'eka_testproblem/D_nodes_{}.pickle'.format(number_stockpile)

with open(str_dic,'wb') as dic_file:
    pickle.dump(D_nodes,dic_file)

# for i in np.arange(4)+1:
#     nx.write_graphml(G_SN[i], 'eka_testproblem/G_SN_1_1_{}.graphml'.format(i))
#     nx.write_graphml(G_NS[i], 'eka_testproblem/G_NS_1_1_{}.graphml'.format(i))
#
# ## plotting the graphs
# determine positions of nodes :: ç
# pos_SN = {}
# for node in G_SN.nodes():
#     node_machine , node_row, node_stockpile , node_bench, node_cut = parse_cut_id(node)
#     pos_node = calc_pos_node(node_stockpile, node_bench ,node_cut)
#     pos_SN[node] = pos_node
#
# pos_NS = {}
# for node in G_NS.nodes():
#     node_machine , node_row, node_stockpile , node_bench, node_cut = parse_cut_id(node)
#     pos_node = calc_pos_node(node_stockpile, node_bench ,node_cut)
#     pos_NS[node] = pos_node
#
#
# nodes = nx.draw_networkx_nodes(G_SN, pos_SN, node_size=10, node_color='black', )
# edges = nx.draw_networkx_edges(G_SN, pos_SN, node_size=10, arrowstyle='->',
#                                arrowsize=3, edge_color='r',
#                                edge_cmap=plt.cm.Blues, width=1)
# labels=nx.draw_networkx_labels(G_SN,pos_SN, font_size=10, font_color='white')
# # nx.draw(G, with_labels = True, node_color='r', node_size=450, font_size='10')
# plt.show() # display
#


