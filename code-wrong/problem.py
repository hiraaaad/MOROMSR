from utils import pickle, np, nx

class Problem:
    def __init__(self, scenario : int, number_demand : int, number_stockpile : int, greedy_factor : int, parcel : bool, local : bool):
        """

        :param scenario: number of scenari
        :param number_demand: number of demands
        :param number_stockpile: number of stockpiles
        :param greedy_factor: greedy factor for RGA
        :param parcel: scenario 3 is activated or not
        """
        self.scenario = scenario
        self.number_demand = number_demand
        if self.scenario > 1:
            self.number_stockpile = 4
        else:
            self.number_stockpile = number_stockpile

        if self.scenario == 1:
            self.parcel = False
        else:
            self.parcel = parcel

        self.local = local
        self.greedy_factor = greedy_factor # lambda in the paper
        self.precision = 6
        ## load dictionaries and graph files

        self.D_cost = pickle.load(open('eka_testproblem/D_cost_{}_SN_SN.pickle'.format(number_stockpile),'rb'))
        self.D_nodes = pickle.load(open('eka_testproblem/D_nodes_{}.pickle'.format(self.number_stockpile),'rb')) # only one stockpile

        # mineral limits :: ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
        self.limits_upper_solution = np.array([2.3886, 100, 0.2191, 0.106, 0.0311, 4.0246])
        self.limits_lower_solution = np.array([0, 61.165, 0, 0, 0, 0])
        self.limits_upper_window   = np.array([2.4677, 100, 0.2686, 0.109, 0.03615, 4.1777])
        self.limits_lower_window   = np.array([0, 60.9636, 0, 0, 0, 0])
        # graph files are disabled for now
        # rang = np.arange(4)+1
        # G_SN = {k: None for k in rang}
        # G_NS = {k: None for k in rang}
        # for i in rang:
        #     G_SN[i] = nx.read_graphml('eka_testproblem/G_SN_1_1_{}.graphml'.format(i))
        #     G_NS[i] = nx.read_graphml('eka_testproblem/G_NS_1_1_{}.graphml'.format(i))
        # self.G_SN = G_SN
        # self.G_NS = G_NS