# -*- coding: utf-8 -*-

from math import sqrt
from networkx.classes.digraph import DiGraph
from networkx import strongly_connected_components_recursive,strongly_connected_components
from networkx.algorithms import shortest_path
import numpy as np

#----------------------------------------
class Motion_MDPST:
    # ----construct probabilistic-labeled MDP----
    def __init__(self, node_dict, edge_dict, state_action, U, C, initial_node, initial_label):
        self.name='motion_MDPST',
        self.U = U
        self.C = C
        self.init_state=initial_node
        self.init_label=initial_label
        self.nodes = node_dict
        self.state_action = state_action
        self.add_edges(edge_dict)
        print("-------Motion MDPST Initialized-------")
        print("%s states and %s edges" %
              (str(len(self.nodes.keys())), str(len(self.edges.keys())))) 
        self.unify_mdpst()

    def add_edges(self, edge_dict):
        edges =dict()
        edges_prob = dict()
        state_action_state = dict()
        for edge, attri in edge_dict.items():
            f_node = edge[0]
            u = edge[1]
            t_node = edge[2]
            edges_prob[(f_node, t_node)] = attri[0]
            state_action_state[edge] = attri[0]
            if (f_node, t_node) in edges.keys():
                prob_cost = edges[(f_node, t_node)]
                prob_cost[tuple(u)] = [attri[0], attri[1]]
            else:
                prob_cost = dict()
                prob_cost[tuple(u)] = [attri[0], attri[1]]
                edges[(f_node, t_node)] = prob_cost
        self.edges = edges
        self.edges_prob = edges_prob
        self.state_action_state = state_action_state
        #print("MDPST edges probability: %s" %edges_prob)
        print("-------Motion MDPST Constructed-------")

    #----
    def dotify(self):
        # ----generate dot diagram for motion_mdp----
        file_dot = open('mdpst.dot', 'w')
        file_dot.write('graph mdpst { \n')
        file_dot.write(
            'graph[rankdir=LR, center=true, margin=0.2, nodesep=0.1, ranksep=0.3]\n')
        file_dot.write(
            'node[shape=circle, fontname="Courier-Bold", fontsize=10, width=0.4, height=0.4, fixedsize=false]\n')
        file_dot.write('edge[arrowsize=0.6, arrowhead=vee]\n')
        for edge in self.edges:
            file_dot.write('"'+str(edge[0])+'"' +
                           '->' + '"' + str(edge[1]) + '"' + ';\n')
        file_dot.write('}\n')
        file_dot.close()
        print("-------motion_mdpst.dot generated-------")
        print("Run 'dot -Tpdf motion_mdpst.dot > mdpst.pdf'")

    def unify_mdpst(self):
        # ----verify the probability sums up to 1----
        nodes = self.nodes
        edges = self.edges
        sum_prob = dict()
        N = dict()
        for f_node in nodes.keys():
            for u in self.U:
                sum_prob[(f_node, u)] = 0
                N[(f_node, u)] = 0

        for (f_node, t_node) in edges.keys():
            for u in self.U:                
                prop = edges[(f_node, t_node)]
                if u in list(prop.keys()):
                    sum_prob[(f_node, u)] += prop[u][0]
                    N[(f_node, u)] += 1
                
        for (f_node, t_node) in edges.keys():
            for u in self.U:               
                if sum_prob[(f_node, u)] < 1.0 and N[(f_node, u)]>0:
                    to_add = (1.0-sum_prob[(f_node, u)])/N[(f_node, u)]
                    prop = edges[(f_node, t_node)]
                    if u in list(prop.keys()):
                        prop[u][0] += to_add
                if sum_prob[(f_node, u)] > 1.0:
                    prop = edges[(f_node, t_node)]
                    if u in list(prop.keys()):
                        prop[u][0] = prop[u][0]/sum_prob[(f_node, u)]
        print('Unify MDPST Done')

# ----------------------------------------------------------------------
#--------------Optimal policy synthesis for MDPST--------------------------
def syn_full_plan_mdpst(mdpst, Acc, SR):
    # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    print("==========[Optimal full plan synthesis start]==========")
    index_prefix, v_new = optimal_plan_prefix_mdpst(mdpst, Acc, SR)
    plan_prefix = index_prefix
    #prefix_risk = 1 - v_new[mdpst.prod_init] 
    if plan_prefix:
        #print("Best plan prefix obtained, risk %s" %str(prefix_risk))
        return plan_prefix, v_new
    else:
        print("No valid plan found")
        return None, None

def optimal_plan_prefix_mdpst(mdp, Acc, SR):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    sf = Acc
    Sr = set()
    v_old = dict()
    #for node in mdp.nodes:
    for node in mdp.prod_nodes:
        if node not in sf:
            Sr.add(node)
            v_old[node] = 0
        else:
            v_old[node] = 1
    print('Number of suffix states: %s' %len(sf))
    print('Number of prefix states: %s' %len(SR))
     # ---------solve vi------------
    print('-----')
    print('Value iteration for prefix starts now')
    print('-----')
    num_iteration = 0
    delta_old = 1
    while delta_old >= 0.01:
        print('Number of interation: %s' %num_iteration)
        v_new, index_prefix, delta_new = optimal_prefix_value_iteration_mdpst(mdp, SR, v_old)
        for s in SR:                    
            v_old[s] = v_new[s]
        for s in sf:
            v_old[s] = 1
        num_iteration += 1           
        delta_old = delta_new
        print(delta_old)
        if delta_old < 0.01:
            print("Prefix Value iteration completed in interations: %s" %num_iteration)
            print("delta: %s" %delta_new)
            return index_prefix, v_new
                      
    # print("Prefix Value iteration completed in interations: %s" %num_iteration)
    # print("delta: %s" %delta_new)

    # return index_prefix, v_new

def optimal_prefix_value_iteration_mdpst(mdpst, Sr, v_old):
    num1 = len(Sr)
    U = mdpst.U
    v_new = dict()
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    index = dict()
    delta = 0
    #print(vlist)
    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U):
            if (s, u) in mdpst.prod_state_action.keys():
                t_set = mdpst.prod_state_action[(s, u)]
                #print(t_set)
                for k, t_group in enumerate(t_set):
                    #print(t_group)
                    #print(k)
                    if len(t_set) == 1:
                        pe = 1
                    else:
                        pe = mdpst.prod_state_action_state[(s, u, t_group)]
                    #print(pe)
                    fd = t_group
                    #print(fd)
                    if len(fd) > 0:
                        v_group = []
                        for t_node in fd[0]: 
                            ss = (t_node, fd[1], fd[2]) 
                            if ss in list(v_old.keys()): 
                                if v_old[ss]:                
                                    v_group.append(v_old[ss])
                        #print(v_group)
                        if len(v_group)>0:
                            vlist[idx][idu] += pe*min(v_group)              
    #print(vlist)
    for idx, s in enumerate(Sr):
        v_new[s], index[s] =  max((value, index) for index, value in enumerate(vlist[idx]))
        error = abs(v_new[s] - v_old[s])
        if error > delta:
            delta = error
    # print(v_new)
    # print(index) 
    # print(delta)  

    return v_new, index, delta


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class Product_MDPST:
    def __init__(self, smdp, dra, PROP): 
        self.name= 'Product_MDPST'
        self.U = smdp.U
        self.C = smdp.C
        self.PROP = PROP
        print("-------Prod MDPST Initialized-------")
        self.add_nodes(smdp, dra)
        self.add_edges(smdp, dra)
        self.add_state_action_dict(smdp, dra)
        print("-------Prod MDPST Constructed-------")
        print("%s states, %s edges" % (
            str(len(self.prod_nodes)), str(len(self.prod_edges.keys()))))

    def add_nodes(self, smdp, dra):
        prod_node = []
        prod_init = []
        for mdp_node in list(smdp.nodes.keys()):
            for mdp_label, label_prob in smdp.nodes[mdp_node].items():
                for dra_node in dra.nodes:
                    node = (mdp_node, mdp_label, dra_node)
                    prod_node.append(tuple(node))
                    if ((mdp_node == smdp.init_state) and
                        (dra_node in dra.graph['initial'])):
                        prod_init.append(tuple(node))
        self.prod_nodes = prod_node
        self.prod_init = prod_init
                        
    def add_edges(self, smdp, dra):
        prod_edges = dict()
        # ----construct full product----
        for (f_mdp_node, t_mdp_node) in list(smdp.edges.keys()):
            for f_mdp_label, f_label_prob in smdp.nodes[f_mdp_node].items():
                for f_dra_node in dra.nodes:
                    f_prod_node = (f_mdp_node, f_mdp_label, f_dra_node)
                    for t_dra_node in dra.successors(f_dra_node):
                        t_prod_node = (t_mdp_node, frozenset(), t_dra_node)                       
                        truth = dra.check_label_for_dra_edge(
                                    f_mdp_label, f_dra_node, t_dra_node)
                        if truth:
                            prob_cost = dict()
                            for u, attri in smdp.edges[(f_mdp_node, t_mdp_node)].items():
                                if attri[0] != 0:
                                    prob_cost[u] = (attri[0], attri[1])
                            if list(prob_cost.keys()):
                                prod_edges[(f_prod_node, t_prod_node)]= prob_cost                                   
        self.prod_edges = prod_edges

    def add_state_action_dict(self, smdp, dra):
        prod_state_action = dict()
        # ----construct full product----
        for f_mdp_node in list(smdp.nodes.keys()):
            for f_mdp_label, f_label_prob in smdp.nodes[f_mdp_node].items():
                for f_dra_node in dra.nodes:
                    f_prod_node = (f_mdp_node, f_mdp_label, f_dra_node)                    
                    for t_dra_node in dra.successors(f_dra_node):
                        for u in self.U:                      
                            truth = dra.check_label_for_dra_edge(
                                        f_mdp_label, f_dra_node, t_dra_node)
                            if truth:
                                t_prod_node_set = []
                                for t_mdp_node in smdp.state_action[(f_mdp_node, u)]:
                                    t_prod_node = (t_mdp_node, frozenset(), t_dra_node)
                                    t_prod_node_set.append(tuple(t_prod_node))
                                prod_state_action[(f_prod_node, u)] = t_prod_node_set                                       
        self.prod_state_action = prod_state_action

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class Product_MDPST_ldba:
    def __init__(self, smdp, ldba, PROP): 
        self.name= 'Product_MDPST'
        self.U_MDP = set(smdp.U)
        self.U_EPS = ldba.graph['Ueps']
        self.U = self.U_MDP.union(ldba.graph['Ueps'])  
        self.C = smdp.C
        self.PROP = PROP
        print("-------Prod MDPST Initialized-------")
        self.add_nodes(smdp, ldba)
        self.add_edges(smdp, ldba)
        self.add_state_action_dict(smdp, ldba)
        print("-------Prod MDPST Constructed-------")
        print("%s states, %s edges" % (
            str(len(self.prod_nodes)), str(len(self.prod_edges.keys()))))

    def add_nodes(self, smdp, ldba):
        prod_node = []
        prod_init = []
        for mdp_node in list(smdp.nodes.keys()):
            for mdp_label, label_prob in smdp.nodes[mdp_node].items():
                for ldba_node in ldba.nodes:
                    node = (mdp_node, mdp_label, ldba_node)
                    prod_node.append(tuple(node))
                    if ((mdp_node == smdp.init_state) and
                        (ldba_node in ldba.graph['initial'])):
                        prod_init.append(tuple(node))
        self.prod_nodes = prod_node
        self.prod_init = prod_init
                        
    def add_edges(self, smdp, ldba):
        prod_edges = dict()
        # ----construct full product----
        for (f_mdp_node, t_mdp_node_set) in list(smdp.edges.keys()):
            for f_mdp_label, f_label_prob in smdp.nodes[f_mdp_node].items():
                for f_ldba_node in ldba.nodes:
                    f_prod_node = (f_mdp_node, f_mdp_label, f_ldba_node)
                    #define non-epsilon transitions
                    for t_mdp_node in t_mdp_node_set:
                        for t_mdp_label, t_label_prob in smdp.nodes[t_mdp_node].items():
                            t_ldba_node = ldba.auto.delta[f_ldba_node][tuple(f_mdp_label)]
                            t_prod_node = (t_mdp_node, t_mdp_label, t_ldba_node)                        
                            prob_cost = dict()                          
                            for u, attri in smdp.edges[(f_mdp_node, t_mdp_node_set)].items():
                                if attri[0] != 0:
                                    prob_cost[u] = (attri[0], attri[1])
                            if list(prob_cost.keys()):
                                prod_edges[(f_prod_node, t_prod_node)]= prob_cost 
                    #define epsilon transitions
                    for t_ldba_node in ldba.nodes:
                        truth = ldba.check_epsilon_transition_for_dra_edge(
                                    f_mdp_label, f_ldba_node, t_ldba_node)
                        if truth:
                            prob_cost = dict()
                            t_prod_node_eps = (f_mdp_node, f_mdp_label, t_ldba_node)
                            prob_cost[self.U_EPS[t_ldba_node]] = (1, 0)
                            prod_edges[(f_prod_node, t_prod_node_eps)]= prob_cost                                 
        self.prod_edges = prod_edges

    def add_state_action_dict(self, smdp, ldba):
        prod_state_action = dict()
        prod_state_action_state = dict()
        # ----construct full product----
        for (f_mdp_node, t_mdp_node_set) in list(smdp.edges.keys()):
            for f_mdp_label, f_label_prob in smdp.nodes[f_mdp_node].items():
                for f_ldba_node in ldba.nodes:
                    f_prod_node = (f_mdp_node, f_mdp_label, f_ldba_node)                    
                    #define non-epsilon transitions
                    for t_mdp_node in t_mdp_node_set:
                        for t_mdp_label, t_label_prob in smdp.nodes[t_mdp_node].items():
                            t_ldba_node = ldba.auto.delta[f_ldba_node][tuple(f_mdp_label)]
                            for u, attri in smdp.edges[(f_mdp_node, t_mdp_node_set)].items():
                                if attri[0] != 0:                      
                                    t_prod_node_set = set()
                                    for t_mdp_node in smdp.state_action[(f_mdp_node, u)]:
                                        t_prod_node = (t_mdp_node, t_mdp_label, t_ldba_node)
                                        prod_state_action_state[(f_prod_node, u, t_prod_node)] = smdp.edges_prob[(f_mdp_node, t_mdp_node)]
                                        t_prod_node_set.add(tuple(t_prod_node))
                                    prod_state_action[(f_prod_node, u)] = t_prod_node_set 
                                    
                    #define epsilon transitions
                    for t_ldba_node in ldba.nodes:
                        truth = ldba.check_epsilon_transition_for_dra_edge(
                                    f_mdp_label, f_ldba_node, t_ldba_node)
                        if truth:
                            t_prod_node_eps_set = set()
                            t_prod_node_eps = (f_mdp_node, f_mdp_label, t_ldba_node)
                            t_prod_node_eps_set.add(t_prod_node_eps)
                            prod_state_action[(f_prod_node, self.U_EPS[t_ldba_node])] = t_prod_node_eps_set
                            prod_state_action_state[(f_prod_node, self.U_EPS[t_ldba_node], t_prod_node_eps)] = 1
        self.prod_state_action = prod_state_action
        self.prod_state_action_state = prod_state_action_state

#----------------------------------------
#--------------Optimal policy synthesis under LTL--------------------------
def syn_full_plan(prod_mdpst, mdpst, AMEC, SR):
    # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    print("==========[Optimal full plan synthesis start]==========")
    Plan = []
    for S_fi in AMEC:
        print("---for one S_fi---")
        plan = []
        for k, MEC in enumerate(S_fi):
            index_prefix, v_new = optimal_plan_prefix(prod_mdpst, mdpst, MEC, SR)
            plan_prefix = index_prefix
            prefix_risk = 1 - v_new[prod_mdpst.prod_init[0]] 
            print("Best plan prefix obtained, risk %s" %
                  str(prefix_risk))
            if plan_prefix:
                plan.append([plan_prefix, v_new, prefix_risk])
        if plan:
            best_k_plan = min(plan, key=lambda p: p[2])
            Plan.append(best_k_plan)
        else:
            print("No valid found!")
    if Plan:
        print("=========================")
        print(" || Final compilation  ||")
        print("=========================")
        best_all_plan = min(Plan, key=lambda p: p[2])
        print('Best plan prefix obtained for %s states in Sr' %
              str(len(best_all_plan[0])))
        print('risk: %s ' %best_all_plan[2])
        return best_all_plan
    else:
        print("No valid plan found")
        return None


def optimal_plan_prefix(prod_mdp, mdp, MEC, SR):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    sf = MEC[0]
    Sr = set()
    v_old = dict()
    for node in prod_mdp.prod_nodes:
        if node not in sf:
            Sr.add(node)
            v_old[node] = 0
        else:
            v_old[node] = 1
    print('Number of suffix states: %s' %len(sf))
    print('Number of prefix states: %s' %len(SR))
    for init_node in prod_mdp.prod_init:
        # ---------solve vi------------
        print('-----')
        print('Value iteration for prefix starts now')
        print('-----')
        num_iteration = 0
        num_num = 0
        delta_old = 1
        for num_iteration in range(5):
            print('Number of interation: %s' %num_iteration)
            if delta_old > 0.001:
                v_new, index_prefix, delta_new = optimal_prefix_value_iteration(prod_mdp, mdp, SR, v_old)
                for s in SR:                    
                    v_old[s] = v_new[s]
                for s in sf:
                    v_old[s] = 1
                num_iteration += 1
                num_num += 1
            else:   
                num_iteration += 1 
            
            delta_old = delta_new
            print(delta_old)
                
        print("Prefix Value iteration completed in interations: %s" %num_num)
        print("delta: %s" %delta_new)
    
    # for node in sf:
    #     v_new[node] = 1

    return index_prefix, v_new

def optimal_prefix_value_iteration(prod_mdp, mdp, Sr, v_old):
    num1 = len(Sr)
    U = prod_mdp.U
    PROP = prod_mdp.PROP
    #print(PROP)
    v_new = dict()
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    index = dict()
    delta = 0
    #print(vlist)
    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U):
            if (s, u) in prod_mdp.prod_state_action.keys():
                t_set = prod_mdp.prod_state_action[(s, u)]
                #print(len(t_set))
                for k, t_group in enumerate(t_set):
                    #print(t_group)
                    #print(k)
                    if len(t_set) == 1:
                        pe = 1
                    else:
                        pe = prod_mdp.prod_state_action_state[(s, u, t_group)]
                    #print(pe)
                    fd = t_group[0]
                    #print(fd)
                    dd = t_group[2]
                    #print(dd)
                    if len(fd) > 0:
                        v_group = []
                        for t_node in fd: 
                            #print(t_node)
                            for fl, prob_fl in mdp.nodes[t_node].items(): 
                                #print(fl)
                                node_valid = (t_node, fl, dd)
                                ss = tuple(node_valid) 
                                if ss in list(v_old.keys()): 
                                    if v_old[ss]:                
                                        v_group.append(v_old[ss])
                        #print(v_group)
                        if len(v_group)>0:
                            vlist[idx][idu] += pe*min(v_group)              
    #print(vlist)
    for idx, s in enumerate(Sr):
        v_new[s], index[s] =  max((value, index) for index, value in enumerate(vlist[idx]))
        error = abs(v_new[s] - v_old[s])
        if error > delta:
            delta = error
    # print(v_new)
    # print(index) 
    # print(delta)  

    return v_new, index, delta

#----------------------------------------
#--------------Cost minimization under LTL constraints--------------------------
def syn_prefix_plan_vi(prod_mdp, mdp, AMEC, suffix_cost, beta, inf_cost, gamma):
     # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    f_plan_prefix_dict = open('to_matlab/data/plan_prefix.dat','w')
    f_plan_probability_dict = open('to_matlab/data/plan_probability.dat','w')
    f_plan_cost_dict = open('to_matlab/data/plan_cost.dat','w')
    f_plan_total_dict = open('to_matlab/data/plan_total.dat','w')
    print("==========[Optimal full plan synthesis start]==========")
    for l, S_fi in enumerate(AMEC):
        print("---for one S_fi---")
        for k, MEC in enumerate(S_fi):            
            plan_prefix, vtotal, v, vcost = syn_plan_prefix(
                prod_mdp, mdp, MEC, beta, suffix_cost, inf_cost, gamma)
            for s in plan_prefix.keys():
                index = plan_prefix[s]
                prob = v[s]
                cost = vcost[s]
                total = vtotal[s]
                f_plan_prefix_dict.write('%s, %s\n' %(s, index))
                f_plan_probability_dict.write('%s, %s\n' %(s, prob))
                f_plan_cost_dict.write('%s, %s\n' %(s, cost))
                f_plan_total_dict.write('%s, %s\n' %(s, total))

    if plan_prefix:
        return [plan_prefix, vtotal, v, vcost]
    else:
        return [None, None, None, None]

def syn_plan_prefix(prod_mdp, mdp, MEC, beta, suffix_cost, inf_cost, gamma):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    # ----with bounded risk and minimal expected total cost----
    print("===========[plan prefix synthesis starts]===========")
    sf = MEC[0]
    Sr = set()
    for node in prod_mdp.prod_nodes:
        if node not in sf:
            Sr.add(node)
    v_old = dict()
    vcost_old = dict()
    v_new = dict()
    vcost_new = dict()
    vtotal_new = dict()
    for init_node in prod_mdp.prod_init:
        # ---------solve vi------------
        print('-----')
        print('Value iteration for prefix starts now')
        print('-----')
        for s in prod_mdp.prod_nodes:
            if s in sf:
                v_old[s] = 1
                v_new[s] = 1
                if s in suffix_cost.keys():
                    vcost_old[s] = suffix_cost[s]
                    vcost_new[s] = suffix_cost[s]
                else:
                    vcost_old[s] = inf_cost
                    vcost_new[s] = inf_cost
            else:
                v_old[s] = 0
                vcost_old[s] = inf_cost

        num_iteration = 0
        num_num = 0
        delta_old = 1
        for num_iteration in range(100):
            if delta_old > 0.001:
                vtotal_new, v_new, vcost_new, index_prefix, delta_new, delta_cost = value_iteration_v2(prod_mdp, mdp, Sr, v_old, vcost_old, beta, inf_cost, gamma)
                for s in Sr:                    
                    v_old[s] = v_new[s]
                    vcost_old[s] = vcost_new[s]
                num_iteration += 1
                delta_old = delta_new
                num_num += 1
            else:   
                num_iteration += 1 
                
        print("Prefix Value iteration completed in interations: %s" %num_num)
        print("delta: %s" %delta_new)
        print("delta cost: %s" %delta_cost)
        return index_prefix, vtotal_new, v_new, vcost_new

def value_iteration(prod_mdp, mdp, Sr, v_old, vcost_old, beta, inf_cost, gamma):
    num1 = len(Sr)
    U = prod_mdp.U
    PROP = prod_mdp.PROP
    print(PROP)
    C = prod_mdp.C
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    vlist_cost = [[inf_cost] * num2 for _ in range(num1)]
    vlist_total = [[beta*0-(1-beta)*inf_cost] * num2 for _ in range(num1)]
    v_new = dict()
    vcost_new =dict()
    vtotal_new =dict()
    index = dict()
    delta = 0
    delta_cost = 0
    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U):
            ce = C[idu]
            vlist_cost[idx][idu] = ce
            if (s, u) in prod_mdp.prod_state_action.keys():
                t_set = prod_mdp.prod_state_action[(s, u)]
                print(t_set)
                for k, t_group in enumerate(t_set):
                    print(t_group)
                    print(k)
                    pe = PROP[idu][k]
                    v_group = []
                    vcost_group = []
                    fd = t_group[0]
                    print(fd)
                    dd = t_group[2]
                    print(dd)
                    if len(fd) > 0:
                        for t_node in fd: 
                            for fl, prob_fl in mdp.nodes[t_node].items(): 
                                print(fl)
                                node_valid = (t_node, fl, dd)
                                ss = tuple(node_valid) 
                                if ss in list(v_old.keys()) and ss in list(vcost_old.keys()): 
                                    if v_old[ss]:                
                                        v_group.append(v_old[ss])
                                    if vcost_old[ss]:
                                        vcost_group.append(vcost_old[ss])
                        if len(v_group)>0:
                            vlist[idx][idu] += pe*min(v_group)
                        if len(vcost_group)>0:
                            vlist_cost[idx][idu] += pe*sum(vcost_group)/len(vcost_group)                

    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U): 
            vlist_total[idx][idu] = beta*vlist[idx][idu] - (1-beta)*vlist_cost[idx][idu]

        vtotal_new[s], index[s] =  max((value, index) for index, value in enumerate(vlist_total[idx]))
        v_new[s] = vlist[idx][index[s]]
        vcost_new[s] = vlist_cost[idx][index[s]]

        error = abs(v_new[s] - v_old[s])
        error_cost = abs(vcost_new[s] - vcost_old[s])
        if error > delta:
            delta = error
        if error_cost > delta_cost:
            delta_cost = error_cost

    return vtotal_new, v_new, vcost_new, index, delta, delta_cost

def value_iteration_v2(prod_mdp, mdp, Sr, v_old, vcost_old, beta, inf_cost, gamma):
    num1 = len(Sr)
    U = prod_mdp.U
    PROP = prod_mdp.PROP
    print(PROP)
    C = prod_mdp.C
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    vlist_cost = [[inf_cost] * num2 for _ in range(num1)]
    vlist_total = [[inf_cost-beta*(0-gamma)] * num2 for _ in range(num1)]
    v_new = dict()
    vcost_new =dict()
    vtotal_new =dict()
    index = dict()
    delta = 0
    delta_cost = 0
    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U):
            ce = C[idu]
            vlist_cost[idx][idu] = ce
            if (s, u) in prod_mdp.prod_state_action.keys():
                t_set = prod_mdp.prod_state_action[(s, u)]
                print(t_set)
                for k, t_group in enumerate(t_set):
                    print(t_group)
                    print(k)
                    pe = PROP[idu][k]
                    v_group = []
                    vcost_group = []
                    fd = t_group[0]
                    print(fd)
                    dd = t_group[2]
                    print(dd)
                    if len(fd) > 0:
                        for t_node in fd: 
                            for fl, prob_fl in mdp.nodes[t_node].items(): 
                                print(fl)
                                node_valid = (t_node, fl, dd)
                                ss = tuple(node_valid) 
                                if ss in list(v_old.keys()) and ss in list(vcost_old.keys()): 
                                    if v_old[ss]:                
                                        v_group.append(v_old[ss])
                                    if vcost_old[ss]:
                                        vcost_group.append(vcost_old[ss])
                        if len(v_group)>0:
                            vlist[idx][idu] += pe*min(v_group)
                        if len(vcost_group)>0:
                            vlist_cost[idx][idu] += pe*sum(vcost_group)/len(vcost_group)                

    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U): 
            vlist_total[idx][idu] =  vlist_cost[idx][idu] - beta*(vlist[idx][idu]-gamma)

        vtotal_new[s], index[s] =  min((value, index) for index, value in enumerate(vlist_total[idx]))
        v_new[s] = vlist[idx][index[s]]
        vcost_new[s] = vlist_cost[idx][index[s]]

        error = abs(v_new[s] - v_old[s])
        error_cost = abs(vcost_new[s] - vcost_old[s])
        if error > delta:
            delta = error
        if error_cost > delta_cost:
            delta_cost = error_cost

    return vtotal_new, v_new, vcost_new, index, delta, delta_cost


