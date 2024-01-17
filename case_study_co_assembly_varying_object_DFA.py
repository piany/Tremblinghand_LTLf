import subprocess

from MDP_TG.MDPST import Motion_MDPST, syn_full_plan_mdpst
from MDP_TG.dfa import Dfa, Product_MDPST_dfa
from networkx import single_source_shortest_path

from networkx.classes.digraph import DiGraph
from ortools.linear_solver import pywraplp
from MDP_TG.state_pruning import generate_matrices, create_matrices, human_robot_co_assembly_scenario

import time

#----real example----
# obj_num_set = [2]
obj_num_set = [2, 3, 4, 5, 6]
human_action_set = [3]
all_matrices_dict = dict()
states_edges_num = dict()
prod_states_edges_num = dict()
construction_synthesis_time = dict()
for obj_num in obj_num_set:
    odd_numbers = []
    even_numbers = []
    for num in range(1, obj_num):
        if num % 2 != 0:
            odd_numbers.append(num)
        else:
            even_numbers.append(num)
    for human_num in human_action_set:
        t0 = time.time()
        if obj_num <= 2:
            set_all_matrix = generate_matrices(obj_num, obj_num+1)
            all_matrices_dict[obj_num] = set_all_matrix
        elif obj_num > 2:
            set_all_matrix = create_matrices(all_matrices_dict[obj_num-1], obj_num)
            all_matrices_dict[obj_num] = set_all_matrix

        valid_matrix, U, U_h, target, obstacle, init_node_robot, human_action_allowed = human_robot_co_assembly_scenario(obj_num, human_num, set_all_matrix)

        robot_nodes = dict()
        for matrix in valid_matrix:
            key = tuple(tuple(row) for row in matrix)
            if key == target:       
                robot_nodes[key] = {frozenset(['target']): 1.0}
            if key in obstacle:
                robot_nodes[key] = {frozenset(['obstacle']): 1.0}
            else:
                robot_nodes[key] = {frozenset(): 1.0}

        #----robot transitions: probabilistic----
        P = dict()
        C = dict()
        for u in U[1:]:
            P[u] = [0.9, 0.1]
            C[u] = 1

        robot_edges = dict()
        robot_state_action =dict()
        for fnode in list(robot_nodes.keys()):
            tnode = [list(row) for row in fnode]
            tnode1 = [list(row) for row in fnode]
            robot_edges[(fnode, U[0], fnode)] = (1, 0)
            robot_state_action[(fnode, U[0])] = [fnode]
            for u in U[1:]:   
                obj = u[0]
                loc = u[1]
                row = tnode[obj]
                for idx in range(len(row)):
                    if row[idx] == 1:
                        tnode[obj][idx] = 0
                        tnode1[obj][idx] = 0
                if loc in odd_numbers:
                    tnode[obj][loc] = 1
                    tnode1[obj][loc+1] = 1
                elif loc in even_numbers:
                    tnode[obj][loc] = 1
                    tnode1[obj][loc-1] = 1
                else:
                    tnode[obj][loc] = 1
                    tnode1[obj][loc] = 1
                tnode_tuple = tuple(tuple(row) for row in tnode)
                tnode_tuple1 = tuple(tuple(row) for row in tnode1)
                #print(tnode_tuple)
                if not tnode_tuple == fnode and tnode_tuple in list(robot_nodes.keys()):
                    if loc == 0 or loc == obj_num:
                        t_nodes = [tnode_tuple, fnode]
                        robot_edges[(fnode, u, tnode_tuple)] = (P[u][0], C[u])
                        robot_edges[(fnode, u, fnode)] = (P[u][1], C[u])
                        robot_state_action[(fnode, u)] = t_nodes
                    else:
                        if not tnode_tuple1 == fnode and tnode_tuple1 in list(robot_nodes.keys()):
                            t_nodes = [tnode_tuple, fnode, tnode_tuple1]
                            robot_edges[(fnode, u, tnode_tuple)] = (P[u][0], C[u])
                            robot_edges[(fnode, u, fnode)] = (P[u][1]/2, C[u])
                            robot_edges[(fnode, u, tnode_tuple1)] = (P[u][1]/2, C[u])
                            robot_state_action[(fnode, u)] = t_nodes
                        else:
                            t_nodes = [tnode_tuple, fnode]
                            robot_edges[(fnode, u, tnode_tuple)] = (P[u][0], C[u])
                            robot_edges[(fnode, u, fnode)] = (P[u][1], C[u])
                            robot_state_action[(fnode, u)] = t_nodes

                tnode = [list(row) for row in fnode]
        print(len(robot_edges))

        state_action_map = dict()
        for fnode in robot_nodes.keys():
            act_set = []
            for u in U:
                if (fnode, u) in robot_state_action.keys() and u in U_h:
                    act_set.append(u)
            state_action_map[fnode] = act_set
        #print(state_action_map)

        #----human transitions: nodeterministic----
        human_edges = dict()
        for fnode in list(robot_nodes.keys()):
            tnode = [list(map(int, item)) for item in fnode]
            tnodes_set = []
            tnodes_set.append(fnode)
            for u in state_action_map[fnode]:
                if not u == tuple('ST'):
                    obj = u[0]
                    loc = u[1]
                    row = tnode[obj]
                    for idx in range(1, len(row)):
                        if row[idx] == 1:
                            tnode[obj][idx] = 0
                    tnode[obj][loc] = 1
                    tnode_tuple = tuple(tuple(row) for row in tnode)
                    tnodes_set.append(tnode_tuple)
            human_edges[fnode] = tnodes_set
        print(len(human_edges))

        #----define MDPST----
        human_robot_nodes = dict()
        acc_states = set()
        AccStates = set()
        obstacle_states = set()
        for fnode, prop in robot_nodes.items():
            for count in range(human_action_allowed+1):
                human_robot_nodes[(fnode, count)] = prop
                if fnode == target:
                    node_count = (fnode, count)
                    acc_states.add(node_count)
                    AccStates.add(node_count)
                if fnode in obstacle:
                    node_count = (fnode, count)
                    obstacle_states.add(node_count)
        print(len(human_robot_nodes))
        #print(human_robot_nodes)

        #----
        human_robot_edges = dict()
        human_robot_state_action = dict()
        for fnode, prop in robot_nodes.items():
            for count in range(human_action_allowed+1):
                fnode_hr = (fnode, count)
                if fnode_hr:
                    for u in state_action_map[fnode]:
                        tnodes = robot_state_action[(fnode, u)]
                        tt_set_ST = []
                        #print(len(tnodes))
                        for tnode in tnodes:
                            (prob, cost) = robot_edges[(fnode, u, tnode)]
                            #print(prob)
                            if count >= human_action_allowed:
                                tnode_hr = (tnode, count)
                                tnode_hr_set = ()
                                if tnode_hr in list(human_robot_nodes.keys()):
                                    tnode_hr_set += (tuple(tnode_hr),)
                                    human_robot_edges[(fnode_hr, u, tnode_hr_set)] = (prob, cost)
                                    tt_set_ST.append(tnode_hr_set)
                            else:
                                tnode_human_set = human_edges[tnode]
                                tnode_hr_set = ()
                                tnode_hr1 = (tnode, count)                                
                                if tnode_hr1 in list(human_robot_nodes.keys()):
                                    tnode_hr_set += (tuple(tnode_hr1),)
                                tnode_human_set2 = set(tnode_human_set).difference(tnode)
                                for tnode_human in tnode_human_set2:
                                    tnode_hr = (tnode_human, count+1)
                                    if tnode_hr in list(human_robot_nodes.keys()):
                                        tnode_hr_set += (tuple(tnode_hr),)
                                tt_set_ST.append(tnode_hr_set)
                                human_robot_edges[(fnode_hr, u, tnode_hr_set)] = (prob, cost)
                        human_robot_state_action[(fnode_hr, u)] = tt_set_ST
        print(len(human_robot_edges))
                    
        #----
        initial_node = (init_node_robot, 0)
        if initial_node in list(human_robot_nodes.keys()):
            initial_label = human_robot_nodes[initial_node].keys()
            print("Initial node OK!")
        else:
            print("Initial node error!")

        #----
        motion_mdpst = Motion_MDPST(human_robot_nodes, human_robot_edges, human_robot_state_action, U, C,
                                initial_node, initial_label)

        t1 = time.time()
        print('MDPST constructed in %s seconds.' %str(t1-t0))
        states_edges_num[(obj_num, human_num)] = (str(len(motion_mdpst.nodes.keys())), str(len(motion_mdpst.edges.keys())))
          
        #----compute DFA----
        #reach_avoid = '! obstacle U target'
        # statenum = 3
        # init = 1
        # edges = {(1, 1): ['00'],
        #         (1, 2): ['01' '11'],
        #         (1, 3): ['10'],
        #         (2, 2): ['00', '01', '10', '11'],
        #         (3, 3): ['00', '01', '10', '11'],
        #         }
        # aps = ['obstacle', 'target']
        # acc = [[{2}]]
        # dfa = Dfa(statenum, init, edges, aps, acc)


        cmd = "./LydiaSyft --spec-file goal.ltlf"
        out = subprocess.check_output(cmd, shell=True)
        lines = str(out).split('\\n')
        print(lines)
        aps = []
        edges = {}

        for line in lines:
            if "Number of states" in line:
                statenum = int(line[-1])
                continue
            if "DFA for formula with free variables:" in line:
                items = line.split("DFA for formula with free variables: ")
                line = items[1]
                for item in line.split(' '):
                    if item != '':
                        aps.append(item.strip(' '))
                continue

            if "Initial state" in line:
                init = int(line[-1])+1
                continue
            if "Accepting states" in line:
                items = line.split("Accepting states: ")
                acclist = []
                for item in items[1].split(' '):
                    if item != '':
                        acclist.append(int(item.strip(' '))+1)
                acc = [[set(acclist)]]
                continue
            if "->" in line:
                items = line.split(' ')
                curr = int(items[1].strip(':'))+1
                con = items[2]
                succ = int(items[5].strip(' '))+1
                cons = [con]
                if 'X' in con:
                    flag = True
                else:
                    flag = False
                while flag == True:
                    c = len(cons)
                    for item in cons:
                        if 'X' in item:
                            item_0 = item.replace('X', '0', 1)
                            item_1 = item.replace('X', '1', 1)
                            cons.remove(item)
                            cons.append(item_0)
                            cons.append(item_1)
                            flag = True
                    if c == len(cons):
                        flag = False
                if (curr, succ) in edges.keys():
                    for item in cons:
                        edges[(curr, succ)].append(item)
                else:
                    edges[(curr, succ)] = cons
                continue
        dfa = Dfa(statenum, init, edges, aps, acc)
        print('DFA done.')

        #----
        prod_mdpst = Product_MDPST_dfa(motion_mdpst, dfa, P)
        prod_states_edges_num[((obj_num, human_num))] = (str(len(prod_mdpst.prod_nodes)), str(len(prod_mdpst.prod_edges)))

        #----
        OBS = set()
        TAR = set()
        nodes_list = set(prod_mdpst.prod_nodes)
        for node in nodes_list:
            if node[0] in obstacle_states or node[2] == 3:
                OBS.add(node)
            elif node[0] in AccStates or node[2] == 2:
                TAR.add(node)
        Srr = nodes_list.difference(OBS)
        Sr = Srr.difference(TAR)
        print("Number of accepting states: %s" %len(TAR))
        print("Number of obstacle states: %s" %len(OBS))
        print("Number of prefix states: %s" %len(Sr))

        #----
        plan_prefix, v_new = syn_full_plan_mdpst(prod_mdpst, TAR, Sr)
        #print(v_new)
        t2 = time.time()
        print('Synthesis completed in %s seconds.' %str(t2-t1))

        construction_synthesis_time[(obj_num, human_num)] = (str(t1-t0), str(t2-t1), str(t2-t0))

print(states_edges_num)
print(construction_synthesis_time)