'''
This file provides functions to create and manipulate a MultiAspect Graph ment to represent patient pathways.
'''

from datetime import datetime
import pandas as pd
import warnings

import networkx as nx

import MAG as mag

# ---------------------------------------------------------------------------------------- 

def convert_df_into_MAG(df,
    aspect_columns,
    timestamp_column,
    patient_id_column,
    create_aspect_sequence = True,
    create_aspect_patient = False,
    timestamp_format = '%Y-%m-%d %H:%M:%S',
    sorting_column = None,
    add_virtual_start_end = True,
    timestamp_as_edge_attribute = False,
    l_min = 0,
    graph_name = ''):
    
    '''
    Convert a pandas DataFrame into a MultiAspect Graph.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
      A Pandas dataframe

    aspect_columns : list
      List of the dataframe columns names that will become aspects of the MAG

    timestamp_column : string
      Name of the dataframe column containing the timestamp of the events

    patient_id_column : string
      Name of the dataframe column containing the patient ID

    create_aspect_sequence : bool, optional (default = True)
      If True, the returned MAG will have the aspect 'Sequence'

    create_aspect_patient : bool, optional (default = False)
      If True, the returned MAG will have the aspect 'Patient'

    timestamp_format : string, optional (default = '%Y-%m-%d %H:%M:%S')
      Specifies the date/timestamp format

    sorting_column : string, optional (default = None)
      Name of a column in the dataframe that can be used to define the order of two events with the same patient and timestamp

    add_virtual_start_end : bool, optional (default = True)
      If True, virtual start and end nodes are included in the MAG

    timestamp_as_edge_attribute : bool (default = False)
      If True, besides 'patient_id' and 'interval', the edges will have the attributes 'origin_timestamp' and 'target_timestamp'

    l_min : integer, optional (default = 0)
      Minimum pathway length for it to be included in the MAG

    graph_name : string, optional (default = '')
      String to be used as the MAG name

    Returns
    -------
    graph : mag.MultiAspectMultiDiGraph

    '''

    # ------------------------------------------------------

    list_of_edges = []

    # ----------
    
    list_of_columns = [timestamp_column] + aspect_columns 
    
    # -----------
    # create list of dataframes with the pathways that will be merged

    list_of_pathways = []
    
    for column in list_of_columns:
        
        temp = pd.DataFrame(
            df.sort_values([timestamp_column]+[sorting_column]
                           if sorting_column is not None
                           else [timestamp_column]
                          ).astype(str).groupby(patient_id_column)[column].apply(';'.join))
        temp.columns = ['pathway_'+column]

        list_of_pathways.append(temp)

    # ----------
    # merge the elements of list_of_pathways
    # the result is a dataframe whose index is the patient_id and the columns contain each type of pathway 
    cont = 0
    for temp_pathway in list_of_pathways:
        
        if cont == 0:
            df_pathways = temp_pathway
        
        else:
            df_pathways = df_pathways.merge(temp_pathway, left_index=True, right_index = True, validate = '1:1')

        cont += 1
        
    df_pathways = df_pathways.reset_index()
    
    del list_of_pathways
    
    # -----------
    
    # --- consider only pathways whose length >= l_min
    if add_virtual_start_end:
        l_min = l_min + 2 # len = 5 : start - 1 - 2 - 3 - end
    
    for i, patient_row in df_pathways.iterrows():
        
        patients_pathways = []

        # patient id
        id_patient = patient_row[patient_id_column]

        # passar as trajetórias em string para lista
        
        for pathway in ['pathway_'+ x for x in list_of_columns] :

            # date pathway 
            if pathway == 'pathway_'+timestamp_column :
                temp_pathway = patient_row[pathway].split(';')
                temp_pathway = [datetime.strptime(date,timestamp_format).date() for date in temp_pathway]
            
            # other pathways
            else:
                # remove commas and parentheses from the strings (otherwise the mag will misunderstand the edgelist)
                temp_pathway = patient_row[pathway].replace(',',"_").replace('(',"_").replace(')',"_").split(';')
            
            # add 'start' and 'end', 
            if add_virtual_start_end == True: 
                temp_pathway = ['start'] + temp_pathway + ['end']
                
            patients_pathways.append(temp_pathway)
        
        # convert the patient's pathway into an edgelist
        cont = 0

        len_patients_pathway = len(patients_pathways[0]) # patients_pathways[0] is the timestamp pathway
        
        if len_patients_pathway >= l_min :  

            for idx_origin in range(len_patients_pathway - 1):

                idx_target = idx_origin + 1

                if (idx_origin == 0) & (add_virtual_start_end == True) : # origin = 'start'
                    interval = 0

                elif (idx_target == len_patients_pathway - 1) & (add_virtual_start_end == True) : # last event before 'end'
                    interval = 0

                else : 
                    # tempo entre o evento de origem e o de destino
                    interval = (patients_pathways[0][idx_target]-patients_pathways[0][idx_origin]).days

                # edge
                origin_node = ((str(id_patient)+",") if create_aspect_patient else '')
                target_node = ((str(id_patient)+",") if create_aspect_patient else '')
                
                for pathway_type in patients_pathways[1:]:
                    
                    origin_node = origin_node + (pathway_type[idx_origin] + ",")
                    target_node = target_node + (pathway_type[idx_target] + ",")
                
                if create_aspect_sequence:
                    origin_node = origin_node + ("S" + str(cont))
                    target_node = target_node + ("S" + str(cont+1))
                else:
                    origin_node = origin_node[:-1] # remove last comma
                    target_node = target_node[:-1] # remove last comma

                        
                if timestamp_as_edge_attribute:
                    
                    origin_timestamp = patients_pathways[0][idx_origin]
                    target_timestamp = patients_pathways[0][idx_target]
                    
                    edge = (tuple(origin_node.split(',')), tuple(target_node.split(',')), 
                              {'patient' : id_patient , 
                               'interval' : float(interval),
                               'origin_timestamp' : str(origin_timestamp),
                               'target_timestamp' : str(target_timestamp)
                              })
                else:
                    edge = (tuple(origin_node.split(',')), tuple(target_node.split(',')), 
                              {'patient' : id_patient ,
                               'interval' : float(interval)})
                    
                list_of_edges.append(edge)

                #assert(cont == idx_origin)
                cont = cont+1

    # ------------------------------------------------------

    graph = mag.MultiAspectMultiDiGraph() 

    # importing edges 
    graph.add_edges_from(list_of_edges)

    # graph name
    graph.name = graph_name

    # ----------------------------------------------------------

    return graph

# --------------------------------------------------------------

def contract_multiMAG_nodes(graph, attribute_name, attribute_value, aspect_to_substitute = 0):
    
    '''
    Contracts the nodes of a MultiAspect MultiDiGraph that represents patient pathways.

    Parameters
    ----------
    graph : mag.MultiAspectMultiDiGraph
      The MultiAspect graph whose nodes will be contracted. It must be a MultiDiGraph.

    attribute_name : string
      The name of the node attribute that will be used to find nodes of the same type.

    attribute_value : string or numeric
      All nodes whose attribute_name equals attribute_value will be contracted.

    aspect_to_substitute : integer, optional (default = 0)
      The index of the aspect whose value should be replaced by attribute_value.

    Returns
    ----------
    H : mag.MultiAspectMultiDiGraph

    '''
        
    # create a copy of the MAG
    H = graph.copy()
    H.name = graph.name + ' with nodes whose attribute ' + attribute_name + ' = ' + attribute_value + ' are grouped'
    
    if (not graph.is_multigraph()) | (not graph.is_directed()):
        raise Exception('The provided graph is not a MultiAspect MultiDiGraph.')

    # scan nodes
    for node, node_att in graph.nodes(data = True):
        
        # if the node is one of those that must be contracted
        if node_att[attribute_name] == attribute_value:
            
            # get the contracted version of the node  
            new_node = list(node)
            new_node[aspect_to_substitute] = attribute_value
            new_node = tuple(new_node)
            
            # add the contracted node to the graph, if it is not there already
            if not(new_node in H.nodes()): 
                H.add_node(new_node)

            # update incoming edges
            for origin, target, key, att in graph.in_edges(node, data = True, keys = True):

                # if the origin node is not in the original graph, it has been contracted
                if not(origin in H.nodes()):
                    origin = list(origin)
                    origin[aspect_to_substitute] = attribute_value
                    origin = tuple(origin)
                # add edge
                H.add_edges_from([(origin, new_node, att)])

            # update outgoing edges        
            for origin, target, key, att in graph.out_edges(node, data = True, keys = True):
                # if the target node is not in the original graph, it has been contracted
                if not(target in H.nodes()):
                    target = list(target)
                    target[aspect_to_substitute] = attribute_value
                    target = tuple(target)
                # add edge
                H.add_edges_from([(new_node, target, att)])

            # add nodes attributes if there is any
            for dict_att in graph.nodes[node].items():
                if dict_att[0] in H.nodes[new_node]:
                    H.nodes[new_node][dict_att[0]].append(dict_att[1])
                else:
                    H.nodes[new_node][dict_att[0]] = [dict_att[1]]

            # remove the non-contracted node from graph H
            H.remove_node(node)
    
    H.compact_aspects_list() # update aspect list
    
    return H

# ----------------------------------------------------------------------

def zeta_zero(zeta):
    if (zeta):
        t = len(zeta)
        if zeta.count(0) == t:
            return True
        return False

def zeta_one(zeta):
    if (zeta):
        t = len(zeta)
        if zeta.count(1) == t:
            return True
        return False
    
def subdetermination(graph, zeta, multi=True, direct=True, loop=False, edge_frequency = False, **attr):

    ''' Function adapted from the subdetermination method to keep the edge attributes when subdetermining a MAG '''
    
    if len(zeta) != graph.number_of_aspects():
        raise ValueError('The number of elements in zeta is incorrect. The number of aspects in MAG is {}, and {} have been given!'.format(graph.number_of_aspects(),len(zeta)))
    
    #Verify the basic cases
    if (zeta_zero(zeta) == True):
        print("All aspects supressed. Null returned")
        return None
    
    if (zeta_one(zeta) == True):
        print("None aspect was suppressed. The same MAG is returned")
        return graph
    
    #variables
    lenz = len(zeta)
    asps = list(zeta)+list(zeta)
    total = len(asps)
 #   naspects = zeta.count(1)
    if multi:
        H = mag.MultiAspectMultiDiGraph(**attr) if direct else mag.MultiAspectMultiGraph(**attr)
    else:
        H = mag.MultiAspectDiGraph(**attr) if direct else mag.MultiAspectGraph(**attr)

    #edge list verification
    for e, datadict in graph.edges.items():
        new_edge = [(e[0][i]) if i<lenz else (e[1][i-lenz]) for i in range (total) if asps[i]!=0]        
        From = tuple(new_edge[0:int(len(new_edge)/2)])
        To = tuple(new_edge[int(len(new_edge)/2):len(new_edge)])
        if (From != To and loop == False) or loop == True:
            if multi:
                H.add_edges_from([(From, To, datadict)])
            else:
                # -----------------------------
                if edge_frequency:
                    if (From,To) in H.edges(): 
                        H[From][To]['freq'] += 1 
                    else: 
                        H.add_edge(From,To) 
                        H[From][To]['freq'] = 1 
                else:
                    H.add_edge(From,To) 
                # ------------------------------   
               
    #node list verification
    for n in graph.nodes():
        node = [n[i] for i in range(0,len(n)) if zeta[i] !=0]
        H.add_node(tuple(node))
      
    #return the subdetermination of MAG
    return H

# ----------------------------------------------------------------

def filter_MAG(graph, nodelist, bool_keep_nodes = False, patient_attribute = 'patient'):
    
    '''
    Filter nodes of a MultiAspect Graph representing patient pathways, 
    while reconnecting the origin and target nodes of the removed one.
    The attribute "interval" of the edges is summed when reconnecting 
    the origin and target nodes of the removed one.

    Parameters
    ----------
    graph : mag.MultiAspectMultiDiGraph
      The MultiAspect graph whose nodes will be filtered. It must be a MultiDiGraph 
      with an attribute identifying the patients in its edges.

    nodelist : list
      The list of nodes to be kept or removed (according to bool_keep_nodes)

    bool_keep_nodes : bool, optional (default = False)
      If True, the nodes in nodelist are kept in graph; otheerwise, they are filtered.

    patient_attribute : string, optional (default = 'patient')
      The name of the edge attribute which identifies its patient.

    Returns
    ----------
    filtered_graph : mag.MultiAspectMultiDiGraph

    '''
    
    filtered_graph = graph.copy()
    if (not filtered_graph.is_multigraph()) | (not filtered_graph.is_directed()):
        raise Exception('The provided graph is not a MultiAspect MultiDiGraph.')
    
    # -------
    if bool_keep_nodes == True:
        nodes_to_remove = [node for node in filtered_graph.nodes() if not(node in nodelist)]
    else:
        nodes_to_remove = nodelist
    # -------
    
    for node in nodes_to_remove:
        
        predecessors = list(filtered_graph.predecessors(node))
        successors = list(filtered_graph.successors(node))
        
        # --- if the node has predecessors and a successors, we must connect them; otherwise, we can simply remove the node
        if ((len(predecessors)!=0)&(len(successors)!=0)):
        
            # -- identify the patients in the edges and the associated edge keys
            # ---- predecessors
            dict_pacients_edgeKeys_predecessors = {}
            for predecessor in predecessors:
                # if there is only 1 edge between the predecessor and the node to be removed (predecessors is a tuple)
                if type(filtered_graph[predecessor][node]) == tuple: 
                    patient = filtered_graph[predecessor][node][patient_attribute]
                    dict_pacients_edgeKeys_predecessors[patient] = dict(predecessor = predecessor,
                                                                        edge_key = 0) # é zero se só tem 1 aresta
                else: # if there are multiple edges (predecessors is a dictionary)
                    for key, att in filtered_graph[predecessor][node].items(): 
                        patient = att[patient_attribute]
                        dict_pacients_edgeKeys_predecessors[patient] = dict(predecessor = predecessor,
                                                                            edge_key = key)    
            # ---- successors
            dict_pacients_edgeKeys_successors = {}
            for successor in successors:
                # if there is only 1 edge between the node to be removed and successor (successors is a tuple)
                if type(filtered_graph[node][successor]) == tuple:
                    patient = filtered_graph[node][successor][patient_attribute]
                    dict_pacients_edgeKeys_successors[patient] = dict(successor = successor,
                                                                      edge_key = 0) 
                else: # if there are multiple edges (successors is a dictionary)
                    for key, att in filtered_graph[node][successor].items():
                        patient = att[patient_attribute]
                        dict_pacients_edgeKeys_successors[patient] = dict(successor = successor,
                                                                          edge_key = key)  
            # ------------------------------------------------------------------------
            
            # -- for each patient:
            patient_set = set(list(dict_pacients_edgeKeys_predecessors.keys())+
                              list(dict_pacients_edgeKeys_successors.keys()))
            
            for patient in patient_set:
                
                # if the start/end nodes are removed, a patient will appear only in the incoming or in the outgoing edge
                # in this case, it is not necessary to create a new edge
                if not ((patient in dict_pacients_edgeKeys_predecessors) &
                        (patient in dict_pacients_edgeKeys_successors)):
                    continue
                
                predecessor = dict_pacients_edgeKeys_predecessors[patient]['predecessor']
                predecessor_key = dict_pacients_edgeKeys_predecessors[patient]['edge_key']
                successor = dict_pacients_edgeKeys_successors[patient]['successor']
                successor_key = dict_pacients_edgeKeys_successors[patient]['edge_key']
                
                try:
                    new_edge_key = len(filtered_graph[predecessor][successor]) # the gretest key is len-1, beecause it counts from zero
                except:
                    new_edge_key = 0
                
                incoming_edge_attributes = filtered_graph[predecessor][node][predecessor_key] 
                outgoing_edge_attributes = filtered_graph[node][successor][successor_key]
                new_edge_attributes = {}

                for att in incoming_edge_attributes.keys():
                    if att == 'interval':
                        new_edge_attributes[att] = float(incoming_edge_attributes[att]) + float(outgoing_edge_attributes[att])
                    elif att == patient_attribute:
                        assert incoming_edge_attributes[att] == outgoing_edge_attributes[att]
                        new_edge_attributes[att] = incoming_edge_attributes[att]
                    else: 
                        new_edge_attributes[att] = (
                            incoming_edge_attributes[att]
                            if isinstance(incoming_edge_attributes[att],list) 
                            else [incoming_edge_attributes[att]]) + (
                            outgoing_edge_attributes[att] 
                            if isinstance(outgoing_edge_attributes[att],list)
                            else [outgoing_edge_attributes[att]]
                        ) 
                    
                filtered_graph.add_edges_from([(predecessor,successor,new_edge_key,new_edge_attributes)])

        filtered_graph.remove_node(node)
        
    filtered_graph.compact_aspects_list() # update aspect list
        
    return filtered_graph          

# -------------------------------------------------------------------

def convert_multiMAG_into_diMAG(multigraph):

    '''
    Converts a MultiAspectMultiDiGraph into a MultiAspectDiGraph. 
    When there are multiple edges between two nodes, their attributes are stored 
    as a list in the edge of the resulting MultiAspectDiGraph,
    and an extra attribute 'freq' specifies the original 
    number of edges between the two nodes.

    Parameters
    ----------
    multigraph : mag.MultiAspectMultiDiGraph

    Returns
    ----------
    filtered_graph : mag.MultiAspectDiGraph
    '''
    
    # create a MultiAspectDiGraph from the MultiAspectMultiDiGraph
    digraph = mag.MultiAspectDiGraph() # empty MultiAspectDiGraph
    digraph.name = 'Digraph from ' + multigraph.name

    # add the multigraph nodes in the digraph
    for node, node_att in multigraph.nodes(data = True):
        digraph.add_node(node)
        
        # add node attributes
        for att in node_att.keys():
            digraph.nodes[node][att] = node_att[att]

    # add the multigraph edges in the digraph
    for origin, target, edge_att in multigraph.edges(data = True):
       
        if digraph.has_edge(origin, target): # if another edge between origin and target has already been added to the digraph

            temp_edge_freq = digraph[origin][target]['freq']  # get the edge frequency until now
            temp_edge_freq = temp_edge_freq + 1 # increment the frequency
            digraph[origin][target]['freq'] = temp_edge_freq # attribute the acummulated frequency
            
            for att in edge_att.keys():
                att_list = digraph[origin][target][att] # get the list of the attribute att of the other edges between origin and target 
                att_list.append(edge_att[att])
                digraph.edges[origin,target].update({att : att_list})

        else: # if it is the first edge between origin and target
            
            temp_edge_freq = 1
            
            # create the edge:
            digraph.add_edge(origin, target, freq = temp_edge_freq)
            
            dict_att = {}
            for att in edge_att.keys():
                              
                dict_att[att] = [edge_att[att]]
                digraph.edges[origin, target].update(dict_att)

    return digraph

# -----------------------------------------------------------------------

def convert_mag_into_graph(mag_graph, convert_nodes_into_string = False):
    
    '''
    Converts a MultiAspect Graph into a NetworkX graph.

    Parameters
    ----------
    mag_graph : mag.MultiAspectMultiGraph, or mag.MultiAspectMultiDiGraph, or mag.MultiAspectDiGraph, or mag.MultiAspectGraph
      The MultiAspect graph that will be converted into a NetworkX graph.

    convert_nodes_into_string : bool, optional (default = False)
      If True, the nodes in the NetworkX are converted into strings instead of tuples. 

    Returns
    ----------
    new_graph : nx.MultiGraph, or nx.MultiDiGraph, or nx.DiGraph, or nx.Graph
    '''

    # create a networkx graph according to the MAG type
    if (mag_graph.is_directed()):
        if(mag_graph.is_multigraph()):
            new_graph = nx.MultiDiGraph()
        else:
            new_graph = nx.DiGraph()
    else:
        if(mag_graph.is_multigraph()):
            new_graph = nx.MultiGraph()
        else:
            new_graph = nx.Graph()
            
    # copy nodes and nodes attributes
    for node, node_att in mag_graph.nodes(data = True):
        if convert_nodes_into_string:
            if len(node)==1:
                temp_node = str(node[0])
            else:
                temp_node = str(node)
        else:
            temp_node = node
            
        new_graph.add_node(temp_node)
        
        for key_at, value_att in node_att.items():
            new_graph.nodes[temp_node][key_at] = value_att
        
    # copy edges
    if(not mag_graph.is_multigraph()): # if mag_graph is not a multigraph, the edge key is not necessary
        
        for origin, target, edge_att in mag_graph.edges(data=True):
            
            if convert_nodes_into_string:
                if len(origin)==len(target)==1:
                    temp_origin = str(origin[0])
                    temp_target = str(target[0])
                else:
                    temp_origin = str(origin)
                    temp_target = str(target)
            else:
                temp_origin = origin
                temp_target = target

            new_graph.add_edge(temp_origin,temp_target)
            
            for key_att, value_att in edge_att.items():
                new_graph[temp_origin][temp_target][key_att] = value_att 
    
    else:
        for origin, target, key, edge_att in mag_graph.edges(data=True, keys= True):
            if convert_nodes_into_string:
                if len(origin)==len(target)==1:
                    temp_origin = str(origin[0])
                    temp_target = str(target[0])
                else:
                    temp_origin = str(origin)
                    temp_target = str(target)
            else:
                temp_origin = origin
                temp_target = target

            new_graph.add_edge(temp_origin,temp_target)
            
            for key_att, value_att in edge_att.items():
                new_graph[temp_origin][temp_target][key][key_att] = value_att 
    
    # return the networkx graph
    return new_graph

# ---------------------------------------------------------------

def normalize_centrality_values(centrality_dictionary):

    '''
    Normalizes the node centrality values stored in a dictionary so that 
    the largest centrality value becomes 1 and the lowest one becomes 0.
    '''
    
    normalized_centrality_dictionary = {}
    
    minimum_value = min(list(centrality_dictionary.values()))
    maximum_value = max(list(centrality_dictionary.values()))
    
    if (maximum_value-minimum_value) == 0:
        warnings.warn("The maximum and the minimum values are equal. The same dictionary is returned.")
        return centrality_dictionary

    for key, value in centrality_dictionary.items():
        normalized_centrality_dictionary[key] = (value - minimum_value)/(maximum_value - minimum_value)
        
    return normalized_centrality_dictionary

# ---------------------------------------------------------------

def mag_centralities(G,
                     dict_closeness_intervention, dict_betweenness_occupation, dict_pagerank_unit,
                     idx_intervention = 0, idx_occupation = 1, idx_unit = 2,
                     weight_intervention_vs_occupation = 0.5, weight_complications = 0.5,
                     bool_remove_start_end = True, use_edge_frequency = True,
                     pagerank_alpha = 0.25, show_start_end = False,
                     start_string = 'start', end_string = 'end',
                     return_centrality_dict = False, bool_normalize_result = True
                     ):

    '''
    Obtain the final relevance of G nodes using the PageRank algorithm.


    Parameters
    ----------
    G : mag.MultiAspectMultiDiGraph
      The MultiAspect Graph representing patient pathways.

    dict_closeness_intervention : dictionary
      Dictionary containing the closeness centrality of the nodes in G's subdetermination which keeps only the aspect Intervention.

    dict_betweenness_occupation : dictionary
      Dictionary containing the betweenness centrality of the nodes in G's subdetermination which keeps only the aspect Occupation.

    dict_pagerank_unit : dictionary
      Dictionary containing the PageRank centrality of the nodes in G's subdetermination which keeps only the aspect Healthcare Unit.

    idx_intervention : integer, optional (default = 0)
      The index of the aspect Intervention

    idx_occupation : integer, optional (default = 1)
      The index of the aspect Occupation

    idx_unit : integer, optional (default = 2)
      The index of the aspect Healthcare Unity

    weight_intervention_vs_occupation : float, optional (default = 0.5)
      Real number between 0 and 1 (inclusive) that weights the importance of the intervention closeness to the 
      relevance of the node as a "typical event"

    weight_complications : float, optional (default = 0.5)
      Real number between 0 and 1 (inclusive) that weights the importance of the complication component of the node to its R0 

    bool_remove_start_end : bool, optional (default = True)
      If True, the artificial start and end nodes are not included in the grpah in which the PageRank algorithm is executed

    use_edge_frequency : bool, optional (default = True)
      If True the edge frequency is used as edge weight in the PageRank algorithm

    pagerank_alpha : float, optional (default = 0.25)
      Value of the alpha parameter in the PageRank algorithm

    show_start_end : bool, optional (default = False)
      If True, the artificial start and end nodes will not be removed from the returned graph, 
      even if they were not included in the PageRank algorithm execution according to bool_remove_start_end

    start_string : string, optional (default = 'start')
      What string is used to label the start node

    end_string : string, optional (default = 'end')
      What string is used to label the end node

    return_centrality_dict : bool, optional (default = False)
      If True, besides the graph, a dictionary whose keys are the nodes and the values are the centralities

    bool_normalize_result : bool, optional (default = True)
      If True, the PageRank centralities are normalized to range between 0 and 1

    Returns
    ----------
    graph : mag.MultiAspectMultiDiGraph
      The MultiAspect Graph representing patient pathways with the relevance of the nodes as an attribute

    dict_pagerank_mag_normalized : dictionary
      A dictionary whose keys are the nodes and the values are the centralities. It is only returned if return_centrality_dict is True

    '''

    graph = G.copy()
    n_aspects_mag = len(graph.get_aspects_list())
    
    # --------------------------------------------------------------------------------------------------
    # garantee centrality values range from 0 to 1

    if (max(dict_closeness_intervention.values())!=1)|(min(dict_closeness_intervention.values())!=0):
        dict_closeness_intervention_NORM = normalize_centrality_values(dict_closeness_intervention)
    else:
        dict_closeness_intervention_NORM = dict_closeness_intervention
    
    if (max(dict_betweenness_occupation.values())!=1)|(min(dict_betweenness_occupation.values())!=0):
        dict_betweenness_occupation_NORM = normalize_centrality_values(dict_betweenness_occupation)
    else:
        dict_betweenness_occupation_NORM = dict_betweenness_occupation

    if (max(dict_pagerank_unit.values())!=1)|(min(dict_pagerank_unit.values())!=0):
        dict_pagerank_unit_NORM = normalize_centrality_values(dict_pagerank_unit)
    else:
        dict_pagerank_unit_NORM = dict_pagerank_unit

    # --------------------------------------------------------------------------------------------------
    
    # initialization weight (R0)
    dict_R0 = {}

    for node in graph.nodes():

        if ((node[0]!=start_string) & (node[0]!=end_string)):

            dict_R0[node] = (
                weight_complications * dict_pagerank_unit_NORM[node[idx_unit]] +
                (1-weight_complications) * (
                    weight_intervention_vs_occupation  * dict_closeness_intervention_NORM[node[idx_intervention]] +
                    (1 - weight_intervention_vs_occupation) * dict_betweenness_occupation_NORM[node[idx_occupation]]
                )
            )
            
        else:
            dict_R0[node] = 0

    # --------------------------------------------------------------------------------------------------

    # remove start/end if requested
    if bool_remove_start_end:
        
        if show_start_end:
            graph_with_start_end = graph.copy()
            
        nodelist = list(graph.nodes())
        for node in nodelist:
            if((node[0] == start_string) | (node[0] == end_string)):
                graph.remove_node(node)
    
    # --------------------------------------------------------------------------------------------------
    
    # convert the MAG into a undirected one (PageRank algorithm will account edges in both directions)
    undirected_graph = convert_mag_into_graph(convert_multiMAG_into_diMAG(graph)).to_undirected()

    # calculate the centrality using R0
    if use_edge_frequency:
        dict_pagerank_mag = nx.pagerank(undirected_graph,
                                                 alpha = pagerank_alpha, 
                                                 personalization = dict_R0,
                                                 max_iter = 1000, tol = 1e-07,
                                                 nstart = None, weight = 'freq', dangling = None)
    else:
        dict_pagerank_mag = nx.pagerank(undirected_graph,
                                                 alpha = pagerank_alpha, 
                                                 personalization = dict_R0,
                                                 max_iter = 1000, tol = 1e-07,
                                                 nstart = None, weight = None, dangling = None)
    del undirected_graph
    
    # normalize centrality values to range between 0 and 1, if requested
    if bool_normalize_result == True:
        dict_pagerank_mag_normalized = normalize_centrality_values(dict_pagerank_mag)
    else:
        dict_pagerank_mag_normalized = dict_pagerank_mag
    
    for node in graph.nodes():
        graph.nodes[node]['centrality'] = dict_pagerank_mag_normalized[node]
    
    # ---------------------------------------------------------------------------------------------------
    
    # add start and end nodes again, if requested
    if (bool_remove_start_end) & (show_start_end):
        for node in graph_with_start_end.nodes:
            if node in graph:
                graph_with_start_end.nodes[node]['centrality'] = graph.nodes[node]['centrality']
            elif ((node[0] == start_string) | (node[0] == end_string)):
                graph_with_start_end.nodes[node]['centrality'] = 1.0
            else:
                return -1
        graph = graph_with_start_end
    
    if not(return_centrality_dict):
        return graph
    else:
        return [graph,dict_pagerank_mag_normalized]

# ----------------------------------------------------------------------------------------------------------
