'''
This file provides functions to compare two sequences whose elements are intercalated with time intervals.
'''

import math

# ---------------------------------------------------------------------------------------- 

def rest(l):
    
    '''
	Return all elements of the list l, except the first one
    '''
    
    return l[1:] if len(l)>1 else []

# -----------------------------------------------------

def head(l, if_empty = math.inf):
    
    '''
    Returns the first element of the list l, or the content of variable *if_empty*, if l is empty
    '''
    
    return l[0] if len(l)>0 else if_empty

# -----------------------------------------------------

def dist_t(t1, t2, epsilon):
    
    '''
    Compares the absolute value of the difference between t1 and t2 with epsilon and returns dist_t
    '''
    
    # if epsilon == 0, return 1 (maximum value) if t1!=t2, or 0 if t1==t2
    if epsilon == 0:
        return 1 if t1 != t2 else 0
    else:
        return abs(t1 - t2)/epsilon 

# -----------------------------------------------------

def dist_actv(x,y):
    
    '''
    Returns dist_atv as the fraction of elements of x and y which are different
    '''

    assert ((type(x)==tuple) | (type(x)==list)) & ((type(y)==tuple) | (type(y)==list))

    cont = 0
    for i in range(len(x)):
    	if x[i]==y[i]:
    		cont += 1
    
    return 1-(cont/len(x))

# -----------------------------------------------------

def dist_intervention_occupation(x,y, idx_intervention = 0, idx_occupation = 1, w_intervention = 0.7, w_occupation = 0.3):
    
    '''
    Return the distance between the activity tuples x and y  based on their interventions and occupations, according to the equation:
    w_intervention * (0 if the interventions are equal, else 1) + w_occupation * (0 if the occupations are equal, else 1)
    '''
    
    return (w_intervention*(0 if x[idx_intervention]==y[idx_intervention] else 1) +
            w_occupation*(0 if x[idx_occupation]==y[idx_occupation] else 1))

# -----------------------------------------------------

def dist_icd_intervention(x,y, idx_icd = 0, idx_intervention = 1, w_icd = 0.7):
    
    '''
    Returns the distance between two activity tuples containing diagnosis and intervention information:
    distA = w_icd * (0 if the icd codes are equal; else 0.3 if the subcategories are equal; otherwise 1) + 
       + (1-w_icd) * (0 sif the interventions are equal; else 1)
    '''
    
    # if the icd codes are equal, check their length  
    if x[idx_icd]==y[idx_icd]:
        if len(x[idx_icd])<=2:
            dist_icd = 1
        elif len(x[idx_icd])==3:
            dist_icd = 0.3
        else:
            dist_icd = 0
    else:
        if x[idx_icd][:3]==y[idx_icd][:3]:
            dist_icd = 0.3
        else:
            dist_icd = 1
    
    return w_icd*dist_icd + (1.0-w_icd)*(0 if x[idx_intervention]==y[idx_intervention] else 1)
        

# -------------------------------------------------------

def dissimilarity(S1,S2,T1,T2,epsilon,delta,alpha,penalty,ta1,ta2,function_distA, function_distT, memo = None):

    '''
    Recursive function to calculate the dissimilarity between two patient pathways. 
    
	Parameters
    ----------
    S1 : list
      First sequence of items or activity tuples to be compared

    S2 : list
      Second sequence of items or activity tuples to be compared 

    T1 : list
      List of time intervals between the elements of S1

    T2 : list
      List of time intervals between the elements of S2

    epsilon : numeric
      Maximum difference between the time elapsed since the last alignment in S1 and S2 for a new alignment to be made

    delta : numeric
      Maximum distance (measured by function_distA) between two items (or activity tuples) that allows their alignment

    alpha : numeric (0<=alpha<=1)
	  Weight that balances function_distA and function_distT to obtain the total alignment cost:
	  total_alignment_cost = alpha * function_distA + (1-alpha)*function_distT

	penalty : numeric
	  Increment in the dissimilarity between two pathways when an item of one of them could not be aligned with any item of the other
	  It should be defined so that the penalty is greater than the maximum total_alignment_cost

	ta1 : numeric
	  Time elapsed since the last alignment for S1 (if there was no alignment, ta1 = Null)

	ta2 : numeric
	  Time elapsed since the last alignment for S2 (if there was no alignment, ta2 = Null)

    function_distA : function
      Function that calculates the distance between two items (or activity tuples) of S1 and S2

    function_distT : function
      Function converts the time difference between the last alignment in both sequences into a value to be used 
      in the total alignment cost

    memo : dictionary, optional (default = None)
      Dictionary that records the results of the calls of function dissimilarity, to optimise its performance

	Returns
    -------
    dissimilarity_value : the dissimilarity between the pathways

    '''

    if memo is None:
        memo = {}
    
    # if both sequences are empty
    if (len(S1) == 0) & (len(S2) == 0):
        return 0 # distA = 0; distT = 0; penalty = 0
    
    # only one of the sequences is empty --> we must call the rest of the other until it becomes empty as well
    elif (len(S1) == 0) | (len(S2) == 0): 
        return penalty + dissimilarity(rest(S1),rest(S2),rest(T1),rest(T2),epsilon,delta,alpha,penalty,None,None,
        	function_distA,function_distT)
    
    # if the comparison is already in *memo*, simply return the result
    if tuple([tuple(S1),tuple(S2),tuple(T1),tuple(T2),ta1,ta2]) in memo:
        return memo[tuple([tuple(S1),tuple(S2),tuple(T1),tuple(T2),ta1,ta2])]
    
    # both sequences are not empty and there was no alignment so far
    if(ta1 == ta2 == None):
        
        # it is not possible to align the current activity tuples
        if function_distA(S1[0],S2[0]) > delta:
            
            resul = min(
            	penalty + dissimilarity(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,None,None,
            		function_distA,function_distT,memo=memo),
            	penalty + dissimilarity(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,None,None,
            		function_distA,function_distT,memo=memo))
            memo[tuple([tuple(S1),tuple(S2),tuple(T1),tuple(T2),ta1,ta2])] = resul
            return resul
        
        # it is possible to align
        else : 
            resul = min((alpha * function_distA(S1[0],S2[0])) + dissimilarity(rest(S1),rest(S2),rest(T1),rest(T2),
                                                                                       epsilon,delta,alpha,penalty,
                                                                                       ta1=head(T1),ta2=head(T2),
                                                                                       function_distA=function_distA,
                                                                                       function_distT=function_distT,memo=memo),
                      penalty + dissimilarity(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,None,None,function_distA,
                      	function_distT,memo=memo),
                      penalty + dissimilarity(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,None,None,function_distA,
                      	function_distT,memo=memo)
                      )
            memo[tuple([tuple(S1),tuple(S2),tuple(T1),tuple(T2),ta1,ta2])] = resul
            return resul

    # both sequences are not empty and an alignment has already been made
    else : 
        
        # it is not possible to align the current activity tuples 
        if (function_distA(S1[0],S2[0]) > delta) | (abs(ta1-ta2) > epsilon): 
            resul = min(penalty + dissimilarity(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,
                                             ta1=ta1+head(T1),ta2=ta2,function_distA=function_distA,
                                             function_distT=function_distT,memo=memo),
                       penalty + dissimilarity(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,
                                             ta1=ta1,ta2=ta2+head(T2),function_distA=function_distA,
                                             function_distT=function_distT,memo=memo)
                      )
            memo[tuple([tuple(S1),tuple(S2),tuple(T1),tuple(T2),ta1,ta2])] = resul
            return resul

        # it is possible to align
        else:
            resul = min((alpha * function_distA(S1[0],S2[0]) + (1-alpha)*function_distT(ta1,ta2,epsilon)) + dissimilarity(rest(S1),rest(S2),
                                                                                                      rest(T1),rest(T2),
                                                                                                      epsilon,delta,alpha,penalty,
                                                                                                      ta1=head(T1),
                                                                                                      ta2=head(T2),
                                                                                                   function_distA=function_distA,
                                                                                                   function_distT=function_distT,
                                                                                                            memo=memo),
                       penalty + dissimilarity(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,
                                             ta1=ta1+head(T1),ta2=ta2,function_distA=function_distA,function_distT=function_distT,
                                             memo=memo),
                       penalty + dissimilarity(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,
                                             ta1=ta1,ta2=ta2+head(T2),function_distA=function_distA,function_distT=function_distT,
                                             memo=memo)
                      )
            memo[tuple([tuple(S1),tuple(S2),tuple(T1),tuple(T2),ta1,ta2])] = resul
            return resul

# -----------------------------------------------------

def obtain_sequences(list_of_edges, without_start_end = True):
    
    '''
	The function receives the list of edges of a patient's pathway and returns its sequence of activity tuples and 
	its sequence of time intervals

    Note: the edges in the list must be such that edge[0] is the origin node, edge[1] is the target node and edge[3] contains
    the edge's attributes, including 'interval'

    Parameters
    ----------
    list_of_edges : list
      List with the edges referring to a particular patient
	
	without_start_end : bool
	  If True, removes the first and the last nodes of the pathway (virtual nodes 'start' and 'end')

	Returns
	----------
	S : list
	  sequence of nodes

	T : list
	   sequence of time intervals

    '''
    
    S = [list_of_edges[0][0]] # origin node
    T = []
        
    for i in range(len(list_of_edges)):   
        S.append(list_of_edges[i][1]) # target node
        T.append(float(list_of_edges[i][3]['interval']))
        
    if without_start_end:
        S = S[1:-1]
        T = T[1:-1]
        
    return S,T

# -----------------------------------------------------

def best_list(alpha, *lists):
    
    '''
    The function takes a set of lists, whose elements are dictionaries with the keys 'match', 'distA', 'distT' and 'penalty',
    and returns the list whose dictionaries lead to the smaller distance based on the folloqing equation:
    dist = alpha * distA + (1-alpha) * distT + penalty
    '''
    
    dist_min = math.inf
    
    for l in lists:
        
        result = 0
        for dict in l:
            result = result + alpha*dict['distA'] + (1-alpha)*dict['distT'] + dict['penalty']
        if result < dist_min:
            best_list = l
            dist_min = result
            
    return best_list         

# -----------------------------------------------------

def dissimilarity_matches(S1,S2,T1,T2,epsilon,delta,alpha,penalty,ta1,ta2,function_distA, function_distT):

    '''
    Recursive function to calculate the dissimilarity between two patient pathways and return the matches between their activity tuples.  
    
    Parameters
    ----------
    S1 : list
      First sequence of items or activity tuples to be compared

    S2 : list
      Second sequence of items or activity tuples to be compared 

    T1 : list
      List of time intervals between the elements of S1

    T2 : list
      List of time intervals between the elements of S2

    epsilon : numeric
      Maximum difference between the time elapsed since the last alignment in S1 and S2 for a new alignment to be made

    delta : numeric
      Maximum distance (measured by function_distA) between two items (or activity tuples) that allows their alignment

    alpha : numeric (0<=alpha<=1)
      Weight that balances function_distA and function_distT to obtain the total alignment cost:
      total_alignment_cost = alpha * function_distA + (1-alpha)*function_distT

    penalty : numeric
      Increment in the dissimilarity between two pathways when an item of one of them could not be aligned with any item of the other
      It should be defined so that the penalty is greater than the maximum total_alignment_cost

    ta1 : numeric
      Time elapsed since the last alignment for S1 (if there was no alignment, ta1 = Null)

    ta2 : numeric
      Time elapsed since the last alignment for S2 (if there was no alignment, ta2 = Null)

    function_distA : function
      Function that calculates the distance between two items (or activity tuples) of S1 and S2

    function_distT : function
      Function converts the time difference between the last alignment in both sequences into a value to be used 
      in the total alignment cost

    Returns
    -------
    A dictionary with the matches (alignments) between the patient pathways.

    '''
    
    # if both sequences are empty
    if (len(S1) == 0) & (len(S2) == 0):
        return [{'match' : '#, #', 'distA' : 0, 'distT' : 0, 'penalty' : 0}]
    
    # only one of the sequences is empty --> we must call the rest of the other until it becomes empty as well
    elif (len(S1) == 0): 
        return [{'match' : '', 'distA' : 0, 'distT' : 0, 'penalty' : penalty*len(S2)}]
    elif (len(S2) == 0):
        return [{'match' : '', 'distA' : 0, 'distT' : 0, 'penalty' : penalty*len(S1)}]
    
    # both sequences are not empty and there was no alignment so far
    elif(ta1 == ta2 == None):

        # it is not possible to align the current activity tuples
        if function_distA(S1[0],S2[0]) > delta:
            return [{'match' : '', 
                     'distA' : 0, 
                     'distT' : 0, 
                     'penalty' : penalty}] + best_list(alpha,
                                                  dissimilarity_matches(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,
                                                             None,None,function_distA, function_distT),
                                                  dissimilarity_matches(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,
                                                             None,None,function_distA, function_distT))
        # it is possible to align
        else : 
            return best_list(alpha,
                                [{'match' : str(S1[0])+','+str(S2[0]), 
                                  'distA' : function_distA(S1[0],S2[0]), 
                                  'distT' : 0,
                                  'penalty' : 0}] + dissimilarity_matches(rest(S1),rest(S2),rest(T1),rest(T2),epsilon,delta,
                                                             alpha,penalty,
                                                             ta1=head(T1),ta2=head(T2),function_distA=function_distA, function_distT=function_distT),
                               [{'match' : '', 
                                 'distA' : 0, 
                                 'distT' : 0, 
                                 'penalty' : penalty}] + dissimilarity_matches(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,
                                                                     None,None,function_distA=function_distA, function_distT=function_distT),
                                [{'match' : '', 
                                 'distA' : 0, 
                                 'distT' : 0, 
                                 'penalty' : penalty}] + dissimilarity_matches(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,
                                                                     None,None,function_distA=function_distA, function_distT=function_distT)
                               )
    # both sequences are not empty and an alignment has already been made
    else :

        # it is not possible to align the current activity tuples 
        if (function_distA(S1[0],S2[0]) > delta) | (abs(ta1-ta2) > epsilon): 
            return [{'match' : '', 
                     'distA' : 0, 
                     'distT' : 0, 
                     'penalty' : penalty}] + best_list(alpha,
                                                  dissimilarity_matches(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,
                                                             ta1=ta1+head(T1),ta2=ta2,function_distA=function_distA, function_distT=function_distT),
                                                  dissimilarity_matches(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,
                                                             ta1=ta1,ta2=ta2+head(T2),function_distA=function_distA, function_distT=function_distT))
        # it is possible to align
        else:
            return best_list(alpha,
                                [{'match' : str(S1[0])+','+str(S2[0]), 
                                  'distA' : function_distA(S1[0],S2[0]), 
                                  'distT' : (function_distT(ta1,ta2,epsilon)),
                                  'penalty' : 0}] + dissimilarity_matches(rest(S1),rest(S2),rest(T1),rest(T2),epsilon,delta,alpha,penalty,
                                                             ta1=head(T1),ta2=head(T2),function_distA=function_distA, function_distT=function_distT),
                               [{'match' : '', 
                                 'distA' : 0, 
                                 'distT' : 0, 
                                 'penalty' : penalty}] + dissimilarity_matches(rest(S1),S2,rest(T1),T2,epsilon,delta,alpha,penalty,
                                                            ta1=ta1+head(T1),ta2=ta2,function_distA=function_distA, function_distT=function_distT),
                                [{'match' : '', 
                                 'distA' : 0, 
                                 'distT' : 0, 
                                 'penalty' : penalty}] + dissimilarity_matches(S1,rest(S2),T1,rest(T2),epsilon,delta,alpha,penalty,
                                                            ta1=ta1,ta2=ta2+head(T2),function_distA=function_distA, function_distT=function_distT)
                               )   
        
# -----------------------------------------------------