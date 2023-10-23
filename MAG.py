# -*- coding: utf-8 -*-

"""



  Implementation of version 0.1 MAG module

  Date: 12/04/2018

  Based upon NetworkX 2.2

"""

import networkx as nx
import collections
from networkx.classes import Graph, DiGraph, MultiGraph, MultiDiGraph
from networkx.exception import NetworkXError

__author__ = """Juliana Z. G. Mascarenhas (julianam@lncc.br), Klaus Wehmuth (Klaus@lncc.br) and Artur Ziviani (ziviani@lncc.br)"""


__all__ = ['MultiAspectGraph',
           'MultiAspectDiGraph',
           'MultiAspectMultiGraph',
           'MultiAspectMultiDiGraph',
           'convert_type_edges_to_datetime',
           'convert_type_edges_to_currency',
           'convert_type_list',
           'convert_type_set',
           'convert_type_edges_to_tuple',
	   'convert_type_edges_to_datetime']



#
#
#
#             NETWORKX CLASSES FOR MAG REPRESENTATION
#
#



# ---------------------------------------------------------------  Graph  -------------------------------------------------------------#

class MultiAspectGraph(Graph):
  """  
    A graph generalization for representing networks of any (finite) order that represents binary relations. Formally, H = (A, E) represents the MAG,
    where E is a finite set of edges and A is a finite list of aspects. This representation is isomorphic to a traditional directed graph.
    This feature allows the use of existing graph algorithms without problems. The number of aspects (p) is the order of the MAG, and each aspect a ∈ A is a finite set.
    The nodes are tuples, and the edges are a tuple with 2p entries. From the list of edges derives the composite vertices, as the cartesian product of all aspects.

    This module has four classes based on NetworkX to represents the MultiAspect Graph (MAG).
    The current class is the MultiAspectGraph(), that is a symmetric version of MAG.
     
    To examplify the MAG there are some examples of a hypothetical transport model bellow.
  
    Edge
    ----
      (from,to) = (Bus,Loc1,T1, Bus,Loc2,T2)
                   ----------   -----------
                    origin     destination
    Nodes
    -----
      from = (Bus,Loc1,T1)
      to = (Bus,Loc1,T2)
            
    Types of edges and examples
    ---------------------------
    spatial:  The edges occur in the same instant of time.
       
             (Bus,Loc1,T1, Subway,Loc1,T1)
              ----------    -----------
                origin      destination
 
    temporal: The edges occur between different two time instants.
    
             (Bus,Loc1,T1, Bus,Loc1,T2)
              ----------   -----------
                origin     destination  
 
    mixer: In this case we have two variations, in time and space.
    
            (Bus,Loc1,T1, Bus,Loc2,T2)
             ----------   -----------
               origin     destination
               
     The aspect list for the example is compose by three aspects with the limited set of elementes.
     
     set_aspect[0] = {'Bus','Subway'}
     set_aspect[1] = {'Loc1','Loc2','Loc3'}
     set_aspect[2] = {'T1','t2','T3'}
     Tau = [2,3,3]
 
     P.S.
     ----
     If any aspect caracther is modify the tuple will be another tuple, as the example below.
     ('Bus','Loc1','T1') != ('Bus','loc1','T1')
   
  """
  aspect_list_factory = list()

  def to_directed_class(self):
    return MultiAspectDiGraph()


  def to_undirected_class(self):
    return MultiAspectGraph()
  
  
  def __init__(self, data=None,**attr):
    """
      Initializer method of MultiAspectGraph().

     Parameters
     ----------
      data:
        Input graph, data is None by default.
        The input can be a list of edges.
      attr:
        Attributes to add to the graph as a key=value pair.
        Is None by default.
          
     Return
     ------
      MultiAspectGraph()
     
     Examples
     --------
        >>> G = MultiAspectGraph()
      >>> e = [(('a',1),('b',1)),(('c',1),('a',2)),(('a',2),('c',2))] 
      >>> G = MultiAspectGraph(e) 
      >>> e = [(('bus','loc1','t1'),('bus','loc2','t2')),(('bus','loc2','t2'),('bus','loc2','t3'))]
      >>> G = MultiAspectDiGraph(e, name='MAG')      
      >>> G = MultiAspectDiGraph(name='MAG')               
      >>> G = MultiAspectDiGraph(name='MAG', Day='Sunday')
      
    """
    self._order = None
    self._aspect = self.aspect_list_factory
    Graph.__init__(self,incoming_graph_data=data,**attr)


  def clear(self):
    """
      Remove all nodes, edges, and aspects from the MultiAspectGraph.
      Also removes the graph, nodes and edges attributes.

      Example
      -------
        >>> G.clear()
      
    """
    self._order = None
    self._aspect.clear()
    Graph.clear(self)

  
  def order(self):
    """
      This method returns the order of the MultiAspectGraph.
      The order is the number of aspects that composes the aspects list in the MultiAspectGraph.
      
      Return
      -------
        int

     Example
     -------
        >>> G.order()
    """
    return self._order
      
  def add_node(self, node, **attr):
    """
    Addition of each node by manual insertion.
    This function will add new nodes to an existing graph.
        
    Parameter
    ---------
      node:
        is a tuple of aspects that compose the node.
          
    Examples
    --------
      >>> node = ('Bus','Loc1','T1')
      >>> G.add_node(node)
      >>> G.add_node(('Bus','Loc2','T2'))

      >>> J.add_node(('North',1))

      >>> M.add_node((1,), pos=1)
      >>> M.add_node((2,), pos=2)
      >>> M.add_node((3,), pos=3)
        
    P.S.
    ----
      There is no restriction on the number of aspects, but must be the same for all nodes.
    """
    if self._order is None:
      initialize_list(self, len(node))
      self._order = len(node)
    if (aspect_node_integrity(self, node) == True):
      update_aspect_list(self,node)
    if node not in self._node:
      self._adj[node] = self.adjlist_inner_dict_factory()
      self._node[node] = attr
    else:
      self._node[node].update(attr)
    
        
  def add_nodes_from(self, nodelist, **attr):
    """
      This function allows the manual insertion of nodes (list of nodes).
      
      Parameters
      ----------
        node_list: the list of nodes that is insert into the MultiAspectGraph.
              Every tuple on the list is a node.
        **attr: Is a dicionary of comon attributes.
              If the attr is not empty, then the insertion of common attributes occurs.   
      
      Examples
      --------
        >>> G.add_nodes_from(('bus','loc1','t1'))
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2',t2')])
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc1','t2')], identification='bus point')
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2','t2'),('bus','loc3','t1'),('subway','loc1','t1')])

    """    
    for n in nodelist:
      try:
        if n not in self._adj:
          self.add_node(n,**attr)
        else:
          self._node[n].update(attr)
      except TypeError:
        nn, data = n
        if nn not in self._adj:
          self.add_node(nn, **data)
        else:
          olddict = self._node[n]
          olddict.update(attr)
          olddict.update(data)

  def add_nodes_from_file(self, name_file):
    """
      This method returns a MultiAspectGraph create by file import of nodes. Each line in the file is a node. Thus, the separation of nodes is given by enter (\n).
      The attributes must be separated by a semicolon (;) and the type must be separated by equal (=) from the value.

      Parameter
      ---------
        name_file = name of an exitent file of nodes

      Returns
      -------
        MultiAspectGraph()
        
      Example
      -------
        >>>  G.add_nodes_from_file('list_of_nodes.txt')
        >>>  G.add_nodes_from_file('list_of_nodes.csv')
            
      Example of nodes format in the file
      -----------------------------------
        Nodes without attributes
      
        (Bus,Loc1,T1)
        (Bus,Loc1,T2)
        (Bus,Loc2,T2)
        
        If there are one or more attributes for nodes, the file format must be as below.
      
        (Bus,Loc1,T1)<attr1=001;attr2=34>
        (Bus,Loc1,T2)<attr2=35>
        (Bus,Loc2,T2)<attr1=009;attr2=36>
        (Bus,Loc3,T3)

    """
    archive = open(name_file,'r')
    if (archive and aspect_integrity(archive,0)):
      file = archive.readline()
      while(file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):     #blank line
            file = archive.readline()
            continue
        node = split_tuple(file)
        node = [type_str_number(n) for n in node]
        if (file.find("<")) != -1 and (file.find(">")) != -1:                #there're nodes attributes
          element = split_weight(file)         
          attr = [e.split('=') for e in element]
          for a in (attr):
            a[0] = type_str_number(a[0])
            a[1] = type_str_number(a[1])
          attr = dict(attr)
          self.add_node(tuple(node),**attr)
        else:
            self.add_node(tuple(node))
            
        file = archive.readline()
    else:
      raise ValueError("ERROR IN THE FILE IMPORT!")
                  
          
  def add_edge(self, u, v, **attr):
    """
    Addition of each edge by manual insertion.

    Parameters
    ----------
      u, v:
          are the nodes
      **attr:
          is a dicionary of weight of the edge
      
    Examples
    --------
      >>> G.add_edge((1,), (2,), weight=4.7)
      >>> A.add_edge((2,0.3),(1,0.005))

      >>> u = ('Bus','Loc1','T1')
      >>> v = ('Bus','Loc2','T3')
      >>> M.add_edge(u, v, weight=2)
      >>> M.add_edge(('Bus','Loc1','T1'),('Bus','Loc2','T2'), time=10)
      
    """
    self.add_node(u, **attr)
    self.add_node(v, **attr)
    datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
    datadict.update(attr)
    self._adj[u][v] =  datadict
    self._adj[v][u] =  datadict


  def add_edges_from(self, edgelist, **attr):
    """
    This method allows the manual insertion of edges.
      
    Parameters
    ----------
      list_of_edges: is a list of edges
            Each element of the list has two tuples, where these tuples are the nodes that compose the edge.
            The elements in the list of edges is compose by (u,v,{weight=value})) or (u,v).
            
      **attr: is a dicionary of edge attributes (edge weight).
            These are common attributes (weights) to the edges on the list.
            
    Examples
    --------
      >>> f = ('Bus','Loc1','T1')
      >>> t = ('Bus','Loc1','T2')
      >>> c = ('Bus','Loc1','T3')
      >>> G.add_edges_from([(f,t,{'time':5}),(t,c,{'time':2})]) 
      >>> G.add_edges_from([(('Bus','Loc2','T1'),('Bus','Loc2','T2'))])
      >>> G.add_edges_from([(('Bus','Loc1','T1'),('Bus','Loc1','T2'))], weight=1) #common attribute
      
    """
    for e in edgelist:
      ne = len(e)
      if ne == 3:
        u, v, dd = e
      elif ne == 2:
        u, v = e
        dd = {}
      else:
        raise NetworkXError("Edge tuple %s must be a 2-tuple or 3-tuple." % (e,))
      self.add_node(u)
      self.add_node(v)
      
#      try:
#        adj = self._adj
#      except:
#        adj = self.adj
          
      datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
      datadict.update(attr)
      datadict.update(dd)
      self._adj[u][v] =  datadict
      self._adj[v][u] =  datadict
  
  
  def add_edges_from_file(self, name_file):
    """
    This method creates a MultiAspectGraph from file import of edges.
    The edges can be weighted or not, and there is no limitation to the number of weights.

          
    Parameter
    ---------
      name_file: name of an existent file of edges.
          If the file doesn't exists an error will be returned.

    Returns
    -------
      MultiAspectGraph()
    
    Examples
    --------
      >>>  G.add_edges_from_file('edges.txt')
      >>>  G.add_edges_from_file('edges.csv')
      
          
    Example of edges formart in the file
    ------------------------------------
      unweighted edges

      (Bus,Loc1,T1,Bus,Loc1,T1)
      (Bus,Loc1,T2,Bus,Loc2,T2)
      (Subway,Loc1,T1,Bus,Loc1,T1)
        
      weighted edges
      
      (Bus,Loc1,T1,Bus,Loc1,T1)<timewait=2>
      (Bus,Loc1,T2,Bus,Loc2,T2)<passengers=10;timepass=15>
      (Subway,Loc1,T1,Bus,Loc1,T1)<timetransfer=5>

    """
    archive = open(name_file, 'r')
    
    if (archive and aspect_integrity(archive,1) == True):
      file = archive.readline()
      Type = identify_edge(file)
      while (file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):
            file = archive.readline()
            continue
        edge = split_tuple(file)
        try:
          edge = [type_str_number(e) for e in edge]
        except:
          raise ValueError("ValueError")
        
        #create the edge
        From = tuple(edge[0:int((len(edge)/2))])
        To = tuple(edge[int((len(edge)/2)):len(edge)])
        self.add_edge(From,To)
        
        if file.find("<") != -1 and file.find(">") != -1:   
          add_edge_weight(self, file, From, To, Type)     #creating the weighted egdes

        file = archive.readline()
    else:
      raise ValueError("ERROR IN THE FILE IMPORT")
    
    
     
  def number_of_aspects(self):
    """
      This method returns the number of aspects in the MultiAspectGraph.

      Return
      ------
        int
       
      Example
      -------
        >>> G.number_of_aspects()
        
    """
    return len(self._aspect)
  

  
  def get_number_of_elements(self, nasp):
    """
      This method returns the number of elements in the chosen aspect.

      Parameter
      ---------
        nasp: number of the aspect
              N = Total number of aspects
              0 < nasp < N

      Return
      -------
        int
        
      Example
      -------
        >>> G.get_number_of_elements(1) #returns the number of elements in the first aspect
        >>> G.get_number_of_elements(3) #returns the number of elements in the third aspect
        
    """
    return len(self._aspect[nasp-1])
  
  
  def get_number_elements_aspects(self):
    """
    This method returns a list with the total number of elements in each aspect.

    Return
    ------
      list of ints
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_number_elements_aspects()
      [2,3,4]  
    """
    len_aspect = len(self._aspect)
    return [len(self._aspect[i]) for i in range (len_aspect)]
 
  
  def get_aspects_list(self):
    """
    Returns the list of aspect.

    Return
    ------
      list of sets
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_aspects_list()
      [{'Subway', 'Bus'}, {'Loc1', 'Loc2', 'Loc3'}, {'T2', 'T3', 'T1'}]
    """
    #return self._aspect.copy()
    return list(self._aspect)
  
  
  def print_aspects_list(self):
    """
    This method will print the MultiAspectGraph aspects list.
    
    Return
    ------
      Print in the screen

    Example
    -------
      >>> G.print_aspects_list()
          
      example: hypothetical transport model
      
      Set[ 0 ]: {'Subway', 'Bus'}
      Set[ 1 ]: {'Loc3', 'Loc1', 'Loc2'}
      Set[ 2 ]: {'T3', 'T1', 'T2'}
      Tau [2, 3, 3]
      
    """
    len_aspect = len(self._aspect)
    
    if len(self._aspect) == 0:
      raise ValueError("Aspect list not initialized!")
    
    if len(self._aspect[0]) == 0:
      print ("The list of aspects set is empty!\n")
    else:
      for i in range (len_aspect):
        print("Set[{}]:{}".format(i,self._aspect[i]))
      print("Tau: {}".format([len(self._aspect[i]) for i in range (len_aspect)]))
    
    
  def compact_aspects_list(self):
    """
    This method refreshs the aspect list. One can use this method after many removals (nodes or edges).
    The aspect list is emptied and updated.
    
    Example
    -------
      >>> G.compact_aspects_list()

    """
    [a.clear() for a in self._aspect]
    for node in self.nodes():
      update_aspect_list(self,node)
  
  
  def is_multigraph(self):
    """
      Returns True if the graph is a multigraph, False otherwise.
    """
    return False


  def is_directed(self):
    """
      Returns True if the graph is directed, False otherwise.
    """
    return False

  
  def is_mag(self):
    """
      Returns True if the graph is a MAG, False otherwise.
    """
    return True
  
  
  def aspects_subgraph(self, aspect_list):
    """
    This function creates an subMags inducing by aspects.

    Parameters
    ----------
        aspect_list: aspect list
            This reduced list derives from the original aspect list.
            The function using this aspect list will create the subMag.

    Returns
    -------
        SubMag induced by aspects - analogous to a subgraph

    Example
    -------
        >>> G.aspects_subgraph(G, aspects): 
    """
    return subMag_aspect_induced(self, aspect_list)

  #---------------------------- MAG SUBDETERMINATION ------------------------#
  
  def subdetermination(self, zeta, multi=False, direct=False, loop=False, **attr):
    """ 
      This method returns the subdeterminartion of the MultiAspectGraph(). This new MultiAspectGraph has a lower order than the original.
      
      Parameters
      ----------
        zeta: is a binary list with the positions related to the aspects of MultiAspectGraph.
              The position of the elements in the list is related to the aspects. For the values 0 and 1, the aspect is suppressed and sustained respectively.

        multi: False by default
            For the result of this method be a multiedge MAG, the tag must be True. Otherwise, must be False.

        direct: False by default
            For the result of this method be a direct MAG, the tag must be True. Otherwise, must be False.
            
        loop: False by default
            True, returning the edges with loops. False, the edges with the same vertice aren`t in the subdetermination of MAG.
        
        **attr: dicionary of graph attributes, such as name.

     Return
     ------
       Class of MAG 
      
     Example
     -------
      example: for a MultiAspectGraph with order equals three
      
      >>> zeta = [1,0,1]
      >>> H = G.subdetermination(zeta)
      >>> type(H)
      __main__.MultiAspectGraph
      >>> zeta = [1,0,0]
      >>> T = G.subdetermination(zeta, multi=True, name='subdetermination', day='today') #with attributes
      >>> type(T)
      __main__.MultiAspectMultiGraph
    """  
    if nx.is_weighted(self):
      Par = subDeterminedEdgePartition(self, zeta)
      
    if len(zeta) != self.number_of_aspects():
      raise ValueError ('The number of elements in zeta is incorrect. The number of aspects in MAG is {}, and {} have been given!'.format(self.number_of_aspects(),len(zeta)))
    
    #Verify the basic cases
#    if (zeta_zero(zeta) == True):
#      print("All aspects supressed. Null returned")
#      return None
#    
#    if (zeta_one(zeta) == True):
#      print("None aspect was suppressed. The same MAG is returned")
#      return self

    #variables
    lenz = len(zeta)
    asps = list(zeta)+list(zeta)
    total = len(asps)
#    naspects = zeta.count(1)
    
    if multi:
      H = MultiAspectMultiDiGraph(**attr) if direct else MultiAspectMultiGraph(**attr)
    else:
      H = MultiAspectDiGraph(**attr) if direct else MultiAspectGraph(**attr)

    #edge list verification
    for e, datadict in self.edges.items():
      new_edge = [(e[0][i]) if i<lenz else (e[1][i-lenz]) for i in range (total) if asps[i]!=0]        
      From = tuple(new_edge[0:int(len(new_edge)/2)])
      To = tuple(new_edge[int(len(new_edge)/2):len(new_edge)])
      if (From != To and loop == False) or loop == True:
        H.add_edge(From,To)
        if nx.is_weighted(self):
            k = list(datadict.keys())
            for w in k:
                if w == "weight":
                    H[From][To][w] = min(Par[From,To])
                else:
                    H[From][To][w] = datadict[w]
                    
    #node list verification
    for n in self.nodes():
      node = [n[i] for i in range(0,len(n)) if zeta[i] !=0]
      H.add_node(tuple(node))
      
    #return the subdetermination of MAG
    return H
  
    
  #------------------------------- Adj Matrix ------------------------------------#  
  
  def sparse_adj_matrix(self, nodelist=None, weight=None, dtype=None):
    """
      This method will convert the MultiAspectGraph in an sparse ajdcency matrix.

      Parameter
      ---------
        nodelist: is a list of nodes (optional). The nodes compose the lines and rows of the adjcency matrix.
        The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).
            
        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None by default, and representes the existence of an edge between two nodes.
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        list of tuples (nodes)
        scipy sparse matrix
        
      Example
      -------
        >>> G.sparse_adj_matrix()
        >>> G.sparse_adj_matrix(weight='weight')
    """  
    try:
      from scipy import sparse
    except:
      raise ImportError('The scipy library is necessary and is not installed!')  
    
    if self.number_of_nodes == 0:
      raise nx.NetworkXError('The list of nodes is empty.')
      
    if nodelist is None:
      nodelist = list(self)
    
    len_nodes = len(nodelist)
    dict_node = dict(zip(nodelist,range(len_nodes)))
    edge_list = dict()
    Edges = list(self.edges(nodelist,data=weight))
    color = dict.fromkeys([(dict_node[u],dict_node[v]) for u,v,w in Edges], 0)

    if weight is None:
      for u,v,w in Edges:
        try:
          uu = dict_node[u]
          vv = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            edge_list.update({(uu,vv):1})
            color[(uu,vv)] = 1
    else:
      for u,v,w in Edges:
        if w == None:
          raise ValueError('All the nodes must have the selected weight')
        try:
          uu = dict_node[u]
          vv = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            edge_list.update({(uu,vv):w})
            color[(uu,vv)] = 1
          else:
            value = edge_list[(uu,vv)]
            edge_list[(uu,vv)] = value+w
                
    #build an symmetric sparse matrix
    key = list(edge_list.keys())
    row = [key[i][0] for i in range(len(key))] + [key[i][1] for i in range(len(key))]
    col = [key[i][1] for i in range(len(key))] + [key[i][0] for i in range(len(key))]
    data = [edge_list[e] for e in key] + [edge_list[e] for e in key]
    
    list_tuples = [a[0] for a in (sorted(dict_node.items(),key=lambda e:e[1]))]
    M = sparse.coo_matrix((data, (row,col)), shape=(len_nodes, len_nodes), dtype=dtype)
  
    return M, list_tuples
  
  
  #---------------------------- Incidence Matrix ----------------------------------# 

  
  def sparse_incidence_matrix(self, edgelist=None, nodelist=None, weight=None):
    """
      This method will convert the MultiAspectGraph in an sparse incidence matrix.

      Parameter
      ---------
        edgelist: is a list of edges (optional). The edges compose the rows in the incidence matrix
            The edgelist determine which edges compose the matrix. If is None, edgelist is equal self.edges().
        
        nodelist: is a list of nodes (optional). The nodes compose the lines of the incidence matrix.
            The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).

        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None (default), then each edge has weight 1
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        list of tuples (nodes)
        scipy sparse matrix
        
      Example
      -------
        >>> G.sparse_adj_matrix()
        >>> G.sparse_adj_matrix(weight='weight')
    """
    try:
      from scipy import sparse
    except:
      raise ImportError('The scipy module is necessary and is not installed!')
      
    if nodelist == None:
      nodelist = list(self)
    len_nodes = len(nodelist)
    
    if edgelist == None:
      edgelist = self.edges()
    len_edges = len(edgelist)
    
    M = sparse.lil_matrix((len_nodes,len_edges))
    dictt_node = dict(zip(nodelist,range(len_nodes)))
    dictt = dict(zip(edgelist,range(len_edges)))
  
    if weight is None:
      for a in dictt:
        try:
          u = dictt_node[a[0]]
          v = dictt_node[a[1]]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        e = dictt[a]
        M[u,e] = 1
        M[v,e] = 1
    else:
      for a in self.edges.data():
        dictt_w = a[2]
        try:
          u = dictt_node[a[0]]
          v = dictt_node[a[1]]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        e = dictt[(a[0],a[1])]
        if weight in dictt_w:
          M[u,e] += dictt_w[weight]
          M[v,e] += dictt_w[weight]
        else:
          raise KeyError('The weight was not found!')
            
    list_node_tuple = [a[0] for a in (sorted(dictt_node.items(), key=lambda e:e[1]))]
    list_edge_tuple = [a[0] for a in (sorted(dictt.items(), key=lambda e:e[1]))]

    return M, list_node_tuple, list_edge_tuple  
  
     
  
# ---------------------------------------------------------------  DiGraph  -------------------------------------------------------------#

class MultiAspectDiGraph(DiGraph):
  """  
    A graph generalization for representing networks of any (finite) order that represents binary relations. Formally, H = (A, E) represents the MAG,
    where E is a finite set of edges and A is a finite list of aspects. This representation is isomorphic to a traditional directed graph.
    This feature allows the use of existing graph algorithms without problems. The number of aspects (p) is the order of the MAG, and each aspect a ∈ A is a finite set.
    The nodes are tuples, and the edges are a tuple with 2p entries. From the list of edges derives the composite vertices, as the cartesian product of all aspects.

    This module has four classes based on NetworkX to represents the MultiAspect Graph (MAG).
    The current class is the MultiAspectDiGraph(), that is a directed version of MAG.
     
    To examplify the MAG there are some examples of a hypothetical transport model bellow.
  
    Edge
    ----
      (from,to) = (Bus,Loc1,T1, Bus,Loc2,T2)
                   ----------   -----------
                    origin      destination
    Nodes
    -----
      from = (Bus,Loc1,T1)
      to = (Bus,Loc1,T2)
            
    Types of edges and examples
    ---------------------------
    spatial:  The edges occur in the same instant of time.
       
             (Bus,Loc1,T1, Subway,Loc1,T1)
              ----------    -----------
                origin      destination
 
    temporal: The edges occur between different two time instants.
    
             (Bus,Loc1,T1, Bus,Loc1,T2)
              ----------   -----------
                origin     destination  
 
    mixer: In this case we have two variations, in time and space.
    
            (Bus,Loc1,T1, Bus,Loc2,T2)
             ----------   -----------
               origin     destination
               
    The aspect list for the example is compose by three aspects with the limited set of elementes.
     
    set_aspect[0] = {'Bus','Subway'}
    set_aspect[1] = {'Loc1','Loc2','Loc3'}
    set_aspect[2] = {'T1','t2','T3'}
    Tau = [2,3,3]
 
    P.S.
    ----
    If any aspect caracther is modify the tuple will be another tuple, as the example below.
    ('Bus','Loc1','T1') != ('Bus','loc1','T1')
   
  """
  def to_directed_class(self, len_aspects):
    return MultiAspectDiGraph()


  def to_undirected_class(self, len_aspects):
    return MultiAspectGraph()

  
  aspect_list_factory = list()
  
  def __init__(self, data=None, **attr):
    """
      Initializer method of MultiAspectGraph().

      Parameters
      ----------
        data: Input graph, data is None by default.
          The input can be a list of edges.

        attr: Attributes to add to the graph as a key=value pair.
          Is None by default.

      Returns
      -------
        MultiAspectDiGraph()de

      Examples
      --------
        >>> G = MultiAspectDiGraph()
        >>> G = MultiAspectDiGraph(name='MAG)
        >>> G = MultiAspectDiGraph(edgelist, name='MAG)
        
    """
    self._order = None
    self._aspect = self.aspect_list_factory
    DiGraph.__init__(self,data,**attr)

  def clear(self):
    """
      Remove all nodes, edges, and aspects from the MultiAspectGraph.
      Also removes the graph, nodes and edges attributes.

      Example
      -------
        >>> G.clear()
      
    """
    self._order = None
    self._aspect.clear()
    DiGraph.clear(self)

  
  def order(self):
    """
      This method returns the order of the MultiAspectDiGraph.
      The order is the number of aspects that composes the aspects list in the MultiAspectDiGraph.
      
      Return
      -------
        int

     Example
     -------
        >>> G.order()
    """
    return self._order

  
  def add_node(self, node, **attr):
    """
    Addition of each node by manual insertion.
    This function will add new nodes to an existing graph.
        
    Parameter
    ---------
      node:
        is a tuple of aspects that compose the node.
          
    Examples
    --------
      >>> node = ('Bus','Loc1','T1')
      >>> G.add_node(node)
      >>> G.add_node(('Bus','Loc2','T2'))

      >>> J.add_node(('North',1))

      >>> M.add_node((1,), pos=1)
      >>> M.add_node((2,), pos=2)
      >>> M.add_node((3,), pos=3)
        
    P.S.
    ----
      There is no restriction on the number of aspects, but must be the same for all nodes.
    """
    if self._order is None:
      initialize_list(self, len(node))
      self._order = len(node)
    if (aspect_node_integrity(self,node) == True):
      update_aspect_list(self, node)          
      node = tuple(node)
      if node not in self._succ:
        self._succ[node] = self.adjlist_inner_dict_factory()
        self._pred[node] = self.adjlist_inner_dict_factory()
        self._node[node] = attr
      else:  
        # update attr even if node already exists
        self._node[node].update(attr)
    else:
      raise ValueError("The number os aspect is incorrect! The MAG has a list with {0} aspects, but {1} have been given.".format(len(self._aspect),len(node)))


    
  def add_nodes_from(self, nodelist, **attr):
    """
      This function allows the manual insertion of nodes (list of nodes).
      
      Parameters
      ----------
        node_list: the list of nodes that is insert into the MultiAspectDiGraph.
              Every tuple on the list is a node.
              
        **attr: Is a dicionary of comon attributes.
              If the attr is not empty, then the insertion of common attributes occurs.   
      
      Examples
      --------
        >>> G.add_nodes_from(('bus','loc1','t1'))
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2',t2')])
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc1','t2')], identification='bus point')
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2','t2'),('bus','loc3','t1'),('subway','loc1','t1')])

    """
    for n in nodelist:
      try:
        if n not in self._succ:
          self.add_node(n,**attr)
        else:
          try:
            self._node[n].update(attr)
          except:
            self.node[n].update(attr)
      except TypeError:
        nn, ndict = n
        self.add_node(nn,**ndict)
        
        
  def add_nodes_from_file(self, name_file):
    """
      This method returns a MultiAspectGraph create by file import of nodes. Each line in the file is a node. Thus, the separation of nodes is given by enter (\n).
      The attributes must be separated by a semicolon (;) and the type must be separated by equal (=) from the value.

      Parameter
      ---------
        name_file = name of an exitent file of nodes

      Returns
      -------
        MultiAspectDiGraph()
        
      Example
      -------
        >>>  G.add_nodes_from_file('list_of_nodes.txt')
        >>>  G.add_nodes_from_file('list_of_nodes.csv')
            
      Example of nodes format in the file
      -----------------------------------
        Nodes without attributes
      
        (Bus,Loc1,T1)
        (Bus,Loc1,T2)
        (Bus,Loc2,T2)
        
        If there are one or more attributes for nodes, the file format must be as below.
      
        (Bus,Loc1,T1)<attr1=001;attr2=34>
        (Bus,Loc1,T2)<attr2=35>
        (Bus,Loc2,T2)<attr1=009;attr2=36>
        (Bus,Loc3,T3)

    """
    archive = open(name_file,'r')
    if (archive and aspect_integrity(archive,0)):
      file = archive.readline()
      while(file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):
            file = archive.readline()
            continue
        node = split_tuple(file)
        node = [type_str_number(n) for n in node]
        if (file.find("<")) != -1 and (file.find(">")) != -1:
          attr = list()
          element = split_weight(file)         
          attr = [e.split('=') for e in element]
          for a in (attr):         
            a[0] = type_str_number(a[0])
            a[1] = type_str_number(a[1])
          attr = dict(attr)
          self.add_node(tuple(node),**attr)
        else:
            self.add_node(tuple(node))
        file = archive.readline()
    else:
      raise ValueError("Error during the file import!")
  
  
  def add_edge(self, u, v, **attr):
    """
    Addition of each edge by manual insertion.

    Parameters
    ----------
      u, v: are the nodes
          
      **attr: is a dicionary of weight of the edge
      
    Examples
    --------
      >>> G.add_edge((1,), (2,), weight=4.7)
      >>> A.add_edge((2,0.3),(1,0.005))

      >>> u = ('Bus','Loc1','T1')
      >>> v = ('Bus','Loc2','T3')
      >>> M.add_edge(u, v, weight=2)
      >>> M.add_edge(('Bus','Loc1','T1'),('Bus','Loc2','T2'), time=10)
      
    """
    self.add_node(u)
    self.add_node(v)
    
    #add the edge
    datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
    datadict.update(attr)
    self._succ[u][v] = datadict
    self._pred[v][u] = datadict

        
  def add_edges_from(self, edgelist, **attr):
    """
    This method allows the manual insertion of edges.
      
    Parameters
    ----------
      edges_list: is a list of edges
            Each element of the list has two tuples, where these tuples are the nodes that compose the edge.
            The elements in the list of edges is compose by (u,v,{weight=value})) or (u,v).
            
      **attr: is a dicionary of edge attributes (edge weight).
            These are common attributes (weights) to the edges on the list.
            
    Examples
    --------
      >>> f = ('Bus','Loc1','T1')
      >>> t = ('Bus','Loc1','T2')
      >>> c = ('Bus','Loc1','T3')
      >>> G.add_edges_from([(f,t,{'time':5}),(t,c,{'time':2})]) 
      >>> G.add_edges_from([(('Bus','Loc2','T1'),('Bus','Loc2','T2'))])
      >>> G.add_edges_from([(('Bus','Loc1','T1'),('Bus','Loc1','T2'))], weight=1) #common attribute
      
    """
    for e in edgelist:
      ne = len(e)
      if ne == 3:
        u, v, dd = e
      elif ne == 2:
        u, v = e
        dd = {}
      else:
        raise NetworkXError("Edge tuple %s must be a 2-tuple or 3-tuple." % (e,))
      
      self.add_node(u)
      self.add_node(v)
      
      datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
      datadict.update(attr)
      datadict.update(dd)
      self._succ[u][v] = datadict
      self._pred[v][u] = datadict
  
  
  def add_edges_from_file(self, name_file):                      
    """
    This method creates a MultiAspectGraph from file import of edges.
    The edges can be weighted or not, and there is no limitation to the number of weights.

    Parameter
    ---------
      name_file: name of an existent file of edges.
          If the file doesn't exists an error will be returned.

    Returns
    -------
      MultiAspectDiGraph()
    
    Examples
    --------
      >>>  G.add_edges_from_file('edges.txt')
      >>>  G.add_edges_from_file('edges.csv')
          
    Example of edges formart in the file
    ------------------------------------
      unweighted edges

      (Bus,Loc1,T1,Bus,Loc1,T1)
      (Bus,Loc1,T2,Bus,Loc2,T2)
      (Subway,Loc1,T1,Bus,Loc1,T1)
        
      weighted edges
      
      (Bus,Loc1,T1,Bus,Loc1,T1)<timewait=2>
      (Bus,Loc1,T2,Bus,Loc2,T2)<passengers=10;timepass=15>
      (Subway,Loc1,T1,Bus,Loc1,T1)<timetransfer=5>

    """      
    archive = open(name_file,'r')
    if (archive and aspect_integrity(archive,1) == True):
      file = archive.readline()
      Type = identify_edge(file)
      
      while (file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):
            file = archive.readline()
            continue
        edge = split_tuple(file)
        try: edge = [type_str_number(e) for e in edge]
        except: raise ValueError()
        From = tuple(edge[0:int((len(edge)/2))])
        To = tuple(edge[int((len(edge)/2)):len(edge)])
        self.add_edge(From,To)
        if file.find("<") != -1 and file.find(">") != -1:
          add_edge_weight(self, file, From, To, Type)   #creating the weighted egdes
          
        file = archive.readline()
    else:
      raise ValueError("Error during the file import!")
  

  def number_of_aspects(self):
    """
      This method returns the number of aspects in the MultiAspectGraph.

      Return
      ------
        int
       
      Example
      -------
        >>> G.number_of_aspects()
        
    """
    return len(self._aspect)
  
  
  def get_number_of_elements(self, nasp):
    """
      This method returns the number of elements in the chosen aspect.

      Parameter
      ---------
        nasp: number of the aspect
              N = Total number of aspects
              0 < nasp < N

      Return
      -------
        int
        
      Example
      -------
        >>> G.get_number_of_elements(1) #returns the number of elements in the first aspect
        >>> G.get_number_of_elements(3) #returns the number of elements in the third aspect
        
    """
    return len(self._aspect[nasp-1])
  
  
  def get_number_elements_aspects(self):
    """
    This method returns a list with the total number of elements in each aspect.

    Return
    ------
      list of ints
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_number_elements_aspects()
      [2,3,4]
      
    """
    return [len(self._aspect[i]) for i in range (len(self._aspect))]
  
  
  def get_aspects_list(self):
    """
    Returns the list of aspect.

    Return
    ------
      list of sets
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_aspects_list()
      [{'Subway', 'Bus'}, {'Loc1', 'Loc2', 'Loc3'}, {'T2', 'T3', 'T1'}]
    """
    return self._aspect.copy()
  
  
  def print_aspects_list(self):
    """
    This method will print the MultiAspectGraph aspects list.
    
    Return
    ------
      Print in the screen

    Example
    -------
      >>> G.print_aspects_list()
          
      example: hypothetical transport model
      
      Set[0]: {'Subway', 'Bus'}
      Set[1]: {'Loc3', 'Loc1', 'Loc2'}
      Set[2]: {'T3', 'T1', 'T2'}
      Tau: [2, 3, 3]
      
    """
    len_aspect = len(self._aspect)
    if len(self._aspect) == 0:
      raise ValueError("Aspect list not initialized!")
    
    if len(self._aspect[0]) == 0:
      print ("The list of aspects set is empty!\n")
    else:
      for i in range (len_aspect):
        print("Set[{}]: {}".format(i,self._aspect[i]))
      print("Tau: {}".format([len(self._aspect[i]) for i in range (len_aspect)]))
    
   
  
  def compact_aspects_list(self):
    """
    This method refreshs the aspect list. One can use this method after many removals (nodes or edges).
    The aspect list is emptied and updated.
    
    Example
    -------
      >>> G.compact_aspects_list()

    """
    [a.clear() for a in self._aspect]    
    for node in self.nodes():
      update_aspect_list(self, node)


  def is_multigraph(self):
    """
      Returns True if the graph is a multigraph, False otherwise.
    """
    return False

  def is_directed(self):
    """
      Returns True if the graph is directed, False otherwise.
    """
    return True
  
  def is_mag(self):
    """
      Returns True if the graph is a MAG, False otherwise.
    """
    return True

  
  def aspects_subgraph(self, aspect_list):
    """
    This function creates an subMags inducing by aspects.

    Parameters
    ----------
        aspect_list: aspect list
            This reduced list derives from the original aspect list.
            The function using this aspect list will create the subMag.

    Returns
    -------
        SubMag induced by aspects - analogous to a subgraph

    Example
    -------
        >>> G.aspects_subgraph(aspects): 
    """
    return subMag_aspect_induced(self, aspect_list)
  
      
  #-------------------------- MAG SUBDETERMINATION -----------------------------------#

  def subdetermination(self, zeta, multi=False, direct=True, loop=False, **attr):
    """ 
      This method returns the subdeterminartion of the MultiAspectDiGraph(). This new MultiAspectDiGraph has a lower order than the original.
      
      Parameters
      ----------
        zeta: is a binary list with the positions related to the aspects of MultiAspectDiGraph.
              The position of the elements in the list is related to the aspects. For the values 0 and 1, the aspect is suppressed and sustained respectively.

        multi: False by default
            For the result of this method be a multiedge MAG, the tag must be True. Otherwise, must be False.

        direct: True by default
            For the result of this method be a direct MAG, the tag must be True. Otherwise, must be False.
            
        loop: False by default
            True, returning the edges with loops. False, the edges with the same vertice aren`t in the subdetermination of MAG.
       **attr: dicionary of graph attributes, such as name.

     Return
     ------
       Class of MAG
      
     Example
     -------
      example: for a MultiAspectDiGraph with order equals three
      
      >>> zeta = [1,0,1]
      >>> H = G.subdetermination(zeta)
      >>> type(H)
      __main__.MultiAspectDiGraph
      >>> zeta = [1,0,0]
      >>> T = G.subdetermination(zeta, multi=True, direct=False, name='subdetermination', day='today') #with attributes
      >>> type(T)
      __main__.MultiAspectMultiGraph
      
    """  
    if nx.is_weighted(self):
      Par = subDeterminedEdgePartition(self, zeta)
    
    if len(zeta) != self.number_of_aspects():
      raise ValueError ('The number of elements in zeta is incorrect. The number of aspects in MAG is {}, and {} have been given!'.format(self.number_of_aspects(),len(zeta)))
    
    #Verify the basic cases
#    if (zeta_zero(zeta) == True):
#      print("All aspects supressed. Null returned")
#      return None
#    
#    if (zeta_one(zeta) == True):
#      print("None aspect was suppressed. The same MAG is returned")
#      return self
    
    #variables
    asps = list(zeta)+list(zeta)
    lenz = len(zeta)
    total = len(asps)
 #   naspects = zeta.count(1)
    if multi:
      H = MultiAspectMultiDiGraph(**attr) if direct else MultiAspectMultiGraph(**attr)
    else:
      H = MultiAspectDiGraph(**attr) if direct else MultiAspectGraph(**attr)

    #edge list verification
    for e,datadict in self.edges.items():
      new_edge = [(e[0][i]) if i<lenz else (e[1][i-lenz]) for i in range (total) if asps[i]!=0]
      From = tuple(new_edge[0:int(len(new_edge)/2)])
      To = tuple(new_edge[int(len(new_edge)/2):len(new_edge)])
      if (From != To and loop == False) or loop == True:
        H.add_edge(From,To)
        if nx.is_weighted(self):
            k = list(datadict.keys())
            for w in k:
                if w == "weight":
                    H[From][To][w] = min(Par[From,To])
                else:
                    H[From][To][w] = datadict[w]
    
    #node list verification
    for n in self.nodes():
      node = [n[i] for i in range(0,len(n)) if zeta[i] !=0]
    H.add_node(tuple(node))
      
    #return the subdetermination of MAG
    return H

  
  #---------------------------- Adj Matrix ----------------------------------# 
  
  
  def sparse_adj_matrix(self, nodelist=None, weight=None, dtype=None):
    """
      This method will convert the MultiAspectDiGraph in an sparse ajdcency matrix.

      Parameter
      ---------
        nodelist: is a list of nodes (optional). The nodes compose the lines and rows of the adjcency matrix.
            The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).
            
        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None by default, and representes the existence of an edge between two nodes.
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        list of tuples (nodes)
        scipy sparse adj matrix
        
      Example
      -------
        >>> G.sparse_adj_matrix()
        >>> G.sparse_adj_matrix(weight='weight')
        
    """
    #here
    try:
      from scipy import sparse
    except:
      raise ImportError('The numpy module is necessary and is not installed!')  
    
    if self.number_of_nodes() == 0:
      raise nx.NetworkXError('The list of nodes on MAG is empty.')
    
    if nodelist is None:
      nodelist = list(self)
    
    len_nodes = len(nodelist)
    dict_node = dict(zip(nodelist,range(len_nodes)))
    edge_list = dict()
    Edges = list(self.edges(nodelist,data=weight))
    color = dict.fromkeys([(dict_node[u],dict_node[v]) for u,v,w in Edges], 0)
    
    if weight is None:
      for u,v,w in Edges:
        try:
          uu = dict_node[u]
          vv = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            edge_list.update({(uu,vv):1})
            color[(uu,vv)] = 1
    else:
      for u,v,w in Edges:
        if w == None:
          raise ValueError('All the nodes must have the selected weight')
        try:
          u = dict_node[u]
          v = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            edge_list.update({(u,v):w})
            color[(uu,vv)] = 1
          else:
            value = edge_list[(u,v)]
            edge_list[(u,v)] = value+w
          
    #creation of the adj matrix
    key = list(edge_list.keys())
    row = [key[i][0] for i in range(len(key))]
    col = [key[i][1] for i in range(len(key))]
    data = [edge_list[e] for e in key]
      
    list_tuples=[a[0] for a in (sorted(dict_node.items(),key=lambda e:e[1]))]
    M = sparse.coo_matrix((data, (row,col)), shape=(len_nodes, len_nodes), dtype=dtype)
  
    return M, list_tuples
  
 
  
  #---------------------------- Incidence Matrix ----------------------------------# 

    
  def sparse_incidence_matrix(self, edgelist=None, nodelist=None, weight=None):
    """
      This method will convert the MultiAspectDiGraph in an sparse incidence matrix.

      Parameter
      ---------
        edgelist: is a list of edges (optional). The edges compose the rows in the incidence matrix
            The edgelist determine which edges compose the matrix. If is None, edgelist is equal self.edges().
        
        nodelist: is a list of nodes (optional). The nodes compose the lines of the incidence matrix.
            The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).

        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None (default), then each edge has weight 1
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        scipy sparse matrix
        list of tuples (nodes)
        list of edges
        
      Examples
      --------
        >>> G.sparse_incidence_matrix()
        >>> G.sparse_incidence_matrix(weight='weight')
        
    """
    try:
      from scipy import sparse
    except:
      raise ImportError('The scipy module is necessary and is not installed!')
      
    if nodelist == None:
      nodelist = list(self)
    len_nodes = len(nodelist)
    
    if edgelist == None:
      edgelist = self.edges()
    len_edges = len(edgelist)
    
    M = sparse.lil_matrix((len_nodes,len_edges))
    dictt_node = dict(zip(nodelist,range(len_nodes)))
    dictt = dict(zip(edgelist,range(len_edges)))
    
    if weight is None:
      for a in dictt:
        try:
          u = dictt_node[a[0]]
          v = dictt_node[a[1]]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        e = dictt[a]
        M[u,e] = -1
        M[v,e] = 1
    else:
      for a in self.edges.data():
        dictt_w = a[2]
            
        try:
          u = dictt_node[a[0]]
          v = dictt_node[a[1]]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        e = dictt[(a[0],a[1])]
      
        if weight in dictt_w:
          M[u,e] += -dictt_w[weight]
          M[v,e] += dictt_w[weight]
        else:
          raise KeyError('The weight was not found!')
            
    list_node_tuple = [a[0] for a in (sorted(dictt_node.items(), key=lambda e:e[1]))]
    list_edge_tuple = [a[0] for a in (sorted(dictt.items(), key=lambda e:e[1]))]
    
    return M, list_node_tuple, list_edge_tuple  
       
      
  
# ---------------------------------------------------------------  MultiGraph  -------------------------------------------------------------#

class MultiAspectMultiGraph(MultiGraph):
  """  
    A graph generalization for representing networks of any (finite) order that represents binary relations. Formally, H = (A, E) represents the MAG,
    where E is a finite set of edges and A is a finite list of aspects. This representation is isomorphic to a traditional directed graph.
    This feature allows the use of existing graph algorithms without problems. The number of aspects (p) is the order of the MAG, and each aspect a ∈ A is a finite set.
    The nodes are tuples, and the edges are a tuple with 2p entries. From the list of edges derives the composite vertices, as the cartesian product of all aspects.

    This module has four classes based on NetworkX to represents the MultiAspect Graph (MAG).
    The current class is the MultiAspectMultiGraph(), that is a symmetric version of MAG with multiedges.
     
    To examplify the MAG there are some examples of a hypothetical transport model bellow.
  
    Edge
    ----
      (from,to) = (Bus,Loc1,T1, Bus,Loc2,T2)
                   ----------   -----------
                    origin     destination
    Nodes
    -----
      from = (Bus,Loc1,T1)
      to = (Bus,Loc1,T2)
            
    Types of edges and examples
    ---------------------------
    spatial:  The edges occur in the same instant of time.
       
             (Bus,Loc1,T1, Subway,Loc1,T1)
              ----------    -----------
                origin      destination
 
    temporal: The edges occur between different two time instants.
    
             (Bus,Loc1,T1, Bus,Loc1,T2)
              ----------   -----------
                origin     destination  
 
    mixer: In this case we have two variations, in time and space.
    
            (Bus,Loc1,T1, Bus,Loc2,T2)
             ----------   -----------
               origin     destination
               
     The aspect list for the example is compose by three aspects with the limited set of elementes.
     
     set_aspect[0] = {'Bus','Subway'}
     set_aspect[1] = {'Loc1','Loc2','Loc3'}
     set_aspect[2] = {'T1','t2','T3'}
     Tau = [2,3,3]
 
     P.S.
     ----
     If any aspect caracther is modify the tuple will be another tuple, as the example below.
     ('Bus','Loc1','T1') != ('Bus','loc1','T1')
   
  """
  def to_directed_class(self):
    return MultiAspectMultiDiGraph()

  def to_undirected_class(self):
    return MultiAspectMultiGraph()
  
  aspect_list_factory = list()
  
  def __init__(self, data=None, **attr):
    """
      Initializer method of MultiAspectGraph().

      Parameters
      ----------
        data: Input graph, data is None by default.
          The input can be a list of edges.
          
        attr: Attributes to add to the graph as a key=value pair.
          Is None by default.

      Returns
      -------
        MultiAspectMultiGraph()

      Examples
      --------
        >>> G = MultiAspectMultiGraph()
        >>> G = MultiAspectMultiGraph(name='MAG)
        >>> G = MultiAspectMultiGraph(edgelist, name='MAG)
    """
    self._order = None
    self._aspect = self.aspect_list_factory
    MultiGraph.__init__(self,data,**attr)  


  def clear(self):
    """
      Remove all nodes, edges, and aspects from the MultiAspectGraph.
      Also removes the graph, nodes and edges attributes.

      Example
      -------
        >>> G.clear()
      
    """
    self._order = None
    self._aspect.clear()
    MultiGraph.clear(self)
  
  def order(self):
    """
      This method returns the order of the MultiAspectMultiGraph.
      The order is the number of aspects that composes the aspects list in the MultiAspectMultiGraph.
      
      Return
      -------
        int

     Example
     -------
        >>> G.order()
    """
    return self._order

  
  def add_node(self, node, **attr):
    """
    Addition of each node by manual insertion.
    This function will add new nodes to an existing MultiAspectMultiGraph.
        
    Parameter
    ---------
      node:
        is a tuple of aspects that compose the node.
          
    Examples
    --------
      >>> node = ('Bus','Loc1','T1')
      >>> G.add_node(node)
      >>> G.add_node(('Bus','Loc2','T2'))

      >>> J.add_node(('North',1))

      >>> M.add_node((1,), pos=1)
      >>> M.add_node((2,), pos=2)
      >>> M.add_node((3,), pos=3)
        
    P.S.
    ----
      There is no restriction on the number of aspects, but must be the same for all nodes.
      
    """
    if self._order is None:
      initialize_list(self, len(node))
      self._order = len(node)
    if (aspect_node_integrity(self, node) == True):
      update_aspect_list(self, node)
      if node not in self._adj:
        self._adj[node] = self.adjlist_inner_dict_factory()
        self._node[node] = attr
      else:
        # update attr even if node already exists
        self._node[node].update(attr)
    else:
      raise ValueError("The number os aspect is incorrect! The MAG has a list with {0} aspects, but {1} have been given.".format(len(self._aspect),len(node)))
  
      
  def add_nodes_from(self, nodelist, **attr):
    """
      This function allows the manual insertion of nodes (list of nodes).
      
      Parameters
      ----------
        node_list: the list of nodes that is insert into the MultiAspectMultiGraph.
              Every tuple on the list is a node.

        **attr: Is a dicionary of comon attributes.
              If the attr is not empty, then the insertion of common attributes occurs.   
      
      Examples
      --------
        >>> G.add_nodes_from(('bus','loc1','t1'))
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2',t2')])
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc1','t2')], identification='bus point')
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2','t2'),('bus','loc3','t1'),('subway','loc1','t1')])

    """
    for n in nodelist:
      try:
        if n not in self._adj:
          self.add_node(n,**attr)
        else:
          self._node[n].update(attr)
      except TypeError:
        nn, data = n
        self.add_node(nn,**data)
  
  
  def add_nodes_from_file(self, name_file):
    """
      This method returns a MultiAspectGraph create by file import of nodes. Each line in the file is a node. Thus, the separation of nodes is given by enter (\n).
      The attributes must be separated by a semicolon (;) and the type must be separated by equal (=) from the value.

      Parameter
      ---------
        name_file = name of an exitent file of nodes

      Returns
      -------
        MultiAspectMultiGraph()
        
      Example
      -------
        >>>  G.add_nodes_from_file('list_of_nodes.txt')
        >>>  G.add_nodes_from_file('list_of_nodes.csv')
            
      Example of nodes format in the file
      -----------------------------------
        Nodes without attributes
      
        (Bus,Loc1,T1)
        (Bus,Loc1,T2)
        (Bus,Loc2,T2)
        
        If there are one or more attributes for nodes, the file format must be as below.
      
        (Bus,Loc1,T1)<attr1=001;attr2=34>
        (Bus,Loc1,T2)<attr2=35>
        (Bus,Loc2,T2)<attr1=009;attr2=36>
        (Bus,Loc3,T3)

    """    
    archive = open(name_file,'r')
    
    if (archive and aspect_integrity(archive,0)):
      file = archive.readline()
      while(file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):
            file = archive.readline()
            continue
        node = split_tuple(file)
        try: node = [type_str_number(n) for n in node]
        except: raise ValueError()
        if (file.find("<")) != -1 and (file.find(">")) != -1:
          #there're attributes
          attr = list()
          element = split_weight(file)         
          attr = [e.split('=') for e in element]
          for a in (attr):         
            a[0] = type_str_number(a[0])
            a[1] = type_str_number(a[1])
          attr = dict(attr)
          self.add_node(tuple(node),**attr)
        else:
            self.add_node(tuple(node))
        file = archive.readline()
    else:
      raise ValueError("Error during the file import!")
  
          
  def add_edge(self, u, v, key = None, **attr):
    """
    Addition of each edge by manual insertion.

    Parameters
    ----------
      u, v: are the nodes
      
      **attr: is a dicionary of weight of the edge
      
    Examples
    --------
      >>> G.add_edge((1,), (2,), weight=4.7)
      >>> A.add_edge((2,0.3),(1,0.005))

      >>> u = ('Bus','Loc1','T1')
      >>> v = ('Bus','Loc2','T3')
      >>> M.add_edge(u, v, weight=2)
      >>> M.add_edge(('Bus','Loc1','T1'),('Bus','Loc2','T2'), time=10)
      
    """
    #add nodes
    self.add_node(u)
    self.add_node(v)
      
    if key is None:
      key = self.new_edge_key(u, v)
    if v in self._adj[u]:
      keydict = self._adj[u][v]
      datadict = keydict.get(key, self.edge_attr_dict_factory())
      datadict.update(attr)
      keydict[key] = datadict
    else:
      # selfloops work this way without special treatment
      datadict = self.edge_attr_dict_factory()
      datadict.update(attr)
      keydict = self.edge_key_dict_factory()
      keydict[key] = datadict
      self._adj[u][v] = keydict
      self._adj[v][u] = keydict
    return key  
  
  def add_edges_from_file(self, name_file):             
    """
    This method creates a MultiAspectGraph from file import of edges.
    The edges can be weighted or not, and there is no limitation to the number of weights.

    Parameter
    ---------
      name_file: name of an existent file of edges.
          If the file doesn't exists an error will be returned.

    Returns
    -------
      MultiAspectMultiGraph()
    
    Examples
    --------
      >>>  G.add_edges_from_file('edges.txt')
      >>>  G.add_edges_from_file('edges.csv')      
          
    Example of edges formart in the file
    ------------------------------------
      unweighted edges

      (Bus,Loc1,T1,Bus,Loc1,T1)
      (Bus,Loc1,T2,Bus,Loc2,T2)
      (Subway,Loc1,T1,Bus,Loc1,T1)
        
      weighted edges
      
      (Bus,Loc1,T1,Bus,Loc1,T1)<timewait=2>
      (Bus,Loc1,T2,Bus,Loc2,T2)<passengers=10;timepass=15>
      (Subway,Loc1,T1,Bus,Loc1,T1)<timetransfer=5>

    """
    archive = open(name_file,'r')
    if (archive and aspect_integrity(archive,1) == True):
      file = archive.readline()
      Type = identify_edge(file)
      while (file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):
            file = archive.readline()
            continue            
        edge = split_tuple(file)
        try: edge = [type_str_number(e) for e in edge]
        except: raise ValueError()
        From = tuple(edge[0:int((len(edge)/2))])
        To = tuple(edge[int((len(edge)/2)):len(edge)])
        key = self.add_edge(From,To)
        if file.find("<") != -1 and file.find(">") != -1:
          add_multi_edge_weight(self, file, From, To, key, Type)   #creating the weighted egdes

        file = archive.readline()
    else:
      raise ValueError("Error during the file import!")
  
  
  def number_of_aspects(self):
    """
      This method returns the number of aspects in the MultiAspectMultiGraph.

      Return
      ------
        int
       
      Example
      -------
        >>> G.number_of_aspects()
        
    """
    return len(self._aspect)
  
  
  def get_number_of_elements(self, nasp):
    """
      This method returns the number of elements in the chosen aspect.

      Parameter
      ---------
        nasp: number of the aspect
              N = Total number of aspects
              0 < nasp < N

      Return
      -------
        int
        
      Example
      -------
        >>> G.get_number_of_elements(1) #returns the number of elements in the first aspect
        >>> G.get_number_of_elements(3) #returns the number of elements in the third aspect
        
    """
    return len(self._aspect[nasp-1])
  
  
  def get_number_elements_aspects(self):
    """
    This method returns a list with the total number of elements in each aspect.

    Return
    ------
      list of ints
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_number_elements_aspects()
      [2,3,4]
      
    """
    return [len(self._aspect[i]) for i in range (len(self._aspect))]
  
  
  def get_aspects_list(self):
    """
    Returns the list of aspect.

    Return
    ------
      list of sets
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_aspects_list()
      [{'Subway', 'Bus'}, {'Loc1', 'Loc2', 'Loc3'}, {'T2', 'T3', 'T1'}]
      
    """
    return self._aspect.copy()
  
  
  def print_aspects_list(self):
    """
    This method will print the MultiAspectGraph aspects list.
    
    Return
    ------
      Print in the screen

    Example
    -------
      >>> G.print_aspects_list()
          
      example: hypothetical transport model
      
      Set[0]: {'Subway', 'Bus'}
      Set[1]: {'Loc3', 'Loc1', 'Loc2'}
      Set[2]: {'T3', 'T1', 'T2'}
      Tau: [2, 3, 3]
      
    """
    len_aspect = len(self._aspect)
    
    if len(self._aspect) == 0:
      raise ValueError("Aspect list not initialized!")
    if len(self._aspect[0]) == 0:
      print ("The list of aspects set is empty!\n")
    else:
      for i in range (len_aspect):
        print("Set[{}]: {}".format(i,self._aspect[i]))
      print("Tau: {}".format([len(self._aspect[i]) for i in range (len_aspect)]))
    
  
  def compact_aspects_list(self):
    """
    This method refreshs the aspect list. One can use this method after many removals (nodes or edges).
    The aspect list is emptied and updated.
    
    Example
    -------
      >>> G.compact_aspects_list()

    """
    [a.clear() for a in self._aspect]
    for node in self.nodes():
      update_aspect_list(self, node)


  def is_multigraph(self):
    """
      Returns True if the graph is a multigraph, False otherwise.
    """
    return True

  def is_directed(self):
    """
      Returns True if the graph is directed, False otherwise.
    """
    return False
  
  def is_mag(self):
    """
      Returns True if the graph is a MAG, False otherwise.
    """
    return True


  def aspects_subgraph(self, aspect_list):
    """
    This function creates an subMags inducing by aspects.

    Parameters
    ----------
        aspect_list: aspect list
            This reduced list derives from the original aspect list.
            The function using this aspect list will create the subMag.

    Returns
    -------
        SubMag induced by aspects - analogous to a subgraph

    Example
    -------
        >>> G.aspects_subgraph(G, aspects): 
    """
    return subMag_aspect_induced(self, aspect_list)


  
  #--------------------- MAG SUBDETERMINATION --------------------------#
 
  def subdetermination(self, zeta, multi=True, direct=False, loop=False, **attr):
    """ 
      This method returns the subdeterminartion of the MultiAspectMultiGraph(). This new MultiAspectMultiGraph has a lower order than the original.
      
      Parameters
      ----------
        zeta: is a binary list with the positions related to the aspects of MultiAspectGraph.
              The position of the elements in the list is related to the aspects. For the values 0 and 1, the aspect is suppressed and sustained respectively.

        multi: True by default
            For the result of this method be a multiedge MAG, the tag must be True. Otherwise, must be False.

        direct: False by default
            For the result of this method be a direct MAG, the tag must be True. Otherwise, must be False.
            
        loop: False by default
            True, returning the edges with loops. False, the edges with the same vertice aren`t in the subdetermination of MAG.
          
       **attr: dicionary of graph attributes, such as name.

     Return
     ------
       Class of MAG
      
     Example
     -------
      example: for a MultiAspectGraph with order equals three
      
      >>> zeta = [1,0,1]
      >>> H = G.subdetermination(zeta)
      >>> type(H)
      __main__.MultiAspectMultiGraph
      >>> zeta = [1,0,0]
      >>> T = G.subdetermination(zeta, multi=False, name='subdetermination', day='today') #with attributes
      >>> type(T)
      __main__.MultiAspectGraph

    """ 
    if nx.is_weighted(self):
      Par = subDeterminedEdgePartition(self, zeta)
      
    if len(zeta) != self.number_of_aspects():
      raise ValueError ('The number of elements in zeta is incorrect. The number of aspects in MAG is {}, and {} have been given!'.format(self.number_of_aspects(),len(zeta)))
      
    #Verify the basic cases
#    if (zeta_zero(zeta) == True):
#      print("All aspects supressed. Null returned")
#      return None
#    
#    if (zeta_one(zeta) == True):
#      print("None aspect was suppressed. The same MAG is returned")
#      return self
    
    #variables
    lenz = len(zeta)
    asps = list(zeta)+list(zeta)
    total = len(asps)
 #   naspects = zeta.count(1)
    if multi:
      H = MultiAspectMultiDiGraph(**attr) if direct else MultiAspectMultiGraph(**attr)
    else:
      H = MultiAspectDiGraph(**attr) if direct else MultiAspectGraph(**attr)

    #edge list verification
    for e,datadict in self.edges.items():
      new_edge = [(e[0][i]) if i<lenz else (e[1][i-lenz]) for i in range (total) if asps[i]!=0]
      From = tuple(new_edge[0:int(len(new_edge)/2)])
      To = tuple(new_edge[int(len(new_edge)/2):len(new_edge)])
      if (From != To and loop == False) or loop == True:
        H.add_edge(From,To)
        if nx.is_weighted(self):
            k = list(datadict.keys())
            for w in k:
                if w == "weight":
                    H[From][To][w] = min(Par[From,To])
                else:
                    H[From][To][w] = datadict[w]
               
    #node list verification
    for n in self.nodes():
      node = [n[i] for i in range(0,len(n)) if zeta[i] !=0]
    H.add_node(tuple(node))

    #return the subdetermination of MAG
    return H

  
  #------------------------------- Adj Matrix ------------------------------------#  
  
  
  def sparse_adj_matrix(self, nodelist=None, weight=None, dtype=None):
    """
      This method will convert the MultiAspectMultiGraph in an sparse ajdcency matrix.

      Parameter
      ---------
        nodelist: is a list of nodes (optional). The nodes compose the lines and rows of the adjcency matrix.
        The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).
            
        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None by default, and representes the existence of an edge between two nodes.
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        scipy sparse adj matrix
        list of tuples (nodes)
        
      Example
      -------
        >>> G.sparse_adj_matrix()
        >>> G.sparse_adj_matrix(weight='weight')
        
    """  
    try:
      from scipy import sparse
    except:
      raise ImportError('The numpy module is necessary and is not installed!')  
    
    if self.number_of_nodes == 0:
      raise nx.NetworkXError('The list of nodes on MAG is empty.')

    if nodelist is None:
      nodelist = list(self)
    len_nodes = len(nodelist)
    dict_node = dict(zip(nodelist,range(len_nodes)))
    edge_list = dict()
    Edges = list(self.edges(nodelist,data=weight))
    color = dict.fromkeys([(dict_node[u],dict_node[v]) for u,v,w in Edges], 0)

    if weight is None:
      for u,v,w in Edges:
        try:
          uu = dict_node[u]
          vv = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            key = len(self._adj[u][v])
            edge_list.update({(uu,vv):key})
            color[(uu,vv)] = 1
    else:
      for u,v,w in Edges:
        if w == None:
          raise ValueError('All the nodes must have the selected weight')
        try:
          uu = dict_node[u]
          vv = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            edge_list.update({(uu,vv):w})
            color[(uu,vv)] = 1
          else:
            value = edge_list[(uu,vv)]
            edge_list[(uu,vv)] = value+w
                
    #symmetric matrix
    key = list(edge_list.keys())
    row = [key[i][0] for i in range(len(key))] + [key[i][1] for i in range(len(key))]
    col = [key[i][1] for i in range(len(key))] + [key[i][0] for i in range(len(key))]
    data = [edge_list[e] for e in key] + [edge_list[e] for e in key]
      
    list_tuples=[a[0] for a in (sorted(dict_node.items(),key=lambda e:e[1]))]
    M = sparse.coo_matrix((data, (row,col)), shape=(len_nodes, len_nodes), dtype=dtype)
  
    return M,list_tuples
  
    

  #---------------------------- Incidence Matrix ----------------------------------# 

  def sparse_incidence_matrix(self, edgelist=None, nodelist=None, weight=None):
    """
      This method will convert the MultiAspectMultiGraph in an sparse incidence matrix.

      Parameter
      ---------
        edgelist: is a list of edges (optional). The edges compose the rows in the incidence matrix
            The edgelist determine which edges compose the matrix. If is None, edgelist is equal self.edges().
        
        nodelist: is a list of nodes (optional). The nodes compose the lines of the incidence matrix.
            The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).

        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None (default), then each edge has weight 1
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        scipy sparse incidence matrix
        list of tuples (nodes)
	list of edges
        
      Example
      -------
        >>> G.sparse_incidence_matrix()
        >>> G.sparse_incidence_matrix(weight='weight')
        
    """
    try:
      from scipy import sparse
    except:
      raise ImportError('The scipy module is necessary and is not installed!')
      
    if nodelist == None:
      nodelist = list(self)
    len_nodes = len(nodelist)
    
    if edgelist == None:
      edgelist = self.edges()
    len_edges = len(edgelist)
    M = sparse.lil_matrix((len_nodes,len_edges))
    edgeset=set()
    
    for a in edgelist:
      s = len(self._adj[a[0]][a[1]])
      if s == 1:
        edgeset.add((a[0],a[1],0))
      else:
        for i in range(s):
          edgeset.add((a[0],a[1],i))

    dictt_node = dict(zip(nodelist,range(len_nodes)))
    dictt = dict(zip(edgeset,range(len_edges)))
    if weight is None:
      for a in dictt:
        # a = (u,v,index)
        try:
          u = dictt_node[a[0]]
          v = dictt_node[a[1]]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        e = dictt[a]
        M[u,e] = 1
        M[v,e] = 1
    else:
      for a in dictt:
        try:
          u = dictt_node[a[0]]
 #         uu = a[0]
          v = dictt_node[a[1]]
 #         vv = a[1]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        w = self._adj[a[0]][a[1]]
        dict_w = w[a[2]]        #index of edge
        try:
          value = dict_w[weight]
        except KeyError:
          raise NetworkXError('All edges must be the weight that was especify')
        e = dictt[a]
        M[u,e] = value
        M[v,e] = value
        
    list_node_tuple = [a[0] for a in (sorted(dictt_node.items(), key=lambda e:e[1]))]
    list_edge_tuple = [a[0] for a in (sorted(dictt.items(), key=lambda e:e[1]))]

    return M, list_node_tuple, list_edge_tuple
  

# --------------------------------------------------------------- MultiDiGraph -------------------------------------------------------------#


class MultiAspectMultiDiGraph(MultiDiGraph):
  """  
    A graph generalization for representing networks of any (finite) order that represents binary relations. Formally, H = (A, E) represents the MAG,
    where E is a finite set of edges and A is a finite list of aspects. This representation is isomorphic to a traditional directed graph.
    This feature allows the use of existing graph algorithms without problems. The number of aspects (p) is the order of the MAG, and each aspect a ∈ A is a finite set.
    The nodes are tuples, and the edges are a tuple with 2p entries. From the list of edges derives the composite vertices, as the cartesian product of all aspects.

    This module has four classes based on NetworkX to represents the MultiAspect Graph (MAG).
    The current class is the MultiAspectMultiDiGraph(), that is a directed version of MAG with multiedges.
     
    To examplify the MAG there are some examples of a hypothetical transport model bellow.
  
    Edge
    ----
      (from,to) = (Bus,Loc1,T1, Bus,Loc2,T2)
                   ----------   -----------
                    origin     destination
    Nodes
    -----
      from = (Bus,Loc1,T1)
      to = (Bus,Loc1,T2)
            
    Types of edges and examples
    ---------------------------
    spatial:  The edges occur in the same instant of time.
       
             (Bus,Loc1,T1, Subway,Loc1,T1)
              ----------    -----------
                origin      destination
 
    temporal: The edges occur between different two time instants.
    
             (Bus,Loc1,T1, Bus,Loc1,T2)
              ----------   -----------
                origin     destination  
 
    mixer: In this case we have two variations, in time and space.
    
            (Bus,Loc1,T1, Bus,Loc2,T2)
             ----------   -----------
               origin     destination
               
     The aspect list for the example is compose by three aspects with the limited set of elementes.
     
     set_aspect[0] = {'Bus','Subway'}
     set_aspect[1] = {'Loc1','Loc2','Loc3'}
     set_aspect[2] = {'T1','t2','T3'}
     Tau = [2,3,3]
 
     P.S.
     ----
     If any aspect caracther is modify the tuple will be another tuple, as the example below.
     ('Bus','Loc1','T1') != ('Bus','loc1','T1')
   
  """

  def to_directed_class(self):
    return MultiAspectDiGraph()

  def to_undirected_class(self):
    return MultiAspectGraph()
  
  aspect_list_factory = list()
  
  def __init__(self, data=None, **attr):
    """
      Initializer method of MultiAspectMultiDiGraph().

      Parameters
      ----------
        data:
          Input MAG, data is None by default.
          The input can be a list of edges.
        attr:
          Attributes to add to the graph as a key=value pair.
          Is None by default.

      Returns
      -------
        MultiAspectMultiDiGraph()

      Examples
      --------
        >>> import MAG as mag
        >>> G = mag.MultiAspectMultiDiGraph()
        >>> G = mag.MultiAspectMultiDiGraph(name='MAG)
        >>> G = mag.MultiAspectMultiDiGraph(edgelist, name='MAG)
        
    """
    self._order = None
    self._aspect = self.aspect_list_factory
    MultiDiGraph.__init__(self,data,**attr)


  def clear(self):
    """
      Remove all nodes, edges, and aspects from the MultiAspectGraph.
      Also removes the graph, nodes and edges attributes.

      Example
      -------
        >>> G.clear()
      
    """
    self._order = None
    self._aspect.clear()
    MultiDiGraph.clear(self)
      
  def order(self):
    """
      This method returns the order of the MultiAspectDiGraph.
      The order is the number of aspects that composes the aspects list in the MultiAspectDiGraph.
      
      Return
      -------
        int

     Example
     -------
        >>> G.order()
    """
    return len(self._aspect)

      
  def add_node(self, node, **attr):
    """
    Addition of each node by manual insertion.
    This function will add new nodes to an existing graph.
        
    Parameter
    ---------
      node:
        is a tuple of aspects that compose the node.
          
    Examples
    --------
      >>> node = ('Bus','Loc1','T1')
      >>> G.add_node(node)
      >>> G.add_node(('Bus','Loc2','T2'))

      >>> J.add_node(('North',1))

      >>> M.add_node((1,), pos=1)
      >>> M.add_node((2,), pos=2)
      >>> M.add_node((3,), pos=3)
        
    P.S.
    ----
      There is no restriction on the number of aspects, but must be the same for all nodes.
      
    """
    if self._order is None:
      initialize_list(self, len(node))
      self._order = len(node)
    if (aspect_node_integrity(self, node) == True):
      update_aspect_list(self, node)

      node = tuple(node)    
      if node not in self._node:
        self._succ[node] = self.adjlist_inner_dict_factory()
        self._pred[node] = self.adjlist_inner_dict_factory()
        self._node[node] = attr
      else:  
        # update attr even if node already exists
        self._node[node].update(attr)
    else:
      raise ValueError("The number os aspect is incorrect! The MAG has a list with {0} aspects, but {1} have been given.".format(len(self._aspect),len(node)))
  
  
  def add_nodes_from(self, nodelist, **attr):
    """
      This function allows the manual insertion of nodes (list of nodes).
      
      Parameters
      ----------
        node_list: the list of nodes that is insert into the MultiAspectMultiDiGraph.
              Every tuple on the list is a node.

        **attr: Is a dicionary of comon attributes.
              If the attr is not empty, then the insertion of common attributes occurs.   
      
      Examples
      --------
        >>> G.add_nodes_from(('bus','loc1','t1'))
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2',t2')])
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc1','t2')], identification='bus point')
        >>> G.add_nodes_from([('bus','loc1','t1'),('bus','loc2','t2'),('bus','loc3','t1'),('subway','loc1','t1')])

    """
    for n in nodelist:
      try:
        if n not in self._adj:
          self.add_node(n,**attr)
        else:
          try:
            self._node[n].update(attr)
          except:
            self.node[n].update(attr)
      except TypeError:
        nn, data = n
        self.add_node(nn,**data)

  
  def add_nodes_from_file(self, name_file):
    """
      This method returns a MultiAspectMultiDiGraph create by file import of nodes. Each line in the file is a node. Thus, the separation of nodes is given by enter (\n).
      The attributes must be separated by a semicolon (;) and the type must be separated by equal (=) from the value.

      Parameter
      ---------
        name_file = name of an exitent file of nodes

      Returns
      -------
        MultiAspectGraph()
        
      Example
      -------
        >>>  G.add_nodes_from_file('list_of_nodes.txt')
        >>>  G.add_nodes_from_file('list_of_nodes.csv')
            
      Example of nodes format in the file
      -----------------------------------
        Nodes without attributes
      
        (Bus,Loc1,T1)
        (Bus,Loc1,T2)
        (Bus,Loc2,T2)
        
        If there are one or more attributes for nodes, the file format must be as below.
      
        (Bus,Loc1,T1)<attr1=001;attr2=34>
        (Bus,Loc1,T2)<attr2=35>
        (Bus,Loc2,T2)<attr1=009;attr2=36>
        (Bus,Loc3,T3)

    """
    archive = open(name_file,'r')
    
    if (archive and aspect_integrity(archive,0)):
      file = archive.readline()
      while(file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):   #linha em branco (enter, espaço, ...)
            break
        node = split_tuple(file)
        try:
          node = [type_str_number(n) for n in node]
        except:
          raise ValueError("ValueError")
        if (file.find("<")) != -1 and (file.find(">")) != -1:                #there're nodes attributes
          attr = list()
          element = split_weight(file)         
          attr = [e.split('=') for e in element]
          for a in (attr):         
            a[0] = type_str_number(a[0])
            a[1] = type_str_number(a[1])
          attr = dict(attr)
          self.add_node(tuple(node),**attr)
        else:
            self.add_node(tuple(node))
        file = archive.readline()
    else:
      raise ImportError("ERROR IN THE FILE IMPORT!")
  
        
  def add_edge(self, u, v, key=None, **attr):
    """
    Addition of each edge by manual insertion.
    The edges to be addition in the graph are not weighted by default.

    Parameters
    ----------
        u,v : nodes
            Nodes can be, for example, tuples of strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        
        key : hashable identifier, optional (default=lowest unused integer)
            Used to distinguish multiedges between a pair of nodes.
        
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

    Return
    ------
      int [key]
  
    Examples
    --------
      >>> G.add_edge((1,), (2,), weight=4.7)
      >>> A.add_edge((2,0.3),(1,0.005))

      >>> u = ('Bus','Loc1','T1')
      >>> v = ('Bus','Loc2','T3')
      >>> M.add_edge(u, v, weight=2)
      >>> M.add_edge(u, v, key=1, weight=5)
      >>> M.add_edge(('Bus','Loc1','T1'),('Bus','Loc2','T2'), time=10)

    """
    self.add_node(u)
    self.add_node(v)
    
    if key is None:
      key = self.new_edge_key(u, v)
      
    if v in self._succ[u]:
      keydict = self._adj[u][v]
      datadict = keydict.get(key, self.edge_key_dict_factory())
      datadict.update(attr)
      keydict[key] = datadict
    else:
      datadict = self.edge_attr_dict_factory()
      datadict.update(attr)
      keydict = self.edge_key_dict_factory()
      keydict[key] = datadict
      self._succ[u][v] = keydict
      self._pred[v][u] = keydict
    return key
  
  
  def add_edges_from(self, edgelist, **attr):
    """
    This function allows the manual insertion of nodes.
      
    This method allows the manual insertion of edges.
      
    Parameters
    ----------
      list_of_edges: is a list of edges
            Each element of the list has two tuples, where these tuples are the nodes that compose the edge.
            The elements in the list of edges is compose by (u,v,{weight=value})) or (u,v).
            
      **attr: is a dicionary of edge attributes (edge weight).
            These are common attributes (weights) to the edges on the list.

    Return
    ------
      list of ints [keys]

    Examples
    --------
      >>> f = ('Bus','Loc1','T1')
      >>> t = ('Bus','Loc1','T2')
      >>> c = ('Bus','Loc1','T3')
      >>> G.add_edges_from([(f,t,{'time':5}),(t,c,{'time':2})]) 
      >>> G.add_edges_from([(('Bus','Loc2','T1'),('Bus','Loc2','T2'))])
      >>> G.add_edges_from([(('Bus','Loc1','T1'),('Bus','Loc1','T2'))], weight=1) #common attribute
 
    """
    keylist = list()
    
    for e in edgelist:
      ne = len(e)
      if ne == 4:
        u,v,key,dd = e
      elif ne == 3:
        u, v, dd = e
      elif ne == 2:
        u, v = e
        dd = {}
      else:
        raise NetworkXError("Edge tuple %s must be a 2-tuple, 3-tuple or 4-tuple." % (e,))
      
      dictt = {}
      dictt.update(attr)
      
      try:
        dictt.update(dd)
      except:
        if ne != 3:
          raise
        key = dd
        
      key = self.add_edge(u,v)
      try:
        self._adj[u][v][key].update(dictt)
      except:
        self.adj[u][v][key].update(dictt)
      keylist.append(key)
    
    return keylist
  
  
  
  def add_edges_from_file(self, name_file):                      
    """
    This method creates a MultiAspectGraph from file import of edges.
    The edges can be weighted or not, and there is no limitation to the number of weights.
          
    Parameter
    ---------
      name_file: name of an existent file of edges.
          If the file doesn't exists an error will be returned.

    Returns
    -------
      MultiAspectMultiDiGraph()
         
    Example
    -------
      example of hypothetical transport model
      
      >>>  G.add_edges_from_file('edges.txt')
        
    Example of edges formart in the file
    ------------------------------------
      unweighted edges

      (Bus,Loc1,T1,Bus,Loc1,T1)
      (Bus,Loc1,T2,Bus,Loc2,T2)
      (Subway,Loc1,T1,Bus,Loc1,T1)
        
      weighted edges

      (Bus,Loc1,T1,Bus,Loc1,T1)<timewait=2>
      (Bus,Loc1,T2,Bus,Loc2,T2)<passengers=10;timepass=15>
      (Bus,Loc1,T2,Bus,Loc2,T2)<passengers=50;timepass=30>
      (Subway,Loc1,T1,Bus,Loc1,T1)<timetransfer=5>
      (Subway,Loc1,T2,Bus,Loc1,T2)
      (Subway,Loc1,T3,Bus,Loc1,T3)
      (Bus,Loc1,T2,Subway,Loc2,T2)<passengers=200;timepass=30>
      ...
          
    """
    archive = open(name_file,'r')
    if (archive and aspect_integrity(archive,1) == True):
      file = archive.readline()
      Type = identify_edge(file)
      while (file):
        if (file == "\n" or file == " \n" or file == "" or file == " "):
            file = archive.readline()
            continue
        edge = split_tuple(file)
        try: edge = [type_str_number(e) for e in edge]
        except: raise ValueError()
        From = tuple(edge[0:int((len(edge)/2))])
        To = tuple(edge[int((len(edge)/2)):len(edge)])
        key = self.add_edge(From,To)
        if file.find("<") != -1 and file.find(">") != -1:
          add_multi_edge_weight(self, file, From, To, key, Type)  #creating the weighted egdes

        file = archive.readline()
    else:
      raise ImportError("Error in the file import!")
      

  def number_of_aspects(self):
    """
      This method returns the number of aspects in the MultiAspectMultiDiGraph.

      Return
      ------
        int
       
      Example
      -------
        >>> G.number_of_aspects()
        
    """
    return len(self._aspect)
  
  def get_number_of_elements(self, nasp):
    """
      This method returns the number of elements in the chosen aspect.

      Parameter
      ---------
        nasp: number of the aspect
              N = Total number of aspects
              0 < nasp < N

      Return
      -------
        int
        
      Example
      -------
        >>> G.get_number_of_elements(1) #returns the number of elements in the first aspect
        >>> G.get_number_of_elements(3) #returns the number of elements in the third aspect
        
    """
    return len(self._aspect[nasp-1])
  
   
  def get_number_elements_aspects(self):
    """
    This method returns a list with the total number of elements in each aspect.

    Return
    ------
      list of ints
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_number_elements_aspects()
      [2,3,4]
      
    """
    return [len(self._aspect[i]) for i in range (len(self._aspect))]
  
  
  def get_aspects_list(self):
    """
    Returns the list of aspect.

    Return
    ------
      list of sets
      
    Example
    -------
      example: hypothetical transport model
      
      >>> G.get_aspects_list()
      [{'Subway', 'Bus'}, {'Loc1', 'Loc2', 'Loc3'}, {'T2', 'T3', 'T1'}]
      
    """
    return self._aspect.copy()
  
  def print_aspects_list(self):
    """
    This method will print the MultiAspectMultiDiGraph aspects list.
    
    Return
    ------
      Print in the screen

    Example
    -------
      >>> G.print_aspects_list()
          
      example: hypothetical transport model
      
      Set[0]: {'Subway', 'Bus'}
      Set[1]: {'Loc3', 'Loc1', 'Loc2'}
      Set[2]: {'T3', 'T1', 'T2'}
      Tau: [2, 3, 3]
    """
    len_aspect = len(self._aspect)
    if len(self._aspect) == 0:
      raise ValueError("Aspect list not initialized!")
    if len(self._aspect[0]) == 0:
      print ("The list of aspects set is empty!\n")
    else:
      for i in range (len_aspect):
        print("Set[{}]:{}".format(i,self._aspect[i]))
      print("Tau: {}".format([len(self._aspect[i]) for i in range (len_aspect)]))
    
   
  def compact_aspects_list(self):
    """
    This method refreshs the aspect list. One can use this method after many removals (nodes or edges).
    The aspect list is emptied and updated.
    
    Example
    -------
      >>> G.compact_aspects_list()
    """
    [a.clear() for a in self._aspect]    
    for node in self.nodes():
      update_aspect_list(self, node)
  
  def is_multigraph(self):
    """
      Returns True if the graph is a multigraph, False otherwise.
    """
    return True

  def is_directed(self):
    """
      Returns True if the graph is directed, False otherwise.
    """
    return True
  
  def is_mag(self):
    """
      Returns True if the graph is a MAG, False otherwise.
    """
    return True


  def aspects_subgraph(self, aspect_list):
    """
    This function creates an subMags inducing by aspects.

    Parameters
    ----------
        aspect_list: aspect list
            This reduced list derives from the original aspect list.
            The function using this aspect list will create the subMag.

    Returns
    -------
        SubMag induced by aspects - analogous to a subgraph

    Example
    -------
        >>> G.aspects_subgraph(aspects): 
    """
    return subMag_aspect_induced(self, aspect_list)
  

  #-------------------------- MAG SUBDETERMINATION -----------------------------------#

  def subdetermination(self, zeta, multi=True, direct=True, loop=False, **attr):
    """ 
      This method returns the subdeterminartion of the MultiAspectMultiDiGraph(). This new MultiAspectMultiDiGraph has a lower order than the original.
      
      Parameters
      ----------
        zeta: is a binary list with the positions related to the aspects of MultiAspectMultiDiGraph.
              The position of the elements in the list is related to the aspects. For the values 0 and 1, the aspect is suppressed and sustained respectively.
              
        multi: True by default
            For the result of this method be a multiedge MAG, the tag must be True. Otherwise, must be False.

        direct: True by default
            For the result of this method be a direct MAG, the tag must be True. Otherwise, must be False.
            
        loop: False by default
            True, returning the edges with loops. False, the edges with the same vertice aren`t in the subdetermination of MAG.
            
       **attr: dicionary of graph attributes, such as name.

     Return
     ------
       Class of MAG
      
     Example
     -------
      example: for a MultiAspectMultiDiGraph with order equals three
      
      >>> zeta = [1,0,1]
      >>> H = G.subdetermination(zeta)
      >>> type(H)
      __main__.MultiAspectMultiDiGraph
      >>> zeta = [1,0,0]
      >>> T = G.subdetermination(zeta, multi=False, name='subdetermination', day='today') #with attributes
      >>> type(T)
      __main__.MultiAspectDiGraph
      
    """ 
    if nx.is_weighted(self):
      Par = subDeterminedEdgePartition(self, zeta)
      
    if len(zeta) != self.number_of_aspects():
      raise ValueError('The number of elements in zeta is incorrect. The number of aspects in MAG is {}, and {} have been given!'.format(self.number_of_aspects(),len(zeta)))
    
    #Verify the basic cases
#    if (zeta_zero(zeta) == True):
#      print("All aspects supressed. Null returned")
#      return None
#    
#    if (zeta_one(zeta) == True):
#      print("None aspect was suppressed. The same MAG is returned")
#      return self
    
    #variables
    lenz = len(zeta)
    asps = list(zeta)+list(zeta)
    total = len(asps)
 #   naspects = zeta.count(1)
    if multi:
      H = MultiAspectMultiDiGraph(**attr) if direct else MultiAspectMultiGraph(**attr)
    else:
      H = MultiAspectDiGraph(**attr) if direct else MultiAspectGraph(**attr)

    #edge list verification
    for e, datadict in self.edges.items():
      new_edge = [(e[0][i]) if i<lenz else (e[1][i-lenz]) for i in range (total) if asps[i]!=0]        
      From = tuple(new_edge[0:int(len(new_edge)/2)])
      To = tuple(new_edge[int(len(new_edge)/2):len(new_edge)])
      if (From != To and loop == False) or loop == True:
        H.add_edge(From,To)
        if nx.is_weighted(self):
            k = list(datadict.keys())
            for w in k:
                if w == "weight":
                    H[From][To][w] = min(Par[From,To])
                else:
                    H[From][To][w] = datadict[w]
               
    #node list verification
    for n in self.nodes():
      node = [n[i] for i in range(0,len(n)) if zeta[i] !=0]
    H.add_node(tuple(node))
      
    #return the subdetermination of MAG
    return H
  
  
  #---------------------------- Adj Matrix ----------------------------------#
  
  
  def sparse_adj_matrix(self, nodelist=None, weight=None, dtype=None):
    """
      This method will convert the MultiAspectMultiDiGraph in an sparse ajdcency matrix.

      Parameter
      ---------
        nodelist: is a list of nodes (optional). The nodes compose the lines and rows of the adjcency matrix.
        The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).
            
        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None by default, and representes the existence of an edge between two nodes.
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        list of tuples (nodes)
        scipy sparse adjcency matrix
        
      Example
      -------
        >>> G.sparse_adj_matrix()
        >>> G.sparse_adj_matrix(weight='weight')
    """
    try:
      from scipy import sparse
    except:
      raise ImportError('The numpy module is necessary and is not installed!')  

    if self.number_of_nodes() == 0:
      raise nx.NetworkXError('The list of nodes on MAG is empty.')

    if nodelist is None:
      nodelist = list(self)
    len_nodes = len(nodelist)
    dict_node = dict(zip(nodelist,range(len_nodes)))
    edge_list = dict()
    Edges = list(self.edges(nodelist,data=weight))
    color = dict.fromkeys([(dict_node[u],dict_node[v]) for u,v,w in Edges], 0)

    if weight is None:
      for u,v,w in Edges:
        try:
          uu = dict_node[u]
          vv = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            key = len(self._adj[u][v])
            edge_list.update({(uu,vv):key})
            color[(uu,vv)] = 1
    else:
      for u,v,w in Edges:
        if w == None:
          raise ValueError('All the nodes must have the selected weight')
        try:
          uu = dict_node[u]
          vv = dict_node[v]
        except:
          raise ValueError('The nodes {} or {} are not in the nodelist!'.format(u,v))
        else:
          if color[(uu,vv)] == 0:
            edge_list.update({(uu,vv):w})
            color[(uu,vv)] = 1
          else:
            value = edge_list[(uu,vv)]
            edge_list[(uu,vv)] = value+w

    key = list(edge_list.keys())
    row = [key[i][0] for i in range(len(key))]
    col = [key[i][1] for i in range(len(key))]
    data = [edge_list[e] for e in key]
    
    list_tuples=[a[0] for a in (sorted(dict_node.items(),key=lambda e:e[1]))]
    M = sparse.coo_matrix((data, (row,col)), shape=(len_nodes, len_nodes), dtype=dtype)
    
    return M, list_tuples
  
   

  #---------------------------- Incidence Matrix ----------------------------------# 

  def sparse_incidence_matrix(self, edgelist=None, nodelist=None, weight=None):
    """
      This method will convert the MultiAspectMultiGraph in an sparse incidence matrix.

      Parameter
      ---------
        edgelist: is a list of edges (optional). The edges compose the rows in the incidence matrix
            The edgelist determine which edges compose the matrix. If is None, edgelist is equal self.edges().
        
        nodelist: is a list of nodes (optional). The nodes compose the lines of the incidence matrix.
            The nodelist determine which nodes compose the matrix. If is None (default), nodelist is equal list(self).

        weight: especifies the weight that will be used in the values of the matrix (optional).
            Is None (default), then each edge has weight 1
            
        dtype: NumPy data-type (optional). A valid NumPy dtype used to initialize the array.
            If None, then the NumPy default is used.
        
      Return
      ------
        scipy sparse incidence matrix
        list of tuples (nodes)
	list of edges
        
      Example
      -------
        >>> G.sparse_incidence_matrix()
        >>> G.sparse_incidence_matrix(weight='weight')
        
    """
    try:
      from scipy import sparse
    except:
      raise ImportError('The scipy module is necessary and is not installed!')
      
    if nodelist == None:
      nodelist = list(self)
    len_nodes = len(nodelist)
    if edgelist == None:
      edgelist = self.edges()
    len_edges = len(edgelist)
    M = sparse.lil_matrix((len_nodes,len_edges))
    edgeset=set()

    for a in edgelist:
      s = len(self._adj[a[0]][a[1]])
      if s == 1:
        edgeset.add((a[0],a[1],0))
      else:
        for i in range(s):
          edgeset.add((a[0],a[1],i))
      
    dictt_node = dict(zip(nodelist,range(len_nodes)))
    dictt = dict(zip(edgeset,range(len_edges)))
  
    if weight is None:
      for a in dictt:
        # a = (u,v,index)
        try:
          u = dictt_node[a[0]]
          v = dictt_node[a[1]]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        e = dictt[a]
        M[u,e] = -1
        M[v,e] = 1
    else:
      for a in dictt:
        try:
          u = dictt_node[a[0]]
#          uu = a[0]
          v = dictt_node[a[1]]
#          vv = a[1]
        except KeyError:
          raise NetworkXError('node %s or %s in edgelist '
                              'but not in nodelist"%(u,v)')
        w = self._adj[a[0]][a[1]]
        dict_w = w[a[2]]        #index of edge       
        try:
          value = dict_w[weight]
        except KeyError:
          raise NetworkXError('All edges must be the weight that was especify')
        e = dictt[a]
        M[u,e] = -value
        M[v,e] = value
        
    list_node_tuple = [a[0] for a in (sorted(dictt_node.items(), key=lambda e:e[1]))]
    list_edge_tuple = [a[0] for a in (sorted(dictt.items(), key=lambda e:e[1]))]
    
    return M, list_node_tuple, list_edge_tuple  
  







#
#
#
#
#                  TYPE CONVERSION OF EDGES WEIGHT
#
#
#
#                
#





  
def convert_type_edges_to_datetime(G,weight_name=None):
  """
    Parameters
    ----------
      weight_name: string, the name of weight that will be convert into datetime object.

    Return
    ------
      A MAG with the update of the weigthed edges for datetime
            
    Examples
    --------
      >>> G.convert_type_datetime('time')

    Example of the conversion
    -------------------------
      '2018-09-21' -> datetime.datetime(2018, 9, 20, 0, 0)

    Format of the string to convert into a datetime object
    ------------------------------------------------------
      %Y-%m-%d
      %Y-%m-%d %H:%M:%S
      %Y/%m/%d
      %Y/%m/%d %H:%M:%S
      %b %d %Y
      %b %d %Y %I:%M%p
      
  """    
#  try:
#    import datetime as dt
#  except:
#    raise ImportError('The datetime module is necessary and is not installed!')
      
  if weight_name is None:
    raise ValueError("The weight_name is None!")
  if G.number_of_edges() == 0:
    raise ValueError("The edge list of MAG is empty!")
   
  for edge in G.edges.data():
    From,To,dictt = edge
    if weight_name in dictt.keys():
      try: dictt[weight_name] = type_str_datetime(dictt[weight_name])
      except: raise ValueError("The value of weigth must be a string!")


def convert_type_edges_to_currency(G, weight_name, currency_type):
  pass
  """
    Parameters
    ----------
      weight_name: string, the value that will be convert. 
        The data type can be string or numeric.
      
      currency_type: is the currency type that will convert into the value.
        Must satisfies the ISO 4217 format, ex: EUR,USD,BRL,...
         
    Return
    ------
      A MAG with the update of the weigthed edges for money.Money

    Examples
    --------
      >>> G.convert_type_currency('value','USD')
      >>> G.convert_type_currency('value','BRL')

    Example of the conversion
    -------------------------
      98789709 -> BRL 98,789,709.00
    
  
  try:
    from money import Money
  except:
    raise ImportError('The money module is necessary and is not installed!')
      
  if weight_name is None:
    raise ValueError("The weight_name is None!")
  if G.number_of_edges() == 0:
    raise ValueError("The edge list of MAG is empty!")
  for edge in G.edges.data():
    From,To,dictt = edge      
    if weight_name in dictt.keys():
      try: dictt[weight_name] = Money(amount = dictt[weight_name], currency = currency_type)
      except: raise ValueError("Was not possible to convert the value to money. Check the parameters")
  """
   
def convert_type_list(G,weight_name):
  """
    Parameter
    ---------
      weight_name: string, this is weight name choose to be conveted into a list. 
        The value of weight must be a string with the values separated by comma.  

    Return
    ------
      A MAG with the update of the weigthed edges for list
    
    Example
    --------  
      >>> G.convert_type_list('List')

    Example of the conversion
    -------------------------
       '5;5;3;5323;t;4;user;something;10' ->  [5,3,5323,'t',4,'user','something',10]

  """
  if weight_name is None:
    raise ValueError("The weight_name is None!")
  if G.number_of_edges() == 0:
    raise ValueError("The edge list of MAG is empty!")
  for edge in G.edges.data():
    f,t,dictt = edge
    if weight_name in dictt.keys():
      try: new_list = dictt[weight_name].split(';')
      except: raise ValueError('The value is not a string or the string do not have separator')
      else:
        size = len(new_list)
        new_list = [type_str_number(new_list[i]) for i in range(size)]
        dictt[weight_name] = list(new_list)

  
def convert_type_set(G,weight_name):
  """
    Parameter
    ---------
      weight_name: string, this is weight name choose to be conveted into a set.
          The value of weight must be a string with the values separated by comma. The values can be a number or string

    Return
    ------
      A MAG with the update of the weigthed edges for set
    
    Example
    -------
      G.convert_type_set('set')

    Example of the conversion
    -------------------------
      '5;5;3;5323;t;4;user;something;10' -> {5,3,5323,'t',4,'user','something',10}

  """
  if weight_name is None:
    raise ValueError("The weight_name is None!")
  if G.number_of_edges() == 0:
    raise ValueError("The edge list of MAG is empty!")
  for edge in G.edges.data():
    f,t,dictt = edge
    if weight_name in dictt.keys():
      try: element = dictt[weight_name].split(';')
      except: raise ValueError('The value is not a string or the string do not have separator')
      else:
        size = len(element)
        element = [type_str_number(element[i]) for i in range(size)]
        dictt[weight_name] = set(element)
  
def convert_type_edges_to_tuple(G,weight_name):
  """
    Parameter:
    ----------
      weight_name: string, this is weight name choose to be conveted into a tuple.
          The value of weight must be a string with the values separated by comma.
    
    Return
    ------
      A MAG with the update of the weigthed edges for set
    
    Example
    -------
       >>> G.convert_type_tuple('Tuple')

    Example of the conversion
    -------------------------
      '5;5;3;5323;t;4;user;something;10' -> (5,3,5323,'t',4,'user','something',10)
  """
  if weight_name is None:
    raise ValueError("The weight_name is None!")
  if G.number_of_edges() == 0:
    raise ValueError("The edge list of MAG is empty!")
  for edge in G.edges.data():
    f,t,dictt = edge
    if weight_name in dictt.keys():
      try: element = dictt[weight_name].split(';')
      except: raise ValueError('The value is not a string or the string do not have separator')
      else:
        size = len(element)
        element = [type_str_number(element[i]) for i in range(size)]
        dictt[weight_name] = tuple(element)







#
#
#
#         SUPPORT FUNCTIONS
#
#
#
#

def subDeterminedEdgePartition(G,zeta):

    """Returns a partition of subdetermined edges, containing the wieghts
    of every edge in all equivalence class.
    The tuple zeta has the same number of
    elements as the number of aspects on the original MAG H, 
    and for each aspect carries a value 1 or 0, so that only 
    the aspects marked with 1 will remain on the resulted MAG. 
    The resulting sub-determined MAG has the edges of the original 
    MAG projected over the reduced aspect structure given by the 
    sub-determination.
    input: MAG H and sub-determination tuple zeta
    output: Sub-determined MAG Hz
    """
    Par = collections.defaultdict(list)
    asps = list(zeta) # + list(zeta)
    n = len(asps)
    E = G.edges
    for e in E:
      ez = [[],[]]
      for i in range(n):
         if asps[i] != 0:
            ez[0].append(e[0][i])
            ez[1].append(e[1][i])            
      if ez[0] != ez[1]: 
         tez = (tuple(ez[0]),tuple(ez[1]))
         w = G[e[0]][e[1]]["weight"]
         Par[tez].append(w)
    return Par


def initialize_list(G, size):
#  G._aspect = [set()] * size !!! This does not work !!!
    G._aspect = list()
    for i in range(0,size):
        G._aspect.append(set())


def aspect_node_integrity(G, node):
    if (len(node) == len(G._aspect)):
        return True
    else:
        return False

def update_aspect_list(G, Tuple):
    len_aspect = len(G._aspect)
    len_tuple = len(Tuple)
    for i in range(len_tuple):
        j = (i % len_aspect)
        asp = Tuple[i]
        G._aspect[j].add(asp)
    
def zeta_one(zeta):
    if (zeta):
        t = len(zeta)
        if zeta.count(1) == t:
          return True
        return False
  
def zeta_zero(zeta):
    if (zeta):
        t = len(zeta)
        if zeta.count(0) == t:
          return True
        return False
  
def split_tuple(name):
    if (name):
        result = name.split('(')[1].split(')')[0].split(',')
        return result
    
def split_weight(name):
    if (name):
        result = name.split('<')[1].split('>')[0].split(';')
        return result

def identify_edge(f):
    if (f.find('=') == -1):
        return False    #unweighted edge
    else:
        return True     #weighted edge


def aspect_integrity(File, type_tuple):
    """
      This function verifies the integrity of the edges or the nodes, in the file to be imported.
    """
    line = File.readline()
    a = split_tuple(line)    
    if type_tuple == 0:
        tuple_size = (len(a))
    elif type_tuple == 1:
        tuple_size = (len(a)/2)
    else:
        raise TypeError("Problems with the tuple format!")
    
    line = File.readline()
    while(line):
      if (line == "\n" or line == " \n" or line == "" or line == " "):
         line = File.readline()
         continue        
      a = split_tuple(line)
      len_a = len(a)
      if (type_tuple == 0):
         if len_a != tuple_size:
            raise ValueError("The number os aspects don't mach. At least one node is incorrect!")
      elif (type_tuple == 1):
         if (len_a/2) != tuple_size:
            raise ValueError("The number os aspects don't mach. At least one edge is incorrect!")
      line = File.readline()
      
    #Returning to the begin of the file
    File.seek(0,0)
    return True


def add_multi_edge_weight(G, f, From, To, key, Type):
    """
      Support function to create weighted edges of the MultiAspectMulti(Di)Graph.
    """
    p = split_weight(f)
    len_p = len(p)
    if Type == True:
      name = [i.split('=')[0] for i in p]
      weight = [i.split('=')[1] for i in p]
      for i in range(len(name)):
        try: name[i] = type_str_number(name[i])
        except: raise ValueError()
        if name[i] == 'datetime':
          weight[i] = type_str_datetime(weight[i])
#        elif name[i] == 'money':
#          weight[i] = type_str_money(weight[i])
        else:
          try: weight[i] = type_str_number(weight[i])
          except: raise ValueError()
        G.edges[From,To,key][name[i]] = weight[i]
    else:
      #weight default. In this case, there is a limitation of one weight type for the edges.
      for i in range (len_p):
        try: p[i] = type_str_number(p[i])
        except: raise ValueError()
        G.edges[tuple(From),tuple(To)]['weight'] = p[i]

def add_edge_weight(self, f, From, To, Type):
    """
      Support function to create weighted edges of the MultiAspect(Di)Graph.
    """
    #insertion of the edge weights
    p = split_weight(f)
    len_p = len(p)
    
    if Type == True:            
      name = [i.split('=')[0] for i in p]
      weight = [i.split('=')[1] for i in p]
      for i in range(len(name)):
        try: name[i] = type_str_number(name[i])
        except: raise ValueError()
        if name[i] == 'datetime':
          weight[i] = type_str_datetime(weight[i])
#        elif name[i] == 'money':
#          weight[i] = type_str_money(weight[i])
        else:
          try: weight[i] = type_str_number(weight[i])
          except: raise ValueError()
        self.edges[From,To][name[i]] = weight[i]
    else:
      #weight by default. In this case, there is a limitation of one weight type for each edge.
      for i in range (len_p):
        try: p[i] = type_str_number(p[i])
        except: raise ValueError()
        self.edges[tuple(From),tuple(To)]['weight'] = p[i]


def type_str(string):
    try:
      string = str(string)
      return string
    except:
      return string

def type_str_money(string):
    pass
"""    
    try:
      from money import Money
    except:
      raise ImportError('The money module is necessary and is not installed!')
    if type(string) is str:
      try:
        Format = string.split(" ")[0]
        value = string.split(" ")[1]
      except:
        raise ValueError('The format or value is not found! \nTo convert to Money the weight must follow the format: <money=formt value> \nex:<money=USD 100>')
      try:
        string = Money(amount = value ,currency = Format)
        return string
      except: raise ValueError("It wasn't possible to convert the string into a currency object (Money).")
    else:
      raise ValueError("To be able return a currency object, the parameter must be a string. If the conversion is not required, change the weight name.")
"""  
  
def type_str_number(string):
    try:
      string = int (string)
      return string
    except:
      try:
        string = float(string)
        return string
      except:
        try:
          string = complex(string)
          return string
        except:
          return string

        
        
def type_str_datetime(string):
 
    try:
      import datetime as dt
    except:
      raise ImportError('The datetime module is necessary and is not installed!')
  
    if type(string) is str:    
      if string.find(':') == -1:
        if string.find('-') != -1:
          Format = '%Y-%m-%d'
        elif string.find('/') != -1:
          Format = '%Y/%m/%d'
        else:
          Format = '%b %d %Y'
      else:
        if string.find('-') != -1:
          Format = '%Y-%m-%d %H:%M:%S'
        elif string.find('/') != -1:
          Format = '%Y/%m/%d %H:%M:%S'
        else:
          Format = '%b %d %Y %H:%M:%S'
        
      try:
        return dt.datetime.strptime(string,Format)
      except:
        raise ValueError("It was not possible to convert. To convert into datetime object, the parameter format must be one of these:\n %Y-%m-%d\n %H:%M:%S\n %Y-%m-%d\n %Y/%m/%d\n %H:%M:%S\n %Y/%m/%d\n %b %d %Y\n %I:%M%p\n %b %d %Y")
    else:
      raise ValueError("To be able return a datetime object, the parameter must be a string. If the conversion is not required, change the weight name.")


def checkAspects(edge, aspect_list):
    if len(edge[0]) == len(aspect_list):
        for i in range(len(aspect_list)):
            if edge[0][i] not in aspect_list[i]:
                return False
            if edge[1][i] not in aspect_list[i]:
                return False
        return True
    return False


def mag_identification(G):
    
    if G.is_mag() and not G.is_directed() and not G.is_multigraph():
        subMag = MultiAspectGraph()
    if G.is_mag() and G.is_directed() and not G.is_multigraph():
        subMag = MultiAspectDiGraph()
    if G.is_mag() and not G.is_directed() and G.is_multigraph():
        subMag = MultiAspectMultiGraph()
    if G.is_mag() and  G.is_directed() and G.is_multigraph():
        subMag = MultiAspectMultiDiGraph()

    return subMag


def subMag_aspect_induced(G, aspect_list):
    """
    This function creates an subMags inducing by aspects.

    Parameters
    ----------
        G: mag
            This Mag will be converted into a subMag.
        aspect_list: aspect list
            This reduced list derives from the original aspect list.
            The function using this aspect list will create the subMag.

    Returns
    -------
        SubMag - analogous to a subgraph

    Example
    -------
        >>> subMag_aspect_induced(G, aspects): 
    """
    if G.number_of_aspects() == len(aspect_list):
        subMag = mag_identification(G)
        subMagEdges = []
        for e in G.edges.data():
            if checkAspects(e, aspect_list):
                subMagEdges.append(e)
        subMag.add_edges_from(subMagEdges)
        return subMag
    return None



