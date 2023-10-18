'''
This file provides a function to plot a MultiAspect Graph ment to represent patient pathways.
'''

import numpy as np 
import math
import matplotlib.cm as colormaps
import pandas as pd
from matplotlib.pyplot import cm
from numpy import sign
from scipy.interpolate import UnivariateSpline

import plotly.graph_objects as go

# ---------------------------------------------------------------------------------------- 

############################################
# AUXILIARY FUNCTIONS

#--------------------------------------------
# Auxiliary function that iterates over edges in multigraphs and digraphs

def edge_iterator(G):  
    if G.is_multigraph():
        for edge in G.edges(keys=True):
            yield edge
    else:
        for edge in G.edges():
            yield edge

#--------------------------------------------

def get_cmap(n, name='gist_rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    # source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    return cm.get_cmap(name, n)

#--------------------------------------------

def color_according_to_cmap(cmap, index):
    color_tuple = tuple(cmap(index))[:3]
    c1 = round(color_tuple[0]*255,5)
    c2 = round(color_tuple[1]*255,5)
    c3 = round(color_tuple[2]*255,5)
    color_string = f'rgb({c1},{c2},{c3})'
    return color_string
    
#--------------------------------------------

def get_color(x, v_min, v_max, n_points = 100, color_scale = 'RdYlGn_r'):
    
    x = float(x)

    if (v_max - v_min + 1) > n_points:
        n_points = int(v_max - v_min + 1)

    i_min = 0
    i_max = int(n_points - 1)

    if x >= v_max:
        index_x = i_max
    elif x <= v_min:
        index_x = i_min
    else:
        index_x = int((i_max - i_min)*((x - v_min)/(v_max - v_min)) + i_min)

    color_tuple = tuple(colormaps.get_cmap(color_scale)(np.linspace(0, 1, n_points))[index_x])[:3]
    c1 = color_tuple[0]*255
    c2 = color_tuple[1]*255
    c3 = color_tuple[2]*255
    color_string = f'rgb({c1},{c2},{c3})'
    
    return color_string

# ---------------------------------------------------------------

# Custom function to create an edge between node x and node y, with a given text and width

def make_edge(x, y, text, width, color, edge_alpha, n_aux_points_edges = 50, shift_edge_arrow = 0, auxiliary_point = False, hover = False):
    return  go.Scatter(x              = x,
                       y              = y,
                       line           = dict(width = width,
                                             color = color),
                       hoverinfo      = 'text' if hover else 'skip',
                       text           = ([text]),
                       mode           = 'lines+markers',
                       marker         =  dict(size= (
                           [0 for i in range(n_aux_points_edges - 1 - shift_edge_arrow)]+
                           [min(5,10*width)]+
                           [0 for i in range(max(0,shift_edge_arrow-1))]
                           if auxiliary_point 
                           else min(5,10*width)),symbol= "arrow-bar-up", angleref="previous"),
                       line_shape     = 'spline',
                       line_smoothing = 1.3,
                       showlegend = False,
                       opacity = edge_alpha)

# ---------------------------------------------------------------

def str_edge_attributes(dict_edge_attributes, list_edge_attributes = None):
    if list_edge_attributes:
        temp = {k:str(v) for k,v in dict_edge_attributes.items() if k in list_edge_attributes}
        return(', '.join(['<b>'+str(k)+'</b>'+':<br>'+str(v) for k,v in temp.items()]))
    else:
        return('')


############################################
# PLOT MAG

def plot_mag(graph, idx_sequence, idx_label_aspect, 
    order_aspect_label = None,
    idx_aspect_yaxis = None, order_aspect_yaxis = None,
    node_alpha = 1.0, edge_alpha = 0.6,
    idx_aspect_color = None, attribute_color_discrete = None,
    attribute_color_continuous = None, node_color = 'violet',
    node_colormap = None,
    node_colorscale = 'YlGn', node_colorscale_limits = [None,None],
    fixed_node_labels = False,
    node_size = 10, node_size_factor = 0.5,
    node_size_power = 1, node_size_attribute = None,
    dict_node_labels = None, list_node_attributes = [],
    factor_bend_edges = 0.01, n_aux_points_edges = 50, colored_edges = False,
    limits_colored_edges = [None,None], list_edge_attributes = [],
    edge_width = 0.5, attribute_edge_width = None,
    edge_width_factor = 0.1, edge_width_power = 1,
    edge_color = 'gray', show_edge_labels = False, shift_edge_arrow = 0,
    figure_title = '',
    dict_colors_aspect_yaxis = None, df_info_aspect_yaxis = None, columns_df_info_aspect_yaxis = [],
    rot_xticks = 0, interval_yaxis = 2, starting_point_xaxis_bar = -0.5,
    figure_size = [900,400],
    figure_margins = [20,20,20,20],
    hide_some_edges = False, attribute_hide_edges = None, minimum_value_show_edges = None,
    text_note = None):
    

    '''
    Plots MultiAspect Graphs (MAG) representing patient pathways. The function supports displaying MAGs with 2, 3 or 4 aspects;
    one aspect is mandatorily the Sequence aspect.

    Parameters
    ----------
    graph : mag.MultiAspectMultiDiGraph or mag.MultiAspectDiGraph
      MAG with patient pathways

    idx_sequence : integer
      Index of the aspect Sequence in graph

    idx_label_aspect : integer
      Index of the aspect that will label the graph nodes

    order_aspect_label : list, optional (default = None)
      List with the elements of the aspect 'label' sorted as desired

    idx_aspect_yaxis : integer, optional (default = None)
      Index of the aspect whose elements will be grouped in stacked horizontal bars 

    order_aspect_yaxis : list, optional (default = None)
      List with the elements of the aspect 'label' sorted as desired

    node_alpha : float, optional (default = 1.0)
      Transparency of the nodes

    edge_alpha : float, optional (default = 0.6)
      Transparency of the edges

    idx_aspect_color : integer, optional (default = None)
      Index of the aspect whose elements will be represented by the node colors

    attribute_color_continuous : string, optional (default = None)
      Name of the attribute with continuous values that will be represented by the node colors. 
      Mind that, if idx_aspect_color is not None, attribute_color_continuous will not be used.

    attribute_color_discrete : string, optional (default = None)
      Name of the attribute with discrete values that will be represented by different node colors. 
      Mind that, if idx_aspect_color is not None or attribute_color_continuous is not None, attribute_color_discrete will not be used.

    node_color : string, optional (default = 'violet')
      Color of the nodes if idx_aspect_color, attribute_color_continuous and attribute_color_discrete are all null

    node_colormap : string, optional (default = None)
      Name of the colormap to color nodes according to an aspect or discrete-valued attribute. If it is not specified, 
      'tab10' is used if no more than 8 different colors are necessary or 'gist_rainbow' otherwise

    node_colorscale : string, optional (default = 'YlGn')
      Name of the color scale to be used to color nodes according to a continuous-valued attribute
    
    node_colorscale_limits : list of lenght 2, optional (default = [None,None])
      If the nodes are colored according to a continuous-valued attribute, the first element of node_colorscale_limits specifies
      the lowest value for the colorscale and  the second element specifies the greatest value for the colorscale

    fixed_node_labels : bool, optional (default = False)
      If True, the nodes of the returned figure will be displayed with their corresponding labels

    node_size_attribute : string, optional (default = None)
      Node attribute whose values are numeric used to determine the node size. If None, a fixed node size is used

    node_size_factor : float, optional (default = 0.5)
      If node_size_attribute is not None, the node size is given by 
      node_size_factor * value of the node_size_attribute ** node_size_power

    node_size_power : float, optional (default = 1.0)  
    If node_size_attribute is not None, the node size is given by 
      node_size_factor * value of the node_size_attribute ** node_size_power

    node_size : numeric, optional (default = 10) 
      Node size to be used if node_size_attribute is None

    dict_node_labels : dictionary, optional (default = None)
      Dictionary that maps each node to a label

    list_node_attributes : list, optional (default = [])
      List with the name of node attributes to be displayed when hovering over the nodes

    factor_bend_edges : float, optional (default = 0.01)
      Specifies how curved the edges should be. If equal to 0, the edges are straight lines. 

    n_aux_points_edges : integer, optional (default = 50)
      Number of auxiliary points to create curved edges and/or pulling the edge arrow

    colored_edges : bool, optional (default = False)
      If True, edges are colored according to their corresponding time interval

    limits_colored_edges : list of length 2, optional (default = [None,None])
      If edges are colored according to the time interval, the first element of limits_colored_edges indicates 
      the lowest value of the colorscale and the second element indicates the highest value of the colorscale.
      When the bounds are not specified, the minimum/maximum value found in the edges are used.

    list_edge_attributes : list, optional (default = [])
      List with the name of edge attributes to be displayed when hovering over them

    attribute_edge_width : string, optional (default = None)
      Edge attribute, whose values are numeric, used to determine the edge width. If None, a fixed edge width is used

    edge_width_factor : float, optional (default = 0.1)  
    If attribute_edge_width is not None, the edge width size is given by 
      edge_width_factor * value of the attribute_edge_width ** edge_width_power

    edge_width_power : float, optional (default = 1.0)  
    If attribute_edge_width is not None, the edge width size is given by 
      edge_width_factor * value of the attribute_edge_width ** edge_width_power

    edge_width : float, optional (default = 0.5)
      Edge width to be used if attribute_edge_width is None
    
    edge_color : string, optional (default = 'gray')
      Edge color if attribute_edge_width is None

    show_edge_labels : bool, optional (default = False)
      If True, information about the edges is displayed when hovering over them

    shift_edge_arrow : integer, optional (default = 0)
      Depending on the node size, the edge arrows may be covered by the nodes. Increasing this parameter pulls the arrow back.
      The arrow displacement depends on n_aux_points_edges. If n_aux_points_edges = 100 and shift_edge_arrow = 50, the arrow
      will appear more or less in the middle of the edge

    figure_title : string, optional (default = '')
      Title of the returned figure

    dict_colors_aspect_yaxis : dictionary, optional (default = None)
      Specifies the colors of the stacked horizontal bars related to the elements of the y-axis aspect. 
      If None, the bars will be orange and white, intercalated

    df_info_aspect_yaxis : pandas.core.frame.DataFrame
      A Pandas dataframe containing extra information about each element of the y-axis aspect. This information 
      will be shown when hovering over the labels of the y-axis aspect.
      The dataframe must be indexed with the elements of the y-axis aspect.
      
    columns_df_info_aspect_yaxis : list, optional (default = [])
      The columns of df_info_aspect_yaxis whose information will be shown when hovering over the labels of the y-axis aspect

    rot_xticks : integer, optional (default = 0)
      Rotates the labels of the aspect Sequence

    interval_yaxis : integer, optional (default = 2)
      Vertical spacient between nodes when changing the elements in the y-axis aspect

    starting_point_xaxis_bar : float, optional (default = -0.5)
      Position in the x-axis where the stacked horizontal bars begin

    figure_size : list of length 2, optional (default = [900,400])
      Size of the returned figure ([width,height])
    
    figure_margins : list of length 4, optional (default = [20,20,20,20])
      Margin around the plot area in the returned figure ([left,right,top,bottom])

    hide_some_edges : bool, optional (default = False)
      If True, edges whose attribute_hide_edges is smaller then minimum_value_show_edges are not displayed

    attribute_hide_edges : string, optional (default = '')
      Name of the attribute based on which edges may be hidden from the visualization (if hide_some_edges = True)

    minimum_value_show_edges : float, optional (default = None)
      Minimum value of the attribute attribute_hide_edges for an edge to appear in the figure (if hide_some_edges = True)

    text_note : string, optional (default = '')
      Text to be displayed in the upper part of the returned figure   

    Returns
    -------
    figure : plotly.graph_objects.Figure
    
    '''

    #################################################################

    annotations = []
    if text_note is not None:
        annotations.append(text_note)

    ##################################################################
    # Elements

    # ----------------------------------------------------------------------
    # Aspect related to the node labels

    elements_label_aspect = list(graph.get_aspects_list()[idx_label_aspect]) # list of elements of the 'label aspect'
    if order_aspect_label: # if an order of the elements was specified
        if (all([a in order_aspect_label for a in elements_label_aspect])): # check if all elements are in the given list
            elements_label_aspect = [a for a in order_aspect_label if a in elements_label_aspect]
    else: # alphabetical order 
        elements_label_aspect = sorted(elements_label_aspect)
 
    # -------------------------------------------------------------------------
    # y-axis aspect
    
    elements_aspect_yaxis = list(graph.get_aspects_list()[idx_aspect_yaxis]) if (idx_aspect_yaxis is not None) else []
    if order_aspect_yaxis:
        if (all([y in order_aspect_yaxis for y in elements_aspect_yaxis])): 
            elements_aspect_yaxis = [y for y in order_aspect_yaxis if y in elements_aspect_yaxis]
    else: # alphabetical order 
        elements_aspect_yaxis = sorted(elements_aspect_yaxis) 
    elements_aspect_yaxis = list(reversed(elements_aspect_yaxis)) # inverse order to plot from bottom to top
    
    # ----------------------------------------------------------------------------
    # aspect Sequence in the x-axis
    
    numbers_sequence_aspect = sorted([int(s[1:]) for s in list(graph.get_aspects_list()[idx_sequence])]) # number of the S elements
    largest_element_sequence_aspect = max(numbers_sequence_aspect)
    sequence = ['S' + str(s) for s in range(largest_element_sequence_aspect+1)] # add an 'S' before each number
    n_elements_aspect_sequence = len(sequence) # number of elements in aspect Sequence
    del numbers_sequence_aspect

    # ----------------------------------------------------------------------------
    
    # interval when changing elements in aspect y
    interval_y = interval_yaxis
    
    # --------------------------------------------------------------------------------
    # number of nodes in each element of the aspect in the y-axis
    
    if idx_aspect_yaxis is not None:
        
        aspect_yaxis_height = {}
        height_beginning_element_y = {}
        nodes_per_element_aspect_yaxis = {}
        temp_height = 0

        for y in elements_aspect_yaxis:
            temp_list =[]
            height_beginning_element_y[y] = temp_height

            for node in graph.nodes():
                if node[idx_aspect_yaxis] == y: # if the node has the current element of the yaxis aspect

                    temp_label = (
                        node[idx_label_aspect]+node[idx_aspect_color] # concatenate info aspect label and aspect color
                        if ((idx_aspect_color is not None) & (idx_aspect_color!=idx_label_aspect))
                        else node[idx_label_aspect] # or only the label aspect info
                    ) 
                    temp_list.append(temp_label) # add to the list

            nodes_per_element_aspect_yaxis[y] = sorted(list(set(temp_list))) # remove duplicates  

            aspect_yaxis_height[y] = interval_y + len(nodes_per_element_aspect_yaxis[y]) + interval_y 
            temp_height = temp_height + aspect_yaxis_height[y]
            
    else: # if no aspect was selected to the y-axis
        
        nodes_per_element_aspect_yaxis = []
        for node in graph.nodes():
            temp_label = (
                        node[idx_label_aspect]+node[idx_aspect_color] # concatenate info aspect label and aspect color
                        if ((idx_aspect_color is not None) & (idx_aspect_color!=idx_label_aspect)) 
                        else node[idx_label_aspect] # or only the label aspect info
                    ) 
            nodes_per_element_aspect_yaxis.append(temp_label) 
            nodes_per_element_aspect_yaxis = sorted(list(set(nodes_per_element_aspect_yaxis))) # remove duplicates
        aspect_yaxis_height = len(nodes_per_element_aspect_yaxis)
            
    # ###############################################################
    # Node position
    
    node_position = {}
    
    for node in graph.nodes():
    
        # index of the element of the y-axis aspect
        i_y = elements_aspect_yaxis.index(node[idx_aspect_yaxis]) if idx_aspect_yaxis else 0
        y_name = node[idx_aspect_yaxis] if (idx_aspect_yaxis is not None) else ''
        
        # node index in the sequence aspect
        i_seq = sequence.index(node[idx_sequence])

        # index of the node in the y-axis element
        label = (node[idx_label_aspect]+node[idx_aspect_color] # concatenate label and color aspects, if they both exist
                  if ((idx_aspect_color is not None) & (idx_aspect_color!=idx_label_aspect)) 
                  else node[idx_label_aspect])
        i_label = (nodes_per_element_aspect_yaxis[y_name].index(label)
                    if idx_aspect_yaxis is not None
                    else nodes_per_element_aspect_yaxis.index(label))
        
        # x position
        pos_x = i_seq 
        
        # y position
        pos_y = i_label + (height_beginning_element_y[y_name] if (idx_aspect_yaxis is not None) else interval_y) + interval_y 
        
        # par ordenado
        node_position[node] = np.array([pos_x,pos_y])
    
    # ###############################################################
    # Node color
       
    dict_node_color = None
    dict_node_shape = None
        
    if idx_aspect_color is not None: # if nodes are colored according to an aspect
        dict_node_color = {} 
        dict_node_shape = {}
        shapes = ['circle','square','diamond','cross','x','star-square','hexagram','star-diamond',
                 'circle-dot','square-dot','diamond-dot','cross-dot','x-dot','star-square-dot','hexagram-dot']
        elements_color_aspect = [node[idx_aspect_color] for node, att in graph.nodes(data=True)]
        elements_color_aspect = list(set(elements_color_aspect))
        if len(elements_color_aspect)<= 8 :
            cmap = get_cmap(len(elements_color_aspect),'tab10' if node_colormap is None else node_colormap) 
        else:
            cmap = get_cmap(len(elements_color_aspect),'gist_rainbow' if node_colormap is None else node_colormap) 
        for node, att in graph.nodes(data=True):
            dict_node_color[node] = color_according_to_cmap(cmap, index = elements_color_aspect.index(node[idx_aspect_color]))
            dict_node_shape[node] = shapes[elements_color_aspect.index(node[idx_aspect_color])%len(shapes)]

    elif attribute_color_discrete: # if the nodes are colored according to an attribute with discrete values
        dict_node_color = {} 
        elements_color_aspect = [att[attribute_color_discrete] for node, att in graph.nodes(data=True)]
        elements_color_aspect = list(set(elements_color_aspect))
        if len(elements_color_aspect)<= 8 :
            cmap = get_cmap(len(elements_color_aspect),'tab10' if node_colormap is None else node_colormap) 
        else:
            cmap = get_cmap(len(elements_color_aspect),'gist_rainbow' if node_colormap is None else node_colormap) 
        for node, att in graph.nodes(data=True):
            dict_node_color[node] = color_according_to_cmap(cmap, index = elements_color_aspect.index(att[attribute_color_discrete]))
            
   
    # ###############################################################
    # Node Trace (plotly)
    
    list_node_traces = []
    
    # if nodes have a fixed color or are colored by a dicrete-valued attribute
    if (idx_aspect_color is None) & (attribute_color_continuous is None): 

        # --------------------------
        # create empty node trace
        node_trace = go.Scatter(x         = [],
                                y         = [],
                                text      = [],
                                hovertext = [],
                                textposition = "middle center",
                                textfont_size = 10,
                                mode      = 'markers+text' if fixed_node_labels else 'markers',
                                marker    = dict(color = [],
                                                 size  = [],
                                                 line  = None,
                                                 opacity = node_alpha),
                               hoverinfo = 'text',
                               showlegend = False)

        # --------------------------
        # fill with the nodes
        for node, att in graph.nodes(data = True):
            x, y = node_position[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y]) 
            node_trace['marker']['size'] += (tuple(
                [node_size_factor*att[node_size_attribute]**node_size_power]) 
                                             if node_size_attribute
                                             else tuple([node_size]))
            node_trace['marker']['color'] += (tuple([dict_node_color[node]])
                                             if dict_node_color
                                             else tuple([node_color]))
            temp_text = [('<b>' + dict_node_labels[node] + '</b>'
                          if dict_node_labels
                          else ('<b>' + list(node)[idx_label_aspect] + '</b>' + 
                                '<br>' +'<br>'.join([str(a) for a in list(node)]))
                         ) ] + ['<br>' + x +' : '+ str(att[x])+'</br>'
                                for x in list_node_attributes]
            temp_text = ''.join(temp_text)
            node_trace['hovertext'] += tuple([temp_text])
            label = (dict_node_labels[node]
                          if dict_node_labels
                          else list(node)[idx_label_aspect]
                         )
            node_trace['text'] += tuple([label])
        
        list_node_traces.append(node_trace)
    
    # if nodes are colored according to a continuos-valued attribute
    elif attribute_color_continuous is not None:
        
        # --------------------------
        # create empty node trace with colorbar
        node_trace = go.Scatter(x         = [],
                                y         = [],
                                text      = [],
                                hovertext = [],
                                textposition = "middle center",
                                textfont_size = 10,
                                mode      = 'markers+text' if fixed_node_labels else 'markers',
                                marker    = dict(color = [],
                                                 size  = [],
                                                 line  = None,
                                                 opacity = node_alpha,
                                                 colorbar = dict(
                                                     title = dict(text = attribute_color_continuous,
                                                                 side = 'right'),
                                                     orientation = 'v',
                                                     x = 1.01 if not colored_edges else 1.03,
                                                     thickness = 15,
                                                     len = 0.7,
                                                     lenmode = 'fraction'
                                                 ),
                                                 colorscale = node_colorscale,
                                                 cmin = node_colorscale_limits[0],
                                                 cmax = node_colorscale_limits[1]
                                                ),
                               hoverinfo = 'text',
                               showlegend = False)

        # --------------------------
        # fill with nodes
        for node, att in graph.nodes(data = True):
            x, y = node_position[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y]) 
            node_trace['marker']['size'] += (tuple(
                [node_size_factor*att[node_size_attribute]**node_size_power]) 
                                             if node_size_attribute
                                             else tuple([node_size]))
            node_trace['marker']['color'] += (tuple([att[attribute_color_continuous]]))
            temp_text = [('<b>' + dict_node_labels[node] + '</b>'
                          if dict_node_labels
                          else ('<b>' + list(node)[idx_label_aspect] + '</b>' + 
                                '<br>' +'<br>'.join([str(a) for a in list(node)]))
                         ) ] + ['<br>' + x +' : '+ str(att[x])+'</br>'
                                for x in list_node_attributes]
            temp_text = ''.join(temp_text)
            node_trace['hovertext'] += tuple([temp_text])
            label = (dict_node_labels[node]
                          if dict_node_labels
                          else list(node)[idx_label_aspect]
                         )
            node_trace['text'] += tuple([label])
        
        list_node_traces.append(node_trace)
        
    # if nodes are colored according to an aspect, we will create a node trace for each color
    else:
        
        for element_aspect_color in elements_color_aspect:
            
            # --------------------------
            # create empty node trace for element_aspect_color
            node_trace = go.Scatter(x         = [],
                                    y         = [],
                                    text      = [],
                                    hovertext = [],
                                    textposition = "middle center",
                                    textfont_size = 10,
                                    mode      = 'markers+text' if fixed_node_labels else 'markers',
                                    marker    = dict(color = [],
                                                     size  = [],
                                                     symbol = [],
                                                     line  = None,
                                                     opacity = node_alpha),
                                   hoverinfo = 'text')

            # --------------------------
            # fill with the nodes
            for node, att in graph.nodes(data = True):
                
                if node[idx_aspect_color] == element_aspect_color:
                    x, y = node_position[node]
                    node_trace['x'] += tuple([x])
                    node_trace['y'] += tuple([y]) 
                    node_trace['marker']['size'] += (tuple(
                        [node_size_factor*att[node_size_attribute]**node_size_power]) 
                                                     if node_size_attribute
                                                     else tuple([node_size]))
                    node_trace['marker']['color'] += (tuple([dict_node_color[node]])
                                                     if dict_node_color
                                                     else tuple([node_color]))
                    node_trace['marker']['symbol'] += (tuple([dict_node_shape[node]])
                                                     if dict_node_shape
                                                     else tuple(['circle']))
                    temp_text = [('<b>' + list(node)[idx_label_aspect] + '</b>' +
                                  '<br>' +'<br>'.join([str(a) for a in list(node)]))
                                 ] + ['<br>' + x +' : '+ str(att[x])+'</br>'
                                        for x in list_node_attributes]
                    temp_text = ''.join(temp_text)
                    node_trace['hovertext'] += tuple([temp_text])
                    label = (dict_node_labels[node]
                          if dict_node_labels
                          else list(node)[idx_label_aspect]
                         )
                    node_trace['text'] += tuple([label])

            node_trace['name']=element_aspect_color
            list_node_traces.append(node_trace)
    
    # ###############################################################
    # Edge trace
        
    # --------------------------------------------------------------- 
    # bend edge according to the number of edges between a pair of nodes

    dict_edge_bend = {}
    
    if(graph.is_multigraph()):
        for origin, target, key, att in graph.edges(data=True,keys=True):
            angle = key*0.005 + factor_bend_edges # each edge has one key; 1 edge=[0]; 2=[0,1], etc
            dict_edge_bend[(origin,target,key)] = angle
    
    # ----------------------------------------------------------------
    # edge color according to the time interval
    
    dict_edge_color = None
    
    if colored_edges:
        
        dict_edge_color = {}
        
        # get all time intervals to define the minimum and maximum values (if the desired ones were not specified)
        if (limits_colored_edges[0] is None) | (limits_colored_edges[1] is None):
            edge_time_intervals = []
            for origin, target, att in graph.edges(data = True):
                edge_time_intervals.append(att['interval'])

        # define the minimum value (for the color scale)
        if limits_colored_edges[0] is not None:
            vmin = limits_colored_edges[0] 
        else:
            vmin = min(edge_time_intervals)

        # define the maximum value (for the color scale)
        if limits_colored_edges[1] is not None:
            vmax = limits_colored_edges[1]
        else :
            vmax = max(edge_time_intervals)

        # define the color for each edge
        for edge in edge_iterator(graph):
            dict_edge_color[edge] = get_color(graph.edges[edge]['interval'],
                                                 vmin, vmax, n_points = 1000, color_scale = 'RdYlGn_r')
            
        # legend
        edge_color_trace = go.Scatter(
            x=[0,1],
            y=[0,1],
            marker=dict(
                size=0,
                cmax=vmax,
                cmin=vmin,
                color=[vmin,vmax],
                colorbar=dict(
                    title= dict(text = "Intervalos das Arestas",
                               side = 'right'),
                    orientation = 'v',
                    #y = -0.2,
                    x = 1.01,
                    #lenmode='fraction', 
                    #len=0.75,
                    thickness = 15
                ),
                opacity = 0.0,
                colorscale="RdYlGn_r"
            ),
            showlegend = False,
            mode="markers")
            
    # --------------------------------------
    # edge width
    
    dict_edge_width = None
    
    if attribute_edge_width:
        
        dict_edge_width = {}

        for edge in edge_iterator(graph):
            dict_edge_width[edge] = (
                edge_width_factor*
                graph.edges[edge][attribute_edge_width]**
                edge_width_power
            )      
            
    # --------------------------------------
    # list of edge traces (each edge is a trace that is added to a list)

    edge_trace = []
    edge_labels = []
    
    for edge in edge_iterator(graph):
        
        if (hide_some_edges):
            if (graph.edges[edge][attribute_hide_edges]<minimum_value_show_edges):
                continue

        origin = edge[0]
        target = edge[1]
        att = graph.edges[edge]
        
        x0, y0 = node_position[origin]
        x1, y1 = node_position[target]

        text   = (origin[idx_label_aspect] +
                   ' > ' + 
                   target[idx_label_aspect] + 
                   ('<br>' if len(list_edge_attributes)>0 else '') + 
                   str_edge_attributes(att, list_edge_attributes)) 

        if ((factor_bend_edges == 0.0)&(shift_edge_arrow == 0)&(not(graph.is_multigraph()))) :
            trace  = make_edge(x = [x0, x1, None], 
                               y = [y0, y1, None], 
                               text = text,
                               width = (dict_edge_width[edge] 
                                        if dict_edge_width 
                                        else edge_width),
                               color = (dict_edge_color[edge]
                                        if dict_edge_color
                                        else edge_color),
                               edge_alpha = edge_alpha,
                               hover = False)
        else:
            factor_bend_edges = dict_edge_bend[edge] if dict_edge_bend else factor_bend_edges
            
            factor_bend_edges = factor_bend_edges * math.sqrt((x0-x1)**2+(y0-y1)**2)
            x_aux = 0.5*(x0+x1) 
            y_aux = 0.5*(y0+y1) + factor_bend_edges *  sign(x0-x1)
            
            x_spline = [x0,x_aux,x1] if x1>x0 else [x1,x_aux,x0] 
            y_spline = [y0,y_aux,y1] if x1>x0 else [y1,y_aux,y0] 
            spl = UnivariateSpline(x_spline, y_spline, k=2, s = 2)
            xs = np.linspace(x0, x1, n_aux_points_edges)
            
            trace  = make_edge(x = xs, 
                               y = spl(xs), 
                               text = text,
                               width = (dict_edge_width[edge] 
                                        if dict_edge_width 
                                        else edge_width),
                               color = (dict_edge_color[edge]
                                        if dict_edge_color
                                        else edge_color),
                               edge_alpha = edge_alpha,
                               auxiliary_point = True,
                               n_aux_points_edges = n_aux_points_edges,
                               shift_edge_arrow = shift_edge_arrow,
                               hover = False)

        edge_trace.append(trace)
    

        # Edge labels --> to be able to hover over (almost) any part of the edge, we must use invisible auxiliary points
        if show_edge_labels:
            
            x_origin = x0
            x_target = x1
            y_origin = y0
            y_target = y1

            label = text

            if ((factor_bend_edges == 0.0)&(shift_edge_arrow == 0)&(not(graph.is_multigraph()))):
                trace_edge_labels = go.Scatter(
                    x = list(np.linspace(x_origin - sign(x_origin)*0.01*abs(x_target-x_origin),
                                         x_target - sign(x_target)*0.01*abs(x_target-x_origin),20)),
                    y = list(np.linspace(y_origin - sign(y_origin)*0.01*abs(y_target-y_origin),
                                         y_target - sign(y_target)*0.01*abs(y_target-y_origin),20)),
                                          opacity=0.0,
                                          mode = 'markers',
                                          marker = dict(color = 'white',
                                                        size = 1),
                                          hoverinfo = 'text',
                                          text = label,
                                          showlegend = False)
            else:
                trace_edge_labels = go.Scatter(x = xs, 
                                          y = spl(xs),
                                          opacity=0.0,
                                          mode = 'markers',
                                          marker = dict(color = 'white',
                                                        size = 1),
                                          hoverinfo = 'text',
                                          text = label,
                                          showlegend = False)

            edge_labels.append(trace_edge_labels)

    
    ###########################################################
    # Plot
    
    # Customize layout
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)', # transparent background
        plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background
        xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines
        yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines
    )

    # Create figure
    fig = go.Figure(layout = layout)
    
    
    # --------------------------------------------------------
    #  Sequence Aspect
        
    # x position of the first element
    pos_x = 0
    
    # identify largest y position
    if isinstance(aspect_yaxis_height, dict):
        max_pos_y = 0
        for y in aspect_yaxis_height.keys():
            max_pos_y += aspect_yaxis_height[y]
    else:
        max_pos_y = aspect_yaxis_height + 2*interval_yaxis
    
    # add text and dashed lines for the Sequence Aspect
    for s in range(n_elements_aspect_sequence):
        
        # text
        temp_annotation = {
            'xref':'x', 'yref':'y',
            'x': pos_x, 'y': -1,
            'showarrow': False,
            'text': sequence[s],
            'font' : {'size': 10, 'color': 'green'},
            'textangle' : rot_xticks}
        annotations.append(temp_annotation)
        
        fig.add_trace(go.Scatter(x=[pos_x for i in range(max_pos_y+1)], 
                                 y=[i for i in range(max_pos_y+1)],
                                 mode='lines',
                                 line = {'color':'green', 'width':1, 'dash':'dash'}, 
                                 opacity = 0.2,
                                 hoverinfo='skip',
                                 showlegend = False
                                ))
        
        # update 'x'
        pos_x += 1

    # --------------------------------------------------------
    # mark the elements of the y-axis elements
    
    if idx_aspect_yaxis is not None:
        
        min_pos_x = starting_point_xaxis_bar
        max_pos_x = len(sequence)-0.5
        
        last_color_yaxis_aspect = 'orange'
        current_color_yaxis_aspect = 'white'
        
        for y in elements_aspect_yaxis:
            
            temp = current_color_yaxis_aspect
            current_color_yaxis_aspect = last_color_yaxis_aspect
            last_color_yaxis_aspect = temp
            
            temp_min_y = height_beginning_element_y[y]
            temp_max_y = height_beginning_element_y[y] + aspect_yaxis_height[y]
            
            fig.add_trace(go.Scatter(x=[min_pos_x, max_pos_x, max_pos_x, min_pos_x], 
                                     y=[temp_min_y, temp_min_y, temp_max_y, temp_max_y], 
                                     fill="toself", 
                                     line={'width':0}, 
                                     mode = 'lines', 
                                     fillcolor = (dict_colors_aspect_yaxis[y] 
                                                  if dict_colors_aspect_yaxis 
                                                  else (current_color_yaxis_aspect if y != 'inicio_fim' else 'gray')),
                                     opacity = 0.1,
                                     hoverinfo='skip',
                                     showlegend = False))

            text = y 
            if isinstance(df_info_aspect_yaxis, pd.DataFrame):
                if y in df_info_aspect_yaxis.index:
                    for column in columns_df_info_aspect_yaxis:
                        text = text + '<br>' + df_info_aspect_yaxis.loc[y][column] 
                elif y.replace('–','-') in df_info_aspect_yaxis.index:
                    y = y.replace('–','-')
                    for column in columns_df_info_aspect_yaxis:
                        text = text + '<br>' + df_info_aspect_yaxis.loc[y][column]
  
            # y-axis aspect element label 
            temp_annotation_asp_y = {
                'x': -1, 'y': temp_min_y + 0.5*(temp_max_y - temp_min_y),
                'text': y, 'showarrow' : False,
                'xanchor':'right',
                'hovertext' : text}#,
                #'font' : {'size': 14, 'color': 'black'}}
            annotations.append(temp_annotation_asp_y)
                

    # --------------------------------------------------------                  
    # Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)

    # Add edge label trace
    if show_edge_labels:
        for trace in edge_labels:
            fig.add_trace(trace)
            
    # Add bar with the edge color
    if colored_edges:
        fig.add_trace(edge_color_trace)

    # Add node trace
    for trace in list_node_traces:
        fig.add_trace(trace)

    # Legenda
    if ((idx_aspect_color is not None)|(attribute_color_discrete is not None)):
        fig.update_layout(
            legend=dict(
                xanchor="left",
                x= (1.15 if colored_edges else 1.01),
                yanchor = 'middle',
                y = 0.5,
#                 orientation="h",
#                 entrywidth=70,
#                 yanchor="top",
#                 y=0.02
            ))

    # Remove tick labels
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)
    
    # add title
    fig.update_layout(
        title=dict(text=figure_title, font=dict(size=20), automargin=True, yref='paper'))
    
    # annotations
    fig.update_layout({'annotations': annotations})
    
    # figure_margins
    fig.update_layout(
        margin=dict(l=figure_margins[0], r=figure_margins[1], t=figure_margins[2], b=figure_margins[3]),
    )
    #figure size
    fig.update_layout(
        autosize = False,
        width = figure_size[0],
        height = figure_size[1])
    
    # Return figure
    return fig