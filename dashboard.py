#!/usr/bin/env python # 

# -------------------------------------------------
# Import packages and functions

import numpy as np
import pandas as pd
import pickle

import dash
import io
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from base64 import b64encode
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import MAG as mag
from create_and_modify_MAG_patient_pathways import contract_multiMAG_nodes
from create_and_modify_MAG_patient_pathways import convert_mag_into_graph
from create_and_modify_MAG_patient_pathways import convert_multiMAG_into_diMAG
from create_and_modify_MAG_patient_pathways import filter_MAG
from create_and_modify_MAG_patient_pathways import mag_centralities 
from create_and_modify_MAG_patient_pathways import subdetermination
from plot_mag import plot_mag

# ------------------------------------------------------
# Import data

# read the dictionary with the edges of each patient
dict_edges_of_patients = pickle.load(open(r"results/dict_edges_of_patients.p", "rb"))

# read the dictionaries with the subdetermied nodes centralities
dict_closeness_intervention_NORM = pickle.load(open("results/dict_closeness_intervention_NORM.p", "rb"))
dict_betweenness_occupation_NORM = pickle.load(open("results/dict_betweenness_occupation_NORM.p", "rb"))
dict_pagerank_unit_NORM = pickle.load(open("results/dict_pagerank_unit_NORM.p", "rb"))

# original data
df = pd.read_csv(r'results/sample_synthetic_data.csv')

# type of the healthcare units
df_unity_type = pd.DataFrame(df.value_counts(['unit','unit_type'])).reset_index().set_index('unit')

type_unity_order = ['Hospital','Secondary Care','Primary Care','end','start']

# -------------------------------------------------------
# Interface

# Create the Dash app
app = dash.Dash()

# Layout
app.layout = html.Div(style = {'marginLeft': 50, 'marginRight': 50},
    children = [
        
        # Title (top row)
        html.H1(style={'display':'block','textAlign': 'center'}, 
        children=['Multi-perspective assessment of pregnancy patient pathways']
        ), # end Titla
        
        # Row with figure information
        html.Hr(),
        html.H3('Figure options'),
        html.Div(style={'display':'flex','width':'100%','height':'160px','verticalAlign' : 'top'},
            children=[

                # column - node color
                html.Div(style={'display':'inline-block', 'width':'30%'},
                children=[
                    html.H4('What should the colour of the nodes indicate?'),
                    dcc.Dropdown(id = 'node_coloring',
                             options= [
                             {'label' : 'Occupation', 'value' : 'occupation'},
                             {'label' : 'Intervention', 'value' : 'intervention'},
                             {'label' : 'Node Relevance', 'value' : 'centrality'},
                             {'label' : 'Nothing', 'value' : 'none'}],
                             value = 'occupation',
                             clearable = False
                             )
                ]), # end column - node color

                # column - figure size
                html.Div(style={'display':'inline-block', 'width':'15%','marginLeft':'50px'},
                children=[
                    html.H4('Figure size:'),
                    dcc.Input(id = 'figure_width', type = 'number', min = 0, value = 1200, required = False, debounce=False,
                    style={"width" : '70px',"height" : '28px'}),
                    dcc.Input(id = 'figure_height', type = 'number', min = 0, value = 700, required = False, debounce=False,
                    style={"margin-left": "15px", "width" : '70px',"height" : '28px'})
                ]), # end column - figure size
                
                # column - edge frequency
                html.Div(style={'display':'inline-block', 'width':'20%','marginLeft':'50px'},
                children=[
                    html.H4('Minimun edge frequency:'),
                    dcc.Input(id = 'edge_min_freq', type = 'number', min = 0, value = 0, required = True, debounce=False,
                    style={"width" : '50px',"height" : '28px'})
                ]), # end column - edge frequency

                # column - other options
                html.Div(style={'display':'inline-block', 'width':'30%','marginLeft':'00px'},
                children=[
                    html.Div(style = {'display':'inline-block'}, 
                        children = [
                        html.H4('Other options:'),
                        dcc.Checklist(
                            id = 'show_node_label',
                            options = ['Show node labels']),
                        dcc.Checklist(
                            id = 'hide_start_end',
                            options = ['Remove Start/End nodes']),
                        dcc.Checklist(
                            id = 'group_units_per_type',
                            options = ['Group healthcare units according to their category'],
                            value = ['Group healthcare units according to their category']),
                        dcc.Checklist(
                            id = 'show_edge_label',
                            options = ['Show edge labels']
                            ),
                        dcc.Checklist(
                            id = 'color_edges',
                            options = ['Colored edges']
                            )
                        ]
                    )
                ]) # end column - other options
            ]), # end Row with figure information

        # Row with sliders
        html.Hr(),
        html.H3('Filter options'),
        html.Div(style={'display':'flex','width':'100%','height':'100px'},
        children=[

            # column - filter slider
            html.Div(style={'display':'inline-block','width':'42%'},
                children=[
                    html.H4('Filter the nodes of the graph according to their relevance:'),
                    dcc.RangeSlider(min=0, max=1, value=[0, 1], marks = {0:'0',0.25:' ',0.5:' ',0.75:' ',1:'1'}, 
                        id='range_centrality')
                ]), # end column - filter slider

            # column - typical event/complications weight
            html.Div(style={'display':'inline-block','marginLeft':'150px','width':'42%'},
                children=[
                    html.H4('What characterises an event as relevant?'),
                    dcc.Slider(min=0, max=1, value=(1/3), marks = {0:'Typical Events', 0.5:' ', 1:'Complications'},
                        id='weight_complications')
                ]) # end column - typical event/complications weight

        ]), # end Row with sliders

        # Row with advanced settings
        html.Hr(),
        html.Div(style={'display':'block','width':'100%'},
        children=[

            # checklist hide or show
            dcc.Checklist(
                id = 'dropdown-to-show_or_hide-element',
                options=['Show advanced options'],
                value = []),

            # element (Div) to hide or show
            html.Div(style = {'marginTop':20, 'display':'100%', 'display':'flex','height':'120px'}, 
                id = 'element-to-hide',
                children = [

                    html.H3('Advanced options'),

                    # column - alpha slider
                    html.Div(style={'display':'inline-block', 'width':'42%'},
                    children=[
                        html.H4('Influence of context and time on the relevance of the nodes:'),
                        dcc.Slider(id = 'pagerank_alpha',
                            min = 0,
                            max = 1,
                            value = 0.3,
                            marks = {0:'0%',
                            1: '100%'})
                    ]), # end column - alpha slider

                    # column slider intervention/occupation weight
                    html.Div(style={'display':'inline-block','width':'42%','marginLeft':'150px'},
                        children = [
                        html.H4('Balance between the contribution of the occupation and of the intervention to the "typical events" relevance value'),
                        dcc.Slider(min=0, max=1, value=0.5, 
                                   marks = {0:'Occupation only', 1:'Intervention only'},id='weight_intervention_occupation')
                        ]) # end column slider intervention/occupation weight
                ]), # end element (Div) to hide or show
        ]), # end Row with advanced settings

        # Row with figure
        html.Hr(),
        html.Div(style={'display':'block', "margin-top": "0px"},
            children=[
                dcc.Graph(id='figure'),
                html.A(
                    html.Button("Download as HTML",
                                id="buttom_download")
                    ),
                dcc.Download(id='download')
            ])# end Row with figure
        
       
    ]) # end Layout

# -------------------------------------------------------
# INPUTS & OUTPUTS

@app.callback(
    Output(component_id = 'figure', component_property='figure'),
    Output(component_id = 'element-to-hide', component_property='style'),
    Input(component_id = 'node_coloring', component_property = 'value'),
    Input(component_id = 'show_node_label', component_property = 'value'),
    Input(component_id = 'hide_start_end', component_property = 'value'),
    Input(component_id = 'range_centrality', component_property = 'value'),
    Input(component_id = 'group_units_per_type', component_property = 'value'),
    Input(component_id = 'figure_width', component_property = 'value'),
    Input(component_id = 'figure_height', component_property = 'value'),
    Input(component_id = 'weight_complications', component_property = 'value'),
    Input(component_id = 'weight_intervention_occupation', component_property = 'value'),
    Input(component_id = 'pagerank_alpha', component_property = 'value'),
    Input(component_id='dropdown-to-show_or_hide-element', component_property='value'),
    Input(component_id = 'show_edge_label', component_property = 'value'),
    Input(component_id = 'edge_min_freq', component_property = 'value'),
    Input(component_id = 'color_edges', component_property = 'value')
)  

def update_plot(node_coloring, show_node_label,
    hide_start_end, range_centrality, group_units_per_type, figure_width, figure_height,
    weight_complications, weight_intervention_occupation, pagerank_alpha, visibility_state, show_edge_label,
    edge_min_freq, color_edges):

    # ----------------------
    # Hide/show advanced options

    if visibility_state == ['Show advanced options']:
        visibility_result = {'display': 'block'}
    else:
        visibility_result = {'display': 'none'}

    # ----------------------
    # Boolean variables

    bool_hide_start_end = False
    if hide_start_end==['Remove Start/End nodes']:
        bool_hide_start_end = True

    bool_group_units_per_type = False
    if group_units_per_type == ['Group healthcare units according to their category']:
        bool_group_units_per_type = True

    bool_show_edge_label = False
    if show_edge_label == ['Show edge labels']:
        bool_show_edge_label = True
        
    bool_color_edges = False
    if color_edges == ['Colored edges']:
        bool_color_edges = True

    bool_node_label = False
    if show_node_label == ['Show node labels']:
        bool_node_label = True

    # ------------------
    # Create the MAG with the dictionary of edges 

    G = mag.MultiAspectMultiDiGraph()
    list_of_patients = list(dict_edges_of_patients.keys())

    for patient in list_of_patients:
        edgelist = dict_edges_of_patients[patient]
        for edge in edgelist:
            G.add_edges_from([(edge[0], edge[1], edge[3])]) #edge[0] is the origin; edge[1] is the target; edge[3] are the attributes

    # --------------------
    # calculate the relevance of each node

    G = mag_centralities(
            G,
            dict_closeness_intervention_NORM, 
            dict_betweenness_occupation_NORM, 
            dict_pagerank_unit_NORM,
            idx_intervention = 0, idx_occupation = 1, idx_unit = 2,
            weight_intervention_vs_occupation = weight_intervention_occupation, 
            weight_complications = weight_complications,
            bool_remove_start_end = True,
            use_edge_frequency = True,
            pagerank_alpha = pagerank_alpha,
            show_start_end = not bool_hide_start_end
        )

    # --------------------
    # filter the MAG

    nodes_to_remove = []

    min_relevance = range_centrality[0]
    max_relevance = range_centrality[1]

    for node, att in G.nodes(data = True):
        if (att['centrality'] < min_relevance)|(att['centrality'] > max_relevance):
            nodes_to_remove.append(node)

    G = filter_MAG(G, nodes_to_remove,  bool_keep_nodes = False, patient_attribute = 'patient')

    # ---------------------
    # annotation

    note_text = f'''The figure displays {len(list_of_patients)} patient pathways.'''
    note_text += f'''<br>{"A minimum value for the relevance of the nodes was not set" if min_relevance == 0.0 else "Showing nodes with minimum relevance value of "+str(min_relevance)}'''
    note_text += f''' and {"a maximum value for the relevance of the nodes was not set" if max_relevance == 1.0 else "showing nodes with maximum relevance value of "+str(max_relevance)}'''
    note_text += f'''<br>The initial relevance of the nodes is influeced in {round(weight_complications*100,2)}% by the possibility of the node being part of a complication of the patient pathway, <br>and in {round((1-weight_complications)*100,2)}% by the possibility of the node being a typical event of the patient pathway.'''
    note_text += f'''<br>To identify typical pathway events, there is a weight of {round(weight_intervention_occupation*100,2)}% for the intervention, and of {round((1-weight_intervention_occupation)*100,2)}% for the occupation.'''
    
    annotation = {'xref': 'x', 'yref': 'paper',
                'x': 0.0, 'y': 1.0,
                'align' : 'left',
                'yanchor':'bottom',
                'xanchor':'left',
                'showarrow': False,
                'text': note_text,
                'font' : {'size': 12,'color': 'black'},
                'bgcolor' : "lightblue"}

    # --------------------
    # group healthcare units, if requested

    if bool_group_units_per_type:
        unit_types = set()
        for node in G.nodes():
            if ((node[2] != 'start') & (node[2] != 'end')):
                G.nodes[node]['unit_type'] = df_unity_type.loc[node[2]]['unit_type']
                unit_types.add(G.nodes[node]['unit_type'])
            else:
                G.nodes[node]['unit_type'] = node[2]
        for t in unit_types:
            G = contract_multiMAG_nodes(G, attribute_name = 'unit_type',
                attribute_value = t, aspect_to_substitute = 2)

    # ---------------------
    # subdetermine the MAG if necessary
    
    max_edge_freq = 1
    if not bool_color_edges: # if edges are not colored, we convert the multiMAG into a diMAG
        
        if node_coloring == 'intervention': # if nodes are colored according to the intervention, we subdetermine to remove Occupation
            G = subdetermination(G, [1,0,1,1], multi=False, direct=True, loop=False, edge_frequency=True)
        
        else:
            G = convert_multiMAG_into_diMAG(G)
        
        # get the maximum edge frequency
        list_edge_freq = [1]
        for u,v,att in G.edges(data=True):
            list_edge_freq.append(att['freq'])
        max_edge_freq = max(list_edge_freq)

    else: # if edges are colored, we keep G as a multiMAG
        
        if node_coloring == 'intervention': # if nodes are colored according to the intervention, we subdetermine to remove Occupation
            G = subdetermination(G, [1,0,1,1], multi=True, direct=True, loop=False, edge_frequency=True)
        
        else:
            G_copy = mag.MultiAspectMultiDiGraph()
            for origin, target, key, att in G.edges(data=True,keys=True):
                G_copy.add_edges_from([(origin, target, att)])
            for node, att in G.nodes(data=True):
                for key in att:
                    G_copy.nodes[node][key] = att[key]
            G = G_copy

    # -----------------------------------
    # add edge labels, if requested

    if bool_show_edge_label:

        for u,v,att in G.edges(data=True):
    
            list_of_intervals = list(att['interval'])
            list_of_intervals = [float(i) for i in list_of_intervals]

            text = f'''{len(list_of_intervals)} patient{''+'s'if len(list_of_intervals)>1 else''}'''
            text += f''' --- {len(set(list_of_intervals))} interval{''+'s'if len(set(list_of_intervals))>1 else''}<br>'''
            text += f'''Min    '''
            text += f'''Q1    ''' if len(list_of_intervals)>=4 else f''''''
            text += f'''Mdn    ''' if len(list_of_intervals)>=2 else f''''''
            text += f'''Q3    ''' if len(list_of_intervals)>=4 else f''''''
            text += f'''Max<br>'''
            text += f'''{round(np.min(list_of_intervals),1)}    '''
            text += f'''{round(np.percentile(list_of_intervals,25),1)}    ''' if len(list_of_intervals)>=4 else f''''''
            text += f'''{round(np.percentile(list_of_intervals,50),1)}    ''' if len(list_of_intervals)>=2 else f''''''
            text += f'''{round(np.percentile(list_of_intervals,75),1)}    ''' if len(list_of_intervals)>=4 else f''''''
            text += f'''{round(np.max(list_of_intervals),1)}<br>'''
        
            G[u][v]['info_intervals'] = text

    # -------------------------
    # use intervention as node label (if fixed node labels were requested)

    if bool_node_label:
        dict_node_labels = {}
        for node in G.nodes():
            dict_node_labels[node] = node[0]
            
    # ----------------------------     
        
    fig = plot_mag(graph = G, 
        idx_sequence = 3 if node_coloring!='intervention' else 2,
        idx_label_aspect = 0, order_aspect_label = None,
        idx_aspect_yaxis = 2 if node_coloring!='intervention' else 1, 
        order_aspect_yaxis = type_unity_order,
        node_alpha = 0.85, edge_alpha = 0.2,
        idx_aspect_color = (0 if node_coloring == 'intervention' else 1 if node_coloring == 'occupation' else None), 
        attribute_color_discrete = None,
        attribute_color_continuous = None if node_coloring!='centrality' else 'centrality',
        node_colorscale = 'Sunset', 
        node_colorscale_limits = [0,1],
        fixed_node_labels = bool_node_label,
        node_size = 10, 
        node_size_factor = 0.5,
        node_size_power = 1, 
        node_size_attribute = None,
        dict_node_labels = None if not bool_node_label else dict_node_labels,
        list_node_attributes = ['centrality'] if node_coloring != 'intervention' else [],
        factor_bend_edges = 0.0 if not bool_color_edges else 0.02, 
        n_aux_points_edges = 50, 
        colored_edges = bool_color_edges,
        limits_colored_edges = [0,90],
        list_edge_attributes = ['info_intervals'] if bool_show_edge_label else [],
        edge_width = 2, 
        attribute_edge_width = 'freq' if not bool_color_edges else None,
        edge_width_factor = (0.2 if max_edge_freq>1000 else 0.3),
        edge_width_power = (0.2 if max_edge_freq>1000 else 0.6),
        edge_color = 'gray', 
        show_edge_labels = bool_show_edge_label, 
        shift_edge_arrow = 0,
        figure_title = '',
        dict_colors_aspect_yaxis = None, 
        df_info_aspect_yaxis = None, 
        columns_df_info_aspect_yaxis = [],
        rot_xticks = 0, 
        interval_yaxis = 2 if bool_group_units_per_type else 1,
        starting_point_xaxis_bar = -0.5,
        figure_size = [figure_width,figure_height],
        figure_margins = [0,0,150,0],
        hide_some_edges = True if not bool_color_edges else False,
        attribute_hide_edges = 'freq' if not bool_color_edges else None, 
        minimum_value_show_edges = edge_min_freq,
        text_note = annotation 
        )

    fig.update_layout(
        margin=dict(pad=50), plot_bgcolor='white',paper_bgcolor='white'
    )
    fig.write_html("results/plotly_graph.html")

    return fig, visibility_result

@app.callback(
    Output('download','data'),
    Input('buttom_download','n_clicks'), prevent_initial_call=True)

def download_html(n):
    return dcc.send_file("results/plotly_graph.html")


# -------------------------------------------------------
# Set the app to run in development mode
if __name__ == '__main__':
    app.run_server(debug=True)