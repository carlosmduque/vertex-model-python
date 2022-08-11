import numpy as np
import pandas as pd
from itertools import chain

from matplotlib import cm

def range_to_list_dict(lst):
    """[summary]

    Args:
        lst ([type]): [description]

    Returns:
        [type]: [description]
    """
    # return {key:val for key,val in enumerate(list)}
    # return dict(np.column_stack((np.arange(len(list),dtype=int),list)))
    return dict(zip(np.arange(len(lst),dtype=int),lst))

def list_to_range_dict(lst):
    """[summary]

    Args:
        list ([type]): [description]

    Returns:
        [type]: [description]
    """
    # return {key:val for val,key in enumerate(list)}
    # return dict(np.column_stack((list,np.arange(len(list),dtype=int))))
    return dict(zip(lst, np.arange(len(lst),dtype=int)))

def add_constant_columns_to_df(df,col_value_pairs):
    """[summary]

    Args:
        df ([type]): [description]
        col_value_pairs ([type]): [description]
    """
    for col_name, const_val in col_value_pairs:
        df[col_name] = const_val

    return

def delete_integer_from_nparray(array,element):
    """[summary]

    Args:
        array ([type]): [description]
        element ([type]): [description]

    Returns:
        [type]: [description]
    """
    index = np.argwhere(array == element)
    return np.delete(array, index)

def add_vertex_to_df(tissue,vert_id,v_pos,
                     fx=0.0,fy=0.0,is_interior=True):

    tissue.vert_df.loc[vert_id] = np.array([vert_id,v_pos[0],v_pos[1],
                                        fx, fy, is_interior],dtype=object)
    
    return

def add_dir_edge_to_df(tissue,edge_id,v_out_id,v_in_id,
                 left_dbond,right_dbond,conj_dbond,dbond_face,
                 is_interior=True,length=1.0,line_tension=0.12):
    
    tissue.edge_df.loc[edge_id] = np.array([edge_id, v_out_id, v_in_id,
                                        is_interior, length, line_tension,
                                        left_dbond, right_dbond, conj_dbond,
                                        dbond_face],dtype=object)
    
    return

def add_face_to_df(tissue,face_id,x,y,num_sides,area,perimeter,
                   A_0=1.0,P_0=0.0,contractility=0.04,
                   active=1,mother=-1):
    
    
    tissue.face_df.loc[face_id] = np.array([face_id, x, y, num_sides,
                                   area,perimeter, mother, A_0, P_0,
                                   contractility,active],dtype=object)
    
    return

def find_face_vertices(tissue,face_dbonds,single_face=False):
    """[summary]

    Args:
        vert_positions ([type]): [description]
        edge_df ([type]): [description]
        face_dbonds ([type]): [description]

    Returns:
        [type]: [description]
    """

    vert_positions = tissue.vert_df[tissue.coord_labels]
    v_out_series = tissue.edge_df['v_out_id']
    
    if single_face:
        vert_indices = v_out_series.loc[face_dbonds]
        face_verts = vert_positions.loc[vert_indices].values
    else:
        face_verts = np.empty(len(face_dbonds), dtype = object)
    
        for i, f_dbonds in enumerate(face_dbonds):
            vert_indices = v_out_series.loc[f_dbonds]
            face_verts[i] = vert_positions.loc[vert_indices].values  

    return face_verts

def find_face_vertices_faster(tissue,face_dbonds,return_positions=True):
    """[summary]

    Args:
        tissue ([type]): [description]
        face_dbonds ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    vert_positions = tissue.vert_df[tissue.coord_labels]
    v_out_series = tissue.edge_df['v_out_id']
    
    flatten_dbonds = list(face_dbonds)
    flatten_dbonds = list(chain(*flatten_dbonds))
    
    v_out_flatten = v_out_series.loc[flatten_dbonds]
    if return_positions:
        v_out_flatten = vert_positions.loc[v_out_flatten].values
    else:
        v_out_flatten = v_out_flatten.values
    
    face_verts = np.empty(len(tissue.face_dbonds),dtype=object)

    counter = 0
    for i, num_sides in enumerate(tissue.face_df['num_sides'].values):
        counter_old = counter
        counter += num_sides
        face_verts[i] = v_out_flatten[counter_old:counter]
    
    return face_verts

def find_directed_edges_per_face(num_sides_list):
    """ Defines and ordered list of directed (half) edge indices
        around a face given a list containing the number of sides of
        each polygonal face.

        Note: The edge indices are defined based on the order in which
        the polygonal faces are listed.
        
        Parameters:
        -----------
        num_sides_list : list, shape: (N,)
            List with the number of sides of each polygonal face.

        Returns:
        -----------
        edge_indices_per_face : dict, (face_id, [sorted dir_edges])
            Dictionary containing the ordered edge indices of each face.
    """
    edge_counter = 0
    edge_indices_per_face = {}

    for idx,num_sides in enumerate(num_sides_list):
        range_list = list(range(edge_counter, edge_counter + num_sides))

        edge_indices_per_face[idx] = range_list
        edge_counter += num_sides

    return edge_indices_per_face

def find_face_per_directed_edge(num_sides_list):
    """ Determines the face to which each directed (half) edge belongs to.

        NOTE: This function should be used with find_directed_edges_per_face
        in order to ensure the consistency of the indices.
        
        Parameters:
        -----------
        num_sides_list : list, shape: (N,)
            List with the number of sides of each polygonal face.

        Returns:
        -----------
        face_per_directed_edge : dict, (edge_id, face_id)
            Dictionary containing the face index that half edge belongs to.
    """

    edge_counter = 0
    face_per_directed_edge = {}

    for idx,num_sides in enumerate(num_sides_list):
        for _ in range(num_sides):
            face_per_directed_edge[edge_counter] = idx
            edge_counter += 1

    return face_per_directed_edge

# def find_left_directed_edges(edge_indices_per_face):
def find_left_and_right_directed_edges(edge_indices_per_face):
    """ Determines the left (neighbor in the anticlockwise sense) half
        edge of each half edge within the mesh.

        Note: find_directed_edges_per_face should be used first in order
        to ensure the consistency of the left indices.
        
        Parameters:
        -----------
        edge_indices_per_face : dict, (face_id, [sorted dir_edges])
            Dictionary containing the ordered edge indices of each face.

        Returns:
        -----------
        left_edge_index : dict, (bond_id, left_bond_id)
            Dictionary containing the left half edge index for each half edge.
    """

    left_edge_index = {}
    right_edge_index = {}

    for face_idx, dir_edges in edge_indices_per_face.items():
        
        # l_dir_edges = np.roll(dir_edges,-1)
        l_dir_edges, r_dir_edges = np.roll(dir_edges,-1), np.roll(dir_edges,1)

        # sub_dict = dict(zip(dir_edges, l_dir_edges))
        left_sub_dict = dict(zip(dir_edges, l_dir_edges))
        right_sub_dict = dict(zip(dir_edges, r_dir_edges))

        # left_edge_index.update(sub_dict)
        left_edge_index.update(left_sub_dict)
        right_edge_index.update(right_sub_dict)


    # return left_edge_index
    return left_edge_index, right_edge_index

def find_conjugate_directed_edges(dir_edges_end_points):
    """ Determines the conjugate (opposite) half edge for each half edge
        within the mesh.

        Note: The order of dir_edges_end_points should be in agreement with
        the edge order defined by find_directed_edges_per_face in order
        to ensure the consistency with other lists, e.g, left_edge_index.
        
        Parameters:
        -----------
        dir_edges_end_points : dict, (bond_id,[out_vertex_id, in_vertex_id])
            Dictionary containing the out(in) vertex indices of each edge.

        Returns:
        -----------
        conjugate_edge_index : dict, (bond_id, conjugate_bond_id)
            Dictionary containing the conjugate (opposite) half edge index
            for each half edge. For boundary edges (no conjugate edge) a -1
            value is assigned.
    """

    sorted_edge_end_points = [tuple(sorted(dir_edges_end_points[i]))
                                for i in range(len(dir_edges_end_points))]


    undirected_edge_end_points = list(set(sorted_edge_end_points))

    fold_dict = {key:val
                for val, key in enumerate(undirected_edge_end_points)}

    # Directed edges belong to a fold that can be composed of 1 or 2 edges
    fold_per_directed_edge = [fold_dict[end_pts]
                            for end_pts in sorted_edge_end_points]

    fold_per_directed_edge = pd.Series(fold_per_directed_edge)

    directed_edges_per_fold = [0
                            for i, _ in enumerate(undirected_edge_end_points)]

    for i, _ in enumerate(undirected_edge_end_points):
        dir_edges_indices = fold_per_directed_edge[fold_per_directed_edge == i]
        directed_edges_per_fold[i] = dir_edges_indices.index.tolist()

    conjugate_edge_index = [-1 for i in range(len(dir_edges_end_points))]

    for _, dir_edges_indices in enumerate(directed_edges_per_fold):
        if len(dir_edges_indices) == 2:
            edge_index_1, edge_index_2 = dir_edges_indices

            conjugate_edge_index[edge_index_1] = edge_index_2
            conjugate_edge_index[edge_index_2] = edge_index_1

    conjugate_edge_index = range_to_list_dict(conjugate_edge_index)

    return conjugate_edge_index

def generate_directed_edges_topology(dir_edges_end_points,face_data):
    """ Wrapper function to generate the relevant directed edges dictionaries.
        
        Parameters:
        -----------
        dir_edges_end_points : dict, (bond_id,[out_vertex_id, in_vertex_id])
            Dictionary containing the out(in) vertex indices of each edge.
        face_data : dict, (face_id, [sorted vertex_id])
            Dictionary containing the ordered vertex indices of each polygon.

        Returns:
        -----------
        edge_indices_per_face : dict, (face_id, [sorted dir_edges])
            Dictionary containing the ordered edge indices of each face.
        face_per_directed_edge : dict, (edge_id, face_id)
            Dictionary containing the face index that half edge belongs to.
        left_edge_index : dict, (bond_id, left_bond_id)
            Dictionary containing the left half edge index for each half edge.
        conjugate_edge_index : dict, (bond_id, conjugate_bond_id)
            Dictionary containing the conjugate (opposite) half edge index
            for each half edge. For boundary edges (no conjugate edge) a -1
            value is assigned.
    """

    num_sides_list = [len(face_data[i]) for i in range(len(face_data))]

    edge_indices_per_face = find_directed_edges_per_face(num_sides_list)
    face_per_directed_edge = find_face_per_directed_edge(num_sides_list)

    left_edge_index, right_edge_index = \
                    find_left_and_right_directed_edges(edge_indices_per_face)
    conjugate_edge_index = find_conjugate_directed_edges(dir_edges_end_points)
 

    return (edge_indices_per_face,face_per_directed_edge,
            left_edge_index,right_edge_index,conjugate_edge_index)

def get_face_ordered_dbonds(tissue,face_list=[]):
    """ Calculate the ordered indices of the edges surrounding the
        faces of each polygonal face.
        
        Parameters:
        -----------
        edge_df : Pandas.DataFrame
            DataFrame containing the topology data of each directed edge. 

        Returns:
        -----------
        edge_indices_per_face : Pandas.Series, (face_id, [sorted dir_edges])
            Series containing the ordered edge indices of each face.
    """
    # Group the edges with respect to their common face and also create
    # smaller (face, edge_id) groups.

    if face_list == []:
        face_groups = tissue.edge_df.groupby(['dbond_face'])
        face_bond_groups = tissue.edge_df.groupby(['dbond_face','id'])
    else:
        edge_df_subset = \
                tissue.edge_df[tissue.edge_df['dbond_face'].isin(face_list)]

        face_groups = edge_df_subset.groupby(['dbond_face'])
        face_bond_groups = edge_df_subset.groupby(['dbond_face','id'])


    edge_indices_per_face = {}
    for face_id,f_group in face_groups:
        dbond = f_group['id'].iat[0]

        face_dbonds = []
        
        for _, _ in enumerate(f_group.index):
            face_dbonds += [dbond]

            fb_group = face_bond_groups.get_group((face_id,dbond))
            left_dbond = fb_group['left_dbond'].iat[0]

            dbond = left_dbond

        # edge_indices_per_face[face_id] = face_dbonds
        edge_indices_per_face[face_id] = np.array(face_dbonds)

    # edge_indices_per_face = pd.Series(edge_indices_per_face)
    if face_list == []:
        tissue.face_dbonds = pd.Series(edge_indices_per_face)
    else:
        tissue.face_dbonds.loc[face_list] = edge_indices_per_face

    return

def find_ordered_vertex_dbonds(tissue,vert_id):
    
    dbonds_out = tissue.edge_df.loc[
                tissue.edge_df['v_out_id'] == vert_id,'id'].values
    
    
    dbond_in_next = tissue.edge_df.loc[
                                    dbonds_out[0],'right_dbond']
    
    dbond_out_next = tissue.edge_df.loc[
                                    dbond_in_next,'conj_dbond']

    vert_dbonds_in = [dbond_in_next]
    vert_dbonds_out = [dbond_out_next]
    
    while dbond_out_next != dbonds_out[0]:

        dbond_in_next = tissue.edge_df.loc[
                                        dbond_out_next,'right_dbond']
        
        dbond_out_next = tissue.edge_df.loc[
                                        dbond_in_next,'conj_dbond']
        
        vert_dbonds_in += [dbond_in_next]
        vert_dbonds_out += [dbond_out_next]
    
    vert_dbonds_in = list(np.roll(vert_dbonds_in, -1))
    
    return vert_dbonds_out, vert_dbonds_in

def find_ordered_vertex_faces(tissue,vert_id):
    
    dbonds_out, _ =  find_ordered_vertex_dbonds(tissue,vert_id)
    vert_faces = tissue.edge_df.loc[dbonds_out,'dbond_face'].values
    
    return vert_faces

def get_face_colors(tissue,color_by='area',min_max_vals=[]):
    
    if color_by == 'area':
        if min_max_vals == []:
            min_area = tissue.face_df['area'].min()
            max_area = tissue.face_df['area'].max()
        else:
            min_area, max_area = min_max_vals
        
        area_diff = max_area - min_area       
        normalized_areas = (tissue.face_df['area'] - min_area)/ area_diff
        
    elif color_by == 'num_sides':
        if min_max_vals == []:
            min_num_sides = tissue.face_df['num_sides'].min()
            max_num_sides = tissue.face_df['num_sides'].max()
        else:
            min_num_sides, max_num_sides = min_max_vals
        
        num_sides_diff = max_num_sides - min_num_sides       
        normalized_areas = \
            (tissue.face_df['num_sides'] - min_num_sides)/ num_sides_diff
        
        
    normalized_areas = normalized_areas.values

    facecolors = [cm.jet(x) for x in normalized_areas]
    return facecolors




# def get_face_ordered_dbonds(edge_df):
#     """ Calculate the ordered indices of the edges surrounding the
#         faces of each polygonal face.
        
#         Parameters:
#         -----------
#         edge_df : Pandas.DataFrame
#             DataFrame containing the topology data of each directed edge. 

#         Returns:
#         -----------
#         edge_indices_per_face : dict, (face_id, [sorted dir_edges])
#             Dictionary containing the ordered edge indices of each face.
#     """
#     # Group the edges with respect to their common face.
#     face_groups = edge_df.groupby(['dbond_face'])

#     edge_indices_per_face = {}
#     for face_id,f_group in face_groups:

#         face_dbonds = []
#         dbond = f_group['id'].iat[0]

#         # Find consecutive left_dbonds belonging to the face.
#         for i, _ in enumerate(f_group.index):
#             face_dbonds += [dbond]

#             left_dbond = f_group[f_group['id'] == dbond]['left_dbond'].iat[0]
#             dbond = left_dbond

#         edge_indices_per_face[face_id] = face_dbonds

#     return edge_indices_per_face

# def find_ordered_vertex_dbonds(tissue,vert_id):
    
#     sub_edge_df_v_out = \
#                     tissue.edge_df[(tissue.edge_df['v_out_id'] == vert_id)]
                    
#     sub_edge_df_v_in = tissue.edge_df[(tissue.edge_df['v_in_id'] == vert_id)]
    
#     dbonds_out = sub_edge_df_v_out['id'].values
    
#     dbond_in_next = sub_edge_df_v_out.loc[
#                         dbonds_out[0],'right_dbond']
    
#     dbond_out_next = sub_edge_df_v_in.loc[
#                         dbond_in_next,'conj_dbond']

#     vert_dbonds_in = [dbond_in_next]
#     vert_dbonds_out = [dbond_out_next]
    
#     while dbond_out_next != dbonds_out[0]:
#         dbond_in_next = sub_edge_df_v_out.loc[
#                         dbond_out_next,'right_dbond']
        
#         dbond_out_next = sub_edge_df_v_in.loc[
#                         dbond_in_next,'conj_dbond']
        
#         vert_dbonds_in += [dbond_in_next]
#         vert_dbonds_out += [dbond_out_next]
    
#     vert_dbonds_in = np.roll(vert_dbonds_in, -1)
    
#     return vert_dbonds_out, vert_dbonds_in

# def dict_to_array(x,dictionary):
#     """[summary]

#     Args:
#         x ([type]): [description]
#         dictionary ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     return np.vectorize(dictionary.get)(x)

# def pad_array_2d(arr,offset):
#     pad_arr = np.pad(arr,((0,offset),(0,0)),'edge')
#     return pad_arr

# def find_face_vertices(vert_positions,edge_df,face_dbonds):
#     """[summary]

#     Args:
#         vert_positions ([type]): [description]
#         edge_df ([type]): [description]
#         face_dbonds ([type]): [description]

#     Returns:
#         [type]: [description]
#     """

#     # face_groups = tissue.edge_df.groupby(['dbond_face'])[['v_out_id']]

#     # face_IDs = tissue.face_df['id']
#     # poly_verts = [
#     #           face_groups.get_group(f_ID).loc[tissue.face_dbonds.loc[f_ID]]   
#     #           ['v_out_id'] for f_ID in face_IDs]

#     v_out_series = edge_df['v_out_id']

#     face_verts = [v_out_series.loc[f_dbonds] 
#                         for f_dbonds in face_dbonds]

#     # face_verts = np.array([vert_positions.loc[vert_indices]
#     #                 for vert_indices in face_verts], dtype=object)
    
#     face_verts = np.array([vert_positions.loc[vert_indices].values
#                     for vert_indices in face_verts], dtype=object)

#     return face_verts


# def pad_polygon_arrays(tissue,face_arr):
    
#     max_num_sides = tissue.face_df['num_sides'].max()
#     offset =  max_num_sides - tissue.face_df['num_sides']
#     reduced_offset = offset[offset>0]
#     indices_to_pad = reduced_offset.index
    
#     face_arr_df = pd.Series(face_arr,
#                             index=tissue.face_df.index,dtype=object)
    
#     face_arr_df.loc[indices_to_pad] = \
#                     face_arr_df.loc[indices_to_pad].combine(
#                     reduced_offset, lambda x, y: pad_array_2d(x,y))
    
#     padded_array = np.array([*face_arr_df.values],dtype=float)
    
#     return padded_array

