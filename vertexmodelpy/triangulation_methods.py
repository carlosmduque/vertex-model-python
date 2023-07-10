import numpy as np
import pandas as pd
from itertools import chain
from .utilities import list_to_range_dict,find_ordered_vertex_faces,generate_directed_edges_topology
from .basic_geometry import euclidean_distance,pol_area,pol_centroid,pol_perimeter


def reindexed_face_vertices(tissue):

    vert_ids = tissue.vert_df['id'].values

    old_to_new_ids_series = \
                pd.Series(list_to_range_dict(vert_ids))
    
    v_out_series = tissue.edge_df['v_out_id']
        
    flatten_dbonds = list(tissue.face_dbonds)
    flatten_dbonds = list(chain(*flatten_dbonds))

    v_out_flatten = v_out_series.loc[flatten_dbonds]
    v_out_flatten = old_to_new_ids_series.loc[v_out_flatten].values

    tissue_faces_relabeled = np.empty(len(tissue.face_dbonds),dtype=object)

    counter = 0
    for i, num_sides in enumerate(tissue.face_df['num_sides'].values):
        counter_old = counter
        counter += num_sides
        tissue_faces_relabeled[i] = v_out_flatten[counter_old:counter]
        
    return tissue_faces_relabeled

def tissue_subcellular_triangulation(tissue):
    
    num_verts, num_faces = len(tissue.vert_df), len(tissue.face_df)
    
    triangulation_vertices = np.concatenate(
                        (tissue.vert_df[['x','y']].values,
                        tissue.face_df[['x','y']].values))
    
    zipped_vertices = [
            np.dstack((vert_list,np.roll(vert_list,-1)))[0]
            for vert_list in reindexed_face_vertices(tissue)]
    
    extra_verts_range = np.arange(num_verts,num_verts+num_faces)

    triangulation_faces = np.array([[v_id,*vert_pair]
                    for _, (v_id, vert_pairs_list) in 
                    enumerate(zip(extra_verts_range,zipped_vertices))
                    for vert_pair in vert_pairs_list])
    
    triangulation_dbonds = np.array([list(pair)
                    for triang_face in triangulation_faces
                    for pair in zip(triang_face, np.roll(triang_face, -1))])
    
    
    return triangulation_vertices,triangulation_dbonds,triangulation_faces

def tissue_dual_triangulation(tissue):
    triangulation_vertices = tissue.face_df[['x','y']].values
    
    face_ids = tissue.face_df['id'].values

    old_to_new_ids_series = \
                pd.Series(list_to_range_dict(face_ids))
    
    interior_vert_ids = tissue.vert_df.loc[
                        tissue.vert_df['is_interior'] == True,'id'].values
    
    triangulation_faces = [old_to_new_ids_series.loc[
                                find_ordered_vertex_faces(tissue,v_id)].values
                                for v_id in interior_vert_ids]
    
    triangulation_dbonds = np.array([list(pair)
                    for triang_face in triangulation_faces
                    for pair in zip(triang_face, np.roll(triang_face, -1))])
    
    # triangulation_faces = np.array(triangulation_faces,dtype=object)
    
    
    return triangulation_vertices,triangulation_dbonds,triangulation_faces

def create_triangulation_vert_df(vert_positions):
    """ Generates a Dataframe with relevant vertex data,
        i.e, ID, (x,y) coordinates, and force components.
        
        Parameters:
        -----------
        vert_positions : Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) coordinates of each vertex.

        Returns:
        -----------
        vert_df : Pandas.DataFrame
            Dataframe containing relevant vertex data.
    """
    id_list = list(range(len(vert_positions)))
    x, y = vert_positions[:,0], vert_positions[:,1]

    initial_data = list(zip(id_list, x, y))
    cols = ['id', 'x', 'y']
    vert_df = pd.DataFrame(initial_data, columns=cols)

    return vert_df


def create_triangulation_edge_df(vert_positions,dir_edges_end_points,
                    face_per_directed_edge,left_edge_index,
                    right_edge_index,conjugate_edge_index):
    """ Generates a Dataframe with edge data characterizing the
        geometry, topology, and energetics (line_tension) of
        the polygonal mesh.
       
        Parameters:
        -----------
        vert_positions: Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) coordinates of each vertex.
        dir_edges_end_points : dict, (bond_id,[out_vertex_id, in_vertex_id])
            Dictionary containing the out(in) vertex indices of each edge.
        face_per_directed_edge : dict, (edge_id, face_id)
            Dictionary containing the face index that half edge belongs to.
        left_edge_index : dict, (bond_id, left_bond_id)
            Dictionary containing the left half edge index for each half edge.
        conjugate_edge_index : dict, (bond_id, conjugate_bond_id)
            Dictionary containing the conjugate (opposite) half edge index
            for each half edge. For boundary edges (no conjugate edge) a -1
            value is assigned.

        Returns:
        -----------
        edge_df : Pandas.DataFrame
            DataFrame containing the geometry, topology, and energy parameters
            of each directed edge of the polygonal mesh.
    """

    number_of_edges = len(dir_edges_end_points)

    id_list = np.arange(number_of_edges)

    vertex_out_id = [dir_edges_end_points[edge_id][0]
                        for edge_id in id_list]

    vertex_in_id = [dir_edges_end_points[edge_id][1]
                        for edge_id in id_list]

    interior_edge = [True if conjugate_edge_index[edge_id] != -1 else False
                        for edge_id in id_list]

    edge_lengths = np.zeros(number_of_edges, dtype=float)

    for edge_id, (v_out, v_in) in enumerate(zip(vertex_out_id,vertex_in_id)):
        p_out, p_in = vert_positions[v_out], vert_positions[v_in]
        edge_lengths[edge_id] = euclidean_distance(p_out,p_in)

    initial_data = list(zip(id_list, vertex_out_id, vertex_in_id,
                            interior_edge,edge_lengths))

    cols = ['id', 'v_out_id', 'v_in_id', 'is_interior', 'length']
    edge_df = pd.DataFrame(initial_data, columns=cols)

    edge_df = edge_df.assign(
            left_dbond = lambda df: df['id'].map(left_edge_index),
            right_dbond = lambda df: df['id'].map(right_edge_index),
            conj_dbond = lambda df: df['id'].map(conjugate_edge_index),
            dbond_face = lambda df: df['id'].map(face_per_directed_edge))


    return edge_df

def create_triangulation_face_df(vert_positions,face_data):
    """ Generates a Dataframe with face data characterizing the
        geometry, such as the number of sides or the area, as well
        as parameters relevant to the energetics like preferred area.
              
        Parameters:
        -----------
        vert_positions : Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) coordinates of each vertex.
        face_data : dict, (face_id, [sorted vertex_id])
            Dictionary containing the ordered vertex indices of each polygon. 

        Returns:
        -----------
        edge_df : Pandas.DataFrame
            DataFrame containing the geometry, topology, and energy parameters
            of each directed edge of the polygonal mesh.
    """
    number_of_faces = len(face_data)

    id_list = np.arange(number_of_faces)
    
    face_vertices = [vert_positions[triang_vert_ids]
                                for triang_vert_ids in face_data]
    
    num_sides = [len(triang_vert_ids)
                                for triang_vert_ids in face_data]

    area_list = np.array([pol_area(f_vert) for f_vert in face_vertices])
    centroid_list = np.array([pol_centroid(f_vert) for f_vert in face_vertices])
    perimeter_list = np.array([pol_perimeter(f_vert)
                            for f_vert in face_vertices])
    
    x, y = centroid_list[:,0], centroid_list[:,1]
    
    initial_data = list(zip(id_list, x, y,
                            area_list, perimeter_list,num_sides))
    
    face_df = pd.DataFrame(initial_data, columns = 
                ['id', 'x', 'y','area', 'perimeter','num_sides'])
    

    return face_df

def generate_triangulation_dataframes(vert_positions,dir_edges_end_points,
                    face_per_directed_edge,left_edge_index,
                    right_edge_index,conjugate_edge_index,face_data):

    
    vert_df = create_triangulation_vert_df(vert_positions)

    edge_df = create_triangulation_edge_df(vert_positions,dir_edges_end_points,
                    face_per_directed_edge,left_edge_index,
                    right_edge_index,conjugate_edge_index)

    face_df = create_triangulation_face_df(vert_positions,face_data)

    return vert_df, edge_df, face_df

def initialize_triangulation(
                            tissue,triangulation_type='dual_triangulation'):
    
    if triangulation_type == 'dual_triangulation':
        triangulation_vertices, triangulation_dbonds,\
            triangulation_faces = tissue_dual_triangulation(tissue)
    elif triangulation_type == 'subcellular_triangulation':
        triangulation_vertices, triangulation_dbonds,\
            triangulation_faces = tissue_subcellular_triangulation(tissue)
            
    face_dbonds, face_per_directed_edge,left_edge_index, right_edge_index, \
    conjugate_edge_index = generate_directed_edges_topology(
                                    triangulation_dbonds,triangulation_faces)
        
    vert_df, edge_df, face_df = \
        generate_triangulation_dataframes(
            triangulation_vertices,triangulation_dbonds,
            face_per_directed_edge,left_edge_index,
            right_edge_index,conjugate_edge_index,triangulation_faces)
        
    face_dbonds = pd.Series(face_dbonds)


    return vert_df, edge_df, face_df, face_dbonds, triangulation_faces