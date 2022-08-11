# import pandas as pd
import numpy as np
import warnings

from utilities import (delete_integer_from_nparray,get_face_ordered_dbonds,
                add_vertex_to_df,add_dir_edge_to_df,add_face_to_df,
                find_face_vertices)

from basic_geometry import (rotation_2d,vertex_to_centroid,displace_vertex,
                            positive_quadrant_angle)
from math import pi

from basic_geometry import (pol_centroid,pol_perimeter,line_axis_intersection_point)

from energetics import vertex_stability

def set_vertex_topology(tissue):
    """[summary]

    Args:
        tissue ([type]): [description]
    """
    # boundary_vertices = \
    #     tissue.edge_df[tissue.edge_df['is_interior'] == False] \
    #                     ['v_out_id'].unique()
                        
    boundary_vertices = \
        tissue.edge_df.loc[tissue.edge_df['is_interior'] == False,
                        'v_out_id'].unique()

    tissue.vert_df['is_interior'] = True
    tissue.vert_df.loc[boundary_vertices, 'is_interior'] = False
    return

def recalculate_vertex_topology(tissue,vert_id):
    
    # sub_edge_df_v_out = \
    #                 tissue.edge_df[(tissue.edge_df['v_out_id'] == vert_id)]
                    
    # conj_dbonds = sub_edge_df_v_out['conj_dbond'].values
    
    # sub_edge_df_v_out = \
    #                 tissue.edge_df[(tissue.edge_df['v_out_id'] == vert_id)]
                    
    conj_dbonds = \
            tissue.edge_df.loc[
                tissue.edge_df['v_out_id'] == vert_id,'conj_dbond'].values
    
    if -1 not in conj_dbonds:
        tissue.vert_df.loc[vert_id,'is_interior'] = True
    else:
        tissue.vert_df.loc[vert_id,'is_interior'] = False
    
    
    return

def rotate_T1_vertices(tissue,edge_id,preferred_length=None):
    """[summary]

    Args:
        tissue ([type]): [description]
        edge_id ([type]): [description]
    """
    vert_IDs = tissue.edge_df.loc[edge_id,['v_out_id','v_in_id']]

    end_points_positions = tissue.vert_df.loc[vert_IDs,['x','y']].values
    edge_mid_point = np.atleast_2d(np.mean(end_points_positions, axis=0))

    new_end_points_positions = rotation_2d(end_points_positions,
                                origin=edge_mid_point, angle=-pi/2)

    if preferred_length != None:
        side_vectors = new_end_points_positions - edge_mid_point
        unit_vectors = side_vectors / np.linalg.norm(side_vectors,axis=1)
        
        new_end_points_positions = edge_mid_point + \
                                        preferred_length*unit_vectors 

    tissue.vert_df.loc[vert_IDs,['x','y']] = new_end_points_positions
    
    return

# def t1_able_edges(tissue):
    
#     v_id_out,v_id_in = tissue.edge_df['v_out_id'],tissue.edge_df['v_in_id']

#     valid_T1_edges = (tissue.vert_df.loc[v_id_out,'is_interior'].values & 
#                 tissue.vert_df.loc[v_id_in,'is_interior'].values)

    
#     t1_edges_IDs = tissue.edge_df.loc[valid_T1_edges,'id']
#     t1_conj_edges_IDs = tissue.edge_df.loc[t1_edges_IDs,'conj_dbond']
    
#     t1_edges_face_IDs = tissue.edge_df.loc[t1_edges_IDs,'dbond_face']
#     t1_conj_edges_face_IDs = \
#                     tissue.edge_df.loc[t1_conj_edges_IDs,'dbond_face']
    
#     t1_edges_face_num_sides = \
#                         tissue.face_df.loc[t1_edges_face_IDs,'num_sides']
#     t1_conj_edges_face_num_sides = \
#                     tissue.face_df.loc[t1_conj_edges_face_IDs,'num_sides']

#     t1_edges_face_criterion = (t1_edges_face_num_sides.values > 3) & \
#                                     (t1_conj_edges_face_num_sides.values > 3)

#     t1_edges_IDs = t1_edges_IDs[t1_edges_face_criterion].values
      
#     return t1_edges_IDs    

def rewire_left_and_right_edges(tissue,dbonds_list,left_dbonds_list):
    """[summary]

    Args:
        tissue ([type]): [description]
        dbonds_list ([type]): [description]
        left_dbonds_list ([type]): [description]
    """
    for dbonds,left_dbonds in zip(dbonds_list,left_dbonds_list):
        tissue.edge_df.loc[dbonds,'left_dbond'] = left_dbonds

    # We simply invert the order to rewire the right directed edges.
    for dbonds,left_dbonds in zip(dbonds_list,left_dbonds_list):
        tissue.edge_df.loc[left_dbonds,'right_dbond'] = dbonds

    return

def rewire_T1_edges(tissue,edge_id):
    """[summary]

    Args:
        tissue ([type]): [description]
        edge_id ([type]): [description]

    Returns:
        [type]: [description]
    """

    # conj_dbond = tissue.edge_df.loc[edge_id]['conj_dbond']
    conj_dbond = tissue.edge_df.loc[edge_id,'conj_dbond']
    dbonds = np.array([edge_id, conj_dbond])

    # We identify the group of edges involved in the T1 rewiring.
    # l_dbonds = tissue.edge_df.loc[dbonds]['left_dbond'].values
    # r_dbonds = tissue.edge_df.loc[dbonds]['right_dbond'].values
    # lc_dbonds = tissue.edge_df.loc[l_dbonds]['conj_dbond'].values
    # lcl_dbonds = tissue.edge_df.loc[lc_dbonds]['left_dbond'].values
    l_dbonds = tissue.edge_df.loc[dbonds,'left_dbond'].values
    r_dbonds = tissue.edge_df.loc[dbonds,'right_dbond'].values
    lc_dbonds = tissue.edge_df.loc[l_dbonds,'conj_dbond'].values
    lcl_dbonds = tissue.edge_df.loc[lc_dbonds,'left_dbond'].values
 
    dbonds_list = [dbonds,lc_dbonds,r_dbonds]
    left_dbonds_list = [lcl_dbonds,dbonds,l_dbonds]

    # We rewire the group of edges involved in the T1 transition.
    rewire_left_and_right_edges(tissue,dbonds_list,left_dbonds_list)

    # NOTE: In order to not mess up the topology of the network, we must make 
    # sure that the order of the elements of left_dbonds_list is consistent 
    # with whatever order we chose for the elements of dbonds_list.

    # Additionally, the end-point-vertex indices of l_dbonds and lc_dbonds
    # need to be reindexed after the T1 transition is completed.
    # out_vertices = tissue.edge_df.loc[dbonds]['v_out_id'].values
    # in_vertices = tissue.edge_df.loc[l_dbonds]['v_in_id'].values
    out_vertices = tissue.edge_df.loc[dbonds,'v_out_id'].values
    in_vertices = tissue.edge_df.loc[l_dbonds,'v_in_id'].values

    l_dbond_vert_IDs = [[v_out,v_in]
                        for v_out,v_in in zip(out_vertices,in_vertices)]

    tissue.edge_df.loc[l_dbonds,['v_out_id','v_in_id']] = l_dbond_vert_IDs
    tissue.edge_df.loc[lc_dbonds,['v_in_id','v_out_id']] = l_dbond_vert_IDs

    return dbonds,lc_dbonds

def simple_t1_rearrangement(tissue,edge_id):
    """[summary]

    Args:
        tissue ([type]): [description]
        edge_id ([type]): [description]
    """

    vert_IDs = tissue.edge_df.loc[edge_id,['v_out_id','v_in_id']]
    vert_inside = tissue.vert_df.loc[vert_IDs,'is_interior'].values

    # We disallow edges connected to boundary vertices.
    if False in vert_inside:
        warnings.warn(
            f"The edge with ID: {edge_id} is connected to a boundary vertex.")
        return

    # The group of edges involved in a full T1 transition is reconnected.
    dbonds, lc_dbonds = rewire_T1_edges(tissue,edge_id)

    # The faces of dbonds change after the T1 transition.
    
    # dbonds_faces = tissue.edge_df.loc[dbonds]['dbond_face'].values
    # lc_dbonds_faces = tissue.edge_df.loc[lc_dbonds]['dbond_face'].values
    dbonds_faces = tissue.edge_df.loc[dbonds,'dbond_face'].values
    lc_dbonds_faces = tissue.edge_df.loc[lc_dbonds,'dbond_face'].values
    tissue.edge_df.loc[dbonds,'dbond_face'] = lc_dbonds_faces

    # The number of sides of the T1-edge faces decrease by 1 and once
    # The edge is fully flipped the conjugate faces get an extra side.
    
    tissue.face_df.loc[dbonds_faces,'num_sides'] -= 1
    tissue.face_df.loc[lc_dbonds_faces,'num_sides'] += 1

    # The dbonds list of the T1 face group is updated after reconnecting
    # the group of directed edges.
    face_list = np.array([dbonds_faces, lc_dbonds_faces]).flatten()
    get_face_ordered_dbonds(tissue, face_list=face_list)
    
    # tissue.update_areas_perimeters_centroids(face_list=face_list)
    
    tissue.update_tissue_geometry(face_list=face_list)
    

    return

def boundary_to_interior_t1_rearrangement(tissue,edge_id,scale_factor=0.1):
    
    vert_IDs = tissue.edge_df.loc[edge_id,['v_out_id','v_in_id']]
    vert_ID_stay, vert_ID_drop = np.sort(vert_IDs)
    pulled_face_ID = tissue.edge_df.loc[edge_id,'dbond_face']
    
    collapse_edge(tissue,edge_id)
    split_boundary_vertex(tissue,vert_ID_stay,scale_factor=scale_factor,
                            preferred_face=pulled_face_ID)
    
    return

# def pure_boundary_t1_rearrangement(tissue,edge_id):
    
#     boundary_face_id = tissue.edge_df.loc[edge_id,'dbond_face']

#     face_dbonds_ordered = np.array(tissue.face_dbonds.loc[boundary_face_id])

#     is_interior_dbonds = \
#         (tissue.edge_df.loc[face_dbonds_ordered,'is_interior']==False).values


#     first_false_position = np.where(is_interior_dbonds == False)[0][0]

#     is_interior_dbonds = np.roll(is_interior_dbonds,-first_false_position)
#     face_dbonds_ordered = np.roll(face_dbonds_ordered,-first_false_position)

#     face_boundary_dbonds = face_dbonds_ordered[is_interior_dbonds]

#     dbonds_number = len(face_boundary_dbonds)

#     position_dbond_id = np.where(face_boundary_dbonds == edge_id)[0][0]

#     if (dbonds_number - (position_dbond_id + 1)) == 0:
#         l_dbond_id = tissue.edge_df.loc[edge_id,'left_dbond']
#         lc_dbond_id = tissue.edge_df.loc[l_dbond_id,'conj_dbond']
#         lcl_dbond_id = tissue.edge_df.loc[lc_dbond_id,'left_dbond']
        
#         add_vertex_to_edge(tissue,lcl_dbond_id)
        
#         # print('This is the last boundary bond')
        
#     collapse_edge(tissue,edge_id)
    
    
#     return

# def pure_boundary_t1_rearrangement(tissue,edge_id,scale_factor=0.1):
    
#     boundary_face_id = tissue.edge_df.loc[edge_id,'dbond_face']

#     face_dbonds_ordered = np.array(tissue.face_dbonds.loc[boundary_face_id])

#     is_interior_dbonds = \
#         (tissue.edge_df.loc[face_dbonds_ordered,'is_interior']==False).values


#     first_false_position = np.where(is_interior_dbonds == False)[0][0]

#     is_interior_dbonds = np.roll(is_interior_dbonds,-first_false_position)
#     face_dbonds_ordered = np.roll(face_dbonds_ordered,-first_false_position)

#     face_boundary_dbonds = face_dbonds_ordered[is_interior_dbonds]

#     dbonds_number = len(face_boundary_dbonds)

#     position_dbond_id = np.where(face_boundary_dbonds == edge_id)[0][0]
    

#     position_bool = (
#             (position_dbond_id == dbonds_number - 1) |
#             (position_dbond_id == 0)
#                     )
    
#     if position_bool:
#         boundary_to_interior_t1_rearrangement(
#             tissue,edge_id,scale_factor=scale_factor)
    
#     return

def pure_boundary_t1_rearrangement(tissue,edge_id,scale_factor=0.1):
      
    boundary_to_interior_t1_rearrangement(
            tissue,edge_id,scale_factor=scale_factor)
          
    return

def collapse_edge(tissue,edge_id,on_midpoint=True):
    """TODO: Add documentation.

    Args:
        tissue ([type]): [description]
        edge_id ([type]): [description]
        on_midpoint (bool, optional): [description]. Defaults to True.
    """
    vert_IDs = tissue.edge_df.loc[edge_id,['v_out_id','v_in_id']]
    
    is_interior = tissue.vert_df.loc[vert_IDs,'is_interior'].values

    end_points_positions = tissue.vert_df.loc[vert_IDs,['x','y']].values
    edge_mid_point = np.mean(end_points_positions, axis=0)

    # conj_dbond = tissue.edge_df.loc[edge_id]['conj_dbond']
    conj_dbond = tissue.edge_df.loc[edge_id,'conj_dbond']
    if conj_dbond == -1:
        dbonds = np.array([edge_id])
    else:
        dbonds = np.array([edge_id, conj_dbond])

    
    dbonds_faces = tissue.edge_df.loc[dbonds,'dbond_face'].values
    tissue.face_df.loc[dbonds_faces,'num_sides'] -= 1

    l_dbonds = tissue.edge_df.loc[dbonds,'left_dbond'].values
    r_dbonds = tissue.edge_df.loc[dbonds,'right_dbond'].values

    dbonds_list = [r_dbonds]
    left_dbonds_list = [l_dbonds]
    rewire_left_and_right_edges(tissue,dbonds_list,left_dbonds_list)

    vert_ID_stay, vert_ID_drop = np.sort(vert_IDs)

    tissue.vert_df.drop([vert_ID_drop],inplace=True)
    tissue.edge_df.drop(dbonds,inplace=True)
    

    # TODO: Make sure this block of code is robust enough that we can avoid the reindexing. 

    in_edge_group = tissue.edge_df[tissue.edge_df['v_in_id'].isin([vert_ID_drop])].index
    out_edge_group = tissue.edge_df[tissue.edge_df['v_out_id'].isin([vert_ID_drop])].index

    tissue.edge_df.loc[out_edge_group,'v_out_id'] = vert_ID_stay
    tissue.edge_df.loc[in_edge_group,'v_in_id'] = vert_ID_stay

    for _, (bond_id,face_id) in enumerate(zip(dbonds, dbonds_faces)):
        dbond_list = tissue.face_dbonds.loc[face_id]
        tissue.face_dbonds.loc[face_id] = \
                    delete_integer_from_nparray(dbond_list,bond_id)

    if on_midpoint:
        tissue.vert_df.loc[vert_ID_stay,['x','y']] = edge_mid_point

    if False in is_interior:
        tissue.vert_df.loc[vert_ID_stay,'is_interior'] = False
        
    # We look for two sides faces which must be removed.
    # sub_face_df = tissue.face_df.loc[dbonds_faces].copy()
    
    # sub_face_df = tissue.face_df.loc[dbonds_faces]
    # two_sided_faces = sub_face_df.loc[sub_face_df['num_sides'] == 2,'id'].values
    
    
    two_sided_faces = tissue.face_df.loc[tissue.face_df['num_sides'] == 2,
                                         'id'].values   
    
    if len(two_sided_faces) > 0:
        for two_face_id in two_sided_faces:
            delete_two_sided_face(tissue,two_face_id)
            remove_two_fold_internal_vertices(tissue)
        
        # dbonds_faces = np.setdiff1d(dbonds_faces,two_sided_faces)
        
        
    # We get the faces sharing vert_ID_stay and update their basic geometry.
    
    # edges_in = tissue.edge_df[tissue.edge_df['v_in_id'] == vert_ID_stay]
    # edges_in = tissue.edge_df[tissue.edge_df['v_in_id'] == vert_ID_stay].copy()
    # vert_faces = edges_in['dbond_face'].values
    
    vert_faces = tissue.edge_df.loc[tissue.edge_df['v_in_id'] == vert_ID_stay
                                    ,'dbond_face'].values 
    
    tissue.update_tissue_geometry(face_list=vert_faces)
    
    
    return

def delete_two_sided_face(tissue,face_id):
    # sub_edge_df = tissue.edge_df[tissue.edge_df['dbond_face'] == face_id]
    
    # f_dbonds = sub_edge_df['id'].values
    # c_dbonds = sub_edge_df['conj_dbond'].values
    
    # f_dbonds = tissue.edge_df.loc[
    #                 tissue.edge_df['dbond_face'] == face_id,'id'].values
    
    # c_dbonds = tissue.edge_df.loc[
    #             tissue.edge_df['dbond_face'] == face_id,'conj_dbond'].values
    
    f_dbonds = tissue.face_dbonds.loc[face_id]
    c_dbonds = tissue.edge_df.loc[f_dbonds,'conj_dbond'].values
       
    
    if -1 not in c_dbonds:
        tissue.edge_df.loc[c_dbonds,'conj_dbond'] = np.flip(c_dbonds)
    elif c_dbonds[0] != -1:
        tissue.edge_df.loc[c_dbonds[0],'conj_dbond'] = -1
        tissue.edge_df.loc[c_dbonds[0],'is_interior'] = False
    else:
        tissue.edge_df.loc[c_dbonds[1],'conj_dbond'] = -1
        tissue.edge_df.loc[c_dbonds[1],'is_interior'] = False
    
    tissue.edge_df.drop(f_dbonds,inplace=True)
    tissue.face_df.drop([face_id],inplace=True)
    tissue.face_dbonds.drop([face_id],inplace=True)
    
    return

def delete_boundary_face(tissue,face_id):
    
    # sub_edge_df = tissue.edge_df[tissue.edge_df['dbond_face'] == face_id]
    # c_dbonds = sub_edge_df['conj_dbond'].values
    
    f_dbonds = tissue.face_dbonds.loc[face_id]
    c_dbonds = tissue.edge_df.loc[f_dbonds,'conj_dbond'].values
    
    c_bonds_reduced = np.setdiff1d(c_dbonds,[-1])
    
    verts_face = tissue.edge_df.loc[f_dbonds,'v_out_id'].values
    
    redundant_vertices = \
            tissue.edge_df.loc[c_bonds_reduced,['v_out_id','v_in_id']]
        
    redundant_vertices = redundant_vertices.values.flatten()
                    
    vert_ID_drop = np.setdiff1d(verts_face,redundant_vertices)
    
    tissue.edge_df.loc[c_bonds_reduced,'conj_dbond'] = -1
    tissue.edge_df.loc[c_bonds_reduced,'is_interior'] = False
    
    vert_indices = \
        tissue.edge_df.loc[c_bonds_reduced,['v_out_id','v_in_id']].values
        
    vert_indices = vert_indices.flatten()
    vert_indices = np.unique(vert_indices)
    
    tissue.vert_df.loc[vert_indices,'is_interior'] = False
          
            
    tissue.vert_df.drop(vert_ID_drop,inplace=True)
    tissue.edge_df.drop(f_dbonds,inplace=True)
    tissue.face_df.drop([face_id],inplace=True)
    tissue.face_dbonds.drop([face_id],inplace=True)
    
    return

def delete_cell(tissue,face_id,remove_boundary=True):
    """ TODO: Add documentation.
        
        FIXME: This is still not working well for boundary cells. In that
        case we must be careful since some edges will be at the boundary.
        We can still simply remove the cell without touching the neighboring
        cells or remove it and collapse the edges to the centroid.
    Args:
        tissue ([type]): [description]
        face_id ([type]): [description]
    """

    centroid = tissue.face_df.loc[face_id,['x','y']]
    # sub_edge_df = tissue.edge_df[tissue.edge_df['dbond_face'] == face_id]
      
    # bond_pairs = sub_edge_df[['id','conj_dbond']].values
    
    # c_dbonds = bond_pairs[:,1]
    
    f_dbonds = tissue.face_dbonds.loc[face_id]
    c_dbonds = tissue.edge_df.loc[f_dbonds,'conj_dbond'].values
    
    bond_pairs = np.dstack((f_dbonds,c_dbonds))
    
    
    # if -1 in c_dbonds:
    #     warnings.warn(
    #         f"The cell with ID: {face_id} is located at the boundary."
    #         f"Boundary cells are not implemented yet.")
    #     return
    
    boundary_bool = -1 in c_dbonds and remove_boundary
    if boundary_bool:
        delete_boundary_face(tissue,face_id)       
        return

    # We rewire the edges of the neighboring cells.
    left_right_bond_pairs = \
                    tissue.edge_df.loc[c_dbonds,['left_dbond','right_dbond']]

    right_dbonds_list = left_right_bond_pairs['right_dbond'].values
    left_dbonds_list = left_right_bond_pairs['left_dbond'].values
    rewire_left_and_right_edges(tissue,right_dbonds_list,left_dbonds_list)  

    # We only keep one vertex which is now shared by the meighboring cells.
    # vert_IDs = sub_edge_df[['v_out_id','v_in_id']].values.flatten()
    vert_IDs = tissue.edge_df.loc[
                        f_dbonds,['v_out_id','v_in_id']].values.flatten()
    
    vert_IDs = np.sort(np.unique(vert_IDs))   
    vert_ID_stay, vert_ID_drop = vert_IDs[0], vert_IDs[1:]
    
        
    bond_list = bond_pairs.flatten()
    bond_list = np.concatenate([bond_list,left_dbonds_list,right_dbonds_list])
    
    for v_id in vert_ID_drop:
        dbonds_in = tissue.edge_df.loc[tissue.edge_df['v_in_id'] == v_id,'id']
        dbonds_out = tissue.edge_df.loc[tissue.edge_df['v_out_id'] == v_id,'id']
        
        dbonds_in_extra = np.setdiff1d(dbonds_in,bond_list)
        dbonds_out_extra = np.setdiff1d(dbonds_out,bond_list)
        
        if len(dbonds_in_extra) > 0:
            tissue.edge_df.loc[dbonds_in_extra,'v_in_id'] = vert_ID_stay
        
        if len(dbonds_out_extra) > 0:
            tissue.edge_df.loc[dbonds_out_extra,'v_out_id'] = vert_ID_stay

    tissue.edge_df.loc[left_dbonds_list,'v_out_id'] = vert_ID_stay
    tissue.edge_df.loc[right_dbonds_list,'v_in_id'] = vert_ID_stay

    # We move the remanining vertex to the centroid of the deleted cell.
    tissue.vert_df.loc[vert_ID_stay,['x','y']] = centroid

    # The neighboring faces lose one side after the cell deletion.
    dbonds_faces = tissue.edge_df.loc[c_dbonds,'dbond_face'].values
    tissue.face_df.loc[dbonds_faces,'num_sides'] -= 1

    # We remove all the other vertices, the cell edges, and the cell itself.
    tissue.vert_df.drop(vert_ID_drop,inplace=True)
    tissue.edge_df.drop(bond_pairs.flatten(),inplace=True)
    tissue.face_df.drop([face_id],inplace=True)
    tissue.face_dbonds.drop([face_id],inplace=True)

    # We finally remove the c_bonds from the edge list of each of
    # the surviving neighboring cells.
    for _, (bond_id,neigh_face_id) in enumerate(zip(c_dbonds, dbonds_faces)):
        dbond_list = tissue.face_dbonds.loc[neigh_face_id]
        tissue.face_dbonds.loc[neigh_face_id] = \
                    delete_integer_from_nparray(dbond_list,bond_id)   

    # We look for two sides faces which must be removed.
    # sub_face_df = tissue.face_df.loc[dbonds_faces]
    # sub_face_df = tissue.face_df.loc[dbonds_faces].copy()
    # two_sided_faces = sub_face_df[sub_face_df['num_sides'] == 2].index.values
    
    two_sided_faces = tissue.face_df.loc[tissue.face_df['num_sides'] == 2,
                                         'id'].values
    
    if len(two_sided_faces) > 0:
        for two_face_id in two_sided_faces:
            delete_two_sided_face(tissue,two_face_id)
            remove_two_fold_internal_vertices(tissue)
        
        dbonds_faces = np.setdiff1d(dbonds_faces,two_sided_faces)
           
    # print(dbonds_faces)
    tissue.update_tissue_geometry(face_list=dbonds_faces)
    
    return

# def split_vertex(tissue,vert_id,scale_factor = 1.0,random_face=True,
#                  preferred_face=None):
def split_vertex(tissue,vert_id,scale_factor = 0.1,preferred_face='random',
                 vector_shift=[],return_vert_ids=False):
    
    is_interior =  tissue.vert_df.loc[vert_id,'is_interior']
    
    if not is_interior:
        warnings.warn(
            f"The vertex with ID: {vert_id} is located at the boundary."
            f"Boundary vertices are not implemented yet.")
        return

    # The ID's of one new vertex and two new directed edges are defined.
    new_vert_id = tissue.num_vertices
    new_dbond_id_1 = tissue.num_edges
    new_dbond_id_2 = tissue.num_edges + 1

    tissue.num_vertices += 1
    tissue.num_edges += 2

    # Given vert_id we first get the directed edges going into it.
    # and out of vert_id as well as the faces sharing vert_id. 
    
    edges_in = \
        tissue.edge_df.loc[tissue.edge_df['v_in_id'] == vert_id,'id'].values
        
    # edges_out = \
    #     tissue.edge_df.loc[tissue.edge_df['v_out_id'] == vert_id,'id'].values
    edges_out = \
        tissue.edge_df.loc[edges_in,'left_dbond'].values
    
    # vert_faces = \
    #     tissue.edge_df.loc[tissue.edge_df['v_in_id'] == vert_id,
    #                        'dbond_face'].values
    vert_faces = \
        tissue.edge_df.loc[edges_in,'dbond_face'].values
    
 
    # A particular face will be pulled away from the vertex. 
    # if random_face:
    #     face_to_pull = np.random.randint(len(vert_faces))
    # else:
    #     face_to_pull = 0
    
    if preferred_face == 'random':
        face_to_pull = np.random.randint(len(vert_faces))
        pulled_face_id = vert_faces[face_to_pull]
    else:
        pulled_face_id = preferred_face
        face_to_pull = np.argwhere(vert_faces == pulled_face_id)
        face_to_pull = face_to_pull[0][0]
               
    # These are the faces and directed edges involved in the vertex splitting.
    # pulled_face_id = vert_faces[face_to_pull]
    dbond_in, dbond_out = edges_in[face_to_pull], edges_out[face_to_pull]

    c_dbond_in = tissue.edge_df.loc[dbond_in,'conj_dbond']
    c_dbond_out = tissue.edge_df.loc[dbond_out,'conj_dbond']

    cr_dbond_in = tissue.edge_df.loc[c_dbond_in,'right_dbond']
    cl_dbond_out = tissue.edge_df.loc[c_dbond_out,'left_dbond']

    c_face_dbond_in = tissue.edge_df.loc[c_dbond_in,'dbond_face']
    c_face_dbond_out = tissue.edge_df.loc[c_dbond_out,'dbond_face']
    
    # We get the position of the new vertex and it to tissue topology.
    if list(vector_shift) == []:
        v_pos, bond_length = vertex_to_centroid(tissue,vert_id,pulled_face_id,
                               scale_factor=scale_factor,return_distance=True)
    else:
        v_pos, bond_length = displace_vertex(tissue,vert_id,
                        vector_shift=vector_shift,return_distance=True)
    
    add_vertex_to_df(tissue,new_vert_id,v_pos)
    
    # We add the two new edges to the tissue topology.
    add_dir_edge_to_df(tissue,new_dbond_id_1,vert_id,new_vert_id,
                    c_dbond_in,cr_dbond_in,new_dbond_id_2,
                    c_face_dbond_in,length=bond_length)

    add_dir_edge_to_df(tissue,new_dbond_id_2,new_vert_id,vert_id,
                    cl_dbond_out,c_dbond_out,new_dbond_id_1,
                    c_face_dbond_out,length=bond_length)
    
    # Once the two new edges and new vertex have been created we
    # rewire the topology locally.
    tissue.edge_df.loc[dbond_in,'v_in_id'] = new_vert_id
    tissue.edge_df.loc[dbond_out,'v_out_id'] = new_vert_id
    
    tissue.edge_df.loc[c_dbond_in,'v_out_id'] = new_vert_id
    tissue.edge_df.loc[c_dbond_out,'v_in_id'] = new_vert_id

    tissue.edge_df.loc[cr_dbond_in,'left_dbond'] = new_dbond_id_1
    tissue.edge_df.loc[c_dbond_in,'right_dbond'] = new_dbond_id_1

    tissue.edge_df.loc[c_dbond_out,'left_dbond'] = new_dbond_id_2
    tissue.edge_df.loc[cl_dbond_out,'right_dbond'] = new_dbond_id_2
      
    # We finally adjust the edge list of the two faces that gained an
    # additional side after creating the two new directed edges.
    face_list = [c_face_dbond_in,c_face_dbond_out]

    tissue.face_df.loc[face_list,'num_sides'] += 1
    get_face_ordered_dbonds(tissue,face_list=face_list)
    
    face_list = [pulled_face_id,c_face_dbond_in,c_face_dbond_out]
    
    # tissue.update_areas_perimeters_centroids(face_list=face_list)
    
    tissue.update_tissue_geometry(face_list=face_list)
    
    if return_vert_ids:
        return [vert_id,new_vert_id]
    else:  
        return

def add_triangular_face_at_boundary(tissue,vert_id,face_id_return=False):
    
        
    edge_out_boundary = \
        tissue.edge_df.loc[(tissue.edge_df['is_interior'] == False) & 
                            (tissue.edge_df['v_out_id'] == vert_id),
                                'id'].values[0]
        
    edge_in_boundary = \
        tissue.edge_df.loc[(tissue.edge_df['is_interior'] == False) & 
                            (tissue.edge_df['v_in_id'] == vert_id),
                                'id'].values[0]
    
    
    ghost_dbond_in_id = tissue.num_edges
    ghost_dbond_out_id = tissue.num_edges + 1
    ghost_dbond_boundary_id = tissue.num_edges + 2
    
    ghost_face_id = tissue.num_faces
    
    tissue.num_edges += 3
    tissue.num_faces += 1
      
    tissue.vert_df.loc[vert_id,'is_interior'] = True
    
    tissue.edge_df.loc[edge_out_boundary,'is_interior'] = True
    tissue.edge_df.loc[edge_in_boundary,'is_interior'] = True
    
    tissue.edge_df.loc[edge_out_boundary,'conj_dbond'] = ghost_dbond_in_id
    tissue.edge_df.loc[edge_in_boundary,'conj_dbond'] = ghost_dbond_out_id
    
    edge_out_length = tissue.edge_df.loc[edge_out_boundary,'length']
    edge_in_length = tissue.edge_df.loc[edge_in_boundary,'length']
    
    vert_ids_edge_out = \
        tissue.edge_df.loc[edge_out_boundary,['v_out_id','v_in_id']].values
        
    vert_ids_edge_in = \
        tissue.edge_df.loc[edge_in_boundary,['v_out_id','v_in_id']].values
    
 
    add_dir_edge_to_df(tissue,ghost_dbond_in_id,
                            vert_ids_edge_out[1],vert_ids_edge_out[0],
                            ghost_dbond_out_id,ghost_dbond_boundary_id,
                            edge_out_boundary,ghost_face_id,
                            length=edge_out_length,
                            line_tension=0.0)
    
    add_dir_edge_to_df(tissue,ghost_dbond_out_id,
                            vert_ids_edge_in[1],vert_ids_edge_in[0],
                            ghost_dbond_boundary_id,ghost_dbond_in_id,
                            edge_in_boundary,ghost_face_id,
                            length=edge_in_length,
                            line_tension=0.0)
    
    pos_out = tissue.vert_df.loc[vert_ids_edge_in[0],['x','y']].values
    pos_in = tissue.vert_df.loc[vert_ids_edge_out[1],['x','y']].values
    pos_vert_id = tissue.vert_df.loc[vert_id,['x','y']].values
    
    length_ghost_dbond_boundary = np.linalg.norm(pos_out-pos_in)
    
    add_dir_edge_to_df(tissue,ghost_dbond_boundary_id,
                            vert_ids_edge_in[0],vert_ids_edge_out[1],
                            ghost_dbond_in_id,ghost_dbond_out_id,
                            -1,ghost_face_id,is_interior=False,
                            length=length_ghost_dbond_boundary,
                            line_tension=0.0)
    
    vert_new_face = np.array([pos_out,pos_in,pos_vert_id],dtype=float)
    centroid_area = pol_centroid(vert_new_face, return_area=True)
    perim_new_cell = pol_perimeter(vert_new_face)
    
    add_face_to_df(tissue,ghost_face_id,centroid_area[0],centroid_area[1],3,
                   centroid_area[2],perim_new_cell,
                   contractility=0.0,
                   active=0)
    
    ghost_dbond_list = \
            [ghost_dbond_in_id,ghost_dbond_out_id,ghost_dbond_boundary_id]
            
    tissue.face_dbonds.loc[ghost_face_id] = ghost_dbond_list
    
    if face_id_return:
        return ghost_face_id
    else:
        return
    

def split_boundary_vertex(tissue,vert_id,scale_factor = 0.1,
                            preferred_face='random',
                            vector_shift=[]):
    
    ghost_face_id = \
        add_triangular_face_at_boundary(tissue,vert_id,face_id_return=True)
    
    if preferred_face == 'ghost':
        pulled_face = ghost_face_id
    else:
        pulled_face = preferred_face 
    
    dbond_vert_ids = split_vertex(tissue,vert_id,
                        scale_factor=scale_factor,
                        preferred_face=pulled_face,
                        vector_shift=vector_shift,
                        return_vert_ids=True)
    
    delete_cell(tissue,ghost_face_id)
    
    recalculate_vertex_topology(tissue,dbond_vert_ids[0])
    recalculate_vertex_topology(tissue,dbond_vert_ids[1])
    
    return 

def resolve_high_order_vertex(tissue,vert_id,scale_factor = 0.1,
                                preferred_face='stable'):
    
    is_interior = tissue.vert_df.loc[vert_id,'is_interior']
    
    if preferred_face == 'stable':
        pulled_face_id, vector_shift = vertex_stability(tissue,vert_id)       
    else:
        pulled_face_id = 'random'
        vector_shift=[]       
    
    if is_interior:
        split_vertex(tissue,vert_id,
                        scale_factor=scale_factor,
                        preferred_face=pulled_face_id,
                        vector_shift=vector_shift)
    else:
        if pulled_face_id not in tissue.face_df['id']:
            pulled_face_id = 'ghost'
        
        split_boundary_vertex(tissue,vert_id,
                        scale_factor=scale_factor,
                        preferred_face=pulled_face_id,
                        vector_shift=vector_shift)
              
    
    return
# def add_vertex_to_edge(tissue,edge_id,v_pos=[]):
def add_vertex_to_edge(tissue,edge_id,v_pos=[],v_offset=0.0):

    v_indices = tissue.edge_df.loc[edge_id,['v_out_id','v_in_id']].values
    if v_pos != []:
        new_vert_pos = v_pos
    else:
        new_vert_pos = tissue.vert_df.loc[v_indices,['x','y']].mean().values

   
    # edge_length = tissue.edge_df.loc[edge_id,'length']
    # new_vert_pos += edge_length*v_offset*np.random.rand(len(new_vert_pos))
    # new_vert_pos += edge_length*v_offset*np.ones(len(new_vert_pos))
        
    new_vert_id = tissue.num_vertices
    tissue.num_vertices += 1

    conj_dbond = tissue.edge_df.loc[edge_id,'conj_dbond']
    if conj_dbond == -1:
        dbonds = np.array([edge_id])
        new_dbond_id = [tissue.num_edges]
        tissue.num_edges += 1   
        add_vertex_to_df(tissue,new_vert_id,new_vert_pos,
                            is_interior=False)
    else:
        dbonds = np.array([edge_id, conj_dbond])
        new_dbond_id = [tissue.num_edges,tissue.num_edges + 1]
        tissue.num_edges += 2  
        add_vertex_to_df(tissue,new_vert_id,new_vert_pos)
        
    l_dbonds = tissue.edge_df.loc[dbonds,'left_dbond'].values
    dbonds_faces = tissue.edge_df.loc[dbonds,'dbond_face'].values

    for i, (dbond,new_dbond,l_dbond,dbond_face) in \
                    enumerate(zip(dbonds,new_dbond_id,l_dbonds,dbonds_faces)):
        
        v_in = tissue.edge_df.at[dbond,'v_in_id']
        tissue.edge_df.loc[dbond,'v_in_id'] = new_vert_id
        tissue.edge_df.loc[dbond,'left_dbond'] = new_dbond
        tissue.edge_df.loc[l_dbond,'right_dbond'] = new_dbond
        
        if tissue.edge_df.loc[dbond,'is_interior']:
            c_bond = tissue.edge_df.loc[dbond,'conj_dbond']
            add_dir_edge_to_df(tissue,new_dbond,new_vert_id,v_in,
                        l_dbond,dbond,c_bond,dbond_face)
            j = (i+1)%2
            tissue.edge_df.loc[dbond,'conj_dbond'] = new_dbond_id[j]
        else:
            add_dir_edge_to_df(tissue,new_dbond,new_vert_id,v_in,
                        l_dbond,dbond,-1,dbond_face,is_interior=False)

    tissue.face_df.loc[dbonds_faces,'num_sides'] += 1
    get_face_ordered_dbonds(tissue,face_list=dbonds_faces)
    
    return

def get_division_edges(tissue,face_id,axis_ang = None,return_angle=True):

    if axis_ang == None:
        rand_angle = np.random.random() * pi
        axis_angles = [rand_angle,rand_angle + pi, rand_angle - pi]
    else:
        axis_angles = [axis_ang, axis_ang + pi, axis_ang - pi]

    face_centroid = tissue.face_df.loc[face_id,['x','y']]
    dbonds_list = tissue.face_dbonds.loc[face_id]

    v_out = tissue.edge_df.loc[dbonds_list,'v_out_id'].values
    v_in = tissue.edge_df.loc[dbonds_list,'v_in_id'].values

    v_min_pos = tissue.vert_df.loc[v_out,['x','y']] - face_centroid
    v_max_pos = tissue.vert_df.loc[v_in,['x','y']] - face_centroid

    v_min_pos, v_max_pos = v_min_pos.values, v_max_pos.values

    bond_indices = []
   
    for i, (v_min,v_max) in enumerate(zip(v_min_pos,v_max_pos)):
        ang_min = positive_quadrant_angle(v_min)
        ang_max = positive_quadrant_angle(v_max)
        
        if ang_min > ang_max:
            ang_min -= 2 * pi
            
        bool_flag = any((axis_ang > ang_min and axis_ang < ang_max)
                            for axis_ang in axis_angles)
        if bool_flag:
            bond_indices += [i]
       
    if return_angle:
        return bond_indices, axis_angles[0]
    else:
        return bond_indices

def get_intersection_points_for_division(tissue,face_id,dbonds_list,
                                                        axis_ang=None):
    if axis_ang == None:
        angle = np.random.random() * pi
    else:
        angle = axis_ang

    face_centroid = tissue.face_df.loc[face_id,['x','y']].values
    
    v_out = tissue.edge_df.loc[dbonds_list,'v_out_id'].values
    v_in = tissue.edge_df.loc[dbonds_list,'v_in_id'].values
    
    v_out = tissue.vert_df.loc[v_out,['x','y']].values
    v_in = tissue.vert_df.loc[v_in,['x','y']].values
    
    end_points_pairs = np.array(list(zip(v_out,v_in)))
    
    intersection_points = line_axis_intersection_point(end_points_pairs,
                                    origin=face_centroid,axis_angle=angle)
    
    return intersection_points
    

# def divide_cell(tissue,face_id,division_angle=None,v_offset=0.0):
def divide_cell(tissue,face_id,division_angle=None,delete_tri_cell=False,
                                        through_centroid=True,v_offset=0.0):
    
    if division_angle == None:
        bond_pos, angle = get_division_edges(tissue,face_id)
    else:
        bond_pos = get_division_edges(tissue,face_id,
                            axis_ang=division_angle,return_angle=False)
        angle = division_angle

    bonds_list = tissue.face_dbonds.loc[face_id]

    dbonds_sides_old = np.take(bonds_list,bond_pos)
    
    new_vert_id = [tissue.num_vertices, tissue.num_vertices + 1]
    
    topology_check = tissue.edge_df.loc[dbonds_sides_old,'is_interior'].values
    
    if topology_check[0]:
        dbonds_sides_new = [tissue.num_edges, tissue.num_edges + 2]
    else:
        dbonds_sides_new = [tissue.num_edges, tissue.num_edges + 1]
           
    if through_centroid:      
        intersection_points = get_intersection_points_for_division(tissue,
                    face_id,dbonds_sides_old,axis_ang=angle)
        
        add_vertex_to_edge(tissue,dbonds_sides_old[0],
                        v_pos=intersection_points[0],v_offset=v_offset)
        add_vertex_to_edge(tissue,dbonds_sides_old[1],
                        v_pos=intersection_points[1],v_offset=v_offset)
        
    else: 
        add_vertex_to_edge(tissue,dbonds_sides_old[0],v_offset=v_offset)
        add_vertex_to_edge(tissue,dbonds_sides_old[1],v_offset=v_offset)

    bonds_list_new = tissue.face_dbonds.loc[face_id]

    idx1 = np.array(np.where(bonds_list_new == dbonds_sides_new[0]))
    idx2 = np.array(np.where(bonds_list_new == dbonds_sides_new[1]))

    new_bond_pos = [idx1.item(), idx2.item()]
    bonds_list_new = np.roll(bonds_list_new,-new_bond_pos[0])

    bonds_list_face_1 = bonds_list_new[0:new_bond_pos[1] - new_bond_pos[0]]
    bonds_list_face_2 = bonds_list_new[new_bond_pos[1] - new_bond_pos[0]:]

    dbonds_division = [tissue.num_edges, tissue.num_edges + 1]

    add_dir_edge_to_df(tissue,dbonds_division[0],new_vert_id[1],
                       new_vert_id[0],dbonds_sides_new[0],
                       dbonds_sides_old[1],dbonds_division[1],face_id)

    new_face_id = tissue.num_faces
    tissue.edge_df.loc[bonds_list_face_2,'dbond_face'] = new_face_id


    add_dir_edge_to_df(tissue,dbonds_division[1],new_vert_id[0],
                       new_vert_id[1],dbonds_sides_new[1],
                       dbonds_sides_old[0],dbonds_division[0],new_face_id)

    tissue.num_edges += 2

    tissue.edge_df.loc[dbonds_sides_old,'left_dbond'] = \
                                                np.roll(dbonds_division,-1)
                                                
    tissue.edge_df.loc[dbonds_sides_new,'right_dbond'] = \
                                                dbonds_division

    bonds_list_face_1 = np.append(bonds_list_face_1,dbonds_division[0])
    bonds_list_face_2 = np.append(bonds_list_face_2,dbonds_division[1])

    tissue.face_dbonds.loc[face_id] = bonds_list_face_1

    num_sides_new = [len(bonds_list_face_1),len(bonds_list_face_2)]

    tissue.face_df.at[face_id,'num_sides'] = num_sides_new[0]
 
    tissue.update_tissue_geometry(face_list=[face_id])
 
    vert_new_face = find_face_vertices(tissue,bonds_list_face_2,
                                                        single_face=True)
    centroid_area = pol_centroid(vert_new_face, return_area=True)
    perim_new_cell = pol_perimeter(vert_new_face)

    tissue.num_faces += 1
    
    add_face_to_df(tissue,new_face_id,centroid_area[0],centroid_area[1],
                num_sides_new[1],centroid_area[2],perim_new_cell,
                    A_0=1.0,P_0=0.0,contractility=0.04,
                    active=1,mother=face_id)

    tissue.face_dbonds.loc[new_face_id] = bonds_list_face_2
    
    if delete_tri_cell:
        if tissue.face_df.loc[face_id,'num_sides'] < 4:
            delete_cell(tissue,face_id)
            
        if tissue.face_df.loc[new_face_id,'num_sides'] < 4:
            delete_cell(tissue,new_face_id)
            
    
    return

def remove_two_fold_internal_vertices(tissue):
    
    internal_vertices = \
        tissue.vert_df.loc[tissue.vert_df['is_interior'].values,'id'].values
    
    nearest_neighbors_num = np.array(
                    [len(tissue.edge_df[tissue.edge_df['v_in_id'] == v_id])
                    for v_id in internal_vertices])
    
    two_fold_vertices = np.argwhere(nearest_neighbors_num == 2)
    
    two_fold_vertices = internal_vertices[two_fold_vertices].flatten()
    
    if len(two_fold_vertices) > 0:
        for v_id in two_fold_vertices:
            dbonds_in = tissue.edge_df.loc[
                tissue.edge_df['v_in_id'] == v_id,'id'].values
            
            dbonds_out = tissue.edge_df.loc[
                tissue.edge_df['v_out_id'] == v_id,'id'].values
            
            l_dbonds = tissue.edge_df.loc[
                            dbonds_out,'left_dbond'].values
            r_dbonds = tissue.edge_df.loc[
                            dbonds_out,'right_dbond'].values
            
            # print(dbonds_in,dbonds_out)
            # print(l_dbonds,r_dbonds)

            dbonds_in_v_out_id = tissue.edge_df.loc[
                            dbonds_in,'v_out_id'].values
            
            # print(dbonds_in_v_out_id)
            
            tissue.edge_df.loc[l_dbonds,'right_dbond'] = r_dbonds
            tissue.edge_df.loc[r_dbonds,'left_dbond'] = l_dbonds

            tissue.edge_df.loc[dbonds_in,'conj_dbond'] = dbonds_in[::-1]
            tissue.edge_df.loc[dbonds_in,'v_in_id'] = dbonds_in_v_out_id[::-1]
            
            face_list = tissue.edge_df.loc[dbonds_out,'dbond_face'].values
            
            tissue.face_df.loc[face_list,'num_sides'] -= 1
            # print(face_list)
            
            
            tissue.vert_df.drop(v_id,inplace=True)
            tissue.edge_df.drop(dbonds_out,inplace=True)
            
            get_face_ordered_dbonds(tissue,face_list=face_list)           
            
        tissue.update_tissue_geometry()
    
    
    return

def round_off_boundary(tissue):
    boundary_faces_IDs =  tissue.edge_df.loc[
            tissue.edge_df['is_interior'] == False,'dbond_face'].unique()
    
    dbonds_boundary_faces = tissue.face_dbonds.loc[boundary_faces_IDs].values

    boundary_dbonds_IDs = tissue.edge_df.loc[
            tissue.edge_df['is_interior'] == False,'id'].values
    
    
    for face_dbonds in dbonds_boundary_faces:
        face_dbonds_ordered = np.array(face_dbonds)
        is_interior_dbonds = np.isin(face_dbonds,boundary_dbonds_IDs)
        first_false_position = np.where(is_interior_dbonds == False)[0][0]
    
        is_interior_dbonds = np.roll(is_interior_dbonds,-first_false_position)
        face_dbonds_ordered = np.roll(face_dbonds_ordered,-first_false_position)
        
        face_boundary_dbonds = face_dbonds_ordered[is_interior_dbonds]
        removable_dbonds = face_boundary_dbonds[0:-1]
        
        for dbond_id in removable_dbonds:
            collapse_edge(tissue,dbond_id)
            
    tissue.update_tissue_geometry(face_list=boundary_faces_IDs)
    
    return
#   ------------------------------------------------------------------------------------------------------
#   This are a bunch of deprecated functions that we don't need to use.

# def reconnect_T1_faces(tissue,dbonds,cl_dbonds):

#     # The faces of dbonds change after the T1 transition.
#     dbonds_faces = tissue.edge_df.iloc[dbonds]['dbond_face'].values
#     cl_dbonds_faces = tissue.edge_df.iloc[cl_dbonds]['dbond_face'].values
#     tissue.edge_df.loc[dbonds,'dbond_face'] = cl_dbonds_faces
    
#     face_dbond_pairs = zip(dbonds_faces,dbonds)  
#     face_dbonds_new = [
#             delete_integer_from_nparray(tissue.face_dbonds[face_id],edge_id)
#             for face_id,edge_id in face_dbond_pairs]

#     tissue.face_dbonds.iloc[dbonds_faces] = face_dbonds_new

#     face_cl_dbonds = tissue.face_dbonds.iloc[cl_dbonds_faces].values
#     len_cl_faces = tissue.face_df.iloc[cl_dbonds_faces]['num_sides'].values

#     cl_dbonds_positions = [
#                         np.argwhere(dbonds_list == dbond_ID)[0,0] for
#                         dbonds_list, dbond_ID in
#                         zip(face_cl_dbonds,cl_dbonds)] 

#     dbonds_positions = [(dbond_pos+1)%len_face for
#                         dbond_pos,len_face in
#                         zip(cl_dbonds_positions,len_cl_faces)]

#     face_cl_dbonds_new = [np.insert(dbonds_list,dbond_pos,dbond_ID) for
#                         dbonds_list,dbond_pos,dbond_ID in
#                         zip(face_cl_dbonds,dbonds_positions,dbonds)]

#     tissue.face_dbonds.iloc[cl_dbonds_faces] = face_cl_dbonds_new

#     tissue.face_df.loc[dbonds_faces,'num_sides'] -= 1
#     tissue.face_df.loc[cl_dbonds_faces,'num_sides'] += 1

#     return


""" NOTE: Don't delete the commented block from below. It works fine but it requires
        that we reindex many vertices and edges. The idea of assigning a unique id is to
        avoid having to reindex. This of course means that we need to keep working with 
        .loc instead of .iloc given the label oriented nature of the former one.
"""
    
# for _, (bond_id,face_id) in enumerate(zip(c_dbonds, dbonds_faces)):
#     dbond_list = tissue.face_dbonds.loc[face_id]
#     tissue.face_dbonds.loc[face_id] = \
#                 delete_integer_from_nparray(dbond_list,bond_id)   

# tissue.vert_df.reset_index(drop=True,inplace=True)
# tissue.edge_df.reset_index(drop=True,inplace=True)

# old_vertex_IDs = tissue.vert_df['id'].values
# vertex_dict = list_to_range_dict(old_vertex_IDs)
# vertex_dict[vert_ID_drop] = vert_ID_stay

# remaining_edge_IDs = tissue.edge_df['id'].values
# edge_dict = list_to_range_dict(remaining_edge_IDs)

# edge_dict[-1] = -1
# for dbond in dbonds:
#     edge_dict[dbond] = -1 

# tissue.vert_df['id'] = tissue.vert_df['id'].map(vertex_dict)

# cols = ['v_out_id','v_in_id']
# for col in cols:
#     tissue.edge_df[col] = tissue.edge_df[col].map(vertex_dict)

# cols = ['id','left_dbond','right_dbond','conj_dbond']
# for col in cols:
#     tissue.edge_df[col] = tissue.edge_df[col].map(edge_dict)

# tissue.face_dbonds = \
#     tissue.face_dbonds.apply(dict_to_array,dictionary=edge_dict)

# get_face_ordered_dbonds(tissue, face_list=dbonds_faces)

# remaining_vertex_IDs = tissue.vert_df['id'].values
# vertex_dict = list_to_range_dict(old_vertex_IDs)
# vertex_dict[vert_ID_drop] = vert_ID_stay

# tissue.update_areas_perimeters_centroids(face_list=dbonds_faces)