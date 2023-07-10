# import pandas as pd
import numpy as np
import warnings
import math

# from basic_topology import (collapse_edge,
#                             simple_t1_rearrangement,rotate_T1_vertices,
#                             delete_cell,resolve_high_order_vertex)

from .basic_topology import (collapse_edge,
                            simple_t1_rearrangement,rotate_T1_vertices,
                            delete_cell,divide_cell,boundary_to_interior_t1_rearrangement,
                            pure_boundary_t1_rearrangement)


from .vertex_stability import resolve_high_order_vertex
from .basic_geometry import dbond_axis_angle


# def find_collapsable_edges(tissue):
#     length_cutoff = tissue.t1_cutoff
    
#     interior_short_dbonds = \
#                 tissue.edge_df.loc[(tissue.edge_df['length'] < length_cutoff) &
#                                 (tissue.edge_df['is_interior'] == True),'id']
                
#     interior_short_c_dbonds = \
#                     tissue.edge_df.loc[interior_short_dbonds,'conj_dbond']
                
#     boundary_short_dbonds = \
#                 tissue.edge_df.loc[(tissue.edge_df['length'] < length_cutoff) &
#                             (tissue.edge_df['is_interior'] == False),'id']
                   
    
#     dbonds_face_IDs = tissue.edge_df.loc[interior_short_dbonds,'dbond_face']
#     c_dbonds_face_IDs = \
#                     tissue.edge_df.loc[interior_short_c_dbonds,'dbond_face']
    
#     dbonds_face_num_sides = tissue.face_df.loc[dbonds_face_IDs,'num_sides']
#     c_dbonds_face_num_sides = \
#                     tissue.face_df.loc[c_dbonds_face_IDs,'num_sides']
                    
#     is_collapsable = (dbonds_face_num_sides.values > 3) & \
#                                     (c_dbonds_face_num_sides.values > 3)
                                    
#     interior_short_dbonds = interior_short_dbonds[is_collapsable].values
    
#     dbonds_face_IDs = tissue.edge_df.loc[boundary_short_dbonds,'dbond_face']
#     dbonds_face_num_sides = tissue.face_df.loc[dbonds_face_IDs,'num_sides']
#     is_collapsable = (dbonds_face_num_sides.values > 3)
    
#     boundary_short_dbonds = boundary_short_dbonds[is_collapsable].values
    
#     short_dbonds = np.concatenate((interior_short_dbonds,
#                                    boundary_short_dbonds)).astype(int)
    
#     return short_dbonds

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

# def t1_interior_to_boundary_able_edges(tissue):
#     v_id_out,v_id_in = tissue.edge_df['v_out_id'],tissue.edge_df['v_in_id']
    
#     v_id_out_interior = tissue.vert_df.loc[v_id_out,'is_interior'].values
#     v_id_in_interior = tissue.vert_df.loc[v_id_in,'is_interior'].values
    

#     t1_edges_interior_to_boundary = (
#             (v_id_out_interior & np.logical_not(v_id_in_interior))|
#             (np.logical_not(v_id_out_interior) & v_id_in_interior)
#             )

#     t1_edges_IDs = \
#                 tissue.edge_df.loc[t1_edges_interior_to_boundary,'id'].values
    
#     return t1_edges_IDs

def t1_able_edges(tissue,t1_type='interior_t1'):
    
    v_id_out,v_id_in = tissue.edge_df['v_out_id'],tissue.edge_df['v_in_id']

    v_id_out_interior = tissue.vert_df.loc[v_id_out,'is_interior'].values
    v_id_in_interior = tissue.vert_df.loc[v_id_in,'is_interior'].values
    
    if t1_type == 'interior_t1':
        valid_T1_edges = v_id_out_interior & v_id_in_interior
    elif t1_type == 'interior_to_boundary_t1':
        valid_T1_edges = (
            (v_id_out_interior & np.logical_not(v_id_in_interior))|
            (np.logical_not(v_id_out_interior) & v_id_in_interior)
            )
    elif t1_type == 'boundary_t1':
        t1_edges_IDs = tissue.edge_df.loc[
            tissue.edge_df['is_interior'] == False,'id'].values

        t1_edges_face_IDs = \
            tissue.edge_df.loc[t1_edges_IDs,'dbond_face'].values

        t1_edges_face_criterion = \
            (tissue.face_df.loc[t1_edges_face_IDs,'num_sides'] > 3).values

        t1_edges_IDs = t1_edges_IDs[t1_edges_face_criterion]
        
        l_t1_edges = tissue.edge_df.loc[t1_edges_IDs,'left_dbond'].values
        r_t1_edges = tissue.edge_df.loc[t1_edges_IDs,'right_dbond'].values

        l_is_interior = tissue.edge_df.loc[l_t1_edges,'is_interior'].values
        r_dis_interior = tissue.edge_df.loc[r_t1_edges,'is_interior'].values

        valid_boundary_t1_edges = l_is_interior | r_dis_interior

        t1_edges_IDs = t1_edges_IDs[valid_boundary_t1_edges]
        
        return t1_edges_IDs
    else:
        warnings.warn(
            f"T1 rearrengement type not recognized")
        
        return

    
    t1_edges_IDs = tissue.edge_df.loc[valid_T1_edges,'id'].values
    
    t1_conj_edges_IDs = tissue.edge_df.loc[t1_edges_IDs,'conj_dbond']
    
    t1_edges_face_IDs = tissue.edge_df.loc[t1_edges_IDs,'dbond_face']
    t1_conj_edges_face_IDs = \
                    tissue.edge_df.loc[t1_conj_edges_IDs,'dbond_face']
    
    t1_edges_face_num_sides = \
                        tissue.face_df.loc[t1_edges_face_IDs,'num_sides']
    t1_conj_edges_face_num_sides = \
                    tissue.face_df.loc[t1_conj_edges_face_IDs,'num_sides']

    t1_edges_face_criterion = (t1_edges_face_num_sides.values > 3) & \
                                    (t1_conj_edges_face_num_sides.values > 3)

    t1_edges_IDs = t1_edges_IDs[t1_edges_face_criterion]
      
    return t1_edges_IDs
    

# def find_collapsable_edges(tissue,dbond_type='all'):
#     length_cutoff = tissue.t1_cutoff
    
#     if dbond_type == 'interior':
#         short_dbonds = \
#             tissue.edge_df.loc[(tissue.edge_df['length'] < length_cutoff) &
#                             (tissue.edge_df['is_interior'] == True),'id']  
                     
#     elif dbond_type == 'boundary':
#         short_dbonds = \
#             tissue.edge_df.loc[(tissue.edge_df['length'] < length_cutoff) &
#                             (tissue.edge_df['is_interior'] == False),'id']
        
#     else:
#         short_dbonds = \
#             tissue.edge_df.loc[(tissue.edge_df['length'] < length_cutoff),'id']                                         
    
#     short_dbonds = short_dbonds.values
    
#     return short_dbonds

def find_collapsable_edges(tissue,dbond_type='all'):
    
    if dbond_type == 'interior':
        length_cutoff = tissue.t1_cutoff
        short_dbonds = \
            tissue.edge_df.loc[(tissue.edge_df['length'] < length_cutoff) &
                            (tissue.edge_df['is_interior'] == True),'id']  
                     
    elif dbond_type == 'boundary':
        length_cutoff = tissue.t1_boundary_cutoff
        short_dbonds = \
            tissue.edge_df.loc[(tissue.edge_df['length'] < length_cutoff) &
                            (tissue.edge_df['is_interior'] == False),'id']
        
    else:
        t1_cutoff_array = np.full(len(tissue.edge_df), tissue.t1_cutoff)
        position_boundary_dbonds = (
                        tissue.edge_df['is_interior'] == False).values
        
        t1_cutoff_array[position_boundary_dbonds] = tissue.t1_boundary_cutoff
        
        
        short_dbonds = \
            tissue.edge_df.loc[
                            (tissue.edge_df['length'] < t1_cutoff_array),'id']                                         
    
    short_dbonds = short_dbonds.values
    
    return short_dbonds

# def find_splittable_vertices(tissue,interior=True):
 
#     size_groups = tissue.edge_df.groupby('v_out_id').size()
    
#     if interior:
#         vert_IDs = size_groups[size_groups > 3].index.values
#         vert_IDs = \
#             vert_IDs[tissue.vert_df.loc[vert_IDs,'is_interior'] == True]
#     else:      
#         vert_IDs = size_groups[size_groups > 2].index.values
#         vert_IDs = \
#             vert_IDs[tissue.vert_df.loc[vert_IDs,'is_interior'] == False]
             
#     return vert_IDs

def find_splittable_vertices(tissue,vertex_group='all'):
 
    size_groups = tissue.edge_df.groupby('v_out_id').size()
    
    if vertex_group == 'interior':
        vert_IDs = size_groups[size_groups > 3].index.values
        vert_IDs = \
            vert_IDs[tissue.vert_df.loc[vert_IDs,'is_interior'] == True]
    elif vertex_group == 'boundary':      
        vert_IDs = size_groups[size_groups > 2].index.values
        vert_IDs = \
            vert_IDs[tissue.vert_df.loc[vert_IDs,'is_interior'] == False]
    else:
        vert_IDs_interior = size_groups[size_groups > 3].index.values
        vert_IDs_boundary = size_groups[size_groups > 2].index.values
        
        vert_IDs_interior = \
            vert_IDs_interior[tissue.vert_df.loc[vert_IDs_interior,
                                                    'is_interior'] == True]
            
        vert_IDs_boundary = \
            vert_IDs_boundary[tissue.vert_df.loc[vert_IDs_boundary,
                                                    'is_interior'] == False]
            
        vert_IDs = np.concatenate((vert_IDs_interior,vert_IDs_boundary))
             
    return vert_IDs

def find_removable_faces(tissue,delete_tri_cell=False):
    
    area_cutoff = tissue.t2_cutoff
    
    if delete_tri_cell:
        face_list = tissue.face_df.loc[(tissue.face_df['area'] < area_cutoff) |
                            (tissue.face_df['num_sides'] < 4),'id']
    else:
        face_list = tissue.face_df.loc[tissue.face_df['area'] < area_cutoff,'id']
    
    face_list = face_list.values
    
    return face_list

def collapse_short_tissue_edges(tissue,return_number_collapsed_edges=False):
    
    shorts_dbonds_list = find_collapsable_edges(tissue)
    
    number_of_collapsed_edges = 0
    
    while len(shorts_dbonds_list) > 0:
        dbond_id = np.random.choice(shorts_dbonds_list)
        
        collapse_edge(tissue,dbond_id)
        shorts_dbonds_list = find_collapsable_edges(tissue)
        print(f"The edge with ID: {dbond_id} was collapsed.")
        number_of_collapsed_edges += 1
    
    if return_number_collapsed_edges:
        return number_of_collapsed_edges
    else:
        return

def t1_able_short_edges(tissue):
    t1_able_ids = t1_able_edges(tissue)
    t1_able_lengths = tissue.edge_df.loc[t1_able_ids,'length']
    length_cutoff = tissue.t1_cutoff
    
    short_t1_edges = t1_able_lengths[t1_able_lengths < length_cutoff].index
    short_t1_edges = short_t1_edges.values
    
    return short_t1_edges
    

def t1_short_edges(tissue):
    
    shorts_dbonds_list = t1_able_short_edges(tissue)
    while len(shorts_dbonds_list) > 0:
        dbond_id = np.random.choice(shorts_dbonds_list)
        
        simple_t1_rearrangement(tissue,dbond_id)
        rotate_T1_vertices(tissue,dbond_id,
                                preferred_length=10*tissue.t1_cutoff)
        
        shorts_dbonds_list = t1_able_short_edges(tissue)
        print(f"The edge with ID: {dbond_id} was T1_ed.")
    
    return

def collapse_small_faces(tissue,return_number_collapsed_faces=False):
    
    removable_faces = find_removable_faces(tissue)
    
    number_of_collapsed_faces = 0
    
    while len(removable_faces) > 0:
        face_id = removable_faces[0]

        delete_cell(tissue,face_id)
        removable_faces = find_removable_faces(tissue)
        print(f"The cell with ID: {face_id} was collapsed.")
        number_of_collapsed_faces += 1
        
    if return_number_collapsed_faces:
        return number_of_collapsed_faces
    else:
        return
        
def resolve_high_order_vertices(tissue):
    
    splittable_vertices = find_splittable_vertices(tissue)
    
    if len(splittable_vertices) > 0:          
        for vert_id in splittable_vertices:
            resolve_high_order_vertex(tissue,vert_id)
    
    return

def divide_large_cells(tissue,division_events):
    
    for i in range(division_events):
        # face_id = np.random.choice(tissue.face_df.index.values)
        
        face_id = tissue.face_df['area'].idxmax()
        print(f"The cell with ID: {face_id} was divided.")
            
        divide_cell(tissue,face_id)
    return

def generic_t1_rearrangement(tissue,dbond_id,dbond_t1_type):

    if dbond_t1_type == 0:
        simple_t1_rearrangement(tissue,dbond_id)
        rotate_T1_vertices(tissue,dbond_id)
    elif dbond_t1_type == 1:
        boundary_to_interior_t1_rearrangement(tissue,dbond_id,scale_factor=0.2)
    else:
        pure_boundary_t1_rearrangement(tissue,dbond_id,scale_factor=0.1)
        
    return
    

