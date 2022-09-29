import numpy as np
import pandas as pd

from  basic_topology import *

def area_energy(tissue):
    cell_areas = tissue.face_df['area']
    preferred_areas = tissue.face_df['A_0']
    
    energy = (cell_areas-preferred_areas)**2
    
    energy = 0.5 * energy.sum()
    
    return energy

def perimeter_energy(tissue):
    cell_perimeters = tissue.face_df['perimeter']
    preferred_perimeters = tissue.face_df['P_0']
    per_contract = tissue.face_df['contractility']
    
    energy = per_contract * (cell_perimeters-preferred_perimeters)**2
    
    energy = 0.5 * energy.sum()
    
    return energy

def bond_tension_energy(tissue):
    dbond_lengths = tissue.edge_df['length']
    line_tension = tissue.edge_df['line_tension']
    
    energy = line_tension * dbond_lengths
    
    energy = 0.5 * energy.sum()
    
    return energy

def tissue_energy(tissue):
    
    energy = area_energy(tissue) + perimeter_energy(tissue) + \
            bond_tension_energy(tissue)
    
            
    return energy

def vertex_model_2d_gradient(tissue):
    vert_IDs = tissue.edge_df['v_in_id'].values
    
    l_dbonds = tissue.edge_df['left_dbond']

    
    v_in_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_in_id'],['x','y']].values
            
    v_out_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_out_id'],['x','y']].values

    left_v_in_IDs = tissue.edge_df.loc[l_dbonds,'v_in_id']
    left_v_out_IDs = tissue.edge_df.loc[l_dbonds,'v_out_id']
    
    v_in_positions_left = tissue.vert_df.loc[left_v_in_IDs,['x','y']].values
                                                            
    v_out_positions_left = tissue.vert_df.loc[left_v_out_IDs,['x','y']].values
                                                            
    side_vectors = v_in_positions - v_out_positions
    side_vectors_left = v_in_positions_left - v_out_positions_left
    
    side_vectors_sum = side_vectors + side_vectors_left
    
    comp_x, comp_y = side_vectors_sum[:,1], -side_vectors_sum[:,0]
  
    
    dbond_lengths = tissue.edge_df['length'].values + 1e-8
    l_dbond_lengths = tissue.edge_df.loc[l_dbonds,'length'].values + 1e-8
    
    dbond_tension = tissue.edge_df['line_tension'].values
    dbond_tension_left = tissue.edge_df.loc[l_dbonds,'line_tension'].values
    
    side_vectors_normalized = side_vectors/dbond_lengths[:,None]
    side_vectors_normalized_left = side_vectors_left/l_dbond_lengths[:,None]
    
    side_vectors_coupling = dbond_tension[:,None]*side_vectors_normalized
    side_vectors_coupling_left = \
                    dbond_tension_left[:,None]*side_vectors_normalized_left
                    
                    
    dbond_face = tissue.edge_df['dbond_face']
    
    dbond_face_area = tissue.face_df.loc[dbond_face,'area']
    dbond_face_preferred_area = tissue.face_df.loc[dbond_face,'A_0']
    
    dbond_face_perimeter = tissue.face_df.loc[dbond_face,'perimeter']
    dbond_face_preferred_perimeter = tissue.face_df.loc[dbond_face,'P_0']
    dbond_face_contractility = tissue.face_df.loc[dbond_face,'contractility']
    
    area_prefactors = 0.5*(dbond_face_area - dbond_face_preferred_area).values
    
    perimeter_prefactors = (dbond_face_contractility *
            (dbond_face_perimeter - dbond_face_preferred_perimeter)).values
    
    area_grad = np.dstack((comp_x,comp_y))[0]
    
    perimeter_grad = side_vectors_normalized - side_vectors_normalized_left
                
    bond_tension_grad = \
                0.5*(side_vectors_coupling - side_vectors_coupling_left)
                
    total_grad = area_prefactors[:,None] * area_grad + \
                perimeter_prefactors[:,None] * perimeter_grad + \
                    bond_tension_grad
    
                    
    grad_components = pd.DataFrame(
        total_grad,index=tissue.edge_df.index,columns=['gx','gy'])
    
    grad_components['v_id'] = vert_IDs
    
    grad_components = grad_components.groupby('v_id')[['gx','gy']].sum()
    grad_components.index.name = None
    
    return grad_components

def tissue_gradient(tissue):
    
    grad_components = vertex_model_2d_gradient(tissue)
        
    grad_components = grad_components.values.flatten()
            
    return grad_components

# def tissue_energy_and_update_pos(x_1d,tissue):
    
#     vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
#     tissue.vert_df[['x','y']] = vert_pos
    
#     energy = tissue_energy(tissue)
            
#     return energy

# def area_gradient(tissue):
#     vert_IDs = tissue.edge_df['v_in_id']
#     l_dbonds = tissue.edge_df['left_dbond']
    
#     v_in_positions = \
#             tissue.vert_df.loc[tissue.edge_df['v_in_id'],['x','y']].values
            
#     v_out_positions = \
#             tissue.vert_df.loc[tissue.edge_df['v_out_id'],['x','y']].values

#     v_in_positions_left = \
#         tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_in_id'],\
#                                                             ['x','y']].values
                                                            
#     v_out_positions_left = \
#         tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_out_id'],\
#                                                             ['x','y']].values
                                                            
#     side_vectors = v_in_positions - v_out_positions
#     side_vectors_left = v_in_positions_left - v_out_positions_left
    
#     side_vectors_sum = side_vectors + side_vectors_left
    
#     grad_x, grad_y = side_vectors_sum[:,1], -side_vectors_sum[:,0]
 
#     grad_components = np.dstack((grad_x,grad_y))[0]

#     dbond_face = tissue.edge_df['dbond_face']
   
#     dbond_face_area = tissue.face_df.loc[dbond_face,'area']
#     dbond_face_preferred_area = tissue.face_df.loc[dbond_face,'A_0']

#     area_prefactors = 0.5*(dbond_face_area - dbond_face_preferred_area).values
    
#     grad_components = area_prefactors[:,None] * grad_components
#     grad_components = pd.DataFrame(
#         grad_components,index=tissue.edge_df.index,columns=['gx','gy'])
    
#     grad_components['v_id'] = vert_IDs
   
#     grad_components = grad_components.groupby('v_id')[['gx','gy']].sum()
    
#     return grad_components

# def perimeter_gradient(tissue):
#     vert_IDs = tissue.edge_df['v_in_id'].values
    
#     l_dbonds = tissue.edge_df['left_dbond'].values

    
#     v_in_positions = \
#             tissue.vert_df.loc[tissue.edge_df['v_in_id'],['x','y']].values
            
#     v_out_positions = \
#             tissue.vert_df.loc[tissue.edge_df['v_out_id'],['x','y']].values

#     v_in_positions_left = \
#         tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_in_id'],\
#                                                             ['x','y']].values
                                                            
#     v_out_positions_left = \
#         tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_out_id'],\
#                                                             ['x','y']].values
                                                            
#     side_vectors = v_in_positions - v_out_positions
#     side_vectors_left = v_in_positions_left - v_out_positions_left
    
#     dbond_lengths = tissue.edge_df['length'].values
#     l_dbond_lengths = tissue.edge_df.loc[l_dbonds,'length'].values
    
#     side_vectors = side_vectors/dbond_lengths[:,None]
#     side_vectors_left = side_vectors_left/l_dbond_lengths[:,None]
    
#     grad_components = side_vectors - side_vectors_left
                    
#     # dbond_face = tissue.edge_df['dbond_face']
#     dbond_face = tissue.edge_df['dbond_face'].values
    
#     dbond_face_perimeter = tissue.face_df.loc[dbond_face,'perimeter']
#     dbond_face_preferred_perimeter = tissue.face_df.loc[dbond_face,'P_0']
#     dbond_face_contractility = tissue.face_df.loc[dbond_face,'contractility']
    
#     perimeter_prefactors = (dbond_face_contractility *
#             (dbond_face_perimeter - dbond_face_preferred_perimeter)).values
                
#     grad_components = perimeter_prefactors[:,None] * grad_components
#     grad_components = pd.DataFrame(
#         grad_components,index=tissue.edge_df.index,columns=['gx','gy'])
    
#     grad_components['v_id'] = vert_IDs
#     # grad_components.loc[:,'v_id'] = vert_IDs
    
#     grad_components = grad_components.groupby('v_id')[['gx','gy']].sum()
    
#     return grad_components

# def bond_tension_gradient(tissue):
#     vert_IDs = tissue.edge_df['v_in_id'].values
    
#     l_dbonds = tissue.edge_df['left_dbond']
    
#     v_in_positions = \
#             tissue.vert_df.loc[tissue.edge_df['v_in_id'],['x','y']].values
            
#     v_out_positions = \
#             tissue.vert_df.loc[tissue.edge_df['v_out_id'],['x','y']].values

#     v_in_positions_left = \
#         tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_in_id'],\
#                                                             ['x','y']].values
                                                            
#     v_out_positions_left = \
#         tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_out_id'],\
#                                                             ['x','y']].values
                                                            
#     side_vectors = v_in_positions - v_out_positions
#     side_vectors_left = v_in_positions_left - v_out_positions_left
    
#     dbond_lengths = tissue.edge_df['length'].values
#     l_dbond_lengths = tissue.edge_df.loc[l_dbonds,'length'].values
    
#     dbond_tension = tissue.edge_df['line_tension'].values
#     dbond_tension_left = tissue.edge_df.loc[l_dbonds,'line_tension'].values
    
#     side_vectors = dbond_tension[:,None]*side_vectors/dbond_lengths[:,None]
#     side_vectors_left = \
#         dbond_tension_left[:,None]*side_vectors_left/l_dbond_lengths[:,None]
    
    
#     grad_components = 0.5*(side_vectors - side_vectors_left)
                    
#     grad_components = pd.DataFrame(
#         grad_components,index=tissue.edge_df.index,columns=['gx','gy'])
    
#     grad_components['v_id'] = vert_IDs
    
#     grad_components = grad_components.groupby('v_id')[['gx','gy']].sum()
    
#     return grad_components


# def tissue_gradient_and_update_pos(x_1d,tissue):
#     # vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
#     # tissue.vert_df[['x','y']] = vert_pos
    
#     grad_components = tissue_gradient(tissue)
            
#     return grad_components

# def tissue_energy_mod(x_1d,*args):
#     # vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
#     # args[0].vert_df[['x','y']] = vert_pos
    
   
#     energy = tissue_energy_and_update_pos(x_1d,args[0])
#     return energy
    
# def tissue_gradient_mod(x_1d,*args):
#     # vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
#     # args[0].vert_df[['x','y']] = vert_pos
    
#     gradient = tissue_gradient_and_update_pos(x_1d,args[0])
#     return gradient

