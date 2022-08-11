import numpy as np
import pandas as pd

from  basic_topology import *

from utilities import find_ordered_vertex_dbonds
# from basic_topology import add_triangular_face_at_boundary,delete_cell

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
    # geometry_quantities=['area','perimeter','length']
    # tissue.update_tissue_geometry(geometry_quantities=geometry_quantities)
    
    energy = area_energy(tissue) + perimeter_energy(tissue) + \
            bond_tension_energy(tissue)
    
            
    return energy

# def tissue_energy(tissue):
#     geometry_quantities=['area','perimeter','length']
#     tissue.update_tissue_geometry(geometry_quantities=geometry_quantities)
    
#     area_perimeter_energy = \
#                     (tissue.face_df['area']-tissue.face_df['A_0'])**2 + \
#                     tissue.face_df['contractility'] * \
#                     (tissue.face_df['perimeter']-tissue.face_df['P_0'])**2  
    
#     bond_tension_energy = \
#                     tissue.edge_df['line_tension']*tissue.edge_df['length']
    
#     energy = 0.5*(area_perimeter_energy.sum() + bond_tension_energy.sum())
     
            
#     return energy

def tissue_energy_and_update_pos(x_1d,tissue):
    
    vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
    tissue.vert_df[['x','y']] = vert_pos
    
    energy = tissue_energy(tissue)
            
    return energy

def area_gradient(tissue):
    vert_IDs = tissue.edge_df['v_in_id']
    l_dbonds = tissue.edge_df['left_dbond']
    
    v_in_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_in_id'],['x','y']].values
            
    v_out_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_out_id'],['x','y']].values

    v_in_positions_left = \
        tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_in_id'],\
                                                            ['x','y']].values
                                                            
    v_out_positions_left = \
        tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_out_id'],\
                                                            ['x','y']].values
                                                            
    side_vectors = v_in_positions - v_out_positions
    side_vectors_left = v_in_positions_left - v_out_positions_left
    
    side_vectors_sum = side_vectors + side_vectors_left
    
    grad_x, grad_y = side_vectors_sum[:,1], -side_vectors_sum[:,0]
 
    grad_components = np.dstack((grad_x,grad_y))[0]

    dbond_face = tissue.edge_df['dbond_face']
   
    dbond_face_area = tissue.face_df.loc[dbond_face,'area']
    dbond_face_preferred_area = tissue.face_df.loc[dbond_face,'A_0']

    area_prefactors = 0.5*(dbond_face_area - dbond_face_preferred_area).values
    
    grad_components = area_prefactors[:,None] * grad_components
    grad_components = pd.DataFrame(
        grad_components,index=tissue.edge_df.index,columns=['gx','gy'])
    
    grad_components['v_id'] = vert_IDs
   
    grad_components = grad_components.groupby('v_id')[['gx','gy']].sum()
    
    return grad_components

def perimeter_gradient(tissue):
    vert_IDs = tissue.edge_df['v_in_id'].values
    
    # l_dbonds = tissue.edge_df['left_dbond']
    l_dbonds = tissue.edge_df['left_dbond'].values

    
    v_in_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_in_id'],['x','y']].values
            
    v_out_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_out_id'],['x','y']].values

    v_in_positions_left = \
        tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_in_id'],\
                                                            ['x','y']].values
                                                            
    v_out_positions_left = \
        tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_out_id'],\
                                                            ['x','y']].values
                                                            
    side_vectors = v_in_positions - v_out_positions
    side_vectors_left = v_in_positions_left - v_out_positions_left
    
    dbond_lengths = tissue.edge_df['length'].values
    l_dbond_lengths = tissue.edge_df.loc[l_dbonds,'length'].values
    
    side_vectors = side_vectors/dbond_lengths[:,None]
    side_vectors_left = side_vectors_left/l_dbond_lengths[:,None]
    
    grad_components = side_vectors - side_vectors_left
                    
    # dbond_face = tissue.edge_df['dbond_face']
    dbond_face = tissue.edge_df['dbond_face'].values
    
    dbond_face_perimeter = tissue.face_df.loc[dbond_face,'perimeter']
    dbond_face_preferred_perimeter = tissue.face_df.loc[dbond_face,'P_0']
    dbond_face_contractility = tissue.face_df.loc[dbond_face,'contractility']
    
    perimeter_prefactors = (dbond_face_contractility *
            (dbond_face_perimeter - dbond_face_preferred_perimeter)).values
                
    grad_components = perimeter_prefactors[:,None] * grad_components
    grad_components = pd.DataFrame(
        grad_components,index=tissue.edge_df.index,columns=['gx','gy'])
    
    grad_components['v_id'] = vert_IDs
    # grad_components.loc[:,'v_id'] = vert_IDs
    
    grad_components = grad_components.groupby('v_id')[['gx','gy']].sum()
    
    return grad_components

def bond_tension_gradient(tissue):
    vert_IDs = tissue.edge_df['v_in_id'].values
    
    l_dbonds = tissue.edge_df['left_dbond']
    
    v_in_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_in_id'],['x','y']].values
            
    v_out_positions = \
            tissue.vert_df.loc[tissue.edge_df['v_out_id'],['x','y']].values

    v_in_positions_left = \
        tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_in_id'],\
                                                            ['x','y']].values
                                                            
    v_out_positions_left = \
        tissue.vert_df.loc[tissue.edge_df.loc[l_dbonds,'v_out_id'],\
                                                            ['x','y']].values
                                                            
    side_vectors = v_in_positions - v_out_positions
    side_vectors_left = v_in_positions_left - v_out_positions_left
    
    dbond_lengths = tissue.edge_df['length'].values
    # l_dbond_lengths = tissue.edge_df.loc[l_dbonds]['length'].values
    l_dbond_lengths = tissue.edge_df.loc[l_dbonds,'length'].values
    
    dbond_tension = tissue.edge_df['line_tension'].values
    # dbond_tension_left = tissue.edge_df.loc[l_dbonds]['line_tension'].values
    dbond_tension_left = tissue.edge_df.loc[l_dbonds,'line_tension'].values
    
    side_vectors = dbond_tension[:,None]*side_vectors/dbond_lengths[:,None]
    side_vectors_left = \
        dbond_tension_left[:,None]*side_vectors_left/l_dbond_lengths[:,None]
    
    
    grad_components = 0.5*(side_vectors - side_vectors_left)
                    
    grad_components = pd.DataFrame(
        grad_components,index=tissue.edge_df.index,columns=['gx','gy'])
    
    grad_components['v_id'] = vert_IDs
    
    grad_components = grad_components.groupby('v_id')[['gx','gy']].sum()
    
    return grad_components

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
  
    
    # dbond_lengths = tissue.edge_df['length'].values
    # l_dbond_lengths = tissue.edge_df.loc[l_dbonds,'length'].values
    
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
    # grad_components.set_index('v_id')
    
    return grad_components

def tissue_gradient(tissue):
    # geometry_quantities=['area','perimeter','length']
    # tissue.update_tissue_geometry(geometry_quantities=geometry_quantities)
    
    # grad_components = area_gradient(tissue) + perimeter_gradient(tissue) + \
    #         bond_tension_gradient(tissue)
    
    grad_components = vertex_model_2d_gradient(tissue)
    
    # tissue.vert_df[['fx','fy']] = - grad_components
    
    grad_components = grad_components.values.flatten()
            
    return grad_components

def tissue_gradient_and_update_pos(x_1d,tissue):
    # vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
    # tissue.vert_df[['x','y']] = vert_pos
    
    grad_components = tissue_gradient(tissue)
            
    return grad_components

def tissue_energy_mod(x_1d,*args):
    # vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
    # args[0].vert_df[['x','y']] = vert_pos
    
   
    energy = tissue_energy_and_update_pos(x_1d,args[0])
    return energy
    
def tissue_gradient_mod(x_1d,*args):
    # vert_pos = np.reshape(x_1d,(int(len(x_1d)/2),2))
    # args[0].vert_df[['x','y']] = vert_pos
    
    gradient = tissue_gradient_and_update_pos(x_1d,args[0])
    return gradient


def vertex_stability(tissue,vert_id,gamma_0=0.12,
                     new_dbond_length='default'):
    
    is_interior = tissue.vert_df.loc[vert_id,'is_interior']
    
    if not is_interior:
        ghost_face_id = \
        add_triangular_face_at_boundary(tissue,vert_id,face_id_return=True)
    
    dbonds_out, dbonds_in = find_ordered_vertex_dbonds(tissue,vert_id)
    
    
    dbond_faces = tissue.edge_df.loc[dbonds_out,'dbond_face'].values
        
    preferred_area_deviation = \
            0.5 * tissue.face_df.loc[dbond_faces,'active'] * \
                    (tissue.face_df.loc[dbond_faces,'area'] - 
                        tissue.face_df.loc[dbond_faces,'A_0'])  
    
    preferred_perimeter_deviation = \
            tissue.face_df.loc[dbond_faces,'contractility'] * \
                    (tissue.face_df.loc[dbond_faces,'perimeter'] - 
                        tissue.face_df.loc[dbond_faces,'P_0'])
    
                        
    preferred_area_deviation = preferred_area_deviation.values
    preferred_perimeter_deviation = preferred_perimeter_deviation.values
                
    dbond_out_tension = tissue.edge_df.loc[dbonds_out,'line_tension'].values
    dbond_in_tension = tissue.edge_df.loc[dbonds_in,'line_tension'].values
                        
    area_pressures = preferred_area_deviation - \
                                np.roll(preferred_area_deviation,1)
                
    perimeter_sum = preferred_perimeter_deviation + \
                                np.roll(preferred_perimeter_deviation,1)
                                
    total_fold_tension = 0.5*(dbond_out_tension + np.roll(dbond_in_tension,1))
    
    verts_in = tissue.edge_df.loc[dbonds_out,'v_in_id'].values
    verts_in_pos = tissue.vert_df.loc[verts_in,['x','y']].values

    vert_out_pos = tissue.vert_df.loc[[vert_id],['x','y']].values
    side_vectors_out = verts_in_pos - vert_out_pos
    
    rotated_side_vectors = \
                np.dstack((-side_vectors_out[:,1],side_vectors_out[:,0]))[0]
    
    dbonds_out_lengths = tissue.edge_df.loc[dbonds_out,'length'].values

    unit_side_vectors_out = side_vectors_out/dbonds_out_lengths[:,None]
    
    fold_forces = area_pressures[:,None]*rotated_side_vectors + \
                    (total_fold_tension + perimeter_sum)[:,None] * \
                        unit_side_vectors_out
    
                        
    pulling_forces_per_face = fold_forces + np.roll(fold_forces,-1,axis=0)
    
    rolled_dbonds_out = np.roll(dbonds_out,-1)   
    dbond_out_pairs = np.dstack((dbonds_out,rolled_dbonds_out))[0]
    
    remaining_pulling_dbonds_out = [
        np.setdiff1d(dbonds_out,dbond_pair,assume_unique=True) 
                                    for dbond_pair in dbond_out_pairs]
    
    force_components = pd.DataFrame(
        fold_forces,index=dbonds_out,columns=['fx','fy'])
    
    opposite_pulling_forces = np.array([
                force_components.loc[dbond_group,['fx','fy']].values      
                            for dbond_group in remaining_pulling_dbonds_out])
    
    opposite_pulling_forces = np.sum(opposite_pulling_forces,axis=1)
    
    pulling_forces = pulling_forces_per_face - opposite_pulling_forces
    
    pulling_magnitudes = np.linalg.norm(pulling_forces,axis=1)
    
    effective_new_fold_tension = \
                        np.roll(preferred_perimeter_deviation,-1) + \
                        np.roll(preferred_perimeter_deviation,1) + \
                        [gamma_0]
    
    stability_criterion = pulling_magnitudes - 2*effective_new_fold_tension
    
    preferred_index = np.argmax(stability_criterion)
    
    preferred_face = dbond_faces[preferred_index]
    preferred_direction = pulling_forces[preferred_index]
    
    preferred_direction /= np.linalg.norm(preferred_direction)
    
    if new_dbond_length == 'default':
        preferred_direction *= 10*tissue.t1_cutoff
    else:
        preferred_direction *= new_dbond_length
        
    if not is_interior:
        delete_cell(tissue,ghost_face_id)   
    
    return preferred_face,preferred_direction


# def vertex_stability(tissue,vert_id,gamma_0=0.12,
#                      new_dbond_length='default'):
    
#     is_interior = tissue.vert_df.loc[vert_id,'is_interior']
    
#     if not is_interior:
#         ghost_face_id = \
#         add_triangular_face_at_boundary(tissue,vert_id,face_id_return=True)
    
#     dbonds_out, dbonds_in = find_ordered_vertex_dbonds(tissue,vert_id)
    
#     sub_edge_df_out = tissue.edge_df.loc[dbonds_out]
#     sub_edge_df_in = tissue.edge_df.loc[dbonds_in]
    
#     dbond_faces = sub_edge_df_out['dbond_face'].values
    
#     sub_face_df = tissue.face_df.loc[dbond_faces]
    
#     preferred_area_deviation = 0.5 * sub_face_df['active'] * \
#                                 (sub_face_df['area'] - sub_face_df['A_0'])
    
#     preferred_perimeter_deviation = sub_face_df['contractility'] * \
#                             (sub_face_df['perimeter'] - sub_face_df['P_0'])
                        
#     preferred_area_deviation = preferred_area_deviation.values
#     preferred_perimeter_deviation = preferred_perimeter_deviation.values
                
#     dbond_out_tension = sub_edge_df_out['line_tension'].values
#     dbond_in_tension = sub_edge_df_in['line_tension'].values
                        
#     area_pressures = preferred_area_deviation - \
#                                 np.roll(preferred_area_deviation,1)
                
#     perimeter_sum = preferred_perimeter_deviation + \
#                                 np.roll(preferred_perimeter_deviation,1)
                                
#     total_fold_tension = 0.5*(dbond_out_tension + np.roll(dbond_in_tension,1))
    
#     verts_in = sub_edge_df_out['v_in_id'].values
#     verts_in_pos = tissue.vert_df.loc[verts_in,['x','y']].values

#     vert_out_pos = tissue.vert_df.loc[[vert_id],['x','y']].values
#     side_vectors_out = verts_in_pos - vert_out_pos
    
#     rotated_side_vectors = \
#                 np.dstack((-side_vectors_out[:,1],side_vectors_out[:,0]))[0]
    
#     dbonds_out_lengths = sub_edge_df_out['length'].values

#     unit_side_vectors_out = side_vectors_out/dbonds_out_lengths[:,None]
    
#     fold_forces = area_pressures[:,None]*rotated_side_vectors + \
#                     (total_fold_tension + perimeter_sum)[:,None] * \
#                         unit_side_vectors_out
                        
      
#     sub_edge_df_out[['fx','fy']] = fold_forces 
                        
#     pulling_forces_per_face = fold_forces + np.roll(fold_forces,-1,axis=0)
    
#     rolled_dbonds_out = np.roll(dbonds_out,-1)   
#     dbond_out_pairs = np.dstack((dbonds_out,rolled_dbonds_out))[0]
    
#     remaining_pulling_dbonds_out = [
#         np.setdiff1d(dbonds_out,dbond_pair,assume_unique=True) 
#                                     for dbond_pair in dbond_out_pairs]
    
#     opposite_pulling_forces = np.array([
#                 sub_edge_df_out.loc[dbond_group,['fx','fy']].values      
#                             for dbond_group in remaining_pulling_dbonds_out])
    
#     opposite_pulling_forces = np.sum(opposite_pulling_forces,axis=1)
    
#     pulling_forces = pulling_forces_per_face - opposite_pulling_forces
    
#     pulling_magnitudes = np.linalg.norm(pulling_forces,axis=1)
    
#     effective_new_fold_tension = \
#                         np.roll(preferred_perimeter_deviation,-1) + \
#                         np.roll(preferred_perimeter_deviation,1) + \
#                         [gamma_0]
    
#     stability_criterion = pulling_magnitudes - 2*effective_new_fold_tension
    
#     preferred_index = np.argmax(stability_criterion)
    
#     preferred_face = dbond_faces[preferred_index]
#     preferred_direction = pulling_forces[preferred_index]
    
#     preferred_direction /= np.linalg.norm(preferred_direction)
    
#     if new_dbond_length == 'default':
#         preferred_direction *= 10*tissue.t1_cutoff
#     else:
#         preferred_direction *= new_dbond_length
        
#     if not is_interior:
#         delete_cell(tissue,ghost_face_id)   
    
#     return preferred_face,preferred_direction