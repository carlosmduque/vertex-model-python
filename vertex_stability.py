import numpy as np
import pandas as pd

from  basic_topology import split_vertex,split_boundary_vertex

from utilities import find_ordered_vertex_dbonds
# from basic_topology import add_triangular_face_at_boundary,delete_cell

from energetics import vertex_stability


# def vertex_stability(tissue,vert_id,gamma_0=0.12,
#                      new_dbond_length='default'):
    
#     is_interior = tissue.vert_df.loc[vert_id,'is_interior']
    
#     if not is_interior:
#         ghost_face_id = \
#         add_triangular_face_at_boundary(tissue,vert_id,face_id_return=True)
    
#     dbonds_out, dbonds_in = find_ordered_vertex_dbonds(tissue,vert_id)
    
    
#     dbond_faces = tissue.edge_df.loc[dbonds_out,'dbond_face'].values
        
#     preferred_area_deviation = \
#             0.5 * tissue.face_df.loc[dbond_faces,'active'] * \
#                     (tissue.face_df.loc[dbond_faces,'area'] - 
#                         tissue.face_df.loc[dbond_faces,'A_0'])  
    
#     preferred_perimeter_deviation = \
#             tissue.face_df.loc[dbond_faces,'contractility'] * \
#                     (tissue.face_df.loc[dbond_faces,'perimeter'] - 
#                         tissue.face_df.loc[dbond_faces,'P_0'])
    
                        
#     preferred_area_deviation = preferred_area_deviation.values
#     preferred_perimeter_deviation = preferred_perimeter_deviation.values
                
#     dbond_out_tension = tissue.edge_df.loc[dbonds_out,'line_tension'].values
#     dbond_in_tension = tissue.edge_df.loc[dbonds_in,'line_tension'].values
                        
#     area_pressures = preferred_area_deviation - \
#                                 np.roll(preferred_area_deviation,1)
                
#     perimeter_sum = preferred_perimeter_deviation + \
#                                 np.roll(preferred_perimeter_deviation,1)
                                
#     total_fold_tension = 0.5*(dbond_out_tension + np.roll(dbond_in_tension,1))
    
#     verts_in = tissue.edge_df.loc[dbonds_out,'v_in_id'].values
#     verts_in_pos = tissue.vert_df.loc[verts_in,['x','y']].values

#     vert_out_pos = tissue.vert_df.loc[[vert_id],['x','y']].values
#     side_vectors_out = verts_in_pos - vert_out_pos
    
#     rotated_side_vectors = \
#                 np.dstack((-side_vectors_out[:,1],side_vectors_out[:,0]))[0]
    
#     dbonds_out_lengths = tissue.edge_df.loc[dbonds_out,'length'].values

#     unit_side_vectors_out = side_vectors_out/dbonds_out_lengths[:,None]
    
#     fold_forces = area_pressures[:,None]*rotated_side_vectors + \
#                     (total_fold_tension + perimeter_sum)[:,None] * \
#                         unit_side_vectors_out
    
                        
#     pulling_forces_per_face = fold_forces + np.roll(fold_forces,-1,axis=0)
    
#     rolled_dbonds_out = np.roll(dbonds_out,-1)   
#     dbond_out_pairs = np.dstack((dbonds_out,rolled_dbonds_out))[0]
    
#     remaining_pulling_dbonds_out = [
#         np.setdiff1d(dbonds_out,dbond_pair,assume_unique=True) 
#                                     for dbond_pair in dbond_out_pairs]
    
#     force_components = pd.DataFrame(
#         fold_forces,index=dbonds_out,columns=['fx','fy'])
    
#     opposite_pulling_forces = np.array([
#                 force_components.loc[dbond_group,['fx','fy']].values      
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
