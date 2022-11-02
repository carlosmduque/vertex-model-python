import os
import sys

parent_path = os.path.dirname(os.path.realpath(__file__))

parent_path = os.path.abspath(os.path.join(parent_path, os.pardir))

sys.path.append(parent_path)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd


from utilities import *
from tissue_generation import *

from basic_topology import *
from topology_triggers import *
from vertex_stability import *

from shape_analysis import *

from energetics import *

from tissue import Tissue

from minimizer import TissueMinimizer

from data_management import *
from dbond_sampling_functions import *


from copy import deepcopy

from IPython.display import clear_output

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def event_manager(
        tissue,t1_events,dt_save,dt_rectangle,
        dt_round_off_boundary,dt_vertex_topology,
        save_path,clear_screen=True):
    
    if (t1_events % dt_save) == 0:     
#        file_name = 'tissue_active_t1_' + str(t1_events) + '.h5'
        file_name = 'tissue_active_t1.h5'
        file_path = os.path.join(save_path, file_name)
        
        save_datasets(file_path, tissue, mode='w')            
        
    # if (t1_events % dt_round_off_boundary) == 0:  
    #     round_off_boundary(tissue)
        
    if (t1_events % dt_vertex_topology) == 0:     
        set_vertex_topology(tissue)
        
    if (t1_events % dt_rectangle) == 0:
        if clear_screen:
            clear_output(wait=True)
            
        l_min, l_max, aspect_ratio, angle = collect_bounding_rectangle_data(tissue)

        print(f"The major axis angle is : {angle}")
        print(f"The aspect ratio is : {aspect_ratio}")
        
        
def minimization_protocol(tissue,number_of_cells,gtol=1e-4,minimization_method='CG',
                         return_number_collapsed_faces_and_edges=True):
    
    number_of_collapsed_faces = 0
    number_of_collapsed_edges = 0
    
    quasi_solver = TissueMinimizer(tissue,tissue_energy,tissue_gradient,
                                   gtol=gtol,minimization_method=minimization_method)
    
    quasi_solver.find_energy_min()
    number_of_collapsed_faces += quasi_solver.number_of_collapsed_faces
    number_of_collapsed_edges += quasi_solver.number_of_collapsed_edges
    
    resolve_high_order_vertices(tissue)
    deleted_cells_number = number_of_cells - len(tissue.face_df)

    if deleted_cells_number > 0:
        divide_large_cells(tissue,deleted_cells_number)
        
    quasi_solver = TissueMinimizer(tissue,tissue_energy,tissue_gradient,
                                   gtol=gtol,minimization_method=minimization_method)
    quasi_solver.find_energy_min()
    number_of_collapsed_faces += quasi_solver.number_of_collapsed_faces
    number_of_collapsed_edges += quasi_solver.number_of_collapsed_edges
    
    resolve_high_order_vertices(tissue)
    quasi_solver = TissueMinimizer(tissue,tissue_energy,tissue_gradient,
                                   gtol=gtol,minimization_method=minimization_method)
    quasi_solver.find_energy_min()
    number_of_collapsed_faces += quasi_solver.number_of_collapsed_faces
    number_of_collapsed_edges += quasi_solver.number_of_collapsed_edges
    
    success = quasi_solver.res.success
    
    if return_number_collapsed_faces_and_edges:
        return success, number_of_collapsed_faces, number_of_collapsed_edges
    else:
        return success

def generate_data_list(tissue,number_of_collapsed_edges=0,number_of_collapsed_faces=0):
    
    box_data = collect_bounding_rectangle_data(tissue)
    
    total_area, total_perimeter = tissue.tissue_area_perimeter()
    mean_geometrical_data, std_geometrical_data = \
                    tissue.mean_geometrical_quantities(return_std=True)  
        
            
    tissue_ener = tissue_energy(tissue)
    
    num_boundary_cells = len(tissue.edge_df[tissue.edge_df['is_interior'] == False])
    
    data_list = [*box_data, total_area, total_perimeter,
                 *mean_geometrical_data,*std_geometrical_data,tissue_ener,
                 num_boundary_cells,number_of_collapsed_edges, number_of_collapsed_faces]
    
    
    data_list = np.array(data_list)
     
        
    return data_list

def generate_file_path(path_tail,Ncells,mu,sigma,ratio,seed):
    
    save_path = path_tail + 'Ncells_' + str(Ncells) 
    # save_path = save_path + '_mu_' + str(mu) + '_sigma_' + str(sigma) + '_ratio_' + str(ratio) + '/'
    # save_path = save_path + str(seed) + '/'
    save_path += '_mu_' + str(mu) + '_sigma_' + str(sigma) + '_ratio_' + str(ratio) + '/'
    save_path += str(seed) + '/'
    
    
    import os
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path))
    
    return save_path

def generate_tissue_disk(N_disk):
    (vertex_positions, dir_edges_end_points, hexagonal_cells) = \
                                            generate_hexagonal_disk(N_disk)


    face_dbonds, face_per_directed_edge, \
        left_edge_index, right_edge_index, conjugate_edge_index = \
        generate_directed_edges_topology(dir_edges_end_points,hexagonal_cells)

    vert_df, edge_df, face_df \
        = generate_dataframes(vertex_positions,dir_edges_end_points,
                        face_per_directed_edge,left_edge_index,
                        right_edge_index,conjugate_edge_index,hexagonal_cells)
        
    tissue  = Tissue(vert_df, edge_df, face_df, face_dbonds)

    quasi_solver = TissueMinimizer(tissue,tissue_energy,tissue_gradient)
    quasi_solver.find_energy_min()

    resolve_high_order_vertices(tissue)

    round_off_boundary(tissue)

    quasi_solver = TissueMinimizer(tissue,tissue_energy,tissue_gradient)
    quasi_solver.find_energy_min()
    
    return tissue

def propagate_tissue(tissue,rng,save_path,number_of_cells,check_point_if_possible=True,
                     mu=0.0,sigma=0.9,
                     dt_save=10,dt_rectangle=10,dt_round_off_boundary=10,dt_vertex_topology=10,
                     gtol=1e-4,
                     anisotropic_isotropic_ratio=0.5,number_of_t1_events_final=100):
    
    number_of_collapsed_faces, number_of_collapsed_edges = 0, 0
    df_file_path = os.path.join(save_path, 'data_df.h5')
    
    if (check_point_if_possible and os.path.isfile(df_file_path)):
        logging_df = pd.read_hdf(df_file_path, 'logging_df')
        
        # tissue_path = save_path + 'tissue_active_t1.h5'      
        tissue_path = os.path.join(save_path, 'tissue_active_t1.h5')
        tissue = load_tissue(tissue_path)
        
        collect_data_counter =  dt_save*logging_df.index[-1] + 1
        
    else:
    
        column_names = ['l_min','l_max','aspect_ratio','box_angle',
                'tissue_area','tissue_perimeter','mean_cell_area','mean_cell_perimeter','mean_dbond_length',
                'std_cell_area','std_cell_perimeter','std_dbond_length','tissue_energy','num_boundary_cells',
                'num_collapsed_edges','num_collapsed_faces']


        logging_df = pd.DataFrame(columns=column_names,dtype=float)
      
    
        data_list = generate_data_list(tissue,number_of_collapsed_faces,
                                   number_of_collapsed_edges)
    
        logging_data(logging_df,data_list,df_file_path)
        collect_data_counter = 0
    
    while (collect_data_counter < number_of_t1_events_final):

        try:
            
            tissue_checkpoint = deepcopy(tissue)
            
            round_off_boundary(tissue)

            event_manager(tissue,collect_data_counter,dt_save,dt_rectangle,
                dt_round_off_boundary,dt_vertex_topology,save_path)

            dbond_id, dbond_t1_type = directed_isotropic_bond(tissue,rng,mu=mu,sigma=sigma,
                                    anisotropic_isotropic_ratio=anisotropic_isotropic_ratio)

            print(f'Event number {collect_data_counter}, dbond_ID = {dbond_id}, T1 event type = {dbond_t1_type}')

            generic_t1_rearrangement(tissue,dbond_id,dbond_t1_type)

            min_success, num_collapsed_faces, num_collapsed_edges = \
                                minimization_protocol(tissue,number_of_cells,gtol=gtol)


            success = (min_success and (not high_order_vertices_remaining(tissue))
                        and (not single_interior_dbond_cells(tissue))
                        and (not non_adjacent_boundary_dbonds_per_cell(tissue))
                        and (not tissue_self_intersect(tissue))
                        and (not floating_faces(tissue)))


            if (not success):
                print('Reverting topology!')
                tissue = deepcopy(tissue_checkpoint)
            else:

                collect_data_counter += 1
                number_of_collapsed_faces += num_collapsed_faces
                number_of_collapsed_edges += num_collapsed_edges

                if (collect_data_counter % dt_save) == 0:

                    data_list = generate_data_list(tissue,
                            number_of_collapsed_faces,number_of_collapsed_edges)
            
                    logging_data(logging_df,data_list,df_file_path)


                    number_of_collapsed_faces, number_of_collapsed_edges = 0, 0
                                
            
        except:
            print('An error occurred! reverting topology...')
            tissue = deepcopy(tissue_checkpoint)
            set_vertex_topology(tissue)
            
    return
    
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------