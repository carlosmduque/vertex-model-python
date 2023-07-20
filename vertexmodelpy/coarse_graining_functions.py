import numpy as np
from .triangulation import Triangulation
from itertools import chain
from numpy import pi

def elongation_tensor_per_cell(tissue):

    tissue_triangulation = Triangulation(tissue,triangulation_type='subcellular_triangulation')

    final_index = np.cumsum(tissue.face_df['num_sides'].values)
    starting_index = final_index - tissue.face_df['num_sides'].values

    subtriangles_per_cell = [np.arange(i_index,f_index) for i_index,f_index in zip(starting_index,final_index)]
    
    elong_tensor_per_triangle = tissue_triangulation.state_tensor()

    area_weighted_elong_tensors = np.array([triang_area*el_tensor
        for triang_area,el_tensor in
            zip(tissue_triangulation.face_df['area'].values,elong_tensor_per_triangle)])

    elong_tensor_per_cell = np.array([np.sum(area_weighted_elong_tensors[triang_indices],axis=0)/cell_area
        for cell_area,triang_indices in zip(tissue.face_df['area'].values,subtriangles_per_cell)])
    
    return elong_tensor_per_cell

def coarse_grained_tensor(tissue,elong_tensor_per_cell,neighbor_patch_per_cell):
    
    area_weighted_elong_tensors_per_cell = np.array([cell_area*el_tensor
    for cell_area,el_tensor in
        zip(tissue.face_df['area'].values,elong_tensor_per_cell)])
    
    patch_area = [tissue.face_df.loc[patch_indices, 'area'].sum()
                    for patch_indices in neighbor_patch_per_cell]
    
    elong_tensor_per_patch = np.array([np.sum(
                area_weighted_elong_tensors_per_cell[patch_indices],axis=0)/p_area
                    for p_area,patch_indices in zip(patch_area,neighbor_patch_per_cell)])
    
    return elong_tensor_per_patch

def calculate_twophi(elong_tensor):
    q_xx, q_xy = elong_tensor[:,0,0], elong_tensor[:,0,1]

    twophi = np.arctan2(q_xy,q_xx)
    
    return twophi

def find_elongation_unit_vectors(twophi):
    
    cos_two_phi, sin_two_phi = np.cos(0.5*twophi), np.sin(0.5*twophi)

    unit_vectors = np.dstack((cos_two_phi,sin_two_phi))[0]
    
    return unit_vectors

def elongation_lines_cells(tissue,twophi,scale=0.5):
# def elongation_lines_cells(tissue,unit_vectors,scale=0.5):

    cell_centers = tissue.face_df[['x','y']].values
    
    unit_vectors = find_elongation_unit_vectors(twophi)
    
    # cos_two_phi, sin_two_phi = np.cos(0.5*twophi), np.sin(0.5*twophi)

    # unit_vectors = np.dstack((cos_two_phi,sin_two_phi))[0]

    end_point_1 = cell_centers + scale*0.5*unit_vectors
    end_point_2 = cell_centers - scale*0.5*unit_vectors

    lines = list(zip(end_point_1,end_point_2))
    
    return lines

def find_boundary_cells_of_patch(tissue,neighbor_list,return_boundary_dbonds=False):

    dbonds_per_cell_in_patch = \
            [list(chain(*tissue.face_dbonds.loc[patch_indices].values))
                    for patch_indices in neighbor_list]

    conj_dbonds_per_patch = \
            [tissue.edge_df.loc[patch_dbonds,'conj_dbond'].values
                    for patch_dbonds in dbonds_per_cell_in_patch]

    boundary_dbonds_per_patch = [np.setdiff1d(dbonds_list,conj_dbond_list)
                    for dbonds_list,conj_dbond_list in
                    zip(dbonds_per_cell_in_patch,conj_dbonds_per_patch)]

    boundary_faces_per_patch = \
            [tissue.edge_df.loc[dbonds_list,'dbond_face'].unique()
                    for dbonds_list in boundary_dbonds_per_patch]
            
    if not return_boundary_dbonds:
        return boundary_faces_per_patch
    else:
        return boundary_faces_per_patch,boundary_dbonds_per_patch
    
def ordered_boundary_contour_per_patch(g,boundary_faces_per_patch):
    
    sub_g_contour = [g.subgraph(face_IDs, implementation='create_from_scratch')
                    for face_IDs in boundary_faces_per_patch]
    
    subg_tid_to_g_tid = [np.array(subg.vs['id'])
                         for subg in sub_g_contour]
    
    patches_contour_ordered_triangles_id =[[]
                            for i in range(len(boundary_faces_per_patch))]
    
    node_coordinates = np.dstack((g.vs['x'], g.vs['y']))[0]
    
    # for patch_id, contour_unsorted_faces_id in enumerate(boundary_faces_per_patch):
    for patch_id, (contour_unsorted_faces_id,subg,id_pairing) in \
            enumerate(zip(boundary_faces_per_patch,sub_g_contour,subg_tid_to_g_tid)):
                
        for start_id in range(len(subg.vs)):
        
            # Find neighbors of start node.
            start_neighbors_subg_id = np.unique(subg.neighbors(start_id))
            # If only one neighbor, continue to next start node.
            if len(start_neighbors_subg_id) < 2:
                continue
            
            # Find all paths between 2 consecutive nodes.
            paths     = subg.get_all_simple_paths(start_id, start_neighbors_subg_id[0])
            paths_len = np.array([len(p) for p in paths])
            long_paths_len = paths_len[ paths_len > len(contour_unsorted_faces_id)/2 ]
            
            if len(long_paths_len) == 0:
                continue
            
            long_paths = [path for path in paths if len(path) > len(contour_unsorted_faces_id)/2]
            contour_faces_ordered_id = id_pairing[ long_paths[long_paths_len.argmin()] ]
            break
        
        
         ### Determine wheter triangles are ordered in clockwisse or counterclockwise direction.
        start, neighbor1 = contour_faces_ordered_id[0], contour_faces_ordered_id[1]
                
        start_to_neighbor1_vector = node_coordinates[neighbor1] - node_coordinates[start]

        # #Define vector from start node to patch centroid. contour_unsorted_triangles_id
        patch_centroid = np.mean(node_coordinates[contour_unsorted_faces_id], axis=0)
        start_to_centroid_vector = patch_centroid - node_coordinates[start]

        # # Find orientation of the contour from start node to first neighbor node.
        # # If contour_orientation = 1, the step start -> neighbor1 is in counterclockwise direction,
        # # This means that the longest path we will find (around the whole contour) is in clockwise direction.
        start_centroid_normal = np.cross(start_to_neighbor1_vector, start_to_centroid_vector)
        
        contour_orientation   = np.sign(np.sum(start_centroid_normal * [0,0,1]))
            
        # In case triangle ordering is clockwise, flip ordered triangle array.
        
        if contour_orientation < 0:
            contour_faces_ordered_id = np.flip(contour_faces_ordered_id)
            
        patches_contour_ordered_triangles_id[patch_id] = contour_faces_ordered_id
        
    return patches_contour_ordered_triangles_id

def find_rotation_angle_contour(ordered_contours_per_patch,twophi):
    
    elongation_unit_vectors = find_elongation_unit_vectors(twophi)
    
    rot_angle_patch = []
    
    for contour_path in ordered_contours_per_patch:
        
        path_unit_vectors = elongation_unit_vectors[contour_path]
                
        path_unit_vectors_shifted = np.roll(path_unit_vectors,-1,axis=0)
        
        consecutive_unit_normals_cross = np.array([np.cross(*vector_pair)
            
            for vector_pair in zip(path_unit_vectors,path_unit_vectors_shifted)])  
        
        consecutive_unit_normals_cross_flipped = -consecutive_unit_normals_cross
        
        
        angle_difference = np.array([np.dot(*vector_pair)
            
            for vector_pair in zip(path_unit_vectors,path_unit_vectors_shifted)])
        
        angle_difference = np.arccos(angle_difference)
        angle_difference_flipped = pi - angle_difference
        
        
        angle_diff_bool = angle_difference < angle_difference_flipped
        
        consecutive_unit_normals_cross = np.array([
            (consecutive_unit_normals_cross[i] if ang_bool
                else consecutive_unit_normals_cross_flipped[i])
            
            for i, ang_bool in enumerate(angle_diff_bool)])
        
        angle_difference = np.minimum(angle_difference,angle_difference_flipped)
        
        consecutive_nodes_rotation_sign = np.sign(consecutive_unit_normals_cross)
        
        rot_angle = np.dot(angle_difference,consecutive_nodes_rotation_sign)/(2*pi)
        
        rot_angle_patch += [rot_angle]
    
    rot_angle_patch = np.array(rot_angle_patch)
    
    return rot_angle_patch