# import pandas as pd
import numpy as np

import scipy as sp

from scipy.spatial import ConvexHull, cKDTree
from topology_triggers import find_splittable_vertices


def bounding_rectangle(boundary_vertices,return_corners=False):
    
    convex_hull = ConvexHull(boundary_vertices)
    convex_hull_vertices = boundary_vertices[convex_hull.vertices]

    convex_hull_side_vectors = np.roll(convex_hull_vertices,-1,axis=0) - convex_hull_vertices

    x_coords, y_coords = convex_hull_side_vectors[:,0], convex_hull_side_vectors[:,1]

    phi_hull = np.arctan2(y_coords,x_coords)
    cos_phi, sin_phi = np.cos(-phi_hull),np.sin(-phi_hull)
    rotation_matrices = np.reshape(np.dstack((cos_phi,-sin_phi,sin_phi,cos_phi)),(-1,2,2))

    translated_vertices = np.array([convex_hull_vertices - conv_hull_vertex for conv_hull_vertex in convex_hull_vertices])
    rotated_points = np.array([rot_mat @ points.T for rot_mat,points in zip(rotation_matrices,translated_vertices)])

    x_min, x_max = np.nanmin(rotated_points[:,0],axis=1), np.nanmax(rotated_points[:,0],axis=1)
    y_min, y_max = np.nanmin(rotated_points[:,1],axis=1), np.nanmax(rotated_points[:,1],axis=1)

    x_len, y_len = x_max - x_min, y_max - y_min
    rectangle_area = x_len*y_len
    min_area_index = np.argmin(rectangle_area)

    minor_length, major_length = np.sort([x_len[min_area_index],y_len[min_area_index]])
    major_length_angle = phi_hull[min_area_index]
    
    if return_corners:
        x_min, x_max = x_min[min_area_index], x_max[min_area_index]
        y_min, y_max = y_min[min_area_index], y_max[min_area_index]

        rectangle_corners = np.array([[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]])

        rectangle_corners = rotation_matrices[min_area_index].T @ rectangle_corners.T

        rectangle_corners = np.dstack(rectangle_corners)[0] + convex_hull_vertices[min_area_index]
        
        return rectangle_corners
    
    else:
        return minor_length, major_length, major_length_angle
    
    
def find_distance_nn(tissue,radius_scale=1.5):
    
    f_IDs = tissue.edge_df.loc[
            tissue.edge_df['is_interior'] == False,'dbond_face'].unique()
    
    dbonds_f_IDs = tissue.face_dbonds.loc[f_IDs].values
    
    c_dbonds_f_IDs = [np.setdiff1d(
                        tissue.edge_df.loc[dbonds,'conj_dbond'].values,[-1])
                        for dbonds in dbonds_f_IDs]

    nn_f_IDs = [np.intersect1d(
                        tissue.edge_df.loc[dbonds,'dbond_face'].values,f_IDs)
                        for dbonds in c_dbonds_f_IDs]

    cell_centers = tissue.face_df.loc[f_IDs,['x','y']].values

    vertices_f_IDs = [tissue.vert_df.loc[
                            tissue.edge_df.loc[dbonds,'v_in_id'],['x','y']].values
                            for dbonds in dbonds_f_IDs]

    max_center_vertex_dist = np.array(
                        [np.max(np.linalg.norm(cell_centers[i] - vert_positions,axis=1))
                            for i,vert_positions in enumerate(vertices_f_IDs)])

    tree = cKDTree(cell_centers)
    nn_cell = tree.query_ball_point(cell_centers, radius_scale*max_center_vertex_dist)
        
    
    distance_nn = [np.setdiff1d(f_IDs[nn],nn_f_IDs[i])
                                for i,nn in enumerate(nn_cell)]

    len_distance_nn = np.array([len(nn) for nn in distance_nn])

    indices_to_check = np.argwhere(len_distance_nn > 1).flatten()
    nn_to_check = [distance_nn[idx] for idx in indices_to_check]

    cells_to_check = f_IDs[indices_to_check]
    nn_to_check = [np.setdiff1d(nn,cells_to_check[i])
                        for i,nn in enumerate(nn_to_check)]
    
    cell_radii = max_center_vertex_dist[indices_to_check]
    
    return cells_to_check,nn_to_check,cell_radii

def check_disk_overlap(tissue,cells_to_check,nn_per_cell_to_check,cell_radii):
    
    for cell_ID,nn_cell_IDs,cell_radius in zip(cells_to_check,nn_per_cell_to_check,cell_radii):
        
        cell_center = tissue.face_df.loc[cell_ID,['x','y']].values
        nn_cell_centers = tissue.face_df.loc[nn_cell_IDs,['x','y']].values
        
        center_nn_distances = np.linalg.norm(cell_center - nn_cell_centers,axis=1)
        
        nn_dbonds_IDs = tissue.face_dbonds.loc[nn_cell_IDs].values
        
        vertices_nn_cells = [tissue.vert_df.loc[
                                tissue.edge_df.loc[dbonds,'v_in_id'],['x','y']].values
                                  for dbonds in nn_dbonds_IDs]
        
        nn_radii = np.array(
                        [np.max(np.linalg.norm(nn_cell_centers[i] - vert_positions,axis=1))
                            for i,vert_positions in enumerate(vertices_nn_cells)])
        
        radii_sum = cell_radius + nn_radii
        
        disk_overlap = center_nn_distances < radii_sum
        
        if True in disk_overlap:
            return True        
    
    return False

def which_side(tissue,cell_ID,perp_vector,vertex_ID_through):
    
    vertex_through = tissue.vert_df.loc[vertex_ID_through,['x','y']].values
    
    cell_dbonds = tissue.face_dbonds.loc[cell_ID]
    
    cell_vertices = tissue.vert_df.loc[
                        tissue.edge_df.loc[cell_dbonds,'v_in_id'],['x','y']].values
    
    diff_vectors = cell_vertices - vertex_through
    
    dot_prods = np.dot(diff_vectors,perp_vector)

    dot_prods = [d_prod for d_prod in dot_prods if abs(d_prod) > 1e-10]
    
    dot_prods_signs = np.sign(dot_prods).astype(int)
    
    dot_prods_signs = np.unique(dot_prods_signs)
    
    if len(dot_prods_signs) > 1:
        return 0
    elif dot_prods_signs[0] == 1:
        return 1
    else:
        return -1
    

def cell_vertices_and_side_normals(tissue,face_id):
    
    face_dbonds_ids = tissue.face_dbonds.loc[face_id]

    vert_end_points = tissue.edge_df.loc[face_dbonds_ids,['v_out_id','v_in_id']].values

    vert_out_IDs, vert_in_IDs = vert_end_points[:,0], vert_end_points[:,1]

    vert_out = tissue.vert_df.loc[vert_out_IDs,['x','y']].values

    vert_in = tissue.vert_df.loc[vert_in_IDs,['x','y']].values
    
    side_vectors = vert_in - vert_out

    x_perp_vec, y_perp_vec = side_vectors[:,1], -side_vectors[:,0]

    perp_vectors = np.dstack((x_perp_vec, y_perp_vec))[0]
    
    return vert_in_IDs,perp_vectors

def test_convex_intersection(tissue,face1_id,face2_id):
    
    vert_in_IDs_1,perp_vectors_1 = \
                cell_vertices_and_side_normals(tissue,face1_id)
    
    vert_in_IDs_2,perp_vectors_2 = \
                cell_vertices_and_side_normals(tissue,face2_id)
    
    for vert_id,perp_vec in zip(vert_in_IDs_1,perp_vectors_1):
        
        if which_side(tissue,face2_id,perp_vec,vert_id) == 1:
            return False
        
    for vert_id,perp_vec in zip(vert_in_IDs_2,perp_vectors_2):
        
        if which_side(tissue,face1_id,perp_vec,vert_id) == 1:
            return False  
    
    return True

def check_possible_intersections(tissue,cells_to_check,nn_per_cell_to_check,cell_radii):
    
    disk_overlap_test = \
            check_disk_overlap(tissue,cells_to_check,
                               nn_per_cell_to_check,cell_radii)
    
    if disk_overlap_test == False:
        return False
    
    for cell_ID,nn_cell_IDs in zip(cells_to_check,nn_per_cell_to_check):
        for nn in nn_cell_IDs:
            convex_test = test_convex_intersection(tissue,cell_ID,nn)
            
            if convex_test == True:
                return True
            
    return False

def tissue_self_intersect(tissue):
    cells_to_check, nn_to_check, cell_radii = find_distance_nn(tissue)

    return check_possible_intersections(tissue,cells_to_check,nn_to_check,cell_radii)

def collect_bounding_rectangle_data(tissue):
    boundary_vertices = tissue.vert_df.loc[
                tissue.vert_df['is_interior'] == False,['x','y']].values


    minor_length, major_length, major_length_angle = \
                        bounding_rectangle(boundary_vertices)
    
    aspect_ratio = minor_length/major_length
    
    return minor_length, major_length, aspect_ratio, major_length_angle

def high_order_vertices_remaining(tissue):
    splittable_vertices = find_splittable_vertices(tissue,vertex_group='interior')
    splittable_boundary_vertices = find_splittable_vertices(tissue,vertex_group='boundary')
    
    if (len(splittable_vertices) > 0 or len(splittable_boundary_vertices) > 0):
        return True
    else:
        return False
    
def single_interior_dbond_cells(tissue):
    boundary_faces_IDs =  tissue.edge_df.loc[
            tissue.edge_df['is_interior'] == False,'dbond_face'].unique()
    
    dbonds_boundary_faces = tissue.face_dbonds.loc[boundary_faces_IDs].values
    
    interior_dbonds_per_boundary_cell = \
                    [tissue.edge_df.loc[dbonds_face,'is_interior'].sum()
                     for dbonds_face in dbonds_boundary_faces]
    
    if 1 in interior_dbonds_per_boundary_cell:
        return True
    else:
        return False
    
def are_elements_adjacent(positions,array_size):
    
    if len(positions) == 1:
        return True
    
    adjacent_positions = [[[(pos-1)%array_size,(pos+1)%array_size]] for pos in positions]
    
    for pos_pair in adjacent_positions:
        if True not in np.isin(pos_pair,positions):
            return False
      
    return True

def non_adjacent_boundary_dbonds_per_cell(tissue):
    boundary_faces_IDs =  tissue.edge_df.loc[
            tissue.edge_df['is_interior'] == False,'dbond_face'].unique()
    
    dbonds_boundary_faces = tissue.face_dbonds.loc[boundary_faces_IDs].values

    boundary_dbonds_IDs = tissue.edge_df.loc[
            tissue.edge_df['is_interior'] == False,'id'].values    

    
    for face_dbonds in dbonds_boundary_faces:
        face_dbonds_ordered = np.array(face_dbonds)
        is_boundary_dbonds = np.isin(face_dbonds,boundary_dbonds_IDs)

        number_of_dbonds = len(face_dbonds_ordered)
        false_positions = np.argwhere(is_boundary_dbonds == True).flatten()
        
        adjacent_bool = are_elements_adjacent(false_positions,number_of_dbonds)
        
        if not adjacent_bool:
            return True
              

    return False

def floating_faces(tissue):
    
    bool_list =  [True in tissue.edge_df.loc[
                    tissue.face_dbonds[face_id],'is_interior'].values
                    for face_id in tissue.face_df['id'].values]
    
    
    return (False in bool_list)