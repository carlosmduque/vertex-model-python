import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

from .utilities import (range_to_list_dict, list_to_range_dict,
                        add_constant_columns_to_df,generate_directed_edges_topology)

from .basic_geometry import pol_area, pol_perimeter, pol_centroid, euclidean_distance
from .tissue import Tissue

import math


# from lloyd import Field


def create_vert_df(vert_positions):
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
    force_list = np.zeros(len(vert_positions))

    initial_data = list(zip(id_list, x, y, force_list, force_list))
    cols = ['id', 'x', 'y', 'fx', 'fy']
    vert_df = pd.DataFrame(initial_data, columns=cols)

    return vert_df

# def create_edge_df(vert_positions,dir_edges_end_points,
#                     face_per_directed_edge,left_edge_index,
#                     conjugate_edge_index):
def create_edge_df(vert_positions,dir_edges_end_points,
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
    line_tensions = np.full(number_of_edges,0.12)

    for edge_id, (v_out, v_in) in enumerate(zip(vertex_out_id,vertex_in_id)):
        p_out, p_in = vert_positions[v_out], vert_positions[v_in]
        edge_lengths[edge_id] = euclidean_distance(p_out,p_in)

    initial_data = list(zip(id_list, vertex_out_id, vertex_in_id,
                            interior_edge,edge_lengths,line_tensions))

    cols = ['id', 'v_out_id', 'v_in_id', 'is_interior', 'length', 'line_tension']
    edge_df = pd.DataFrame(initial_data, columns=cols)

    edge_df = edge_df.assign(
            left_dbond = lambda df: df['id'].map(left_edge_index),
            right_dbond = lambda df: df['id'].map(right_edge_index),
            conj_dbond = lambda df: df['id'].map(conjugate_edge_index),
            dbond_face = lambda df: df['id'].map(face_per_directed_edge))

    # edge_df = edge_df.assign(
    #         left_dbond = lambda df: df['id'].map(left_edge_index),
    #         conj_dbond = lambda df: df['id'].map(conjugate_edge_index),
    #         dbond_face = lambda df: df['id'].map(face_per_directed_edge))


    return edge_df

def create_face_df(vert_positions,face_data):
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

    # face_vertices = np.array(
    #                     [np.take(vert_positions,vert_indices,axis=0)
    #                     for _, vert_indices in face_data.items()])
    face_vertices = np.array(
                        [np.take(vert_positions,vert_indices,axis=0)
                        for _, vert_indices in face_data.items()],dtype=object)

    num_sides = np.array([len(vert_indices)
                        for _, vert_indices in face_data.items()])

    area_list = np.array([pol_area(f_vert) for f_vert in face_vertices])
    centroid_list = np.array([pol_centroid(f_vert) for f_vert in face_vertices])
    perimeter_list = np.array([pol_perimeter(f_vert)
                            for f_vert in face_vertices])
    
    x, y = centroid_list[:,0], centroid_list[:,1]

    # initial_data = list(zip(id_list, x, y, num_sides,
    #                         area_list, perimeter_list))

    # face_df = pd.DataFrame(initial_data,
    #             columns=['id', 'x', 'y','num_sides', 'area', 'perimeter'])

    # col_names = ['A_0', 'P_0', 'contractility', 'active', 'mother']
    
    # const_vals = [1.0, 0.0, 0.04, 1, -1]
    
    initial_data = list(zip(id_list, x, y, num_sides,
                            area_list, perimeter_list, id_list))
    
    face_df = pd.DataFrame(initial_data, columns = 
                ['id', 'x', 'y','num_sides', 'area', 'perimeter', 'mother'])
    
    col_names = ['A_0', 'P_0', 'contractility', 'active']
    
    const_vals = [1.0, 0.0, 0.04, 1]

    col_value_pairs = list(zip(col_names,const_vals))

    add_constant_columns_to_df(face_df,col_value_pairs)

    return face_df

def generate_dataframes(vert_positions,dir_edges_end_points,
                    face_per_directed_edge,left_edge_index,
                    right_edge_index,conjugate_edge_index,face_data):

    # TODO: Finish documentation.
    
    vert_df = create_vert_df(vert_positions)

    edge_df = create_edge_df(vert_positions,dir_edges_end_points,
                    face_per_directed_edge,left_edge_index,
                    right_edge_index,conjugate_edge_index)

    face_df = create_face_df(vert_positions,face_data)

    return vert_df, edge_df, face_df

def generate_triangular_lattice(Lx, Ly,rotation='x'):
    """ Generates a set of points on the vertices of a triangular lattice.
        
        Parameters:
        -----------
        Lx : int
            Number of points along the x-axis.
        Ly : int
            Number of points along the y-axis.

        Returns:
        -----------
        triang_lattice_points: Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) vertex-coordinates of the lattice.
    """
    hex_length = 1
    height, width = hex_length * 3/2, hex_length * 3 ** 0.5

    triang_lattice_points = []

    # Create triangular lattice
    
    if rotation == 'x':
        for i, y in enumerate(range(Ly)):
            x_offset = (i % 2) * width/2
            for x in range(Lx):
                triang_lattice_points += [[x*width - x_offset, y*height]]
    else:        
        for i, x in enumerate(range(Lx)):
            y_offset = (i % 2) * width/2
            for y in range(Ly):
                triang_lattice_points += [[x*height, y*width - y_offset]]

    return np.array(triang_lattice_points)

def get_closed_regions(vor_region):
    """ finds the hexagonal cells of the voronoi region of a 
        triangular lattice.

        Note: For the moment this function is only designed to work with
        honeycomb lattices.
        
        Parameters:
        -----------
        vor_region : scipy.spatial.qhull.Voronoi
            Voronoi region of a triangular lattice.

        Returns:
        -----------
        hexagonal_vertices: Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) vertex-coordinates of vor_region.
        hexagonal_faces : List, shape: (N, 6)
            List containing the ordered vertex indices of each hexagon.
    """

    vor_subregions = vor_region.regions

    # We pick only the regions corresponding to hexagons.
    cells_indices = [i for i in range(len(vor_subregions)) if 
                    len(vor_subregions[i]) == 6]

    hexagonal_faces = [vor_subregions[i] for i in cells_indices]
    hexagonal_vertices = vor_region.vertices

    # We find the area sign and order the cell indices anticlokwisely.
    for i, face_indices in enumerate(hexagonal_faces):
        vert_positions = np.array([hexagonal_vertices[idx]
                                    for idx in face_indices])

        cell_area = pol_area(vert_positions)
        if cell_area < 0:
            hexagonal_faces[i] = hexagonal_faces[i][-1::-1]

    return hexagonal_vertices, hexagonal_faces


def generate_honeycomb_patch(Lx=10, Ly=10,rotation='x'):
    """ Generates a honeycomb lattice by calculating the Voronoi region
        of a list of points on the vertices of a triangular lattice.
        
        Parameters:
        -----------
        Lx : int
            Number of hexagons along the x-axis.
        Ly : int
            Number of hexagons along the y-axis.

        Returns:
        -----------
        hexagonal_vertices: Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) vertex-coordinates of the honeycomb.
        dir_edges_end_points : dict, (bond_id,[out_vertex_id, in_vertex_id])
            Dictionary containing the out(in) vertex indices of each edge.
        hexagonal_faces : dict, (face_id, [sorted vertex_id])
            Dictionary containing the ordered vertex indices of each hexagon.
    """

    # We need to compensate for the boundaries faces by adding + 2.
    triang_lattice_points = generate_triangular_lattice(Lx+2, Ly+2,
                                                        rotation=rotation)

    # The Voronoi region of the triangular lattice is a honeycomb.
    vor_region = Voronoi(triang_lattice_points)

    # vertices and ordered faces of the honeycomb lattice.
    vertices, faces = get_closed_regions(vor_region)

    # We collect the vertex indices of the vertices forming the faces.
    vertex_indices = list(set([vert_idx for hex_cell in faces
                    for vert_idx in hex_cell]))

    hexagonal_vertices = np.array([vertices[idx] for idx in vertex_indices])

    vertex_index_dict = list_to_range_dict(vertex_indices)

    hexagonal_faces = [[vertex_index_dict[vert_idx]
                        for vert_idx in hex_cell]
                        for hex_cell in faces]
                        

    dir_edges_end_points = [list(pair)
                            for hex_cell in hexagonal_faces
                            for pair in zip(hex_cell, np.roll(hex_cell, -1))]

    hexagonal_faces = range_to_list_dict(hexagonal_faces)
    dir_edges_end_points = range_to_list_dict(dir_edges_end_points)

    return hexagonal_vertices, dir_edges_end_points, hexagonal_faces

def circle(num_t, radius=1.0, phase=0.0):
    """Returns x and y positions of `num_t` points regularly placed around a circle
    of radius `radius`, shifted by `phase` radians.

    Parameters
    ----------
    num_t : int
        the number of points around the circle
    radius : float, default 1.
        the radius of the circle
    phase : float, default 0.0
        angle shift w/r to the x axis in radians

    Returns
    -------
    points : np.Ndarray of shape (num_t, 2), the x, y positions of the points

    """
    if not num_t:
        return np.zeros((1, 2))

    tau = 2.0*math.pi
    theta = np.arange(0, tau, tau / num_t)
    return np.vstack([radius * np.cos(theta + phase), radius * np.sin(theta + phase)]).T


def hexa_disk(num_t, radius=1):
    """Returns an arrays of x, y positions of points evenly spread on
    a disk with num_t points on the periphery.

    Parameters
    ----------
    num_t : int
        the number of poitns on the disk periphery, the rest of the disk is
        filled automaticaly
    radius : float, default 1.
        the radius of the disk

    """
    
    tau = 2.0*math.pi

    n_circles = int(np.ceil(num_t / tau) + 1)
    if not n_circles:
        return np.zeros((1, 2))

    num_ts = np.linspace(num_t, 0, n_circles, dtype=int)
    rads = radius * num_ts / num_t
    phases = np.pi * num_ts / num_t
    return np.concatenate(
        [circle(n, r, phi) for n, r, phi in zip(num_ts, rads, phases)]
    )
    
def sun_flower_disk(nun_points,spacing=1):
    
    golden_ratio = 1.6180339887498948482
    tau = 2.0*math.pi
    
    points = [[tau*(i*golden_ratio % spacing),math.sqrt(i)]
                                            for i in range(nun_points)]
    
    points = [[point[1]*math.cos(point[0]),point[1]*math.sin(point[0])]
                            for point in points]
    
    points = np.array(points)
    
    return points

# def random_points_in_disk(num_points,iterations=30):
#     sqr_r = np.sqrt(np.random.rand(num_points))
#     theta = 2*np.pi*(np.random.rand(num_points))

#     p_x = sqr_r*np.cos(theta)
#     p_y = sqr_r*np.sin(theta)

#     points = np.dstack((p_x,p_y))[0]

#     # create a lloyd model on which one can perform iterations
#     field = Field(points)

#     for i in range(iterations):   
#     # the .relax() method performs lloyd relaxation, which spaces the points apart
#         field.relax()
        
#     disk_points = \
#         field.points[np.sqrt(field.points[:,0]**2+field.points[:,1]**2) < 0.5]
        
#     return disk_points
    
def get_closed_regions_modified(vor_region):
    """ finds the hexagonal cells of the voronoi region of a 
        triangular lattice.

        Note: For the moment this function is only designed to work with
        honeycomb lattices.
        
        Parameters:
        -----------
        vor_region : scipy.spatial.qhull.Voronoi
            Voronoi region of a triangular lattice.

        Returns:
        -----------
        hexagonal_vertices: Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) vertex-coordinates of vor_region.
        hexagonal_faces : List, shape: (N, 6)
            List containing the ordered vertex indices of each hexagon.
    """

    vor_subregions = vor_region.regions

    # We pick only the regions corresponding to hexagons.
    cells_indices = [i for i in range(len(vor_subregions)) if 
                (len(vor_subregions[i]) >= 3 and -1 not in vor_subregions[i])]

    hexagonal_faces = [vor_subregions[i] for i in cells_indices]
    hexagonal_vertices = vor_region.vertices

    # We find the area sign and order the cell indices anticlokwisely.
    for i, face_indices in enumerate(hexagonal_faces):
        vert_positions = np.array([hexagonal_vertices[idx]
                                    for idx in face_indices])

        cell_area = pol_area(vert_positions)
        if cell_area < 0:
            hexagonal_faces[i] = hexagonal_faces[i][-1::-1]

    return hexagonal_vertices, hexagonal_faces


def generate_hexagonal_disk(num_t, radius=1):
    """ Generates a honeycomb lattice by calculating the Voronoi region
        of a list of points on the vertices of a triangular lattice.
        
        Parameters:
        -----------


        Returns:
        -----------
        hexagonal_vertices: Numpy.ndarray, shape: (N, 2)
            Array containing the (x,y) vertex-coordinates of the honeycomb.
        dir_edges_end_points : dict, (bond_id,[out_vertex_id, in_vertex_id])
            Dictionary containing the out(in) vertex indices of each edge.
        hexagonal_faces : dict, (face_id, [sorted vertex_id])
            Dictionary containing the ordered vertex indices of each hexagon.
    """

    # We need to compensate for the boundaries faces by adding + 2.
    points_in_disk = hexa_disk(num_t,radius)
    # points_in_disk = sun_flower_disk(num_t)
    # points_in_disk = random_points_in_disk(num_t)

    # The Voronoi region of the triangular lattice is a honeycomb.
    vor_region = Voronoi(points_in_disk)

    # vertices and ordered faces of the honeycomb lattice.
    vertices, faces = get_closed_regions_modified(vor_region)

    # We collect the vertex indices of the vertices forming the faces.
    vertex_indices = list(set([vert_idx for hex_cell in faces
                    for vert_idx in hex_cell]))

    hexagonal_vertices = np.array([vertices[idx] for idx in vertex_indices],
                                  dtype=object)

    vertex_index_dict = list_to_range_dict(vertex_indices)

    hexagonal_faces = [[vertex_index_dict[vert_idx]
                        for vert_idx in hex_cell]
                        for hex_cell in faces]
                        

    dir_edges_end_points = [list(pair)
                            for hex_cell in hexagonal_faces
                            for pair in zip(hex_cell, np.roll(hex_cell, -1))]

    hexagonal_faces = range_to_list_dict(hexagonal_faces)
    dir_edges_end_points = range_to_list_dict(dir_edges_end_points)

    return hexagonal_vertices, dir_edges_end_points, hexagonal_faces

def convert_to_tissue_class(tissue_generating_func,*args,**kwargs):
    (vertex_positions, dir_edges_end_points, polygonal_cells) = \
                                            tissue_generating_func(*args,**kwargs)
                                            
    face_dbonds, face_per_directed_edge, \
        left_edge_index, right_edge_index, conjugate_edge_index = \
        generate_directed_edges_topology(dir_edges_end_points,polygonal_cells)

    vert_df, edge_df, face_df \
        = generate_dataframes(vertex_positions,dir_edges_end_points,
                        face_per_directed_edge,left_edge_index,
                        right_edge_index,conjugate_edge_index,polygonal_cells)
        
    tissue  = Tissue(vert_df, edge_df, face_df, face_dbonds)
    
    return tissue
