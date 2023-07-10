#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import math
import numpy as np
import pandas as pd

from utilities import *
from basic_geometry import (euclidean_distance,
                            pol_area_faster,pol_perimeter_faster,
                            pol_centroid_faster,triangle_centroid_faster,
                            calculate_triangle_state_tensor_symmetric_antisymmetric,
                            triangulation_Q_norm)

from basic_topology import set_vertex_topology

from triangulation_methods import initialize_triangulation


class Triangulation():
    # Our collection of cells, edges, and vertices along with functions

    def __init__(self, tissue,triangulation_type='dual_triangulation'):
    # def __init__(self, vert_df, edge_df, face_df):

        self.tissue = tissue
        self.triangulation_type = triangulation_type
        
        self.coord_labels = ['x','y']
        
        vert_df, edge_df, face_df, face_dbonds, face_verts = \
            initialize_triangulation(
                    self.tissue,triangulation_type=self.triangulation_type)
        
        self.vert_df = vert_df
        self.edge_df = edge_df
        self.face_df = face_df
        self.face_dbonds = face_dbonds
        self.face_verts = face_verts


        self.num_vertices = len(vert_df)
        self.num_edges = len(edge_df)
        self.num_faces = len(face_df)
        
            
        self.reset_vertex_topology()
        self.triangle_side_vectors()
        self.state_tensor()

    
    def reset_dbonds_per_face(self):
        get_face_ordered_dbonds(self)
        return

    def reset_vertex_topology(self):
        set_vertex_topology(self)
        return
     
    def update_tissue_geometry(self,face_list=[],
                geometry_quantities=['area','perimeter','centroid','length']):
             
        if face_list == []:
            face_verts = find_face_vertices_faster(self,self.face_dbonds)
            
            vert_positions_df = pd.Series(face_verts,
                                index=self.face_df.index,dtype=object)
            
            unique_num_sides = self.face_df['num_sides'].unique()
        else:
            f_dbonds = self.face_dbonds.loc[face_list]
            face_verts = find_face_vertices(self,f_dbonds)
            
            vert_positions_df = pd.Series(face_verts,
                                index=face_list,dtype=object)
            
            unique_num_sides = \
                            self.face_df.loc[face_list,'num_sides'].unique()
        
        for num_sides in unique_num_sides:
            if face_list == []: 
                face_ids = self.face_df.loc[
                                self.face_df['num_sides'] == num_sides,
                                'num_sides']
                face_ids = face_ids.index
            else:
                face_ids = self.face_df.loc[
                            self.face_df.index.isin(face_list) & 
                            (self.face_df['num_sides'] == num_sides).values,
                                'num_sides']
                face_ids = face_ids.index
            
            same_num_sides_vert_pos = vert_positions_df.loc[face_ids]
            same_num_sides_vert_pos = np.array(
                                [*same_num_sides_vert_pos.values],dtype=float)
            
            if 'perimeter' in geometry_quantities:
                perimeter_list = pol_perimeter_faster(same_num_sides_vert_pos)
                self.face_df.loc[face_ids,'perimeter'] = perimeter_list
                
            if 'area' in geometry_quantities:
                perimeter_list = pol_area_faster(same_num_sides_vert_pos)
                self.face_df.loc[face_ids,'area'] = perimeter_list
                
            if 'centroid' in geometry_quantities:
                if num_sides > 3:
                    centroid_list = \
                                pol_centroid_faster(same_num_sides_vert_pos)
                else:
                    centroid_list = \
                            triangle_centroid_faster(same_num_sides_vert_pos)
                # centroid_list = \
                #             triangle_centroid_faster(same_num_sides_vert_pos)
                    
                self.face_df.loc[face_ids,self.coord_labels] = centroid_list
            
            
        if 'length' in geometry_quantities:
            if face_list == []:
                self.update_edge_lengths()
            else:
                edge_list = f_dbonds.explode().values
                self.update_edge_lengths(edge_list=edge_list)
                    
        return
    
    def update_edge_lengths(self,edge_list=[]):
             
        vert_positions = self.vert_df[self.coord_labels]

        if edge_list == []:
            edge_lengths = np.zeros(len(self.edge_df), dtype=float)
            v_out_ids = self.edge_df['v_out_id']
            v_in_ids = self.edge_df['v_in_id']
            
        else:
            edge_lengths = np.zeros(len(edge_list), dtype=float)
            v_out_ids = self.edge_df.loc[edge_list, 'v_out_id']
            v_in_ids = self.edge_df.loc[edge_list, 'v_in_id']
        
        p_out = vert_positions.loc[v_out_ids].values
        p_in = vert_positions.loc[v_in_ids].values
        
        edge_lengths = euclidean_distance(p_out,p_in,multiple_distance=True)
       
        if edge_list == []:
            self.edge_df['length'] = edge_lengths
        else:
            self.edge_df.loc[edge_list, 'length'] = edge_lengths

        return  

    def normalize_tissue(self,area_normalization=1):
        mean_area = self.face_df['area'].mean()
        self.vert_df[self.coord_labels] = \
            (area_normalization/mean_area)**0.5 * \
                        self.vert_df[self.coord_labels]

        self.update_tissue_geometry()

        return
    
    def triangle_side_vectors(self):
    
        triangulation_verts = self.vert_df[['x','y']].values
        face_vertices = np.array([triangulation_verts[triang_vert_ids]
                        for triang_vert_ids in self.face_verts])
        
        side_vectors = np.roll(face_vertices,-1,axis=1) - face_vertices
        side_vectors_21, side_vectors_32 = side_vectors[:,0], side_vectors[:,1]
        
        side_vectors_21_x, side_vectors_21_y = side_vectors_21[:,0], side_vectors_21[:,1]
        side_vectors_32_x, side_vectors_32_y = side_vectors_32[:,0], side_vectors_32[:,1]
        
        self.face_df['dr_21_x'] = side_vectors_21_x
        self.face_df['dr_21_y'] = side_vectors_21_y
        self.face_df['dr_32_x'] = side_vectors_32_x
        self.face_df['dr_32_y'] = side_vectors_32_y
    
        return
    
    def state_tensor(self,A_0=1.0):
        state_tensor = \
            calculate_triangle_state_tensor_symmetric_antisymmetric(self,A_0=A_0)
            
        # self.face_df[['theta','AabsSq','twophi','BabsSq',]] = state_tensor
        AabsSq, BabsSq = state_tensor[:,1], state_tensor[:,3]
        Q_norms = triangulation_Q_norm(AabsSq, BabsSq)
        
        theta, twophi = state_tensor[:,0], state_tensor[:,2]
        
        self.face_df['theta'] = theta
        self.face_df['twophi'] = twophi
        self.face_df['elong_tensor_norm'] = Q_norms
        
        cos_two_phi, sin_two_phi = np.cos(twophi), np.sin(twophi)
        
        
        q_xx, q_xy = Q_norms * cos_two_phi, Q_norms * sin_two_phi
        
        
        q_row_0, q_row_1 = np.dstack((q_xx,q_xy)), np.dstack((q_xy,-q_xx))

        triangle_elongation_tensor = np.stack((q_row_0,q_row_1),axis=2)[0]
        
        # cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
        # r_xx, r_xy = cos_theta, sin_theta
        # rot_row_0 = np.dstack((r_xx,-r_xy))
        # rot_row_1 = np.dstack((r_xy,r_xx))

        # rotation_matrix_z_axis = np.stack((rot_row_0,rot_row_1),axis=2)[0]
        
        # triangle_elongation_tensor = triangle_elongation_tensor @ rotation_matrix_z_axis
        
               
        # return triangle_elongation_tensor, rotation_matrix_z_axis
        return triangle_elongation_tensor
            
        
    
        
       
