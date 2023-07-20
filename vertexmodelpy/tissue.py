#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .utilities import *
from .basic_geometry import (pol_area, pol_centroid, pol_perimeter, euclidean_distance,
                            pol_area_faster,pol_perimeter_faster,
                            pol_centroid_faster,triangle_centroid_faster)
from .basic_topology import set_vertex_topology


class Tissue():
    # Our collection of cells, edges, and vertices along with functions

    def __init__(self, vert_df, edge_df, face_df, face_dbonds=[],
                 normalize_area=True):
    # def __init__(self, vert_df, edge_df, face_df):

        self.vert_df = vert_df
        self.edge_df = edge_df
        self.face_df = face_df

        if face_dbonds == []:
            self.reset_dbonds_per_face()
        else:
            self.face_dbonds = pd.Series(face_dbonds)

        self.coord_labels = ['x','y']

        self.num_vertices = len(vert_df)
        self.num_edges = len(edge_df)
        self.num_faces = len(face_df)
        
        self.t1_cutoff = 1e-4#1e-6
        self.t1_boundary_cutoff = 1e-4
        self.t2_cutoff = 1e-4

        # self.reset_dbonds_per_face()
        if normalize_area:
            self.normalize_tissue(area_normalization=0.5756985420555952)
            
        self.reset_vertex_topology()

         # Time step tracker
        self.step = 0

    def reset_dbonds_per_face(self):
        get_face_ordered_dbonds(self)
        return

    def reset_vertex_topology(self):
        set_vertex_topology(self)
        return
     
    def update_tissue_geometry(self,face_list=[],
                geometry_quantities=['area','perimeter','centroid','length']):
             
        if face_list == []:
            face_verts = find_face_vertices_faster(self)
            
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
        
    def tissue_area_perimeter(self):
        
        tissue_area = self.face_df['area'].sum()
        tissue_perimeter = self.edge_df.loc[
                        self.edge_df['is_interior'] == False,'length'].sum()
        
        return tissue_area, tissue_perimeter
    
    def mean_geometrical_quantities(self,return_std=False):
        
        mean_cell_area = self.face_df['area'].mean()
        mean_cell_perimeter = self.face_df['perimeter'].mean()
    
        dbond_conj_dbond_pairs = self.edge_df.loc[
                    self.edge_df['is_interior'] == True, ['id','conj_dbond']].values
               
        unrepeated_interior_edge_indices = np.unique(
                    np.sort(dbond_conj_dbond_pairs)[:,0])
        
        boundary_edge_indices = self.edge_df.loc[
                    self.edge_df['is_interior'] == False,'id'].values

        dbond_indices_unrepeated = np.concatenate(
                    (unrepeated_interior_edge_indices,boundary_edge_indices))
        
        mean_dbond_length = self.edge_df.loc[
                                    dbond_indices_unrepeated,'length'].mean()
        
        mean_geometrical_data = [mean_cell_area,mean_cell_perimeter,
                                        mean_dbond_length]
        
        if return_std:
            std_cell_area = self.face_df['area'].std(ddof=0)
            std_cell_perimeter = self.face_df['perimeter'].std(ddof=0)
            std_dbond_length = self.edge_df.loc[
                        dbond_indices_unrepeated,'length'].std(ddof=0)
            
            std_geometrical_data = [std_cell_area,std_cell_perimeter,
                                        std_dbond_length]
            
            return mean_geometrical_data, std_geometrical_data
        
        else:
            return mean_geometrical_data
                  
       
