import numpy as np
import math

from basic_geometry import dbond_axis_angle
from topology_triggers import t1_able_edges

    
 
def find_tissue_dbond_angles(tissue,pref_axis_angle):
    
    dbond_angle_diffs = np.array([
        dbond_axis_angle(tissue,dbond_id,axis_angle=pref_axis_angle)
                for dbond_id in tissue.edge_df['id'].values])
    
    dbond_angle_diffs /= math.pi/2.0
    
    
    return dbond_angle_diffs
   
def sample_dbonds(tissue,rng,dbond_angle_diffs,mu,sigma):
    
    rand_num = abs(rng.normal(mu, sigma))
    
    rand_num_normalized = 1.0-math.exp(-rand_num)
    
    abs_diff = np.abs(rand_num_normalized - dbond_angle_diffs)
    
    dbond_index = np.argmin(abs_diff)
    
    dbond_IDs = tissue.edge_df['id'].values
    
    dbond_index = dbond_IDs[dbond_index]
    
    return dbond_index

def t1_dbond_selector_directed(tissue,rng,pref_axis_angle,mu,sigma):
    
    
    dbond_angle_diffs = find_tissue_dbond_angles(tissue,pref_axis_angle)
    
    t1_able_dbonds = t1_able_edges(tissue) 
    t1_int_to_bound_able_dbonds = t1_able_edges(tissue,t1_type='interior_to_boundary_t1')
    t1_boundary_able_dbonds = t1_able_edges(tissue,t1_type='boundary_t1')
    
    dbond_bool = False
    
    while not dbond_bool:
        dbond_id = sample_dbonds(tissue,rng,dbond_angle_diffs,mu,sigma)
        
        if dbond_id in t1_able_dbonds:
            dbond_t1_type = 0
            dbond_bool = True
        elif dbond_id in t1_int_to_bound_able_dbonds:
            dbond_t1_type = 1
            dbond_bool = True
        elif dbond_id in t1_boundary_able_dbonds:
            dbond_t1_type = 2
            dbond_bool = True
        
    
    return dbond_id, dbond_t1_type

def t1_dbond_selector_uniform(tissue,rng):
    t1_able_dbonds = t1_able_edges(tissue) 
    t1_int_to_bound_able_dbonds = t1_able_edges(tissue,t1_type='interior_to_boundary_t1')
    t1_boundary_able_dbonds = t1_able_edges(tissue,t1_type='boundary_t1')
    
    num_t1_dbonds = len(t1_able_dbonds)//2
    num_t1_int_bound_dbonds = len(t1_int_to_bound_able_dbonds)//2
    num_t1_boundary_dbonds = len(t1_boundary_able_dbonds)
    
    number_of_edges = np.array([num_t1_dbonds,num_t1_int_bound_dbonds,num_t1_boundary_dbonds])
    total_number_of_edges = np.sum(number_of_edges)
    
    edge_probability = number_of_edges/total_number_of_edges
    
    dbond_t1_type = rng.choice(3, 1, p=edge_probability)[0]
    
    if dbond_t1_type == 0:
        dbond_id = rng.choice(t1_able_dbonds)
    elif dbond_t1_type == 1:
        dbond_id = rng.choice(t1_int_to_bound_able_dbonds)
    else:
        dbond_id = rng.choice(t1_boundary_able_dbonds)   
    
    return dbond_id, dbond_t1_type

def t1_dbond_selector(tissue,rng,distribution_type='uniform',
                      pref_axis_angle=np.pi/2.0,mu=0.0,sigma=0.2):
    
    if distribution_type == 'uniform':
        dbond_id, dbond_t1_type = t1_dbond_selector_uniform(tissue,rng)
    elif distribution_type == 'directed':
        dbond_id, dbond_t1_type = t1_dbond_selector_directed(tissue,rng,pref_axis_angle,mu,sigma)
    
    return dbond_id, dbond_t1_type

def directed_isotropic_bond(tissue,rng,pref_axis_angle=np.pi/2.0,
                            mu=0.0,sigma=0.2,anisotropic_isotropic_ratio=0.5):
    rand_num = rng.random()
    if rand_num > anisotropic_isotropic_ratio:
        dbond_id, dbond_t1_type = t1_dbond_selector(tissue,rng,distribution_type='uniform')
        # print("Isotropic T1!")
    else:
        dbond_id, dbond_t1_type = t1_dbond_selector(tissue,rng,distribution_type='directed',
                                pref_axis_angle=pref_axis_angle,mu=mu,sigma=sigma)
        # print("Directed T1!")
        
    return dbond_id, dbond_t1_type

