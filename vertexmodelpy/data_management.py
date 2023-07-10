import os
import pandas as pd

from .tissue import Tissue

def load_datasets(h5store, data_names=["vert_df", "edge_df", "face_df"]):
    if not os.path.isfile(h5store):
        raise FileNotFoundError("file %s not found" % h5store)
    with pd.HDFStore(h5store) as store:
        data = {name: store[name] for name in data_names if name in store}
    return data

def save_datasets(h5store, tissue, 
                data_names=["vert_df", "edge_df", "face_df"],mode='w'):
    
    with pd.HDFStore(h5store,mode=mode) as store:
        for key in data_names:
            store.put(key, getattr(tissue, f"{key}"))
            
    return
            
def load_tissue(h5store,normalize_area=False):
    tissue_df = load_datasets(h5store)
    
    vert_df = tissue_df['vert_df']
    edge_df = tissue_df['edge_df']
    face_df = tissue_df['face_df']
    
    tissue = Tissue(vert_df,edge_df,face_df,normalize_area=normalize_area)
    
    tissue.num_vertices = vert_df['id'].max() + 1
    tissue.num_edges = edge_df['id'].max() + 1
    tissue.num_faces = face_df['id'].max() + 1
    
    return tissue

def logging_data(df,data_list,df_file_path=''):
    
    idx = len(df)
    df.loc[idx] = data_list
    
    df_row = df.iloc[[-1]]

    
    if df_file_path != '':
        if idx == 0:
            df_row.to_hdf(df_file_path,key='logging_df', mode='w',format='t')
        else:
            df_row.to_hdf(df_file_path,key='logging_df', append=True, mode='r+', format='t')
     
        
    return