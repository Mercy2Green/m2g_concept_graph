from conceptgraph.cg_process.obj_edge_processor import ObjEdgeProcessor

import os
import json
import h5py
import numpy as np
from tqdm import tqdm

def load_viewpoint_ids(connectivity_dir, connectivity_file_name, exclude_scans_file_name="eval_scans.txt"):
    viewpoint_ids = []
    
    if exclude_scans_file_name is None:
        eval_scans = []
    else:
        with open(os.path.join(connectivity_dir, exclude_scans_file_name)) as f:
            eval_scans = [x.strip() for x in f] 
    
    with open(os.path.join(connectivity_dir, connectivity_file_name)) as f:
        scans = [x.strip() for x in f]      # load all scans
        
    filtered_scans = list(set(scans) - set(eval_scans))
    
    for scan in filtered_scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

class ObjEdgeCorresponder():
    
    def __init__(
        self, 
        obj_edge_processor: ObjEdgeProcessor,
        base_save_path,
        base_save_name
        ):
        
        self.obj_edge_processor = obj_edge_processor
        
        self.obj_edge_processor.allobjs_dict, self.obj_edge_processor.alledges_dict, _ = self.obj_edge_processor.load_obj_edge_vp()

        self.scan_list = []
        
        self.base_save_path = base_save_path
        self.base_save_name = base_save_name
        
        viewpoint_ids = self.obj_edge_processor.load_viewpoint_ids(self.obj_edge_processor.connectivity_dir, self.obj_edge_processor.connectivity_file_name, self.obj_edge_processor.exclude_scans_file_name)
        scan_list = list(set(scan for scan, _ in viewpoint_ids))
        self.scan_list = scan_list
        
    def main_process(self):
        
        ## get the all scan
        scan_list = self.scan_list
        
        ## check if the obj_dict and the edge_dict is already loaded
        
        if self.obj_edge_processor.alledges_dict== {} or self.obj_edge_processor.allobjs_dict == {}:
            self.obj_edge_processor.load_obj_edge_vp()
        
        ## For loop each scan to generate the correspondence_dict and save these correspondence_dict
        
        for scan in tqdm(scan_list):
            
            all_objs = self.obj_edge_processor.allobjs_dict[scan]
            #'observed_info' and "bbox_np" are the keys in the all_objs, bbox_np is the center of the bbox
            
            edges_objs = self.obj_edge_processor.alledges_dict[scan]
            # a set or list of tuples, 0 is a list of the eight corner of the bbox of all edge_filterd obj , 1 is the relation in this scan
            
            all_objs_positions = []
            for obj in all_objs:
                all_objs_positions.append(obj[self.obj_edge_processor.obj_pose_name])
            
            edges_objs_positions = []    
            for edge_position in edges_objs[0]:
                edge_dict = {}
                edge_dict[self.obj_edge_processor.obj_pose_name] = edge_position
                edges_objs_positions.append(edge_dict)
        
            correspondence_dict = self.obj_edge_processor.find_correspondece_similarity_bboxcenter(edges_objs_positions, all_objs_positions)
            
            saved_dict = {}
            saved_dict[scan] = correspondence_dict
            
            save_file_path = self.base_save_path + '/' + self.base_save_name + "_" + scan + '.hdf5'
            
            save_correspondence_dict_to_hdf5(saved_dict, save_file_path)

    def merge_savefile_into_one(self):
        
        ## merge all the saved files into one
        
        all_saved_dict = {}
        
        for scan in self.scan_list:
            load_file_path = self.base_save_path + '/' + self.base_save_name + "_" + scan + '.hdf5'
            
            saved_dict = load_correspondence_dict_from_hdf5(load_file_path)
            all_saved_dict[scan] = saved_dict[scan]
        
        save_correspondence_dict_to_hdf5(all_saved_dict, self.base_save_path + self.base_save_name)

def save_correspondence_dict_to_hdf5(correspondence_dict, file_path):
    with h5py.File(file_path, 'w') as hdf5_file:
        for scan, data in correspondence_dict.items():
            # Convert scan to string if it's not already
            scan_str = str(scan)
            # Create a group for each scan
            scan_group = hdf5_file.create_group(scan_str)
            for key, value in data.items():
                # Convert key to string if it's not already
                key_str = str(key)
                # Assuming value is a numpy array or can be converted to one
                scan_group.create_dataset(key_str, data=np.array(value))

def load_correspondence_dict_from_hdf5(file_path):
    correspondence_dict = {}
    with h5py.File(file_path, 'r') as hdf5_file:
        for scan in hdf5_file.keys():
            scan_group = hdf5_file[scan]
            data = {}
            for key in scan_group.keys():
                data[key] = np.array(scan_group[key])
            correspondence_dict[scan] = data
    return correspondence_dict

def main():

    base_save_path = '/data2/vln_dataset/obj_edge_correspondence_dict'
    base_save_name = 'obj_edge_correspondence_dict'
    
    obj_edge_processor = ObjEdgeProcessor(
        objs_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/finetune_cg_hdf5",
        objs_hdf5_save_file_name="finetune_cg_data.hdf5",
        edges_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/edges_hdf5",
        edges_hdf5_save_file_name="edges.hdf5",
        connectivity_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity",
        connectivity_file_name="scans.txt",
        exclude_scans_file_name="eval_scans.txt",
        obj_feature_name="clip",
        obj_pose_name="bbox_np",
        allobjs_dict=[],
        alledges_dict={},
        allvps_pos_dict={}
        )
    
    obj_edge_corresponder = ObjEdgeCorresponder(obj_edge_processor, base_save_path, base_save_name)
    
    obj_edge_corresponder.main_process()
    
    obj_edge_corresponder.merge_savefile_into_one()
    
if __name__ == '__main__':
    main()