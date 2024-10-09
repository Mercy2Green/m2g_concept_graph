import os
import numpy as np
import h5py
import json
from os import path
from numpy import save
from requests import get
from omegaconf import OmegaConf
from typing import Dict, Any, List

import argparse
from pathlib import Path
import re
from PIL import Image
import cv2

import open_clip

import torch
import torchvision
import supervision as sv

# from conceptgraph.dataset.datasets_common import GradSLAMDataset, as_intrinsics_matrix

# from conceptgraph.utils.model_utils import compute_clip_features
# import torch.nn.functional as F

# from gradslam.datasets import datautils
# from conceptgraph.slam.utils import gobs_to_detection_list
# from conceptgraph.slam.cfslam_pipeline_batch import BG_CLASSES

# # Local application/library specific imports
# from conceptgraph.dataset.datasets_common import get_dataset
# from conceptgraph.utils.vis import OnlineObjectRenderer
# from conceptgraph.utils.ious import (
#     compute_2d_box_contained_batch
# )
# from conceptgraph.utils.general_utils import to_tensor

# from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
# from conceptgraph.slam.utils import (
#     create_or_load_colors,
#     merge_obj2_into_obj1, 
#     denoise_objects,
#     filter_objects,
#     merge_objects, 
#     gobs_to_detection_list,
# )
# from conceptgraph.slam.mapping import (
#     compute_spatial_similarities,
#     compute_visual_similarities,
#     aggregate_similarities,
#     merge_detections_to_objects
# )

import gc

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


# try: 
#     from groundingdino.util.inference import Model
#     from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# except ImportError as e:
#     print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
#     raise e

# # Set up some path used in this script
# # Assuming all checkpoint files are downloaded as instructed by the original GSA repo
# if "GSA_PATH" in os.environ:
#     GSA_PATH = os.environ["GSA_PATH"]
# else:
#     raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
# import sys
# if "TAG2TEXT_PATH" in os.environ:
#     TAG2TEXT_PATH = os.environ["TAG2TEXT_PATH"]
    
# EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
# sys.path.append(GSA_PATH) # This is needed for the following imports in this file
# sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
# sys.path.append(EFFICIENTSAM_PATH)

# import torchvision.transforms as TS
# try:
#     from ram.models import ram
#     from ram.models import tag2text
#     from ram import inference_tag2text, inference_ram
# except ImportError as e:
#     print("RAM sub-package not found. Please check your GSA_PATH. ")
#     raise e

# # Disable torch gradient computation
# # torch.set_grad_enabled(False)
# # Don't set it in global, just set it in the function that needs it.
# # Using with torch.set_grad_enabled(False): is better.
# # Or using with torch.no_grad(): is also good.
    
# # GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# # Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# # Tag2Text checkpoint
# TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
# RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

# FOREGROUND_GENERIC_CLASSES = [
#     "item", "furniture", "object", "electronics", "wall decoration", "door"
# ]

# FOREGROUND_MINIMAL_CLASSES = [
#     "item"
# ]

def time_logger(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

class ObjEdgeProcessor():
    def __init__(
        self,
        objs_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/finetune_cg_hdf5",
        objs_hdf5_save_file_name="finetune_cg_data.hdf5",
        edges_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/edges_hdf5",
        edges_hdf5_save_file_name="edges.hdf5",
        connectivity_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity",
        connectivity_file_name="scans.txt",
        exclude_scans_file_name="eval_scans.txt",
        obj_feature_name="clip",
        obj_pose_name="bbox_np",
        allobjs_list=[],
        alledges_dict={},
        allvps_pos_dict={}
        ):
        
        # vps
        
        self.connectivity_dir = connectivity_dir
        self.connectivity_file_name = connectivity_file_name
        self.exclude_scans_file_name = exclude_scans_file_name
        
        self.allvps_pos_dict = allvps_pos_dict # a dict of dict, scan dict.  usage: allvps[scan]
        
        # objs
        self.objs_hdf5_save_dir = objs_hdf5_save_dir
        self.objs_hdf5_save_file_name = objs_hdf5_save_file_name
        self.obj_feature_name = obj_feature_name
        self.obj_pose_name = obj_pose_name
        
        self.allobjs_list_list = allobjs_list # a dict of dict, scan dict.  usage: allobjs[scan]
        
        # edges
        self.edges_hdf5_save_dir = edges_hdf5_save_dir
        self.edges_hdf5_save_file_name = edges_hdf5_save_file_name
        
        self.alledges_dict = alledges_dict
        
    @time_logger    
    def get_merge_feature_edges_relationship(self, scan, path):

        filtered_objs, filtered_objs_feature, filtered_objs_pose = self.merge_feature(scan, path, self.obj_feature_name, self.obj_pose_name)
        edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend = self.get_obj_edges_finetune(scan, filtered_objs_pose, filtered_objs, self.alledges_dict)
        
        # return filtered_objs, objs_feature, objs_pose, edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
        return filtered_objs, filtered_objs_feature, filtered_objs_pose, edges_objects_extend, edges_relationship_extend
    
    @time_logger
    def load_obj_edge_vp(self):
        self.load_allobjs_from_hdf5()
        self.get_edges_from_hdf5()
        self.load_allvps_pos_from_connectivity_json()
        return self.allobjs_list, self.alledges_dict, self.allvps_pos_dict
    
    @time_logger
    def merge_feature(self, scan, path, valname_feature="clip", valname_pose="bbox_np"):
        
        if self.allobjs_list == []:
            self.load_allobjs_from_hdf5()
            Warning("Loading objs from hdf5 file can be very slow! Shouldn't be used in the mergeing process.")
        all_objects = self.allobjs_list[scan]
    
        filtered_objs = []
        objs_feature = []
        objs_pose = []
        
        for obj in all_objects:
            filterd_obj = {}
            vp_in_path = []
            obj_observation = obj.get("observed_info")
            obj_observation_vps = list(obj_observation.keys())
            for vp in obj_observation_vps:
                if vp in path:
                    vp_in_path.append(vp)
                    #if filterd_obj[valname_feature] is None: ######!!!! This is a wrong way to judge the key is exist or not.
                    if valname_feature not in filterd_obj: # This is the right way.
                        filterd_obj[valname_feature] = obj_observation[vp]
                    else:
                        filterd_obj[valname_feature] += obj_observation[vp]
                    
            if filterd_obj != {}:
                filterd_obj[valname_feature] = filterd_obj[valname_feature] / len(vp_in_path)
                filterd_obj[valname_pose] = obj[valname_pose]
                filtered_objs.append(filterd_obj)
                
                # obj_feature_tensor = torch.tensor(filterd_obj[valname_feature])
                # obj_feature_normalize = F.normalize(obj_feature_tensor, p=2, dim=0)
                # objs_feature.append(obj_feature_normalize.numpy())
                objs_feature.append(filterd_obj[valname_feature]) # The feature is already normalized.
                
                objs_pose.append(filterd_obj[valname_pose])
        
        # obj_clip_fts = F.normalize(obj_clip_fts, p=2, dim=-1)
        
        return filtered_objs, objs_feature, objs_pose
        
    def get_obj_edges_finetune(self, scan, objs_pose, objs, edges_dict):
        edges_objects = None
        edges_relationship = None
        edges_relationship_extend = None
        
        _, edges_relationship = self.get_edges_from_dict(edges_dict, scan)
        
        correspondence_dict = self.find_correspondece_similarity_bboxcenter(objs, objs_pose)
        # Correspondec_dict is a dict, the key is the index of the objs, and the value is the index of obj_pos.
        
        edges = []  
        for relationship in edges_relationship:
            edge = []
            if relationship[0] in correspondence_dict and relationship[1] in correspondence_dict and relationship[2] != "none of these":
                edge.append(correspondence_dict[relationship[0]])
                edge.append(correspondence_dict[relationship[1]])
                edge.append(relationship[2])
                edges.append(edge)
            else:
                continue
            
        # edges_objs is edges first two columns
        edges_objects = [edge[:2] for edge in edges]
        # edges_relationship is the third column
        edges_relationship = [self.edge_encode(edge[2], False) for edge in edges]
        
        # exntend
        _edges_objects_extend = [[edge[1], edge[0]] for edge in edges]
        _edges_relationship_extend =  [self.edge_encode(edge[2], True) for edge in edges]
        
        edges_objects_extend = []
        edges_relationship_extend = []
        edges_objects_extend.extend(edges_objects)
        edges_objects_extend.extend(_edges_objects_extend)
        edges_relationship_extend.extend(edges_relationship)
        edges_relationship_extend.extend(_edges_relationship_extend)
        
        # return edges_objects, edges_relationship, edges_objects_extend, edges_relationship_extend
        return np.array(edges_objects), np.array(edges_relationship), np.array(edges_objects_extend), np.array(edges_relationship_extend)
        
    def find_similar_vp(self, scan, cur_pos, similarity_threshold=0.2, decimals=5):
        import numpy as np
        """Find the most similar viewpoint based on the similarity of vp position."""
        
        vps_pos_dict = self.allvps_pos_dict[scan]
        # Key is the vp name, value is the vp position
        
        distance_list = []
        min_distance = None
        
        # Iterate over each center in allvps
        for vp_name, vp_pos in vps_pos_dict.items():
            # Calculate Euclidean distance to determine similarity.
            distance = np.linalg.norm(np.around(cur_pos, decimals=decimals) - np.around(vp_pos, decimals=decimals))
            distance_list.append(distance)
            if min_distance is None or distance <= min_distance:
                if distance == min_distance:
                    Warning("There are two vp have the same distance to the current vp. Should deal with this situation.")
                min_distance = distance
                closest_vp = vp_name
        
        if min_distance > similarity_threshold:
            Warning("Compromising. The most similar vp is not similar enough. The distance is larger than the threshold.")

        return closest_vp, min_distance
    
    def load_allobjs_from_hdf5(self):
        hdf5_filename = os.path.join(self.objs_hdf5_save_dir, self.objs_hdf5_save_file_name)
        save_objs = {}
        with h5py.File(hdf5_filename, 'r') as hdf5_file:
            for scan in hdf5_file:
                scan_group = hdf5_file[scan]
                save_objs[scan] = []
                for obj_idx in scan_group:
                    obj_group = scan_group[obj_idx]
                    observed_info_group = obj_group["observed_info"]
                    observed_info = {}
                    for key in observed_info_group:
                        np_array = observed_info_group[key][()]
                        observed_info[key] = np_array  # Assuming direct usage of numpy arrays; convert as needed
                    bbox_np = obj_group["bbox_np"][()]
                    obj_data = {"observed_info": observed_info, "bbox_np": bbox_np}
                    save_objs[scan].append(obj_data)
        self.allobjs_list = save_objs
        return save_objs 
    
    def get_edges_from_hdf5(self):
        import h5py
        
        hdf5_path = self.edges_hdf5_save_dir
        hdf5_file_name = self.edges_hdf5_save_file_name
        
        edges_dict = {}
        hdf5_path = os.path.join(hdf5_path, hdf5_file_name)
        
        with h5py.File(hdf5_path, 'r') as f:
            for scan in f.keys():
                # Initialize an empty list to store combined edges for the current scan
                combined_edges = []
                
                # Load the 'objs' dataset as is
                objs = f[scan]['objs'][:]
                
                # Check if the 'edges_integers' and 'edges_strings' datasets exist
                if 'edges_integers' in f[scan] and 'edges_strings' in f[scan]:
                    edges_integers = f[scan]['edges_integers'][:]
                    edges_strings = f[scan]['edges_strings'][:]
                    
                    # Combine integers and strings back into the original mixed structure
                    for i in range(len(edges_integers)):
                        combined_edge = list(edges_integers[i]) + [edges_strings[i].decode('utf-8')]
                        combined_edges.append(combined_edge)
                else:
                    # Fallback if the original 'edges' dataset exists without splitting
                    combined_edges = f[scan]['edges'][:]
                
                # Store the loaded data in the dictionary
                edges_dict[scan] = (objs, combined_edges)
        self.alledges_dict = edges_dict
        return edges_dict
    
    def load_allvps_pos_from_connectivity_json(self):
        
        allvps_pos_dict = {}    
        
        connectivity_dir = self.connectivity_dir
        connectivity_file_name = self.connectivity_file_name
        exclude_scans_file_name = self.exclude_scans_file_name
        
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
                allvps_pos_dict[scan] = {}
                for item in data:
                    if item['included']:
                        allvps_pos_dict[scan][item['image_id']] = np.array([item['pose'][3], item['pose'][7], item['pose'][11]])
                        # This is form the "def load_nav_graphs(connectivity_dir):" of the common.py 
        
        self.allvps_pos_dict = allvps_pos_dict
        
        return allvps_pos_dict
    
    def load_viewpoint_ids(self, connectivity_dir, connectivity_file_name, exclude_scans_file_name="eval_scans.txt"):
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
        
    def find_correspondece_similarity_bboxcenter(self, objs, obj_pos, similarity_threshold=0.2, decimals=5):
        import numpy as np
        """Find correspondence based on the similarity of bbox centers."""
        correspondence_dict = {}
        
        # Calculate centers for each bbox in objs
        objs_centers = [self.calculate_center(obj["bbox_np"]) for obj in objs]
        
        # Iterate over each center in obj_pos
        for i, pos_center in enumerate(obj_pos):
            # Compare with each center in objs
            for j, obj_center in enumerate(objs_centers):
                # Calculate Euclidean distance to determine similarity
                distance = np.linalg.norm(np.around(pos_center, decimals=decimals) - np.around(obj_center, decimals=decimals))
                # If distance is within the threshold, add to correspondence_dict
                if distance < similarity_threshold:
                    correspondence_dict[j] = i
                    break  # Assuming one-to-one correspondence, stop after the first match
        return correspondence_dict
    
    def calculate_center(self, bbox_corners):
        """Calculate the center of a bbox given its eight corners."""
        if len(bbox_corners) == 3:
            bbox_center = bbox_corners
            return bbox_center
        
        return np.mean(bbox_corners, axis=0)
    
    def get_edges_from_dict(self, dict_scan_edges, scan):
        objs_bbox_list, edges_relationship = dict_scan_edges[scan]
        objs = [{"bbox_np": obj} for obj in objs_bbox_list]
        return objs, edges_relationship
    
    def edge_encode(self, edge_relationship, revert_flag = False):
    
        if revert_flag == False:
            if edge_relationship == "a on b":
                encoded_relationship = 0
            elif edge_relationship == "a in b":
                encoded_relationship = 1
            elif edge_relationship == "b on a":
                encoded_relationship = 2
            elif edge_relationship == "b in a":
                encoded_relationship = 3
            # elif edge_relationship == "none of these":
            #     encoded_relationship = 4
        elif revert_flag == True:
            if edge_relationship =="a on b":
                encoded_relationship = 2
            elif edge_relationship == "a in b":
                encoded_relationship = 3
            elif edge_relationship == "b on a":
                encoded_relationship = 0
            elif edge_relationship == "b in a":
                encoded_relationship = 1
            # elif edge_relationship == "none of these":
            #     encoded_relationship = 4
        
        return encoded_relationship
   
