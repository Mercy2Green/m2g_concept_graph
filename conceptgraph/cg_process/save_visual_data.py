from tracemalloc import start
from cv2 import FileNode_NAMED, merge
import numpy as np
from conceptgraph.slam.cfslam_pipeline_batch import *
from sqlalchemy import all_
from torch import gt
from cg_processing import *
import os
import time



def get_obs_mask_per_obj(obj_vps: list, paths_masks: np.ndarray, all_vp_ids, vp_id_to_index: dict):
    '''
    return (n_obs_comb, n_vp) tensor
    '''
    # generate the obs_masks
    # obs_masks = np.zeros(len(all_vp_ids), dtype=int)
    obs_masks = torch.zeros((1,len(vp_id_to_index)), dtype=torch.int).cuda()
    for vp_id in obj_vps:
        # idx = all_vp_ids.index(vp_id)
        idx = vp_id_to_index[vp_id]
        obs_masks[0,idx] = 1
    # obs_masks = np.expand_dims(obs_masks, axis=0)
    # obs_masks: (1, n_vp) paths_masks: (n_paths, n_vp)
    obs_comb = obs_masks & paths_masks # 
    # get unique obs combination
    obs_comb = torch.unique(obs_comb, dim=0, return_inverse=False) # (n_obs_comb, n_vp)
    if torch.all(obs_comb == 0):
        return None,None
    # transfer obs_comb from global vp to local vp
    # local_obs_comb = np.zeros((obs_comb.shape[0],len(obj_vps)), dtype=int)
    # one_indices = np.where(obs_comb == 1)
    local_obs_comb = torch.zeros(obs_comb.shape[0], len(obj_vps), dtype=torch.int).cuda()
    one_indices = torch.where(obs_comb == 1)
    indices_list = list(zip(one_indices[0], one_indices[1]))
    for one_item in indices_list:
        obs_comb_idx =one_item[0]
        vp_idx_inAlllist = one_item[1]
        vp_id = all_vp_ids[vp_idx_inAlllist]
        # assert vp_id in obj_vps, "The vp_id is not in obj_vps"
        local_vp_idx = obj_vps.index(vp_id)
        local_obs_comb[obs_comb_idx, local_vp_idx] = 1
    return local_obs_comb,obs_comb
def get_all_path_masks(paths:list, vp_id_to_index:dict):
    '''
    input:
        paths: list of path, each path is a list of vp_ids
        all_vp_ids: list of all vp_ids
    '''
    assert len(paths) > 0, "The paths is empty"
    
    n_paths = len(paths)
    n_vps = len(vp_id_to_index)
    
    # 创建一个全零的数组，形状为 (n_paths, n_vps)
    # path_masks = np.zeros((n_paths, n_vps), dtype=int)
    path_masks = torch.zeros(n_paths, n_vps, dtype=torch.int).cuda()
    # 遍历每一条路径
    for i, path in enumerate(paths):
        for vp_id in path:
            # assert vp_id in all_vp_ids, "The vp_id is not in all_vp_ids"
            # idx = all_vp_ids.index(vp_id)
            idx = vp_id_to_index[vp_id]
            path_masks[i, idx] = 1
    return path_masks
def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
def pad_tensors_2dim(tensors):
    # 计算最大长度
    max_len_1 = max(t.size(0) for t in tensors)
    max_len_2 = max(t.size(1) for t in tensors)
    
    # 创建新的tensor并复制原始数据
    padded_tensors = []
    for t in tensors:
        # 创建新的tensor，初始化为0
        padded_tensor = torch.zeros(max_len_1, max_len_2, dtype=t.dtype, device=t.device)
        # 复制原始数据到新的tensor中
        padded_tensor[:t.size(0), :t.size(1)] = t
        padded_tensors.append(padded_tensor)
    
    # 使用torch.stack将列表中的tensors合并为一个新的tensor
    return torch.stack(padded_tensors)
def new_merge_function(objects: MapObjectList, paths:list, all_vp_ids:list):
    '''
    faster version all merge function
    '''
    # for object in objects scene map to get the obs_masks
    batch_obs_masks = []
    batch_obs_gloabl_masks = []
    batch_clips = []
    batch_pos = []
    vp_id_to_index = {vp_id: idx for idx, vp_id in enumerate(all_vp_ids)}
    all_path_masks = get_all_path_masks(paths, vp_id_to_index)
    num_obj = 0
    for obj in tqdm(objects, desc="processing clip merge"):
        observed_info = obj.get('observed_info')
        assert observed_info is not None, "The observed_info is None"
        vp_ids = list(observed_info.keys())
        # assert vp_ids in all_vp_ids, "The vp_ids is not in all_vp_ids"
        local_obs_mask,global_obs_masks = get_obs_mask_per_obj(vp_ids, all_path_masks,all_vp_ids, vp_id_to_index)
        if local_obs_mask is None:
            continue
        # batch_obs_masks.append(torch.from_numpy(local_obs_mask))
        # batch_obs_gloabl_masks.append(torch.from_numpy(global_obs_masks))
        batch_obs_masks.append(local_obs_mask)
        batch_obs_gloabl_masks.append(global_obs_masks)
        # stack clip
        vp_clips = [observed_info[vp_id] for vp_id in vp_ids]
        stacked_vp_clips = torch.stack(vp_clips)
        # create the position information
        obj_pos = np.round(obj['bbox'].center,5)
        
        batch_clips.append(stacked_vp_clips)
        batch_pos.append(obj_pos)
        
        # num_obj += 1
        # print(f" in merge function, {num_obj} objects have been processed")
    
    # pad the masks and clips to have the same shape
    batch_obs_gloabl_masks = pad_tensors(batch_obs_gloabl_masks)
    batch_clips = pad_tensors(batch_clips).cuda()
    batch_pos = np.array(batch_pos)
    batch_obs_masks = pad_tensors_2dim(batch_obs_masks)
    # check the space size of the tensors
    # assert batch_obs_gloabl_masks.shape[0] == batch_clips.shape[0], "The batch_clips and batch_obs_masks have different num of objects"
    # assert batch_obs_gloabl_masks.shape[1] == batch_obs_masks.shape[1], "The batch_clips and batch_obs_masks have different num of vps"
    # assert batch_obs_gloabl_masks.shape[2] == len(all_vp_ids), "The batch_clips and batch_obs_masks have different num of vps"
    
    # assert batch_clips.shape[0] == batch_obs_masks.shape[0], "The batch_clips and batch_obs_masks have different num of objects"
    # assert batch_clips.shape[1] == batch_obs_masks.shape[2], "The batch_clips and batch_obs_masks have different num of vps"
    # merge clip feature
    if batch_obs_masks.shape[0] < 150:
        merged_clips = batch_obs_masks.unsqueeze(-1) * batch_clips.unsqueeze(1)
    else:
    # 获取 batch_clips 的形状
        n_obj, n_obj_vps, clip_lens = batch_clips.shape

        batch_size = 20
        # 计算每个批次的大小
        # batch_size = n_obj // batch_num
        torch.cuda.empty_cache()
        # 存储结果的列表
        merged_clips_list = []
        # 分批处理
        # for start_idx in range(0, n_obj, batch_size):
        for start_idx in tqdm(range(0, n_obj, batch_size), desc="merging multiplications"):
            end_idx = min(start_idx + batch_size, n_obj)  # 确保最后一个批次包含所有剩余的元素
        # for i in range(batch_num):
        #     start_idx = i * batch_size
        #     end_idx = (i + 1) * batch_size if i < batch_num-1 else n_obj  # 确保最后一个批次包含所有剩余的元素
            
            # 获取当前批次的 batch_obs_masks 和 batch_clips
            batch_obs_masks_batch = batch_obs_masks[start_idx:end_idx]
            batch_clips_batch = batch_clips[start_idx:end_idx]
            
            # 进行乘法计算
            merged_clips_batch = batch_obs_masks_batch.unsqueeze(-1) * batch_clips_batch.unsqueeze(1)
            
            # 将结果添加到列表中
            merged_clips_batch_cpu = merged_clips_batch.cpu()
            merged_clips_list.append(merged_clips_batch_cpu)
            # del merged_clips_batch
            # del batch_obs_masks_batch
            # del batch_clips_batch
            # torch.cuda.empty_cache()
            # 将所有批次的结果合并起来
        merged_clips = torch.cat(merged_clips_list, dim=0)
        print(f" the space size of merged_clips is {merged_clips.nbytes/1024/1024/1024}GB")
    assert merged_clips.shape[0] == batch_clips.shape[0], "The merged_clips and batch_clips have different num of objects"
    assert merged_clips.shape[1] == batch_obs_masks.shape[1], "The merged_clips and batch_obs_masks have different num of paths"
    
    # mean
    merged_clips = torch.mean(merged_clips, dim=2)
    assert merged_clips.shape[2] == batch_clips.shape[2], "The merged_clips and batch_clips have different clip length"
    # normalize
    merged_clips = F.normalize(merged_clips, dim=2)
    print(f" the space size of merged_clips is {merged_clips.nbytes/1024/1024/1024}GB")
    
    return merged_clips, batch_pos, batch_obs_gloabl_masks, all_path_masks          
def merger_clip4all_combinations_test(objects: MapObjectList, paths:list, all_vp_ids:list, device='cuda'):
    '''
    given all possible paths, which provide all obserbation combination for all object
    '''
    batch_masks = []
    batch_clips = []
    
    max_combinations = 0
    max_vps = 0
    
    observed_combinations_list = []  # Store the results of find_union_of_viewpoints
    
    batch_clip_masks = []
    batch_obj_pos = []
    for obj in objects:
        observed_info = obj.get('observed_info')
        if observed_info is None:
            continue
        
        vp_ids = list(observed_info.keys())
        
        # This is the list of all possible combinations of viewpoints
        observed_combinations = find_union_of_viewpoints(vp_ids, paths) #这样的循环不能接受
        # observed_combinations = get_all_length_combinations(vp_ids)
        
        # observed_combinations_list.append(observed_combinations) 
        combination_masks = create_merge_masks(vp_ids, observed_combinations)
        if combination_masks is None:
            continue
        vp_clips = [observed_info[vp_id] for vp_id in vp_ids]
        stacked_vp_clips = torch.stack(vp_clips)
        
        batch_masks.append(combination_masks)
        batch_clips.append(stacked_vp_clips)
        
        max_combinations = max(max_combinations, len(observed_combinations))
        max_vps = max(max_vps, len(vp_ids))
        ## create masks for saving the whole information
        clip_masks = create_merge_masks(all_vp_ids, observed_combinations)
        batch_clip_masks.append(clip_masks)

        # create the position information
        obj_pos = np.round(obj['bbox'].center,5)
        batch_obj_pos.append(obj_pos)
        #for now
        # clips is 1*m tensor list,
        # masks is n*m tensor list,
        # n is path(combination) number
        # m is vp number
    masks_paded = batch_masks
    clips_paded = batch_clips
    # transfer obj_pos from list to np
    # obj_pos_np = np.array(batch_obj_pos)
    
    # Pad the masks and clips to have the same shape
    batch_clip_masks = pad_tensors(batch_clip_masks)
    paths_masks = create_merge_masks(all_vp_ids, paths)
    # for i in range(len(batch_masks)):
    #     max_combinations = max(max_combinations, batch_masks[i].shape[0])
    #     max_vps = max(max_vps, batch_clips[i].shape[0])

    # Pad the masks and clips to have the same shape
    for i in range(len(batch_masks)):
        combination_pad_size = max_combinations - batch_masks[i].shape[0]
        vp_pad_size = max_vps - batch_clips[i].shape[0]
        masks_paded[i] = F.pad(batch_masks[i], (0, 0, 0, vp_pad_size, 0, combination_pad_size))
        # masks_paded[i] = F.pad(masks_paded[i], (0, 0, 0, 0, 0, vp_pad_size))
        clips_paded[i] = F.pad(batch_clips[i], (0, 0, 0, vp_pad_size))
        # print(masks_paded[i].shape)
    
    
    tensor_masks_paded = torch.stack(masks_paded)
    tensor_clips_paded = torch.stack(clips_paded)
    
    # To device
    tensor_masks_paded = tensor_masks_paded.to(device)
    tensor_clips_paded = tensor_clips_paded.to(device)
    # tensor_masks_paded is objnum * path(combination)num * vpnum * 1
    # tensor_clips_paded is objnum * vpnum * feature_dim
    # For example :
    # tensor_masks_paded is 16 5 3 1
    # tensor_clips_paded is 16 3 1024

    tensor_masks_paded = tensor_masks_paded.squeeze(-1).unsqueeze(2).permute(0,3,2,1)
    #tensor_masks_paded is objnum * vpnum * 1 * path(combination)num
    tensor_clips_paded = tensor_clips_paded.unsqueeze(3)
    #tensor_clips_paded is objnum * vpnum * feature_dim * 1
    tensor_merged_clips = tensor_masks_paded * tensor_clips_paded
    # tensor_merged_clips is objnum * vpnum * feature_dim * path(combination)num

    # obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
    #                    obj2['clip_ft'] * n_obj2_det) / (
    #                    n_obj1_det + n_obj2_det)
    # obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)
    
    # Merge the clips
    merged_clips = torch.mean(tensor_merged_clips, dim=1)
    merged_clips = merged_clips.permute(0, 2, 1)
    merged_clips = F.normalize(merged_clips, dim=2)
    # merged_clips is objnum * path(combination)num * feature_dim
    # Every Vp have one object view. So we don't need the weight.

    ## change merged_clips into numpy
    merged_clips_np = merged_clips.cpu().numpy()
    # print the space of merged_clips
    # print("#############################################")
    # print(f"Space of merged_clips for the whole scence: {merged_clips.nbytes}")
    # print("#############################################")
    # # Store the merged clips in the MapObjectList
    # for i, obj in enumerate(objects):
    #     obj['multi_clip'] = {}
    #     observed_info = obj.get('observed_info')
    #     if observed_info is None:
    #         continue
    #     vp_ids = list(observed_info.keys())
    #     observed_combinations = observed_combinations_list[i]  # Retrieve the results
    #     for j, combination in enumerate(observed_combinations):
    #         obj['multi_clip'][combination] = merged_clips[i, j]

    return merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos
def get_scene_map_etp(batch_obj_clip_np, batch_obj_pos_np, path_map_index):
    obj_indices = np.arange(batch_obj_clip_np.shape[0])
    valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
    valid_obs_combination = np.squeeze(valid_obs_combination)

    selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]

    mask = np.squeeze(path_map_index > 0)
    obj_clip = selected_obj_clip[mask]
    obj_pos = batch_obj_pos_np[mask]

    return obj_clip, obj_pos
def save_testing_data(scan, path_vps, vps_pos, gt_obj_pos):
    '''
    given the scan id and path idxs, using the machanism in etp to get the object pos
    and corresponding visual data
    '''
    assert gt_obj_pos is not None, "The gt_obj_pos is None"
    # read the cg files
    with h5py.File('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/cgobj_clip_90scans.hdf5', 'r') as f:
        batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
        paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
        batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)
    # get path_idx
    with open('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/all_paths_90scans.json', 'r') as f:
        all_paths_indices = json.load(f)
    paths_indices = all_paths_indices[scan]
    path_indx = paths_indices.index(path_vps)
      ## check the path_idx
    assert path_vps == paths_indices[path_indx], "The path idx is not correct"
    # get the map idx
    map_idx = paths_map_indices[path_indx]      #肯定是这个map数据出错了
    # get the obj pos correponding to the map
    def get_scene_map_etp(batch_obj_clip_np, batch_obj_pos_np, path_map_index):
        obj_indices = np.arange(batch_obj_clip_np.shape[0])
        valid_obs_combination = np.where(path_map_index >= 0, path_map_index, 0).astype(int)
        valid_obs_combination = np.squeeze(valid_obs_combination)

        selected_obj_clip = batch_obj_clip_np[obj_indices, valid_obs_combination,:]

        mask = np.squeeze(path_map_index > 0)
        # mask = path_map_index >= 0
        # print(f"mask.shape: {mask.shape}")
        
        obj_clip = selected_obj_clip[mask]
        obj_pos = batch_obj_pos_np[mask]

        return obj_clip, obj_pos
    _, obj_pos = get_scene_map_etp(batch_obj_clips, batch_obj_pos, map_idx)
    assert obj_pos.shape[0] == len(gt_obj_pos), f"The object pos is not correct, obj_pos size:{obj_pos.shape[0]}, gt_obj_pos size:{len(gt_obj_pos)}"
    # use the object pos to get the visual data
    save_visual_data_etp(obj_pos, vps_pos, scan)
def get_unique_vps(scan):
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
                "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
                "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
                "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]
    all_trajectory_info = get_all_paths4all_scans(json_dir)
    paths = get_paths4scan(scan, all_trajectory_info)
    print(f" get all the paths for scan {scan}, the length of paths is {len(paths)}")
    unique_viewpoints = get_unique_viewpoints(paths)
    return unique_viewpoints, paths
def load_gt_objlist():
    '''
    given scan and load the whole scene map
    '''
    scene_map_file = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data/D7N2EKCX4Sj_visual_cg_newdata.pkl.gz"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_map_file", type=str, default=scene_map_file)
    args = parser.parse_args()
    # from concept-graphs.conceptgraph.scenegraph.build_scenegraph_cfslam import load_scene_map
    # whole_scene_map = MapObjectList()
    print(f"Loading scene map gt object list ...")
    # load_scene_map_here(args,whole_scene_map)
    whole_scene_map, _, class_colors = load_result(args.scene_map_file)
    print("finish loading the gt object list")
    return whole_scene_map, class_colors
def saving_visual_data(scan, vps_pos, object_pos, whole_scene_map,class_colors,file_name = None):
    # load the whole scene map
    scene_map_folder = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    scene_map_file = scene_map_folder + f"/{scan}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    save_folder = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data"
    save_path = save_folder + f"/{scan}_{file_name}.pkl.gz"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_map_file", type=str, default=scene_map_file)
    parser.add_argument("--savefile", type=str, default=save_path)
    args = parser.parse_args()
    vis_fg = MapObjectList()
    print("Start searching the object in front ground...")
    get_vis_scene_map(whole_scene_map, vis_fg, object_pos)
    # print("Start searching the object in back ground...")
    # get_vis_scene_map(bg_objects, vis_bg, batch_obj_pos)
    # get viewpoint coordinates
    vps_coordinates = vps_pos
    # transform the pcd and bbox of objects
    # vis_fg, vps_coordinates = transform_pcd_bbox(vis_fg, vps_coordinates, vps_coordinates[-1],scan,path)
    # concate the result
    vis_result = {
        "objects": vis_fg.to_serializable(),
        "bg_objects": None,
        "class_colors": class_colors,
        "viewpoints": vps_coordinates,
    }
    # save the vis_scene_map
    if not os.path.exists(os.path.dirname(args.savefile)):
        os.makedirs(os.path.dirname(args.savefile), exist_ok=True)
    with gzip.open(args.savefile, "wb") as f:
        pickle.dump(vis_result, f)
        print(f"Save vis_scene_map to {args.savefile} successfully!")
def load_clip_pos(scan):
    '''
    given scan, load clip and pos from hdf5 files
    '''
    with h5py.File('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/cgobj_clip_90scans.hdf5', 'r') as f:
        batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
        # paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
        batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)
    return batch_obj_clips, batch_obj_pos
def generate_path_mask(path, all_vp):
    path_mask = torch.zeros((1,len(all_vp))).cuda()
    for path_vp in path:
        vp_indx = all_vp.index(path_vp)
        path_mask[0, vp_indx] = 1
    return path_mask
###############################################################
# 这就是主测试函数   
def get_correct_obj_pos(scan, path_idxs,vps_pos):
    '''
    achieve a correct mask technique for the object retrieval
    '''
    # load gt object list

    gt_object_list, _ = load_gt_objlist()
    # get the whole scene map
    whole_scene_map, class_colors = get_whole_scene_map(scan)
    # get all vp id in this scan
    unique_viewpoints, all_paths = get_unique_vps(scan)
    # testing path
    # path_testing = [path_idxs]
    # generate obj_obs_mask, paths_mask
    # _, _, path_testing_mask, gt_obj_pos = merger_clip4all_combinations_test(whole_scene_map, path_testing, unique_viewpoints)
    # _, test_obj_pos, _ = new_merge_function(whole_scene_map, path_testing, unique_viewpoints)
    # print(f"get {test_obj_pos.shape[0]} objects")
    # assert test_obj_pos.shape[0] == len(gt_object_list), "The object pos is not correct"
    # processing the whole scene map
    # scan_clips, scan_pos, scan_obs_masks, path_masks = new_merge_function(whole_scene_map, all_paths, unique_viewpoints)
    # generate path mask
    # testing_path_mask = generate_path_mask(path_idxs,unique_viewpoints)
    # get the obj indices 
    # save scan_clips, scan_pos, scan_obs_masks, path_masks
    # torch.save(scan_clips.cpu(), '/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/scan_clips.pt')
    # np.save('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/scan_pos.npy',scan_pos)
    # torch.save(scan_obs_masks.cpu(), '/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/scan_obs_masks.pt')
    # torch.save(path_masks.cpu(), '/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/path_masks.pt')
    # read the data
    scan_clips = torch.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/scan_clips.pt')
    scan_pos = np.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/scan_pos.npy')
    scan_obs_masks = torch.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/scan_obs_masks.pt')
    path_masks = torch.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/temp_data/path_masks.pt')
    ####
    all_obj_indices = get_obj_indices(scan_obs_masks.cuda(), path_masks.cuda())
    # build a path dict
    path_dict = {}
    for index, path in enumerate(all_paths):
        path_key = ','.join(path)
        path_dict[path_key] = index
    # get the path index
    path_key_to_find = ','.join(path_idxs)
    path_index = path_dict[path_key_to_find]

    map_idx = all_obj_indices[path_index]
    # check the non-negative count is equal to the gt_obj_pos or not
    non_negative_count = (map_idx != -1).sum().item()
    if non_negative_count == len(gt_object_list):
        print(" we may success!!!")
        print(" start to get the map")
        _, obj_pos_test = get_scene_map_etp(scan_clips.cpu().numpy(), scan_pos, map_idx.cpu())
        saving_visual_data(scan, vps_pos, obj_pos_test, whole_scene_map, class_colors, 'all_inducing')
    else:
        print(f"非 -1 的值的数量: {non_negative_count}")
    # save the visual data
    # saving_visual_data(scan, vps_pos, test_obj_pos, whole_scene_map, class_colors, 't_new_merge')
    # saving_visual_data(scan, path_idxs, vps_pos, test_obj_pos, 'new_merge')
    # generate scene map data for the whole scene
    # print("start to generate the scene map data for the whole scene")
    # merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos = merger_clip4all_combinations_test(whole_scene_map, all_paths, unique_viewpoints)
    # get the clip and pos
    # batch_obj_clips, batch_obj_pos = load_clip_pos(scan)
    # #
    # print("end to generate the scene map data for the whole scene")
    # # print space size of merged_clips_np, batch_clip_masks, paths_masks, batch_obj_pos in MB
    # print(f"merged_clips_np space size: {batch_obj_clips.nbytes / 1024 / 1024} MB")
    # # print(f"batch_clip_masks space size: {batch_clip_masks.nbytes / 1024 / 1024} MB")
    # # print(f"paths_masks space size: {paths_masks.nbytes / 1024 / 1024} MB")
    # print(f"batch_obj_pos space size: {batch_obj_pos.nbytes / 1024 / 1024} MB")
    
    # get obj indices
    # obj_indices = get_obj_indices(batch_clip_masks, paths_masks)

    # assert non_negative_count == len(gt_obj_pos), "The object pos is not correct"
    #
    # get obj pos
    # _, obj_pos_testing = get_scene_map_testing(scan, obj_indices)
    # assert obj_pos_testing.shape[0] == len(gt_obj_pos), f"The object pos is not correct, obj_pos size:{obj_pos_testing.shape[0]}, gt_obj_pos size:{len(gt_obj_pos)}"
    # print(" the size is the same!!! we success!!!")
    # # using the object pos to get the visual data
    # saving_visual_data(scan, vps_pos, obj_pos_testing, whole_scene_map, class_colors)
    
def get_scene_map_testing(scan, obj_indices):
    '''
    load clip and pos hdf5 file, and get the object clip and object pos corresponding to the obj_indices
    '''
    # load the hdf5 file
    with h5py.File('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/cgobj_clip_90scans.hdf5', 'r') as f:
        batch_obj_clips = f[f'{scan}_obj_clip_ft'][...].astype(np.float32)
        paths_map_indices = f[f'{scan}_paths_indices'][...].astype(np.float32)
        batch_obj_pos = f[f'{scan}_obj_pos'][...].astype(np.float32)
    _, obj_pos = get_scene_map_etp(batch_obj_clips, batch_obj_pos, obj_indices)
    return obj_pos
def get_whole_scene_map(scan):
    '''
    given scan and load the whole scene map
    '''
    scene_map_folder = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    scene_map_file = scene_map_folder + f"/{scan}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    save_folder = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data"
    save_path = save_folder + f"/{scan}_visual_cg.pkl.gz"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_map_file", type=str, default=scene_map_file)
    parser.add_argument("--savefile", type=str, default=save_path)
    args = parser.parse_args()
    # from concept-graphs.conceptgraph.scenegraph.build_scenegraph_cfslam import load_scene_map
    # whole_scene_map = MapObjectList()
    print(f"Loading scene map {scan} ...")
    # load_scene_map_here(args,whole_scene_map)
    whole_scene_map, _, class_colors = load_result(args.scene_map_file)
    print("finish loading the scene map")
    return whole_scene_map, class_colors

def save_visual_data_cg(path_idxs, scan):
    '''
    this function is for saving the visual data for input from cg project
    given vp_idxs and scan id, and then generate the visual data
    '''
    
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]
    all_trajectory_info = get_all_paths4all_scans(json_dir)
    paths = get_paths4scan(scan, all_trajectory_info)
    print(f" get all the paths for scan {scan}, the length of paths is {len(paths)}")
    unique_viewpoints = get_unique_viewpoints(paths)
    gt_obj_pos = save_scene_map(scan, path_idxs, unique_viewpoints)
    assert gt_obj_pos is not None, "The gt_obj_pos is None"
    
    return gt_obj_pos
    
def save_visual_data_etp(object_pos, vps_pos, scan, file_name = None):
    '''
    this function is for saving the visual data for input from etp project
    '''
    # load the whole scene map
    scene_map_folder = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    scene_map_file = scene_map_folder + f"/{scan}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    save_folder = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data"
    save_path = save_folder + f"/{scan}_{file_name}.pkl.gz"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_map_file", type=str, default=scene_map_file)
    parser.add_argument("--savefile", type=str, default=save_path)
    args = parser.parse_args()
    # from concept-graphs.conceptgraph.scenegraph.build_scenegraph_cfslam import load_scene_map
    # whole_scene_map = MapObjectList()
    print(f"Loading scene map {scan} ...")
    # load_scene_map_here(args,whole_scene_map)
    whole_scene_map, bg_objects, class_colors = load_result(args.scene_map_file)
    print(f"Loaded {len(whole_scene_map)} objects")    
    vis_fg = MapObjectList()
    vis_bg = MapObjectList()
    print("Start searching the object in front ground...")
    get_vis_scene_map(whole_scene_map, vis_fg, object_pos)
    # print("Start searching the object in back ground...")
    # get_vis_scene_map(bg_objects, vis_bg, batch_obj_pos)
    # get viewpoint coordinates
    vps_coordinates = vps_pos
    # transform the pcd and bbox of objects
    # vis_fg, vps_coordinates = transform_pcd_bbox(vis_fg, vps_coordinates, vps_coordinates[-1],scan,path)
    # concate the result
    vis_result = {
        "objects": vis_fg.to_serializable(),
        "bg_objects": None if vis_bg is None else vis_bg.to_serializable(),
        "class_colors": class_colors,
        "viewpoints": vps_coordinates,
    }
    # save the vis_scene_map

    if not os.path.exists(os.path.dirname(args.savefile)):
        os.makedirs(os.path.dirname(args.savefile), exist_ok=True)
    with gzip.open(args.savefile, "wb") as f:
        pickle.dump(vis_result, f)
        print(f"Save vis_scene_map to {args.savefile} successfully!")
def get_obj_indices(batch_obj_obs_mask, paths_mask):
    '''
    batch_obj_obs_mask: tensor shape (n_objects, n_obs_combinations, n_vp)
    paths_mask: tensor shape (n_paths, n_vp)

    Returns:
        A tensor of shape (n_paths, n_objects) indicating the index of the observation
        combination with the maximum observations that is a subset of each path.
    '''
    # # Expand dimensions to match for broadcasting
    # batch_obj_obs_mask = batch_obj_obs_mask.unsqueeze(0)  # (1, n_objects, n_obs_combinations, n_vp)
    # paths_mask = paths_mask.unsqueeze(1).unsqueeze(1)  # (n_paths, 1, 1, n_vp)

    # # Check if each observation combination is a subset of each path
    # non_zero_mask = batch_obj_obs_mask.any(dim=3)
    # is_subset = (batch_obj_obs_mask <= paths_mask).all(dim=3) & non_zero_mask  # Shape: (n_paths, n_objects, n_obs_combinations)
    # Calculate the number of observations in each combination
    obs_counts = batch_obj_obs_mask.sum(dim=-1)  # Shape: (1, n_objects, n_obs_combinations)
    ## batch processing version
    n_paths = paths_mask.shape[0]
    n_objects = batch_obj_obs_mask.shape[0]
    n_obs_combinations = batch_obj_obs_mask.shape[1]
    # 定义批次大小
    batch_size = 20

    # 计算批次数量
    num_batches = (n_paths + batch_size - 1) // batch_size

    # 初始化结果张量
    paths_map_indices = torch.zeros((n_paths, n_objects), dtype=torch.int).cuda()
    # is_subset_result = torch.zeros((n_paths, n_objects, n_obs_combinations), dtype=torch.bool).cuda()
    # valid_obs_counts = torch.zeros((n_paths, n_objects, n_obs_combinations), dtype=torch.int).cuda()
    batch_obj_obs_mask_expanded = batch_obj_obs_mask.unsqueeze(0)  # (1, n_objects, n_obs_combinations, n_vp)
    # 计算非零掩码
    non_zero_mask = batch_obj_obs_mask_expanded.any(dim=3)
    # 循环处理每个批次
    for i in tqdm(range(num_batches), desc="Processing paths map searching"):
        # start_time = time.time()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_paths)
        
        # 当前批次的张量
        paths_mask_batch = paths_mask[start_idx:end_idx]
        
        # 扩展维度以匹配广播
        # batch_obj_obs_mask_expanded = batch_obj_obs_mask.unsqueeze(0)  # (1, n_objects, n_obs_combinations, n_vp)
        paths_mask_batch_expanded = paths_mask_batch.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, n_vp)
        
        # 计算子集掩码
        is_subset_batch = (batch_obj_obs_mask_expanded <= paths_mask_batch_expanded).all(dim=3) & non_zero_mask  # Shape: (batch_size, n_objects, n_obs_combinations)
        valid_obs_counts_batch = obs_counts * is_subset_batch.float()  
        # get the sub result
        valid_obs_counts_batch[~is_subset_batch] = -1 # Shape: (n_paths, n_objects, n_obs_combinations)
        best_combination_indices_batch = valid_obs_counts_batch.argmax(dim=2)
        all_invalid_batch = (valid_obs_counts_batch == -1).all(dim=2)
        best_combination_indices_batch[all_invalid_batch] = -1
        # 存储结果
        paths_map_indices[start_idx:end_idx, :] = best_combination_indices_batch
        # is_subset_result[start_idx:end_idx, :, :] = is_subset_batch
        # valid_obs_counts[start_idx:end_idx, :, :] = valid_obs_counts_batch # Shape: (n_paths, n_objects, n_obs_combinations)
        # del is_subset_batch
        # del paths_mask_batch
        # del valid_obs_counts_batch
        # print(f" time for {i} batch is {time.time() - start_time} ")
 
    # is_subset_1 = torch.all((batch_obj_obs_mask & paths_mask) == batch_obj_obs_mask, dim=3)
    # print(is_subset)
    # print(is_subset_1)

    # Use the subset mask to mask out combinations that are not subsets
    # valid_obs_counts = obs_counts * is_subset_result.float()  # Shape: (n_paths, n_objects, n_obs_combinations)


    # valid_obs_counts[~is_subset_result] = -1

    # # Find the index of the combination with the maximum observations for each path and object
    # best_combination_indices = valid_obs_counts.argmax(dim=2).squeeze(dim=-1)  # Shape: (n_paths, n_objects)
    # 找到最大观测组合的索引
    # best_combination_indices = valid_obs_counts.argmax(dim=2)

    # 检查并调整全为 -1 的情况
    # all_invalid = (valid_obs_counts == -1).all(dim=2)
    # best_combination_indices[all_invalid] = -1
    return paths_map_indices
# test the combination function
def test_combination():
    # generate the test data
    obj_obs_mask = np.array([
        [[0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0]],
        [[0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0]],
        [[1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    ])
    print(f'obj_obs_mask shape: {obj_obs_mask.shape}')
    # generate path mask
    path_mask = np.array([[0,1,0,1,0]])
    # transform the mask to tensor
    obj_obs_mask = torch.from_numpy(obj_obs_mask)
    path_mask = torch.from_numpy(path_mask)
    best_combination_indices = get_obj_indices(obj_obs_mask, path_mask)
    print(f'best_combination_indices: {best_combination_indices}')
def get_all_paths4all_scans_here(json_list):
    all_paths_info = []
    scan_list = []
    # read each json file in json_list
    for json_file in json_list:
        with open(json_file, 'r') as f:
            for line in f:
            # data = json.load(f)
                data = json.loads(line)  # 解析每一行为JSON对象
                scan = data.get('scan')
                scan_list.append(scan)
            # for item in data:
            #     if "instructions" in item:
            #         print(" there is instructions in the json file !!!!!!!!!!!!!!")
                all_paths_info.append(data)  # 将解析后的数据添加到列表中
    return all_paths_info, scan_list
def main():
    '''
    this function is used to save the visual data for the scene
    given input as :D
        object pos
        vps pos
        scan id
    '''
    #read the object global pos for npy file
    object_pos = np.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data/object_global_save_new.npy')
    # read the vp global pos for npy file
    vps_pos = np.load('/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data/path_global_save.npy')
    # scan id
    scan = 'D7N2EKCX4Sj'
    # save the visual data
    save_visual_data_etp(object_pos, vps_pos, scan, "newest")
    # using the old but correct method to visualize
    path_idxs = ['e3c67078918d48a8a37abdbd38c61839', '1d7b7a08654f46df87604e7ae30f06b5', 
                 '61e6284b6ef541e59a87efa918514255', 'b3ea270a560d4fc784e7c7d4ca0e2248', 
                 '5bc65c559e2c4edc92ac6e9832d28ab1', '0a447b165b724cc8a73b00aafb9f8997']
    # get gt_obj_pos
    # gt_obj_pos = save_visual_data_cg(path_idxs, scan)
    # reproducing the visual data
    # save_testing_data(scan, path_idxs, vps_pos, gt_obj_pos)
    # get_correct_obj_pos(scan, path_idxs, vps_pos)
    # visualize the whole scen and sub path
    # visualize_full_scene_path(scan,vps_pos)
######################################
# 可视化完整场景和部分path
def visualize_full_scene_path(scan, path_pos):
    # initialize the param
    scene_map_folder = "/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R"
    scene_map_file = scene_map_folder + f"/{scan}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz"
    save_folder = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/visualize_data"
    save_path = save_folder + f"/{scan}_whole_scene_subpath.pkl.gz"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_map_file", type=str, default=scene_map_file)
    parser.add_argument("--savefile", type=str, default=save_path)
    args = parser.parse_args()
    # load the whole scene 
    whole_scene_map, class_colors = get_whole_scene_map(scan)
    # concate the result
    vis_result = {
        "objects": whole_scene_map.to_serializable(),
        "bg_objects": None,
        "class_colors": class_colors,
        "viewpoints": path_pos,
    }
    # save the vis_scene_map
    if not os.path.exists(os.path.dirname(args.savefile)):
        os.makedirs(os.path.dirname(args.savefile), exist_ok=True)
    with gzip.open(args.savefile, "wb") as f:
        pickle.dump(vis_result, f)
        print(f"Save vis_scene_map to {args.savefile} successfully!")

######################################
# add new data to hdf5 file, not cover it!
def save_path_ObjList2hdf5_add(scan, obj_clips:np.array, maps_indices:np.array, batch_obj_pos:np.array,file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'a') as f:
        clip_np = f.create_dataset(f'{scan}_obj_clip_ft', data=obj_clips)
        #save the paths_indices
        maps_indices_np = f.create_dataset(f'{scan}_maps_indices', data=maps_indices)
        # save the pos infomation
        obj_pos_np = f.create_dataset(f'{scan}_obj_pos', data=batch_obj_pos)
    
    print(f"Save obj_clips and paths_indices to {file_path} successfully!")
#####################################
# sometimes the produce processing can break down, so we need to reproduce the visual data
# this function is used to continue the processing
def get_left_scan_list():
    hdf5_file = '/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/training_cg_data.hdf5'
    # 打开 HDF5 文件
    with h5py.File(hdf5_file, 'r') as f:
        scan_list = set()
        
        # 遍历所有数据集
        for dataset_name in f.keys():
            # 提取 scan 部分
            scan = dataset_name.split('_')[0]
            scan_list.add(scan)
            print(f" scan: {scan} has been proessed")
    return list(scan_list)


######################################
# this function for producing all the training data
def save_new_cg_data():
    # get training scan list
    json_dir =["/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_train_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_prevalent_aug_train_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_seen_enc.jsonl",
        "/data0/vln_datasets/matterport3d/BEVBert_dataset/datasets/R2R/annotations/pretrain_map/R2R_val_unseen_enc.jsonl"]
    
    _, full_scan_list = get_all_paths4all_scans_here(json_dir) 
    full_scan_list = list(set(full_scan_list)) # only contain 72 scans
    # delete the scan that has been processed
    processed_scan_list = get_left_scan_list()
    scan_list = list(set(full_scan_list) - set(processed_scan_list))
    print(f"there are {len(scan_list)} scans left to process")

    hdf5_file_path = '/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/' + 'training_cg_data.hdf5'
    # original_time = time.time()
    for scan in tqdm(scan_list, desc="Processing scans"):
        # scan_start_time = time.time()
        # load the whole scene map: object list
        whole_scene_map, _ = get_whole_scene_map(scan)
        # get all the paths for the scan, including the sub paths, and unique viewpoints
        unique_viewpoints, all_full_paths = get_unique_vps(scan)
        all_paths = generate_subpaths(all_full_paths) # including full paths and sub paths
        # get the merge the clips for all the combinations
        scan_clips, scan_pos, scan_obs_masks, path_masks = new_merge_function(whole_scene_map, all_paths, unique_viewpoints)
        # generate the paths dict
        all_map_indices = get_obj_indices(scan_obs_masks.cuda(), path_masks.cuda())

        # print the space size of training_path_dict
        # print(f"Space of training_path_dict: {sys.getsizeof(training_path_dict) / 1024 / 1024} MB")
        # save hdf5 file for clip and pos, pkl file for path dict
        save_path_ObjList2hdf5_add(scan, scan_clips.cpu().numpy(), all_map_indices.cpu().numpy(), scan_pos, hdf5_file_path)
        # print(f"scan {scan} finished, time cost: {time.time() - scan_start_time}")
    # save the path dict
    training_path_dict = {}
    dict_file_path = '/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/cg_process/all_cg_data/' + 'scan_map_dict.pkl'
    os.makedirs(os.path.dirname(dict_file_path), exist_ok=True)
    for scan in tqdm(full_scan_list, desc="Saving path dict"):
        unique_viewpoints, all_full_paths = get_unique_vps(scan)
        all_paths = generate_subpaths(all_full_paths) # including full paths and sub paths
        # build a path dict
        path_dict = {}
        for index, path in enumerate(all_paths):
            path_key = ','.join(path)
            path_dict[path_key] = index
        training_path_dict[scan] = path_dict
    # save the path dict to the pkl file
    with open(dict_file_path, 'wb') as file:
        pickle.dump(training_path_dict, file)
    print(" finish all data processing!!!!!!!!!!!!!!!")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # save_new_cg_data()
    main()
    # test_combination()