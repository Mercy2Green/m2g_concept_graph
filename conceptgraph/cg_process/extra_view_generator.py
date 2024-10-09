import os
import sys
import math
import json
from grpc import Compression
from tqdm import tqdm
import numpy as np
import h5py
from progressbar import ProgressBar
import torch.multiprocessing as mp
import argparse
import cv2

# MatterSim
sys.path.insert(0, '/home/lg1/peteryu_workspace/BEV_HGT_VLN/Matterport3DSimulator_opencv4/build')  # please compile Matterport3DSimulator using cpu_only mode
sys.path.append('/home/lg1/lujia/VLN_HGT/bevbert/')

import MatterSim

# export PYTHONPATH=$PYTHONPATH:/home/lg1/lujia/VLN_HGT/bevbert/
from precompute_features.utils.habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R

from configparser import ConfigParser

# VIEWPOINT_SIZE = 12
# WIDTH = 224
# HEIGHT = 224
# VFOV = 90
# HFOV = 90

class SENSOR():
    
    def __init__(
        self,
        ID,
        TYPE,
    ):
        pass
    
    def get_data(self):
        pass

class VIEW_GENERATOR():
    
    def __init__(
        self, 
        ):
        
        self.sim = None
        self.habitat_sim = None
        
        self.camera_intrinsics = None
        self.sensor_config = None
        self.save_file_config = None
        
        self.depth_scale = 255
        
        self.get_sensor_config(
            sensor_config_path=None,
            VIEWPOINT_SIZE=12,
            IMG_WIDTH=224,
            IMG_HEIGHT=224,
            VFOV=90,
            HFOV=90
        )
        
        if self.get_sensor_config is not None:
            self.get_camera_intrinsics()
        
        self.get_save_file_config(
            save_file_config_path=None,
            connectivity_dir='/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity',
            scan_dir='/data0/vln_datasets/mp3d/v1/tasks/mp3d',
            output_file=None,
            num_workers=1,
            img_type='rgb',
            save_data_root='/data2/vln_dataset/preprocessed_data/preprocessed_habitiat_R2R',
            )    
        
    def get_sensor_config(
        self,
        sensor_config_path=None,
        VIEWPOINT_SIZE=12,
        IMG_WIDTH=224,
        IMG_HEIGHT=224,
        VFOV=90,
        HFOV=90,
    ):
            
        if sensor_config_path is not None:
            raise NotImplementedError
        else:
            sensor_config_dict = {}
            sensor_config_dict['VIEWPOINT_SIZE'] = VIEWPOINT_SIZE
            sensor_config_dict['IMG_WIDTH'] = IMG_WIDTH
            sensor_config_dict['IMG_HEIGHT'] = IMG_HEIGHT
            sensor_config_dict['VFOV'] = VFOV
            sensor_config_dict['HFOV'] = HFOV
        
        self.sensor_config = sensor_config_dict
        
        return self.sensor_config
    
    # def get_camera_intrinsics(self):
        
    #     HFOV = self.sensor_config['HFOV']
    #     VFOV = self.sensor_config['VFOV']
    #     HEIGHT = self.sensor_config['IMG_HEIGHT']
    #     WIDTH = self.sensor_config['IMG_WIDTH']
        
    #     camera_intrinsics = np.array([
    #         [((1 / np.tan(math.radians(HFOV) / 2.))*WIDTH) / 2, 0., WIDTH/2, 0.],
    #         [0., ((1 / np.tan(math.radians(HFOV) / 2.))*HEIGHT) / 2, HEIGHT/2, 0.],
    #         [0., 0., 1, 0],
    #         [0., 0., 0, 1]])
        
    #     self.camera_intrinsics = camera_intrinsics
        
    #     return self.camera_intrinsics
    
    def get_camera_intrinsics(self):
        
        HFOV = self.sensor_config['HFOV']
        VFOV = self.sensor_config['VFOV']
        HEIGHT = self.sensor_config['IMG_HEIGHT']
        WIDTH = self.sensor_config['IMG_WIDTH']
        
        # Calculate the focal lengths using HFOV and VFOV
        fx = (1 / np.tan(math.radians(HFOV) / 2.)) * (WIDTH / 2)
        fy = (1 / np.tan(math.radians(VFOV) / 2.)) * (HEIGHT / 2)
        
        # Construct the camera intrinsics matrix
        camera_intrinsics = np.array([
            [fx, 0., WIDTH / 2, 0.],
            [0., fy, HEIGHT / 2, 0.],
            [0., 0., 1, 0],
            [0., 0., 0, 1]
        ])
        
        self.camera_intrinsics = camera_intrinsics
        
        return self.camera_intrinsics
            
        
    def get_save_file_config(
        self,
        save_file_config_path=None,
        connectivity_dir='/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity',
        scan_list_file='scans.txt',
        scan_dir='/data0/vln_datasets/mp3d/v1/tasks/mp3d',
        output_file=None,
        num_workers=1,
        img_type='rgb',
        save_data_root='/data2/vln_dataset/preprocessed_data/preprocessed_habitiat_R2R',
        ):
        
        if save_file_config_path is not None:
            raise NotImplementedError
        else:
            save_file_config_dict = {}
            save_file_config_dict['connectivity_dir'] = connectivity_dir
            save_file_config_dict['scan_dir'] = scan_dir
            save_file_config_dict['output_file'] = output_file
            save_file_config_dict['num_workers'] = num_workers
            save_file_config_dict['img_type'] = img_type
            save_file_config_dict['save_data_root'] = save_data_root
            save_file_config_dict['scan_list_file'] = scan_list_file
        
        self.save_file_config = save_file_config_dict
        
        return self.save_file_config 
            
    def build_simulator(self):
        sim = MatterSim.Simulator()
        sim.setNavGraphPath(self.save_file_config['connectivity_dir'])
        sim.setDatasetPath(self.save_file_config['scan_dir'])
        sim.setCameraResolution(self.sensor_config['IMG_WIDTH'], self.sensor_config['IMG_HEIGHT'])
        sim.setCameraVFOV(math.radians(self.sensor_config['VFOV']))
        sim.setDiscretizedViewingAngles(True)
        sim.setRenderingEnabled(False)
        sim.setDepthEnabled(False)
        sim.setPreloadingEnabled(False)
        sim.setBatchSize(1)
        sim.initialize()
        
        self.sim = sim
        
        return self.sim
    
    def init_habitat_sim(self, glb_path):
        
        if self.sim == None:
            self.build_simulator()
        
        if self.habitat_sim != None:
            self.habitat_sim.sim.close()
        
        habitat_sim = HabitatUtils(glb_path, int(0), self.sensor_config['HFOV'], self.sensor_config['IMG_HEIGHT'], self.sensor_config['IMG_WIDTH'])
        
        self.habitat_sim = habitat_sim
        
        return self.habitat_sim
    
    def get_R2R_vp_views(self):
        
        # End 2 End R2R vp views Viewpoint Generator
        
        self.build_img_file()
    
    def get_img(self, proc_id, out_queue, scanvp_list):
        
        HFOV = self.sensor_config['HFOV']
        HEIGHT = self.sensor_config['IMG_HEIGHT']
        WIDTH = self.sensor_config['IMG_WIDTH']
        VIEWPOINT_SIZE = self.sensor_config['VIEWPOINT_SIZE']
        
        print('start proc_id: %d' % proc_id)

        # Set up the simulator
        sim = self.build_simulator(self.save_file_config['connectivity_dir'], self.save_file_config['scan_dir']) #MatterSim

        # print(scanvp_list)
        
        pre_scan = None
        habitat_sim = None
        for scan_id, viewpoint_id in scanvp_list:
            if scan_id != pre_scan:
                if habitat_sim != None:
                    habitat_sim.sim.close()
                habitat_sim = HabitatUtils(f'{self.save_file_config.scan_dir}/{scan_id}/{scan_id}.glb', 
                                        int(0), HFOV, HEIGHT, WIDTH)
                pre_scan = scan_id

            camera_intrinsics = np.array([
                [((1 / np.tan(math.radians(HFOV) / 2.))*WIDTH) / 2, 0., WIDTH/2, 0.],
                [0., ((1 / np.tan(math.radians(HFOV) / 2.))*HEIGHT) / 2, HEIGHT/2, 0.],
                [0., 0., 1, 0],
                [0., 0., 0, 1]])
            
            # # create a .txt file save the camera_intrinsics, and only save fx, fy, cx, cy
            # os.makedirs(f"/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R/{scan_id}", exist_ok=True)
            # with open(f"/data0/vln_datasets/preprocessed_data/preprocessed_habitiat_R2R/{scan_id}/camera_intrinsics_{scan_id}.txt", 'w') as f:
            #     f.write(f"{camera_intrinsics[0, 0]} {camera_intrinsics[1, 1]} {camera_intrinsics[0, 2]} {camera_intrinsics[1, 2]}")

            transformation_matrix_list = []
            cg_transformation_matrix_list = []
            mattersim_transformation_matrix_list = []
            test_transformation_matrix_list = []
            images = []
            depths = []
            images_name_list = []
            depths_name_list = []
            for ix in range(VIEWPOINT_SIZE): #Start MatterSim
                if ix == 0:
                    # sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
                    sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
                # elif ix % 12 == 0:
                #     sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [1.0], [0])
                state = sim.getState()[0]
                # assert state.viewIndex == ix

                x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
                habitat_position = [x, z-1.25, -y]
                mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
                mp3d_e = np.array([e, 0, 0])
                rotvec_h = R.from_rotvec(mp3d_h)
                rotvec_e = R.from_rotvec(mp3d_e)
                habitat_rotation = (rotvec_h * rotvec_e).as_quat()
                habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)
                
                # This is for Test transformation # This is the finaly transformation.
                test_x_position = habitat_position
                test_h = mp3d_h
                test_e = np.array([e+math.pi, 0, 0])
                test_rotation = (R.from_rotvec(test_h) * R.from_rotvec(test_e)).as_quat()
                test_T = np.zeros([4, 4])
                test_T[:3, :3] = R.from_quat(test_rotation).as_matrix()
                test_T[:3, 3] = np.array(test_x_position)
                test_T[3, 3] = 1
                test_T = test_T.reshape(-1)
                test_T = test_T.tolist()
                test_transformation_matrix_list.append(test_T)
                

                ## This is the concept graph transformation
                cg_x = x
                cg_y = -(z-1.25)
                cg_z = y 
                cg_h = h # in concept graph is not counter-clock heading 
                cg_e = e
                
                cx = np.cos(cg_e)
                sx = np.sin(cg_e)
                cy = np.cos(cg_h)
                sy = np.sin(cg_h)

                cg_T = np.zeros([4, 4])
                cg_T[0,0] = cy
                cg_T[0,1] = sx*sy
                cg_T[0,2] = cx*sy
                cg_T[0,3] = cg_x
                cg_T[1,0] = 0
                cg_T[1,1] = cx
                cg_T[1,2] = -sx
                cg_T[1,3] = cg_y
                cg_T[2,0] = -sy
                cg_T[2,1] = cy*sx
                cg_T[2,2] = cy*cx
                cg_T[2,3] = cg_z
                cg_T[3,3] = 1
                cg_T = cg_T.astype(np.float32)
                
                cg_T = cg_T.reshape(-1)
                cg_T = cg_T.tolist()
                cg_transformation_matrix_list.append(cg_T)
                
                ## This is habitat transformation
                # x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
                # habitat_position = [x, z-1.25, -y]
                # mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
                # mp3d_e = np.array([e, 0, 0])
                # rotvec_h = R.from_rotvec(mp3d_h)
                # rotvec_e = R.from_rotvec(mp3d_e)
                # habitat_rotation = (rotvec_h * rotvec_e).as_quat()
                # habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)
                rotation = R.from_quat(habitat_rotation)
                rotation_matrix = rotation.as_matrix()
                
                T = np.eye(4)
                T[:3, :3] = rotation_matrix
                T[:3, 3] = np.array(habitat_position)
                T[3, 3] = 1
                
                T = T.reshape(-1)
                T = T.tolist()
                transformation_matrix_list.append(T)
                
                ## This is matterport3D transformation

                # mp3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
                # mp3d_e = np.array([e, 0, 0])
                # rotvec_h = R.from_rotvec(mp3d_h)
                # rotvec_e = R.from_rotvec(mp3d_e)
                mattersim_position = [x, y, z]
                mattersim_h = np.array([0, h, 0]) # clock heading
                mattersim_e = np.array([e, 0, 0])
                mattersim_rotation = (R.from_rotvec(mattersim_h) * R.from_rotvec(mattersim_e)).as_quat()
                
                mattersim_T = np.zeros([4, 4])
                mattersim_T[:3, :3] = R.from_quat(mattersim_rotation).as_matrix()
                mattersim_T[:3, 3] = np.array(mattersim_position)
                mattersim_T[3, 3] = 1
                
                mattersim_T = mattersim_T.reshape(-1)
                mattersim_T = mattersim_T.tolist()
                
                mattersim_transformation_matrix_list.append(mattersim_T)
                
                # image and depth
                image = habitat_sim.render('rgb')[:, :, ::-1]
                image_name = f"{viewpoint_id}_i_{ix}.jpg"
                
                depth = habitat_sim.render('depth')
                # each depth value is in [0, 1], we need to convert it to [0, 255]
                depth = (depth * self.depth_scale).astype(np.uint8)
                depth_name = f"{viewpoint_id}_d_{ix}.png"

                images_name_list.append(image_name)
                depths_name_list.append(depth_name)

                images.append(image)
                depths.append(depth)
            images = np.stack(images, axis=0)
            out_queue.put((scan_id, viewpoint_id, images, images_name_list, depths, depths_name_list, transformation_matrix_list, camera_intrinsics, cg_transformation_matrix_list, mattersim_transformation_matrix_list, test_transformation_matrix_list))

        out_queue.put(None)

    def build_img_file(self):
        
        # scanvp_list = load_viewpoint_ids(args.connectivity_dir)
        scanvp_list = self.load_viewpoint_ids(self.save_file_config['connectivity_dir'], self.save_file_config['scan_list_file'])

        num_workers = min(self.save_file_config['num_workers'], len(scanvp_list))
        num_data_per_worker = len(scanvp_list) // num_workers

        out_queue = mp.Queue()
        processes = []
        for proc_id in range(num_workers):
            sidx = proc_id * num_data_per_worker
            eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

            process = mp.Process(
                target=self.get_img,
                args=(proc_id, out_queue, scanvp_list[sidx: eidx])
            )
            process.start()
            processes.append(process)
        
        num_finished_workers = 0
        num_finished_vps = 0

        progress_bar = ProgressBar(max_value=len(scanvp_list))
        progress_bar.start()

        config = ConfigParser()
        
        save_root = self.save_file_config['save_data_root']
        os.makedirs(f"{save_root}", exist_ok=True)

        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, images, images_name_list, depths, depths_name_list, transformation_matrix_list, camera_intrinsics, cg_transformation_matrix_list, mattersim_transormation_matrix_list, test_transformation_matrix_list = res
                # create a .txt file save the camera_intrinsics, and only save fx, fy, cx, cy
                os.makedirs(f"{save_root}/{scan_id}", exist_ok=True)
                os.makedirs(f"{save_root}/{scan_id}/camera_parameter", exist_ok=True)
                os.makedirs(f"{save_root}/{scan_id}/color_image", exist_ok=True)
                os.makedirs(f"{save_root}/{scan_id}/depth_image", exist_ok=True)

                # Add parameters to config
                config[viewpoint_id] = {
                    'scan_id': scan_id,
                    'viewpoint_id': viewpoint_id,
                    'images_name_list': images_name_list,
                    'depths_name_list': depths_name_list,
                    'habitat_poses_list': transformation_matrix_list,
                    'cg_poses_list': cg_transformation_matrix_list,
                    'ms_poses_list': mattersim_transormation_matrix_list,
                    'poses_list': test_transformation_matrix_list,
                    'camera_intrinsics': camera_intrinsics.tolist(),  # convert numpy array to list

                }
                with open(f"{save_root}/{scan_id}/camera_parameter/camera_parameter_{scan_id}.conf", 'w') as f:
                    config.write(f)

                for idx in range(len(images_name_list)):
                    cv2.imwrite(f"{save_root}/{scan_id}/color_image/{images_name_list[idx]}", images[idx])
                    cv2.imwrite(f"{save_root}/{scan_id}/depth_image/{depths_name_list[idx]}", depths[idx])

                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

        progress_bar.finish()
        for process in processes:
            process.join()


    def get_connectivity_dict(self, connectivity_dir, connectivity_file_name='scans.txt'):
        
        ## The output is a dict of dict
        ## The first key is the scan_id
        ## The second key is the viewpoint_id
        
        connectivity_dict = {}
        
        viewpoint_ids = []
        
        with open(os.path.join(connectivity_dir, connectivity_file_name)) as f:
            scans = [x.strip() for x in f] # load all scans
        
        for scan in scans:                
            connectivity_dict[scan] = {}
            with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
                data = json.load(f)
                for x in data:
                    viewpoint_ids.append(x['image_id'])
                    connectivity_dict[scan][x['image_id']] = {}
                    connectivity_dict[scan][x['image_id']]['unobstructed'] = x['unobstructed']
                    connectivity_dict[scan][x['image_id']]['visible'] = x['visible']
                    connectivity_dict[scan][x['image_id']]['included'] = x['included']
        return connectivity_dict
    
    def get_viewpoint_neighbors(self, connectivity_dict, scan_id, viewpoint_id, key='visible'):
        ## The output is a list of viewpoint_id
        viewpoint_list = list(connectivity_dict[scan_id].keys())
        neighbors = []
        # The connectivity_dict[scan_id][viewpoint_id][key] is a list of true or false
        for i, is_neighbor in enumerate(connectivity_dict[scan_id][viewpoint_id][key]):
            if is_neighbor:
                neighbors.append(viewpoint_list[i])
        return neighbors
    
    def load_extra_position_dict(self, extra_position_dict_path):
        """
        Load the extra_position_dict from a .json file.
        
        :param extra_position_dict_path: Path to the .json file containing the extra positions and rotations.
        :return: Dictionary containing extra positions and rotations.
        """
        with open(extra_position_dict_path, 'r') as json_file:
            extra_position_dict = json.load(json_file)
        
        return extra_position_dict
    
    def multi_process_extra_view(self, proc_id, out_queue, sub_extra_position_dict, save_root=None):
        
        # The extra_position_dict should be one scan dict
        
        print('start proc_id: %d' % proc_id)

        self.extra_view_generate(sub_extra_position_dict, True, save_root)
        
        out_queue.put(None)
    
    def multi_process_extra_view_generate(self, extra_position_dict, save_root):
        
        # Load the extra_position_dict
        # extra_position_dict = self.load_extra_position_dict(self.save_file_config['extra_position_dict_path'])
        scan_list = list(extra_position_dict.keys())
        
        num_workers = min(self.save_file_config['num_workers'], len(scan_list))
        num_data_per_worker = len(scan_list) // num_workers
        
        out_queue = mp.Queue()
        processes = []
        for proc_id in range(num_workers):
            sidx = proc_id * num_data_per_worker
            eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker
            process = mp.Process(
                target=self.multi_process_extra_view,
                args=(proc_id, out_queue, {scan: extra_position_dict[scan] for scan in scan_list[sidx: eidx]}, save_root)
            )
            process.start()
            processes.append(process)
        
        num_finished_workers = 0
        num_finished_scans = 0
        
        progress_bar = ProgressBar(max_value=len(scan_list))
        progress_bar.start()
        
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan, interp_viewpoint_id, rgb_image_list, rgb_image_name_list, depth_image_list, depth_image_name_list, camera_intrinsics, pose_list, cg_pose_list, habitat_pose_list, mattersim_pose_list = res
                self.save_extra_view(
                    save_root, 
                    scan, 
                    interp_viewpoint_id, 
                    rgb_image_list, 
                    rgb_image_name_list,
                    depth_image_list, 
                    depth_image_name_list,
                    camera_intrinsics, 
                    pose_list, 
                    cg_pose_list, 
                    habitat_pose_list, 
                    mattersim_pose_list)
                
                num_finished_scans += 1
                progress_bar.update(num_finished_scans)

    
    def extra_view_generate(self, extra_position_dict, save_flag=False, save_root=None):
        
        HFOV = self.sensor_config['HFOV']
        HEIGHT = self.sensor_config['IMG_HEIGHT']
        WIDTH = self.sensor_config['IMG_WIDTH']
        
        habitat_sim = None
        
        camera_intrinsics = self.get_camera_intrinsics()
        
        # Set up the simulator
        sim = self.build_simulator()
        
        for scan in tqdm(list(extra_position_dict.keys()), desc="Processing scans"):
            
            if habitat_sim is not None:
                habitat_sim.sim.close()
            _scan_dir = self.save_file_config['scan_dir']
            _path = f'{_scan_dir}/{scan}/{scan}.glb'
            habitat_sim = HabitatUtils(_path, int(0), HFOV, HEIGHT, WIDTH)
            
            rgb_image_list = []
            rgb_image_name_list = []
            depth_image_list = []
            depth_image_name_list = []
            pose_list = []
            cg_pose_list = []
            habitat_pose_list = []
            mattersim_pose_list = []
            
            for interp_viewpoint_id in list(extra_position_dict[scan].keys()):
                for data in extra_position_dict[scan][interp_viewpoint_id]:
                    
                    mattersim_position = [data['mattersim_x'], data['mattersim_y'], data['mattersim_z']]
                    mattersim_h = data['mattersim_h']
                    mattersim_e = data['mattersim_e']
                    interpolation = data['interpolation']
                    view = data['view']
                    start_vp = data['start_vp']
                    end_vp = data['end_vp']

                    habitat_position = [mattersim_position[0], mattersim_position[2] - 1.25, -mattersim_position[1]]
                    mp3d_h = np.array([0, 2 * math.pi - mattersim_h, 0])  # counter-clock heading
                    mp3d_e = np.array([mattersim_e, 0, 0])
                    rotvec_h = R.from_rotvec(mp3d_h)
                    rotvec_e = R.from_rotvec(mp3d_e)
                    habitat_rotation = (rotvec_h * rotvec_e).as_quat()

                    habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)
                    
                    # image and depth
                    rgb_image = habitat_sim.render('rgb')[:, :, ::-1]
                    depth_image = habitat_sim.render('depth')
                    # each depth value is in [0, 1], we need to convert it to [0, 255]
                    depth_image = (depth_image * self.depth_scale).astype(np.uint8)
                    
                    rgb_image_name = f"{interp_viewpoint_id}_i_{interpolation}_v_{view}.jpg"
                    depth_image_name = f"{interp_viewpoint_id}_i_{interpolation}_v_{view}.png"
                    
                    # Pose
                    pose = self.get_pose(mattersim_position, mattersim_h, mattersim_e)
                    cg_pose = self.get_cg_pose(mattersim_position, mattersim_h, mattersim_e)
                    habitat_pose = self.get_habitat_pose(mattersim_position, mattersim_h, mattersim_e)
                    mattersim_pose = self.get_mattersim_pose(mattersim_position, mattersim_h, mattersim_e)
                    
                    # list append
                    rgb_image_list.append(rgb_image)
                    rgb_image_name_list.append(rgb_image_name)
                    depth_image_list.append(depth_image)
                    depth_image_name_list.append(depth_image_name)
                    pose_list.append(pose)
                    cg_pose_list.append(cg_pose)
                    habitat_pose_list.append(habitat_pose)
                    mattersim_pose_list.append(mattersim_pose)
            
            # save the data
            rgb_image_list = np.stack(rgb_image_list,axis=0)
            
            if save_flag:
                self.save_extra_view(
                    save_root, 
                    scan, 
                    interp_viewpoint_id, 
                    rgb_image_list, 
                    rgb_image_name_list,
                    depth_image_list, 
                    depth_image_name_list,
                    camera_intrinsics, 
                    pose_list, 
                    cg_pose_list, 
                    habitat_pose_list, 
                    mattersim_pose_list)
            
            # return scan, interp_viewpoint_id, rgb_image_list, rgb_image_name_list, depth_image_list, depth_image_name_list, camera_intrinsics, pose_list, cg_pose_list, habitat_pose_list, mattersim_pose_list
            
    def save_extra_view(
        self, 
        save_path_root, 
        scan_id, 
        interp_viewpoint_id, 
        rgb_image_list,
        rgb_image_name_list, 
        depth_image_list, 
        depth_image_name_list,
        camera_intrinsics, 
        pose_list, 
        cg_pose_list, 
        habitat_pose_list, 
        mattersim_pose_list):
        
        os.makedirs(save_path_root, exist_ok=True)
        os.makedirs(f"{save_path_root}/{scan_id}", exist_ok=True)
        os.makedirs(f"{save_path_root}/{scan_id}/color_image", exist_ok=True)
        os.makedirs(f"{save_path_root}/{scan_id}/depth_image", exist_ok=True)
        os.makedirs(f"{save_path_root}/{scan_id}/camera_parameter", exist_ok=True)
        
        config = ConfigParser()
        config[interp_viewpoint_id] = {
            'scan_id': scan_id,
            'interp_viewpoint_id': interp_viewpoint_id,
            'images_name_list': rgb_image_name_list,
            'depths_name_list': depth_image_name_list,
            'poses_list': pose_list,
            'habitat_poses_list': habitat_pose_list,
            'cg_poses_list': cg_pose_list,
            'ms_poses_list': mattersim_pose_list,
            'camera_intrinsics': camera_intrinsics.tolist(),  # convert numpy array to list
        }
        
        with open(f"{save_path_root}/{scan_id}/camera_parameter/camera_parameter_{scan_id}.conf", 'w') as f:
            config.write(f)
            
        for idx in range(len(rgb_image_list)):
            cv2.imwrite(f"{save_path_root}/{scan_id}/color_image/{rgb_image_name_list[idx]}", rgb_image_list[idx])
            cv2.imwrite(f"{save_path_root}/{scan_id}/depth_image/{depth_image_name_list[idx]}", depth_image_list[idx])     
                    
                    
    def get_view_from_postion(self, position, heading, elevation):
        # position is [x, y, z] in Matterport3D
        # Will return the RGB-D and transformation matrix
        
        if self.sim == None:
            raise ValueError('Simulator is not initialized')
        
        habitat_position = [position[0], position[2] - 1.25, -position[1]]
        mp3d_h = np.array([0, 2 * math.pi - heading, 0])  # counter-clock heading
        mp3d_e = np.array([elevation, 0, 0])
        rotvec_h = R.from_rotvec(mp3d_h)
        rotvec_e = R.from_rotvec(mp3d_e)
        habitat_rotation = (rotvec_h * rotvec_e).as_quat()
        self.habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)
        
        pose = self.get_pose(position, heading, elevation)
        cg_pose = self.get_cg_pose(position, heading, elevation)
        habitat_pose = self.get_habitat_pose(position, heading, elevation)
        mattersim_pose = self.get_mattersim_pose(position, heading, elevation)
        
        # image and depth
        rgb_image = self.habitat_sim.render('rgb')[:, :, ::-1]
        depth_image = self.habitat_sim.render('depth')
        # each depth value is in [0, 1], we need to convert it to [0, 255]
        depth_image = (depth_image * self.depth_scale).astype(np.uint8)
        
        return rgb_image, depth_image, pose, cg_pose, habitat_pose, mattersim_pose
        

    def extra_position_generate(self, VIEWPOINT_SIZE, num_interpolations=4, scan_save_flag=False):
        
        HFOV = self.sensor_config['HFOV']
        HEIGHT = self.sensor_config['IMG_HEIGHT']
        WIDTH = self.sensor_config['IMG_WIDTH']
        
        extra_position_dict = {}
        
        connectivity_dict = self.get_connectivity_dict(self.save_file_config['connectivity_dir'], self.save_file_config['scan_list_file'])
        # The first key is the scan_id, the second key is the viewpoint_id
        
        habitat_sim = None
        
        # Set up the simulator
        sim = self.build_simulator() # MatterSim
        
        for scan in tqdm(list(connectivity_dict.keys()), desc="Processing scans"):
            # ### This is for test
            # if scan != '17DRP5sb8fy':
            #     continue
            
            extra_position_dict[scan] = {}
            
            if habitat_sim is not None:
                habitat_sim.sim.close()
            _scan_dir = self.save_file_config['scan_dir']
            _path = f'{_scan_dir}/{scan}/{scan}.glb'
            habitat_sim = HabitatUtils(_path, int(0), HFOV, HEIGHT, WIDTH)
            
            for viewpoint in list(connectivity_dict[scan].keys()):
                if connectivity_dict[scan][viewpoint]['included'] == False:
                    continue
                # set the mattersim
                sim.newEpisode([scan], [viewpoint], [0], [0])
                state = sim.getState()[0]
                x, y, z = state.location.x, state.location.y, state.location.z
                
                # get neighbors
                neighbors = self.get_viewpoint_neighbors(connectivity_dict, scan, viewpoint)
                for nv in neighbors:
                    if connectivity_dict[scan][nv]['included'] == False:
                        continue       
                        
                    interp_viewpoint_id = f'{viewpoint}_{nv}'
                    extra_position_dict[scan][interp_viewpoint_id] = []
                    
                    sim.newEpisode([scan], [nv], [0], [0])
                    nv_state = sim.getState()[0]
                    nv_x, nv_y, nv_z = nv_state.location.x, nv_state.location.y, nv_state.location.z
                    
                    # Linear interpolation between previous and current positions
                    for i in range(1, num_interpolations):
                        t = i / (num_interpolations + 1)
                        interp_x = x + t * (nv_x - x)
                        interp_y = y + t * (nv_y - y)
                        interp_z = z + t * (nv_z - z)
                        
                        for ix in range(VIEWPOINT_SIZE): 
                            if ix == 0:
                                sim.newEpisode([scan], [viewpoint], [0], [0])
                            else:
                                sim.makeAction([0], [1.0], [0])
                            view_state = sim.getState()[0]
                            h, e = view_state.heading, view_state.elevation

                            # interp_habitat_position = [interp_x, interp_z - 1.25, -interp_y]
                            # interp_mp3d_h = np.array([0, 2 * math.pi - h, 0])
                            # interp_mp3d_e = np.array([e, 0, 0])
                            # interp_rotvec_h = R.from_rotvec(interp_mp3d_h)
                            # interp_rotvec_e = R.from_rotvec(interp_mp3d_e)
                            # interp_habitat_rotation = (interp_rotvec_h * interp_rotvec_e).as_quat()
                            
                            # interp_mattersim_position = [interp_x, interp_y, interp_z]
                            # interp_mp3d_h_mattersim = np.array([0, h, 0])
                            # interp_rotvec_h_mattersim = R.from_rotvec(interp_mp3d_h_mattersim)
                            # interp_mattersim_rotation = (interp_rotvec_h_mattersim * interp_rotvec_e).as_quat()
                            
                            _data = {
                                'scan': scan,
                                'mattersim_x': interp_x,
                                'mattersim_y': interp_y,
                                'mattersim_z': interp_z,
                                'mattersim_h': h,
                                'mattersim_e': e,
                                'start_vp': viewpoint,
                                'end_vp': nv,
                                'interpolation': i,
                                'view': ix
                            }
                            
                            extra_position_dict[scan][interp_viewpoint_id].append(_data)
                            
            if scan_save_flag:
                save_path = f"/data2/vln_dataset/preprocessed_data/extra_view_config/{scan}_extra_view_config.json"
                save_dict = {scan: extra_position_dict[scan]}
                self.save_extra_position_dict(save_dict, save_path)
        
        return extra_position_dict

    def save_extra_position_dict(self, extra_position_dict, save_path):
        """
        Save the extra_position_dict as a .json file.
        
        :param extra_position_dict: Dictionary containing extra positions and rotations.
        :param save_path: Path to save the .json file.
        """
        # Convert the dictionary to a JSON string
        json_data = json.dumps(extra_position_dict, indent=4)
        
        # Write the JSON string to a file
        with open(save_path, 'w') as json_file:
            json_file.write(json_data)
        
        print(f"Configuration saved to {save_path}")
        
    def get_cg_pose(self, position, heading, elevation):
        # position is [x, y, z] in Matterport3D
        x, y, z = position
        cg_x = x
        cg_y = -(z - 1.25)
        cg_z = y
        cg_h = heading  # in concept graph is not counter-clock heading 
        cg_e = elevation

        cx = np.cos(cg_e)
        sx = np.sin(cg_e)
        cy = np.cos(cg_h)
        sy = np.sin(cg_h)

        cg_T = np.zeros([4, 4])
        cg_T[0, 0] = cy
        cg_T[0, 1] = sx * sy
        cg_T[0, 2] = cx * sy
        cg_T[0, 3] = cg_x
        cg_T[1, 0] = 0
        cg_T[1, 1] = cx
        cg_T[1, 2] = -sx
        cg_T[1, 3] = cg_y
        cg_T[2, 0] = -sy
        cg_T[2, 1] = cy * sx
        cg_T[2, 2] = cy * cx
        cg_T[2, 3] = cg_z
        cg_T[3, 3] = 1

        return cg_T.astype(np.float32).reshape(-1).tolist()

    def get_habitat_pose(self, position, heading, elevation):
        # position is [x, y, z] in Matterport3D
        x, y, z = position
        habitat_position = [x, z - 1.25, -y]
        mp3d_h = np.array([0, 2 * math.pi - heading, 0])  # counter-clock heading
        mp3d_e = np.array([elevation, 0, 0])
        rotvec_h = R.from_rotvec(mp3d_h)
        rotvec_e = R.from_rotvec(mp3d_e)
        habitat_rotation = (rotvec_h * rotvec_e).as_quat()

        rotation = R.from_quat(habitat_rotation)
        rotation_matrix = rotation.as_matrix()

        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = np.array(habitat_position)
        T[3, 3] = 1

        return T.reshape(-1).tolist()

    def get_mattersim_pose(self, position, heading, elevation):
        # position is [x, y, z] in Matterport3D
        x, y, z = position
        mattersim_position = [x, y, z]
        mattersim_h = np.array([0, heading, 0])  # clock heading
        mattersim_e = np.array([elevation, 0, 0])
        mattersim_rotation = (R.from_rotvec(mattersim_h) * R.from_rotvec(mattersim_e)).as_quat()

        mattersim_T = np.zeros([4, 4])
        mattersim_T[:3, :3] = R.from_quat(mattersim_rotation).as_matrix()
        mattersim_T[:3, 3] = np.array(mattersim_position)
        mattersim_T[3, 3] = 1

        return mattersim_T.reshape(-1).tolist()
        
    def get_pose(self, position, heading, elevation):
        # position is [x, y, z] in Matterport3D
        x, y, z = position
        habitat_position = [x, z - 1.25, -y]
        mp3d_h = np.array([0, 2 * math.pi - heading, 0])  # counter-clock heading
        mp3d_e = np.array([elevation + math.pi, 0, 0])
        rotation = (R.from_rotvec(mp3d_h) * R.from_rotvec(mp3d_e)).as_quat()
        
        T = np.zeros([4, 4])
        T[:3, :3] = R.from_quat(rotation).as_matrix()
        T[:3, 3] = np.array(habitat_position)
        T[3, 3] = 1
        
        return T.reshape(-1).tolist()

    def save_intrinsic_as_yaml(self, save_path, save_name='camera_setup'):
        import yaml
        # Extract intrinsic parameters from the camera intrinsics matrix
        fx = float(self.camera_intrinsics[0, 0])
        fy = float(self.camera_intrinsics[1, 1])
        cx = float(self.camera_intrinsics[0, 2])
        cy = float(self.camera_intrinsics[1, 2])
        
        # Create the dictionary to be saved as YAML
        data = {
            'dataset_name': 'r2r',
            'camera_params': {
                'image_height': self.sensor_config['IMG_HEIGHT'],
                'image_width': self.sensor_config['IMG_WIDTH'],
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'png_depth_scale': 25.5
            }
        }
        
        # Add HFOV, VFOV and depth_scale to the save_name
        _save_name = f"{save_name}_{self.sensor_config['HFOV']}_{self.sensor_config['VFOV']}_{self.depth_scale}"
    
        
        # Save the dictionary as a YAML file
        with open(f"{save_path}/{_save_name}.yaml", 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        
    def load_viewpoint_ids(self, connectivity_dir, connectivity_file_name, exclude_scans_file_name="eval_scans.txt"):
        
        ## Usage
        # viewpoint_ids = self.obj_edge_processor.load_viewpoint_ids(self.obj_edge_processor.connectivity_dir, self.obj_edge_processor.connectivity_file_name, self.obj_edge_processor.exclude_scans_file_name)
        # scan_list = list(set(scan for scan, _ in viewpoint_ids))
        # self.scan_list = scan_list
        
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


def save_camera_intrinsics():
    
    generator = VIEW_GENERATOR()
    generator.get_camera_intrinsics()
    generator.save_intrinsic_as_yaml('/home/lg1/lujia/VLN_HGT/concept-graphs/conceptgraph/dataset/dataconfigs/R2R', 'camera_setup')


def save_extra_position():
    
    generator = VIEW_GENERATOR()

    extra_position_dict = generator.extra_position_generate(12, 5, True)
    
    save_path = '/data2/vln_dataset/preprocessed_data/extra_view_config/extra_view_config.json'
    
    generator.save_extra_position_dict(extra_position_dict, save_path)


def save_extra_view():
    
    generator = VIEW_GENERATOR()
    
    # test_scan = '17DRP5sb8fy'
    
    # /data2/vln_dataset/preprocessed_data/extra_view_config/17DRP5sb8fy_extra_view_config.json
    
    # test_extra_position_dict_path = f"/data2/vln_dataset/preprocessed_data/extra_view_config/{test_scan}_extra_view_config.json"
    
    extra_position_dict_path = "/data2/vln_dataset/preprocessed_data/extra_view_config/extra_view_config.json"
    
    data_save_root = "/data2/vln_dataset/preprocessed_data/extra_view_R2R"
    
    extra_position_dict = generator.load_extra_position_dict(extra_position_dict_path)
    
    generator.extra_view_generate(extra_position_dict, True, data_save_root)
    
def multi_process_extra_view():
        
    generator = VIEW_GENERATOR()
    
    extra_position_dict_path = "/data2/vln_dataset/preprocessed_data/extra_view_config/extra_view_config.json"
    
    data_save_root = "/data2/vln_dataset/preprocessed_data/extra_view_R2R_multi"
    
    extra_position_dict = generator.load_extra_position_dict(extra_position_dict_path)
    
    # Set num_workers
    generator.save_file_config['num_workers'] = 5
    
    generator.multi_process_extra_view_generate(extra_position_dict, data_save_root)

def main():
    
    # save_extra_position()    
    
    # save_extra_view()
    
    multi_process_extra_view()
    
    
main()