"""
A PyTorch dataset class used for the training/inference of the Vernier Scale Network.
To play with a demo of the dataset inputs/outputs, see the test_training_pair_generation method.

Author: Shichao Li
"""

import os

import numpy as np
import cv2
# import zarr
import torch
# import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import snvc.visualization.points as vp
import snvc.dataset.kitti_util as kitti_utils

# from . import preprocess
from snvc.utils.numpy_utils import clip_boxes
# from snvc.utils.numba_utils import *
from snvc.dataset.kitti_dataset import kitti_dataset
from snvc.dataset.KITTILoader3D import get_kitti_annos
from snvc.utils.bounding_box import construct_mesh_cuboid
from snvc.utils.img_proc import get_affine_transform, kpts2cs, affine_transform

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return np.load(path).astype(np.float32)

# def mask_loader(path):
#     return torch.from_numpy(zarr.load(path) != 0)

def convert_to_viewpoint_torch(alpha, z, x):
    return alpha + torch.atan2(z, x) - np.pi / 2

def convert_to_ry_torch(alpha, z, x):
    return alpha - torch.atan2(z, x) + np.pi / 2

class refinementDataset(data.Dataset):
    """
    Implementation of the PyTorch dataset class used for the training/inference 
    of the Vernier Scale Network.
    """
    def __init__(self, 
                 left, 
                 right, 
                 left_disparity, 
                 training, 
                 loader=default_loader, 
                 dploader=disparity_loader,
                 # mloader=mask_loader,
                 split=None, 
                 cfg=None, 
                 generate_target=False
                 ):
        self.left = left
        self.right = right
        self.split = split
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        # self.mloader = mloader
        self.training = training
        self.cfg = cfg
        # initialize relevant parameters
        self.purturb_params = {'rotation':cfg.rot_aug,
                               'location':cfg.loc_aug,
                               'dimension':cfg.dim_aug,
                               'std_rot': cfg.std_rot,
                               'std_loc': cfg.std_loc,
                               'std_dim': cfg.std_dim,
                               'check_fov': cfg.check_fov
                               }
        self.roi_params = {'resolution':cfg.resolution,
                           'aspect_ratio':cfg.aspect_ratio
                           }
        self.df_params = {'spacing': np.array(cfg.spacing),
                          'grid_resolution': np.array(cfg.grid_resolution),
                          'range':cfg.grid_range,
                          'sigma':cfg.sigma,
                          'x_range':cfg.x_range,
                          'y_range':cfg.y_range,
                          'z_range':cfg.z_range
                          }
        if 'train.txt' in split:
            # the train split in the training set
            self.kitti_dataset = kitti_dataset('train').train_dataset 
        elif 'val.txt' in split:
            # the val split in the training set
            self.kitti_dataset = kitti_dataset('train').val_dataset 
        elif 'trainval.txt' in split:
            # trainval use all images in the training split
            self.kitti_dataset = kitti_dataset('trainval').train_dataset 
        elif 'test.txt' in split:
            # the testing split that has not label
            self.kitti_dataset = kitti_dataset('trainval').val_dataset 
        self.valid_classes = getattr(self.cfg, 'valid_classes', None)
        self._init_db()
        # initialize a 3D grid used for sampling 2D features
        self._init_3d_grid()
        # parameters relevant to PyTorch image transformation operations
        normalize = transforms.Normalize(mean=cfg.img_mean, std=cfg.img_std)
        transform_list = [transforms.ToTensor(), normalize]
        self.pth_trans = transforms.Compose(transform_list) 
        # uncomment to see a demonstration of input/output
        # self._test_training_pair_generation()
    
    def _init_db_from_gt(self):
        """
        Initialize the record for each object instance.
        """       
        db = []
        depth_range = self.cfg.depth_range
        for img_idx, left_path in enumerate(self.left):
            image_index = int(left_path.split('/')[-1].split('.')[0])
            pc_path = os.path.join(self.kitti_dataset.lidar_dir, "{:06d}.bin".format(image_index))
            cl = self.kitti_dataset.get_calibration(image_index)
            cr = self.kitti_dataset.get_right_calibration(image_index)  
            # load labels
            labels = self.kitti_dataset.get_label_objects(image_index)
            # filter raw labels
            boxes, box3ds, ori_classes = get_kitti_annos(labels,
                                                         valid_classes=self.valid_classes,
                                                         depth_range=depth_range,
                                                         truncation_threshold=0.8
                                                         )  
            for instance_idx in range(len(box3ds)):
                db.append({'lp': left_path,
                           'rp': self.right[img_idx],
                           'cl':cl,
                           'cr':cr,
                           'pc':pc_path,
                           'label':box3ds[instance_idx]
                           }
                          )
        self.db = db
        print("Initialized dataset from ground truth instances.")
        print("There are totally {:d} instances in {:s}".format(len(self.db),
                                                                self.split
                                                                )
              )
        return
    
    def _init_db_for_debug(self, pred_dir, file_name, instance_idx):
        # USED ONLY FOR DEBUGGING. 
        """
        Specify one 3D box for debugging purpose.
        """           
        db = []
        image_index = int(file_name.split('.')[0])
        image_name = file_name[:-4] + ".png"
        left_path = os.path.join(self.kitti_dataset.image_dir, image_name)
        right_path = os.path.join(self.kitti_dataset.right_image_dir, image_name)
        pc_path = os.path.join(self.kitti_dataset.lidar_dir, "{:06d}.bin".format(image_index))
        cl = self.kitti_dataset.get_calibration(image_index)
        cr = self.kitti_dataset.get_right_calibration(image_index)              
        # load predictions
        pred_labels = kitti_utils.read_label(os.path.join(pred_dir, file_name))
        pred_boxes, pred_box3ds, pred_classes, scores = get_kitti_annos(pred_labels,
                                                                valid_classes=self.valid_classes,
                                                                ignore_truncation=False,
                                                                ret_scores=True
                                                                )        
        db.append({'lp': left_path,
                   'rp': right_path,
                   'cl':cl,
                   'cr':cr,
                   'pc':pc_path,
                   'pred':pred_box3ds[instance_idx],
                   'box2d':pred_boxes[instance_idx].reshape(1,4),
                   'score':scores[instance_idx]
                   }
                  )            
        self.db = db
        print("Initialized dataset from predicted instances.")
        print("There are totally {:d} predicted instances in {:s}".format(len(self.db),
                                                                self.split
                                                                )
              )        
        return
    
    def _init_db_from_pred(self, pred_dir):
        """
        Initialize the instance record from 3D object detection results.
        """   
        db = []
        file_names = os.listdir(pred_dir)
        # depth_range = self.cfg.depth_range
        for file_name in file_names:
            if not file_name.endswith(".txt"):
                continue
            image_index = int(file_name.split('.')[0])
            image_name = file_name[:-4] + ".png"
            left_path = os.path.join(self.kitti_dataset.image_dir, image_name)
            right_path = os.path.join(self.kitti_dataset.right_image_dir, image_name)
            pc_path = os.path.join(self.kitti_dataset.lidar_dir, "{:06d}.bin".format(image_index))
            cl = self.kitti_dataset.get_calibration(image_index)
            cr = self.kitti_dataset.get_right_calibration(image_index)  
            # # load gt labels (optional for the train/val splits)
            # labels = self.kitti_dataset.get_label_objects(image_index)
            # # filter raw labels (optional for the train/val splits)
            # boxes, box3ds, ori_classes = get_kitti_annos(labels,
            #                                              valid_classes=self.valid_classes,
            #                                              depth_range=depth_range,
            #                                              truncation_threshold=0.8
            #                                              )              
            # load predictions
            pred_labels = kitti_utils.read_label(os.path.join(pred_dir, file_name))
            pred_boxes, pred_box3ds, pred_classes, scores = get_kitti_annos(pred_labels,
                                                                            valid_classes=self.valid_classes,
                                                                            ignore_truncation=False,
                                                                            ret_scores=True
                                                                            )        
            for instance_idx in range(len(pred_box3ds)):
                db.append({'lp': left_path,
                           'rp': right_path,
                           'cl':cl,
                           'cr':cr,
                           'pc':pc_path,
                           'pred':pred_box3ds[instance_idx],
                           'box2d':pred_boxes[instance_idx].reshape(1,4),
                           'score':scores[instance_idx]
                           }
                          )            
        self.db = db
        print("Initialized dataset from predicted instances.")
        print("There are totally {:d} predicted instances in {:s}".format(len(self.db),
                                                                pred_dir
                                                                )
              )
        return
    
    def _init_db(self):
        """
        Initialize the data record for each instance depending on the experimental setting.
        """
        if self.cfg.usage == 'train' and self.cfg.sup_type == 'synthetic':
            # the supervision type is synthetic
            # In this case, current prediction is sampled from the ground truth
            # locations with a Gaussian noise
            self._init_db_from_gt()
        elif self.cfg.usage == 'train' and self.cfg.sup_type == 'real':
            # In this case, current prediction is provided by the main scale 
            # network
            self._init_db_from_pred(self.cfg.pred_dir)
        elif self.cfg.usage == 'inference' and self.cfg.sup_type == 'real':
            self._init_db_from_pred(self.cfg.pred_dir)
            # FOR DEBUGGING
            # self._init_db_for_debug(self.cfg.pred_dir, '006759.txt', 0)
        elif self.cfg.usage == 'inference' and self.cfg.sup_type == 'synthetic':
            self._init_db_from_gt()            
        else:
            raise NotImplementedError
        return
    
    def _init_3d_grid(self):
        """
        Initialize a 3D grid (the Vernier Scale) used for sampling 2D features.
        """
        x_min, x_max = self.cfg.x_range
        y_min, y_max = self.cfg.y_range
        z_min, z_max = self.cfg.z_range
        x_pts = np.linspace(x_min, x_max, self.cfg.grid_resolution[1])
        y_pts = np.linspace(y_min, y_max, self.cfg.grid_resolution[0])
        z_pts = np.linspace(z_min, z_max, self.cfg.grid_resolution[2])
        grid_x, grid_y, grid_z = np.meshgrid(x_pts, y_pts, z_pts, indexing='xy')
        # the grid is of shape [h (y-axis), w (x-axis), l(z_axis)]
        self.grid_3d = np.concatenate([grid_x[None, :], grid_y[None, :], grid_z[None, :]])
        self.grid_bev = self.grid_3d.copy()[:,0,:,:].squeeze()
        self.grid_bev_flat = np.transpose(self.grid_bev, (2,1,0)).reshape(-1, 3)
        return
    
    def _generate_noise(self, params):
        """
        Producing gaussian noise to 3D bounding box parameters.
        """        
        noise = np.zeros(7) # size (3), location (3), orientation (1)
        if params['rotation']:
            std_rot = np.array(params['std_rot'])*np.pi/180.,              
            noise[6] = np.random.randn(1) * std_rot
        if params['location']:
            std_loc = np.array(params['std_loc'])   
            noise[3:6] = np.random.randn(3) * std_loc
        if params['dimension']:
            std_dim = np.array(params['std_dim'])   
            noise[:3] = np.random.randn(3) * std_dim
        return noise
    
    def _purturb_3D_box(self, 
                        base,
                        params=None,
                        calib_left=None,
                        calib_right=None,
                        max_trials=10
                        ):
        """
        Purturb a 3D bounding box to simulate a coarse proposal.
        """    
        params = self.purturb_params if params is None else params
        if params['check_fov']:
            assert calib_left is not None and calib_right is not None
        cnt, success = 0, False
        while cnt < max_trials and not success:
            noise = self._generate_noise(params)
            sample = base + noise
            if params['check_fov']:
                success = self._check_fov(sample)
            else:
                success = True
        return sample, success
    
    def _sample_3D_box(self,
                       current_pred=None,
                       gt_label=None,
                       augment=False,
                       calib_left=None,
                       calib_right=None,
                       augment_times=1
                       ):
        """
        Generate coarse proposals used for training the Vernier scale network.
        If current prediction is none, then sample a position around the 
        ground truth to simulate the detection error.
        """        
        if current_pred is None:
            assert gt_label is not None
            base = gt_label.copy()
        else:
            # simply return the current prediction as sample
            return current_pred[None, :]
        # perturb the current prediction as data augmentation
        times = augment_times if augment else 1
        samples = []
        for i in range(times):
            sample, success = self._purturb_3D_box(base, 
                                                   calib_left=calib_left,
                                                   calib_right=calib_right
                                                   )            
            if success:
                samples.append(sample[None, :])
        return np.concatenate(samples)
    
    def test_training_pair_generation(self):
        """
        This is a test visualizaing the inputs and outputs of this dataset.
        """   
        idx = np.random.randint(0, len(self.left))
        lp, rp = self.left[idx], self.right[idx]
        print("Left image path: ", lp)
        image_index = int(lp.split('/')[-1].split('.')[0])
        # you can specify a particular image with image_index
        # image_index = 2619
        cl = self.kitti_dataset.get_calibration(image_index)
        cr = self.kitti_dataset.get_right_calibration(image_index)  
        pc = os.path.join(self.kitti_dataset.lidar_dir, "{:06d}.bin".format(image_index))
        # load labels
        labels = self.kitti_dataset.get_label_objects(image_index)
        # filter raw labels
        boxes, box3ds, ori_classes = get_kitti_annos(labels,
                                                     valid_classes=self.valid_classes,
                                                     truncation_threshold=0.8
                                                     ) 
        if len(boxes) == 0: # no valid annotation for this image
            return                        
        gt_label = box3ds[np.random.randint(0, len(box3ds))]
        left_rois, right_rois, targets, meta_data = self._prepare_pairs(lp, 
                                                                        rp, 
                                                                        pc,
                                                                        cl, 
                                                                        cr, 
                                                                        gt_label, 
                                                                        debug=True
                                                                        )        
        # plotting
        import matplotlib.pyplot as plt
        left, right = left_rois[0], right_rois[0]
        plt.figure()
        ax = plt.subplot(131)
        # left image and projected corners of a sampled 3D box
        ax.set_title("Left ROI")
        ax.imshow(left)
        kpts_l_sample = meta_data['kpts_2d_l_local'][0]   
        kpts_l_gt, kpts_r_gt = meta_data['kpts_gt_l'], meta_data['kpts_gt_r']
        ax.plot(kpts_l_sample[:, 0], kpts_l_sample[:, 1], 'ro')
        vp.plot_3d_bbox(ax, kpts_l_sample[1:, :], color='r')      
        ax.plot(kpts_l_gt[:, 0], kpts_l_gt[:, 1], 'go')
        vp.plot_3d_bbox(ax, kpts_l_gt[1:, :], color='k')           
        ax = plt.subplot(132)
        ax.set_title("Right ROI")
        # right image and projected corners of a sampled 3D box
        ax.imshow(right)
        kpts_r_sample = meta_data['kpts_2d_r_local'][0]
        ax.plot(kpts_r_sample[:, 0], kpts_r_sample[:, 1], 'ro')
        vp.plot_3d_bbox(ax, kpts_r_sample[1:, :], color='r')        
        ax.plot(kpts_r_gt[:, 0], kpts_r_gt[:, 1], 'go')
        vp.plot_3d_bbox(ax, kpts_r_gt[1:, :], color='k')     
        ax = plt.subplot(133, projection='3d')
        ax.set_title("3D Plot")
        sample = meta_data['samples'][0]
        sample_kpts_3d = self._get_cam_cord(sample).T
        box_3d = sample.copy()
        # the refinement space is represented as a 3D box, centered at the 
        # current sample and has the same orientation as the sample
        old_center_y = box_3d[4] - box_3d[0] * 0.5
        box_3d[:3] = self.df_params['range'] # range of the refinement space
        box_3d[4] = old_center_y + box_3d[0] * 0.5 # update the bottom center y-coordinate
        # project to image and get 2D boxes
        field_kpts_3d = self._get_cam_cord(box_3d).T       
        gt = meta_data['gt']
        gt_kpts_3d = self._get_cam_cord(gt).T
        n_h, n_w, n_l = self.cfg.grid_resolution
        grid_kpts_3d = meta_data['grid_3d'][0].reshape(n_h, n_w, n_l, 3) # the points of the 3D grid
        # down-sample for display
        grid_kpts_3d_ds = grid_kpts_3d[::5, ::5, ::5].reshape(-1, 3)
        limit_points = np.vstack([sample_kpts_3d, gt_kpts_3d, field_kpts_3d])
        vp.plot_3d_points(ax, grid_kpts_3d_ds, color='b', size=1)
        vp.plot_3d_points(ax, sample_kpts_3d, color='r', size=15)
        vp.plot_lines(ax, sample_kpts_3d[1:,], vp.plot_3d_bbox.connections, dimension=3, c='r')
        vp.plot_3d_points(ax, field_kpts_3d, color='y', size=15)
        vp.plot_lines(ax, field_kpts_3d[1:,], vp.plot_3d_bbox.connections, dimension=3, c='y')
        vp.plot_3d_points(ax, gt_kpts_3d, color='k', size=15)
        vp.plot_lines(ax, gt_kpts_3d[1:,], vp.plot_3d_bbox.connections, dimension=3, c='k')         
        vp.set_3d_axe_limits(ax, limit_points)
        # random point cloud inside the cuboid
        plt.figure()
        ax = plt.subplot(121, projection='3d')
        vp.test_visualize_cuboid_mesh(field_kpts_3d, ax, color='y')
        ax.set_title('3D RoI and random point clouds')
        ax = plt.subplot(122, projection='3d')
        vp.test_visualize_cuboid_mesh(sample_kpts_3d, ax, color='r')
        ax.set_title('Current 3D box and random point clouds')
        # real point cloud inside the cuboid
        plt.figure()
        ax = plt.subplot(131, projection='3d')
        vp.test_visualize_cuboid_mesh(field_kpts_3d, ax, color='y', pc=meta_data['pc_in_roi'][0])
        ax.set_title('3D RoI and point cloud')
        ax = plt.subplot(132, projection='3d')
        vp.test_visualize_cuboid_mesh(sample_kpts_3d, ax, color='r', pc=meta_data['pc_in_roi_fg'][0])
        ax.set_title('Current 3D box and foreground point cloud')   
        ax = plt.subplot(133, projection='3d')
        vp.test_visualize_cuboid_mesh(sample_kpts_3d, ax, color='r', pc=meta_data['pc_in_roi'][0])
        ax.set_title('Current 3D box and point cloud')       
        # occupancy for the 3D grid
        plt.figure()
        ax = plt.subplot(111, projection='3d') 
        occupancy = meta_data['occupancy'][0]
        # foreground points
        i, j, k = np.nonzero(occupancy == 1.0)
        dy, dx, dz = self.df_params['spacing']
        i, j, k = (i*dy).reshape(len(i), 1), (j*dx).reshape(len(j), 1), (k*dz).reshape(len(k), 1)
        fg = np.hstack([j, i, k])
        vp.plot_3d_points(ax, fg, color='r', size=15)
        # background points
        i, j, k = np.nonzero(occupancy == 0)
        dy, dx, dz = self.df_params['spacing']
        i, j, k = (i*dy).reshape(len(i), 1), (j*dx).reshape(len(j), 1), (k*dz).reshape(len(k), 1)
        bg = np.hstack([j, i, k])[::50, :]
        vp.plot_3d_points(ax, bg, color='y', size=1, alpha=0.5)   
        limit_points = np.vstack([fg, bg])
        vp.set_3d_axe_limits(ax, limit_points)
        # plot target
        field = targets[0]
        if len(field.shape) == 4:
            # 3D displacement field
            for part_id in range(len(field)): # for each part
                # plot the (mean) projection of 3D displacement field
                plt.figure()
                ax = plt.subplot(131)
                ax.imshow(field[part_id].mean(axis=0))
                ax.set_title('ZX plane, Part {:d}'.format(part_id + 1))
                ax = plt.subplot(132)
                ax.imshow(field[part_id].mean(axis=1))
                ax.set_title('ZY plane, Part {:d}'.format(part_id + 1))
                ax = plt.subplot(133)
                ax.imshow(field[part_id].mean(axis=2))
                ax.set_title('XY plane, Part {:d}'.format(part_id + 1))  
        elif len(field.shape) == 3:
            # 2D part heatmaps
            plt.figure()
            num_cols = 3
            num_rows = int(np.round(len(field) / num_cols))
            for part_id in range(len(field)): # for each part
                ax = plt.subplot(num_rows, num_cols, part_id + 1)
                ax.imshow(field[part_id])
                ax.invert_yaxis()
                ax.set_title('XZ plane, Part {:d}'.format(part_id + 1))
        return

    def _crop_instance(self,
                       img,
                       params,
                       box_2d=None,
                       kpts_2d=None,
                       ):
        """
        Crop an image patch of an object with cv2.warpAffine.
        """ 
        reso = params['resolution']
        local_coord = kpts_2d.copy()
        c, s, _, _ = kpts2cs(kpts_2d, target_ar=params['aspect_ratio'])
        # s: [width, height] of the crop
        trans = get_affine_transform(c, s, 0.0, reso, absolute=True)
        ret = cv2.warpAffine(img,
                             trans,
                             (int(reso[0]), int(reso[1])),
                             flags=cv2.INTER_LINEAR
                             )     
        # local coordinates within the patch
        local_coord = affine_transform(kpts_2d, trans)        
        return ret, local_coord.T, trans

    def _construct_box_3d(self, l, h, w):
        """
        Compute the coordinates of the vertices of a 3D bounding box in an
        object coordinate system.
        """ 
        x_corners = [0.5*l, l, l, l, l, 0, 0, 0, 0]
        y_corners = [0.5*h, 0, h, 0, h, 0, h, 0, h]
        z_corners = [0.5*w, w, w, 0, 0, w, w, 0, 0]
        x_corners += - np.float32(l) / 2
        y_corners += - np.float32(h)
        z_corners += - np.float32(w) / 2
        corners_3d = np.array([x_corners, y_corners, z_corners])     
        return corners_3d
    
    def _get_cam_cord(self, box_3d):
        """
        Compute the camera coordinates of the vertices of a 3D bounding box.
        """         
        h, w, l = box_3d[0], box_3d[1], box_3d[2]
        corners_3d_fixed = self._construct_box_3d(l, h, w)
        # translation
        x, y, z = box_3d[3:6]
        ry = box_3d[6]
        # rotation
        rot_maty = np.array([[np.cos(ry), 0, np.sin(ry)],
                            [0, 1, 0],
                            [-np.sin(ry), 0, np.cos(ry)]])
        corners_3d = np.matmul(rot_maty, corners_3d_fixed)
        # translation
        corners_3d += np.array([x, y, z]).reshape([3, 1])
        return corners_3d
    
    def _generate_rois(self, 
                       samples, 
                       left_img_path,
                       right_img_path,
                       calib_left,
                       calib_right,
                       params=None,
                       gt_label=None,
                       pth_trans=None
                       ):
        """
        Generate 2D regions of interests for specified samples and input images.
        """        
        roi_params = self.roi_params if params is None else params
        # load images
        left_img =  cv2.imread(left_img_path, 1 | 128 )   
        right_img =  cv2.imread(right_img_path, 1 | 128 )   
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        # create a 3D box to represent the range of displacement field
        boxes_3d = []
        left_rois = []
        right_rois = []
        meta = {'kpts_2d_l':[],
                'kpts_2d_r':[],
                'kpts_2d_l_local':[], # coordinates in a local patch
                'kpts_2d_r_local':[],
                'trans_l':[],
                'trans_r':[]
                }
        for sample in samples:
            box_3d = sample.copy()
            # the local refinement space is represented as a 3D box, centered at the 
            # current sample and has the same orientation as the sample
            old_center_y = box_3d[4] - box_3d[0] * 0.5
            box_3d[:3] = self.df_params['range'] # range of the local refinement space
            box_3d[4] = old_center_y + box_3d[0] * 0.5 # update the bottom center y-coordinate
            boxes_3d.append(boxes_3d)
            # project to image and get 2D boxes
            kpts_3d = self._get_cam_cord(box_3d).T
            kpts_2d_l = calib_left.project_rect_to_image(kpts_3d)
            kpts_2d_r = calib_right.project_rect_to_image(kpts_3d)
            # crop rois
            left_roi, local_l, trans_l = self._crop_instance(left_img, roi_params, kpts_2d=kpts_2d_l)
            right_roi, local_r, trans_r = self._crop_instance(right_img, roi_params, kpts_2d=kpts_2d_r)
            if pth_trans is not None:
                # post-processing (e.g., normalizing) the pixels with PyTorch transformations
                left_roi, right_roi = pth_trans(left_roi), pth_trans(right_roi)
            left_rois.append(left_roi)
            right_rois.append(right_roi)
            meta['kpts_2d_l'].append(kpts_2d_l[None, :, :])
            meta['kpts_2d_r'].append(kpts_2d_r[None, :, :])
            meta['kpts_2d_l_local'].append(local_l[None, :, :])
            meta['kpts_2d_r_local'].append(local_r[None, :, :])
            meta['trans_l'].append(trans_l[None, :, :])
            meta['trans_r'].append(trans_r[None, :, :])
        for key in meta:
            meta[key] = np.concatenate(meta[key])
        # optional processing for the ground truth label
        if gt_label is not None:
            kpts_3d_gt = self._get_cam_cord(gt_label).T
            kpts_2d_l_gt = calib_left.project_rect_to_image(kpts_3d_gt)
            kpts_2d_r_gt = calib_right.project_rect_to_image(kpts_3d_gt)
            local_l_gt = affine_transform(kpts_2d_l_gt, trans_l)
            local_r_gt = affine_transform(kpts_2d_r_gt, trans_r)
            meta['kpts_gt_l'], meta['kpts_gt_r'] = local_l_gt.T, local_r_gt.T                                     
        return left_rois, right_rois, meta
    
    def _draw_heatmaps_3d(self,
                          re,
                          center,
                          sigma,
                          verbose=False
                          ):
        """
        Generate 3D object part heatmaps used in training the Vernier scale network.
        """    
        field = np.zeros(re)
        #TODO: draw gaussian probability at with sub-pixel accuracy
        mu_y, mu_x, mu_z = center
        tmp_size = sigma * 3
        field_size = field.shape
        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        x0 = size // 2
        dif_y = ((x-x0)**2).reshape((size, 1, 1))
        dif_x = dif_y.copy().reshape((1, size, 1))
        dif_z = dif_x.copy().reshape((1, 1, size))        
        g = np.exp(- (dif_x + dif_y + dif_z) / (2 * sigma ** 2))
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size), int(mu_z - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_z + tmp_size + 1)]
        # gaussian range
        g_x = max(0, -ul[0]), min(br[0], field_size[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], field_size[0]) - ul[1]
        g_z = max(0, -ul[2]), min(br[2], field_size[2]) - ul[2]
        # field range
        field_x = max(0, ul[0]), min(br[0], field_size[1])
        field_y = max(0, ul[1]), min(br[1], field_size[0])
        field_z = max(0, ul[2]), min(br[2], field_size[2])
        if g_y[1] > g_y[0] and g_x[1] > g_x[0] and g_z[1] > g_z[0]:
            field[field_y[0]:field_y[1], 
                  field_x[0]:field_x[1], 
                  field_z[0]:field_z[1]] = g[g_y[0]:g_y[1], 
                                             g_x[0]:g_x[1], 
                                             g_z[0]:g_z[1]]        
        elif verbose:
            print('The Gaussian dot lies out of the range.')
        return field[None, :,:,:]

    def _draw_heatmaps_2d(self,
                          re,
                          center,
                          sigma,
                          verbose=False
                          ):
        """
        Generate 2D object part heatmaps used in training the Vernier scale network.
        """    
        field = np.zeros((re[2], re[1])) # w and l direction
        #TODO: draw gaussian probability at with sub-pixel accuracy
        mu_x, mu_z = center
        tmp_size = sigma * 3
        field_size = field.shape
        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        x0 = size // 2
        dif_x = ((x-x0)**2).reshape((1, size))
        dif_z = dif_x.copy().reshape((size, 1))        
        g = np.exp(- (dif_x + dif_z) / (2 * sigma ** 2))
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_z - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_z + tmp_size + 1)]
        # gaussian range
        g_x = max(0, -ul[0]), min(br[0], field_size[1]) - ul[0]
        g_z = max(0, -ul[1]), min(br[1], field_size[0]) - ul[1]
        # field range
        field_x = max(0, ul[0]), min(br[0], field_size[1])
        field_z = max(0, ul[1]), min(br[1], field_size[0])
        if g_x[1] > g_x[0] and g_z[1] > g_z[0]:
            field[field_z[0]:field_z[1],
                  field_x[0]:field_x[1]] = g[g_z[0]:g_z[1], 
                                             g_x[0]:g_x[1]]        
        elif verbose:
            print('The Gaussian dot lies out of the range.')
        return field[None,:,:]
    
    def _get_basis(self, sample):
        """
        Get basis vectors representing three directions (imagine you sitting 
        in the car and heading front):
        [your right hand, gravity direction, your front]
        """
        # basis before rotation
        basis = np.array([[0, 0, -1], # w direction
                          [0, 1, 0],  # h direction
                          [1, 0, 0]]  # l direction
                         )
        # basis after rotation
        ry = sample[6]
        rot_maty = np.array([[np.cos(ry), 0, np.sin(ry)],
                            [0, 1, 0],
                            [-np.sin(ry), 0, np.cos(ry)]])        
        return rot_maty @ basis.T
    
    def _construct_neural_confidence_field(self, sample, gt_label, params):
        """
        Generate one type of training targets for the Vernier scale network.
        The target update is encoded as a local displacement field/heatmap for
        each object part.
        """
        # construct a grid around the center of the sample where x, y and z
        # directions correspond to w, h, and l directions. z points to the 
        # head of the vehicle.
        spa, re = params['spacing'], params['grid_resolution']        
        # transform the gt_label to the local object coordinate system
        # translation -> displacement relative to
        num_parts = self.cfg.num_parts
        kpts_3d_gt = self._get_cam_cord(gt_label).T
        kpts_3d_sample = self._get_cam_cord(sample).T
        offset = kpts_3d_gt[:num_parts, :] - kpts_3d_sample[[0]]
        # rotation -> inner product with basis vectors defined by the sample
        basis = self._get_basis(sample)
        gt_corners_local = offset @ basis
        x, y, z = np.split(gt_corners_local, 3, axis=1)      
        # express the ground truth box in the local coordinate system
        # in the format of (x, y, w, h, alpha) used for IoU loss
        gt_box_local = np.zeros((1,5), dtype=np.float32)
        gt_box_local[0,0] = x[0]
        gt_box_local[0,1] = z[0]
        gt_box_local[0,2] = gt_label[2]
        gt_box_local[0,3] = gt_label[1]
        gt_box_local[0,4] = 0.5 * np.pi - (sample[6] - gt_label[6])
        # convert location to field index        
        (ny, nx, nz) = 0.5 * (np.array(re) - 1)
        indices = (np.floor((y + ny*spa[0])/spa[0]), # h direction
                   np.floor((x + nx*spa[1])/spa[1]), # w direction
                   np.floor((z + nz*spa[2])/spa[2])  # l direction
                   )
        # record the offset from each voxel to the ground truth object part
        # this is the corner coordinates in the local coordinate system (direciton: width, height, length)
        # corners_local_coord = offset @ basis
        # corners_local_coord = (corners_local_coord.T)[:,:,None,None,None]
        # offsets_local = corners_local_coord - self.grid_3d.copy()[:,None,:,:,:]
        # import matplotlib.pyplot as plt
        # plt.imshow(offsets_local[0,0,16,:,:])
        # normalize
        # roi_3d_size = re * spa
        # offsets_local /= np.array([roi_3d_size[1], roi_3d_size[0], roi_3d_size[2]]).reshape(3, 1, 1, 1, 1)
        field = []
        for part_id in range(num_parts):
            # draw Gaussian gots to represent the gt_label
            if self.cfg.grid_type == '3D': # y, x, z field
                index = [indices[i][part_id][0] for i in range(3)]
                field_part = self._draw_heatmaps_3d(re, index, params['sigma'])
            elif self.cfg.grid_type == '2D': # x, z map
                index = [indices[i][part_id][0] for i in range(1, 3)]
                field_part = self._draw_heatmaps_2d(re, index, params['sigma'])
            field.append(field_part)
        # return np.concatenate(field, axis=0)[None, :], offsets_local[None,:], gt_corners_local[None,:], gt_box_local
        return np.concatenate(field, axis=0)[None, :], None, gt_corners_local[:num_parts,:][None,:], gt_box_local

    def _get_point_cloud(self, pc, sample, gt_label, grid_3d):
        """
        Fetch the object points in a point cloud and compute foreground occupancy.
        """
        spa, re = self.df_params['spacing'], self.df_params['grid_resolution']     
        roi_3d = sample.copy()
        roi_3d[:3] = self.df_params['range']
        kpts_3d = self._get_cam_cord(roi_3d).T
        kpts_3d_gt = self._get_cam_cord(gt_label).T
        mesh = construct_mesh_cuboid(kpts_3d)
        mesh_gt = construct_mesh_cuboid(kpts_3d_gt)
        flag_roi = mesh.in_mesh(pc)
        flag_gt = mesh_gt.in_mesh(pc)
        # point cloud inside the 3D RoI
        pc_in_roi = pc[flag_roi] 
        # point cloud inside the 3D RoI and inside the gt 3D bounding box (foreground) 
        pc_in_roi_fg = pc[np.logical_and(flag_roi, flag_gt)]    
        # construct a 3D grid to represent the point clouds
        basis = self._get_basis(sample)
        offset = pc_in_roi_fg - kpts_3d[0].reshape(1, 3)
        x, y, z = np.split(offset @ basis, 3, axis=1)
        # convert location to field index        
        (ny, nx, nz) = 0.5 * (np.array(re) - 1)
        i, j, k = (np.floor((y + ny*spa[0])/spa[0]), # h direction
                   np.floor((x + nx*spa[1])/spa[1]), # w direction
                   np.floor((z + nz*spa[2])/spa[2])  # l direction
                   )
        if (i >= re[0]).sum() > 0:
            print('warning: i out of range')
            i[i>=re[0]] = re[0] - 1
        if (j >= re[1]).sum() > 0:
            print('warning: j out of range')
            j[j>=re[1]] = re[1] - 1
        if (k >= re[2]).sum() > 0:
            print('warning: k out of range')
            k[k>=re[2]] = re[2] - 1
        i, j, k = [i.squeeze().astype(np.int32).tolist(),
                   j.squeeze().astype(np.int32).tolist(),
                   k.squeeze().astype(np.int32).tolist()
                   ]
        occupancy = -1. * np.ones(re, dtype=np.float32)  # -1 -> undefined 1-> foreground 0-> background
        # foreground
        occupancy[i, j, k] = 1.
        # background
        n_h, n_w, n_l = self.cfg.grid_resolution
        flag_grid = mesh_gt.in_mesh(grid_3d).reshape(n_h, n_w, n_l)
        occupancy[np.logical_not(flag_grid)] = 0.
        return pc_in_roi, pc_in_roi_fg, occupancy[None,:,:,:]
    
    def _to_cam(self, pts_3d, sample):
        """
        Convert grid coordinates from the local object coordinate system to 
        the camera coordinate system.
        
        pts_3d: [3, N]
        """        
        # the label assumes the heading right before rotation
        # here the points head front before rotation -> +0.5*pi
        ry = sample[6] + 0.5 * np.pi 
        # rotate around y-axis 
        rot_maty = np.array([[np.cos(ry), 0, np.sin(ry)],
                            [0, 1, 0],
                            [-np.sin(ry), 0, np.cos(ry)]])         
        # move to the location
        x, y, z = sample[3:6]
        center_y = y - sample[0] * 0.5
        pts_3d = rot_maty @ pts_3d + np.array([[x], [center_y], [z]])
        return pts_3d
    
    def _generate_grid_proj(self, 
                           samples, 
                           calib_left, 
                           calib_right, 
                           meta_roi,
                           debug=False
                           ):
        """
        Generate the 2D coordinates of the projected 3D grid on left/right ROIs.
        """
        pts_3d = self.grid_3d.copy().reshape(3, -1)
        grid_3d = []
        coord_l, coord_r = [], []
        for idx, sample in enumerate(samples):
            pts_3d_cam = self._to_cam(pts_3d, sample).T
            grid_3d.append(pts_3d_cam[None, :, :])
            pts_2d_l = calib_left.project_rect_to_image(pts_3d_cam)
            pts_2d_r = calib_right.project_rect_to_image(pts_3d_cam)
            coord_l.append(affine_transform(pts_2d_l, meta_roi['trans_l'][idx])[None, :, :])
            coord_r.append(affine_transform(pts_2d_r, meta_roi['trans_r'][idx])[None, :, :])
        return np.concatenate(coord_l), np.concatenate(coord_r), np.concatenate(grid_3d)
    
    def _generate_displacement_field(self,
                                     samples,
                                     gt_label,
                                     calib_left,
                                     grid_3d,
                                     pc_path,
                                     params=None
                                     ):
        """
        Generate the trainging targets for the Vernier scale network.
        """
        params = self.df_params if params is None else params
        fields, offsets, gt_corners_local, gt_box_local, meta = [], [], [], [], {'pc_in_roi':[], 'pc_in_roi_fg':[], 'occupancy':[]}
        assert self.cfg.num_parts <= 9, "Only support less than or equal to 9 object parts"
        # point cloud for this image
        pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
        # transform to the camera coordinate system
        pc = calib_left.project_velo_to_rect(pc[:,:3])
        for idx, sample in enumerate(samples):
            ret = self._construct_neural_confidence_field(sample, gt_label, params)
            fields.append(ret[0])
            offsets.append(ret[1])
            gt_corners_local.append(ret[2])
            gt_box_local.append(ret[3])
            pc_in_roi, pc_in_roi_fg, occupancy = self._get_point_cloud(pc, sample, gt_label, grid_3d[idx])
            meta['pc_in_roi'].append(pc_in_roi)
            meta['pc_in_roi_fg'].append(pc_in_roi_fg)
            meta['occupancy'].append(occupancy)
        meta['occupancy'] = np.concatenate(meta['occupancy']).astype(np.float32)
        # meta['offset'] = np.concatenate(offsets).astype(np.float32)
        meta['gt_corners_local'] = np.concatenate(gt_corners_local).astype(np.float32)
        # meta['gt_box_local'] = np.concatenate(gt_box_local)
        fields = np.concatenate(fields).astype(np.float32)
        return fields, meta
    
    def _prepare_pairs(self,
                       left_img_path,
                       right_img_path,
                       point_cloud_path,
                       calib_left,
                       calib_right,
                       gt_label=None,
                       current_pred=None,
                       augment=False,
                       augment_times=1,
                       debug=False,
                       pth_trans=None
                       ):
        """
        Prepare the left-right region of intererts based on the current 3D proposal 
        and generate a representation of the displacement residual.
        If current prediction is none, then sample a position around the 
        ground truth to simulate the localization error.
        
        During training, gt_label needs to be specified.
        
        During inference, gt_label is not required.
        """
        assert (gt_label is not None) or (current_pred is not None)
        # generate sampled 3D boxes
        samples = self._sample_3D_box(current_pred=current_pred,
                                      gt_label=None if current_pred is not None else gt_label,
                                      augment=augment,
                                      augment_times=augment_times,
                                      calib_left=calib_left,
                                      calib_right=calib_right
                                      )
        # generate inputs 
        left_rois, right_rois, meta_roi = self._generate_rois(samples, 
                                                              left_img_path, 
                                                              right_img_path, 
                                                              calib_left, 
                                                              calib_right,
                                                              gt_label=gt_label if debug else None,
                                                              pth_trans=pth_trans
                                                              )        
        # generate the projection of the 3d grids, which are used to gather
        # image feature
        left_coord, right_coord, grid_3d = self._generate_grid_proj(samples, 
                                                                    calib_left, 
                                                                    calib_right,
                                                                    meta_roi,
                                                                    debug=debug
                                                                    )

        # generate outputs
        if gt_label is not None:
            targets, meta_field = self._generate_displacement_field(samples, 
                                                                    gt_label, 
                                                                    calib_left,
                                                                    grid_3d,
                                                                    pc_path=point_cloud_path
                                                                    )
        else:
            targets, meta_field = None, {}
        
        if pth_trans is not None:
            left_rois, right_rois = torch.stack(left_rois), torch.stack(right_rois)
        if pth_trans is not None and gt_label is not None:
            targets = torch.from_numpy(targets)
            meta_field['occupancy'] = torch.from_numpy(meta_field['occupancy'])
        # meta_data
        meta_data = {**meta_roi,
                     **meta_field,
                     'lp':left_img_path,
                     'samples':samples, 
                     'gt':gt_label,
                     'grid_proj_left': left_coord,
                     'grid_proj_right': right_coord,
                     'grid_3d':grid_3d,
                     'calib_left':calib_left,
                     'calib_right':calib_right
                     }
        return left_rois, right_rois, targets, meta_data
    
    def _process_bbox(self, 
                      boxes,
                      box3ds,
                      left_img,
                      ori_classes
                      ):
        """
        A bounding box processing helper function.
        """
        boxes[:, [2,3]] = boxes[:, [0,1]] + boxes[:, [2,3]]
        boxes = clip_boxes(boxes, left_img.size, remove_empty=False)

        # sort(far -> near)
        inds = box3ds[:, 5].argsort()[::-1]
        box3ds = box3ds[inds]
        boxes = boxes[inds]
        ori_classes = ori_classes[inds]

        # sort by classes
        inds = ori_classes.argsort(kind='stable')
        box3ds = box3ds[inds]
        boxes = boxes[inds]
        ori_classes = ori_classes[inds]

        # guard against no boxes
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  
        box3ds = torch.as_tensor(box3ds).reshape(-1, 7)
        return boxes, box3ds, ori_classes
    
    def get_neighbor(self, image_path, query):
        """
        From all the GT 3D boxes specified by image_index, find the one that
        is closest to the query 3D box.
        """
        image_index = int(image_path.split('/')[-1].split('.')[0])
        labels = self.kitti_dataset.get_label_objects(image_index)
        # filter raw labels
        boxes, box3ds, ori_classes = get_kitti_annos(labels,
                                                     valid_classes=self.valid_classes,
                                                     truncation_threshold=0.8
                                                     )
        displacement = np.linalg.norm(query[3:6].reshape(1,3) - box3ds[:, 3:6], axis=1)
        idx = np.argmin(displacement, axis=0)
        return box3ds[idx]
    
    def __getitem__(self, index):
        """
        The PyTorch dataset method that loads a batch of data.
        """
        # record dictionary for a ground truth instance or predicted instance
        # from the main scale
        ins_dict = self.db[index]
        if self.cfg.usage == 'train':
            assert 'label' in ins_dict, "each instance must has a target during training"
            gt_label = ins_dict['label']
            current_pred = None
            augment = self.cfg.augment
        elif self.cfg.usage == 'inference' and self.cfg.sup_type == 'real':
            current_pred = ins_dict['pred']
            gt_label = None
            augment = False
        elif self.cfg.usage == 'inference' and self.cfg.sup_type == 'synthetic':
            gt_label = ins_dict['label']
            current_pred = None
            augment = self.cfg.augment            
        left_rois, right_rois, targets, meta_data = self._prepare_pairs(ins_dict['lp'], 
                                                                        ins_dict['rp'], 
                                                                        ins_dict['pc'],
                                                                        ins_dict['cl'], 
                                                                        ins_dict['cr'], 
                                                                        gt_label=gt_label,
                                                                        current_pred =current_pred,
                                                                        augment=augment,
                                                                        augment_times=self.cfg.augment_times,
                                                                        pth_trans=self.pth_trans
                                                                        )  
        for key in ['box2d', 'score']:
            if key in ins_dict:
                meta_data[key] = ins_dict[key]
        return left_rois, right_rois, targets, meta_data

    def __len__(self):
        return len(self.db)