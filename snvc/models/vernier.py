"""
Vernier scale network that refines a 3D prediction from the proposal given 
by the main scale model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# plotting
import matplotlib.pyplot as plt

from torch.nn.functional import grid_sample, cosine_similarity
from scipy.spatial.transform import Rotation

import snvc.models.hrnet as hrnet
# import snvc.models.hrnetv2 as hrnetv2
import snvc.visualization.points as vp

from snvc.models.submodule import convbn_3d, convbn, hourglass, hourglass2d, hourglass_downsample_16, hourglass2d_downsample_16
from snvc.utils.img_proc import get_affine_transform, kpts2cs, affine_transform
from snvc.utils.transformation import compute_rigid_transform
from snvc.models.FCmodel import get_fc_model

class VernierScale(nn.Module):
    def __init__(self, cfg, is_train=False):
        super(VernierScale, self).__init__()
        self.cfg = cfg
        self.is_train = is_train
        # predict 2D evidence from 2D features
        self.use_2d = cfg.use_2d
        # construct cost volume and perform depth estimation
        self.estimate_depth = cfg.estimate_depth
        self._init_3d_net()
        self._init_grid()
        if self.cfg.vernier_type in ['BEV_type3']:
            self._init_coord_head()
            if getattr(self.cfg, 'use_bbox_head', False):            
                self._init_bbox_head()
        # TODO: check the weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu'
                                        )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='relu'
                                        )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
        self._init_feat_extract(is_train)
            
    def _init_feat_extract(self, is_train):
        """
        Initialize 2D feature extraction network.
        """
        cfg_feat_net = getattr(self.cfg, self.cfg.backbone)
        # self.feat_net = hrnetv2.get_model(is_train=is_train)
        self.feat_net = get_feat_extraction(cfg=cfg_feat_net,
                                            is_train=self.is_train
                                            )
        return

    def _init_coord_head(self):
        """
        Initialize 2D coordinate regression head.
        """
        BasicBlock = hrnet.BasicBlock
        basicdownsample = hrnet.basicdownsample
        num_chan = self.cfg.num_parts
        modules = []
        modules.append(BasicBlock(num_chan+2, 
                                  num_chan*2, 
                                  stride=2,
                                  downsample=basicdownsample(num_chan+2, num_chan*2)
                                  )
                       )
        num_ds = int(4 - np.log2(192 / self.cfg.grid_resolution[2]))
        for _ in range(num_ds):
            modules.append(BasicBlock(num_chan*2, 
                                      num_chan*2, 
                                      stride=2,
                                      downsample=basicdownsample(num_chan*2, num_chan*2)
                                      )
                           )       
        modules.append(nn.Conv2d(num_chan*2, num_chan*2, kernel_size=(6, 4)))
        modules.append(nn.Sigmoid())
        self.coord_head = nn.Sequential(*modules)
        return
    
    def _init_bbox_head(self):
        self.bbox_head = get_fc_model()
        return
    
    def _init_grid(self):
        """
        Initialize a 3D grid used for sampling 2D features
        """
        # coordinate convolution makes arg-max easier
        map_height, map_width = self.cfg.grid_resolution[2], self.cfg.grid_resolution[1]
        x_map = np.tile(np.linspace(0, 1, map_width), (map_height, 1))
        x_map = x_map.reshape(1, 1, map_height, map_width)
        z_map = np.linspace(0, 1, map_height).reshape(map_height, 1)
        z_map = np.tile(z_map, (1, map_width))
        z_map = z_map.reshape(1, 1, map_height, map_width)
        self.coor_maps = np.concatenate([x_map, z_map], axis=1).astype(np.float32)
        self.coor_maps = torch.from_numpy(self.coor_maps)  
        self.xrange = self.cfg.x_range[1] - self.cfg.x_range[0]
        self.zrange = self.cfg.z_range[1] - self.cfg.z_range[0]
        return
    
    def _init_3d_net(self):
        """
        Initialize 3D convolutional network.
        """
        # input of shape [N_batch, F, N_sample_h, N_sample_w, N_sample_l], e.g., [1, 64, 30, 150, 150]
        # output of shape [N_bach, N_pred, N_sample_h, N_sample_w, N_sample_l]
        # N_pred = 1: only predicts the confidence of the 3D center
        if self.cfg.vernier_type in ['3D', 'BEV']:
            dim = int(self.cfg.hrfeat.output_channel * 2)
        else:
            dim = self.cfg.hrfeat.output_channel
        num_parts = getattr(self.cfg, 'num_parts', 9)
        if self.cfg.vernier_type == '3D':
            self.conv1 = nn.Sequential(convbn_3d(dim, dim, 3, 1, 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv2 = nn.Sequential(convbn_3d(dim, dim, 3, 1, 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.hg_conv = hourglass(dim, gn=self.cfg.gn)
            self.classifier = nn.Conv3d(dim, 
                                       1, 
                                       kernel_size=1, 
                                       padding=0, 
                                       stride=1,
                                       bias=False
                                       )
        elif self.cfg.vernier_type == 'BEV':
            self.conv1 = nn.Sequential(convbn_3d(dim, dim, 3, (2,1,1), 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv2 = nn.Sequential(convbn_3d(dim, dim, 3, (2,1,1), 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.pool_3d = torch.nn.AvgPool3d((2, 1, 1), stride=(2, 1, 1))
            self.conv3 = nn.Sequential(convbn(dim*4, dim*2, 3, 1, 1, 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )            
            self.hg_conv2d = hourglass2d(dim*2, gn=self.cfg.gn)
            # occupancy prediction
            self.occu_conv1 = nn.Sequential(convbn(dim*2, 
                                                   dim*2, 
                                                   kernel_size=3, 
                                                   stride=1, 
                                                   pad=1,
                                                   dilation=1,
                                                   gn=self.cfg.gn
                                                   ),
                                            nn.ReLU(inplace=True)
                                            )
            self.occu_conv2 = nn.Sequential(nn.Conv2d(dim*2, 
                                                      self.cfg.grid_resolution[0], 
                                                      3, 
                                                      1, 
                                                      1,
                                                      bias=False
                                                      ),
                                            nn.Sigmoid()
                                            )            
            # heatmap prediction
            # self.hm1 = hourglass2d(dim*2, gn=self.cfg.gn)
            self.hm1 = nn.Sequential(convbn(dim*2, dim*4, 3, 2, 1, 1, gn=self.cfg.gn),
                                     nn.ReLU(inplace=True)                                    
                                     )      
            self.hm2 = get_feat_extraction(cfg=getattr(self.cfg, self.cfg.backbone),
                                           is_train=self.is_train,
                                           head_type="heatmap_regression"
                                           )
            # self.classifier = nn.Conv2d(dim*2, 
            #                             9, 
            #                             kernel_size=1, 
            #                             padding=0, 
            #                             stride=1,
            #                             bias=False
            #                             )
        elif self.cfg.vernier_type == 'BEV_type2':
            self.vimg_feat = nn.Sequential(convbn_3d(2*dim, dim, 1, 1, 0, gn=self.cfg.gn),
                                           nn.ReLU(inplace=True)
                                           )
            self.conv1 = nn.Sequential(convbn_3d(2*dim, dim, 7, 1, 3, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv2 = nn.Sequential(convbn_3d(dim, dim, 5, 1, 2, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv3 = nn.Sequential(convbn_3d(dim, dim, 5, 1, 4, dilation=2, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv4 = nn.Sequential(convbn_3d(2*dim, dim, 3, 1, 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            if self.cfg.n_sample_w <= 16:
                self.hg_conv3d = hourglass(dim, gn=self.cfg.gn)
            else:
                self.hg_conv3d = hourglass_downsample_16(dim, gn=self.cfg.gn)
            self.fg_cls_head = nn.Sequential(convbn_3d(dim, 
                                                       dim, 
                                                       3, 
                                                       1, 
                                                       1, 
                                                       gn=self.cfg.gn),
                                             nn.ReLU(inplace=True),
                                             nn.Conv3d(dim, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid()
                                             )
            # self.part_reg_head = nn.Sequential(convbn_3d(dim, 
            #                                               dim, 
            #                                               3, 
            #                                               1, 
            #                                               1, 
            #                                               gn=self.cfg.gn),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Conv3d(dim, 27, 1, 1, 0, bias=False)
            #                                     )            
            self.pool_3d = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
            self.conv5 = nn.Sequential(convbn(dim*8, 64, 3, 1, 1, 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )                         
            # self.hm = get_feat_extraction(cfg=getattr(self.cfg, self.cfg.backbone),
            #                                is_train=self.is_train,
            #                                head_type="heatmap_regression"
            #                                )
            if self.cfg.n_sample_w <= 16:
                self.hm1 = hourglass2d(64, gn=self.cfg.gn)
            else:
                self.hm1 = hourglass2d_downsample_16(64, gn=self.cfg.gn)
            self.hm2 = nn.Conv2d(64, 
                                 num_parts, 
                                 3, 
                                 1, 
                                 1,
                                 bias=False
                                 )
        elif self.cfg.vernier_type == 'BEV_type3':
            self.vimg_feat = nn.Sequential(convbn_3d(2*dim, dim, 1, 1, 0, gn=self.cfg.gn),
                                           nn.ReLU(inplace=True)
                                           )
            self.conv1 = nn.Sequential(convbn_3d(2*dim, dim, 7, 1, 3, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv2 = nn.Sequential(convbn_3d(dim, dim, 5, 1, 2, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv3 = nn.Sequential(convbn_3d(dim, dim, 5, 1, 4, dilation=2, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            self.conv4 = nn.Sequential(convbn_3d(2*dim, dim, 3, 1, 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )
            if self.cfg.n_sample_w <= 16:
                self.hg_conv3d = hourglass(dim, gn=self.cfg.gn)
            else:
                self.hg_conv3d = hourglass_downsample_16(dim, gn=self.cfg.gn)
            self.fg_cls_head = nn.Sequential(convbn_3d(dim, 
                                                       dim, 
                                                       3, 
                                                       1, 
                                                       1, 
                                                       gn=self.cfg.gn),
                                             nn.ReLU(inplace=True),
                                             nn.Conv3d(dim, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid()
                                             )
            if getattr(self.cfg, 'use_part_reg_head', False):
                self.part_reg_head = nn.Sequential(convbn_3d(dim, 
                                                              dim, 
                                                              3, 
                                                              1, 
                                                              1, 
                                                              gn=self.cfg.gn),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(dim, 27, 1, 1, 0, bias=False)
                                                    )            
            self.pool_3d = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
            if self.cfg.grid_resolution[0] == 32:
                dim_height = 256
            elif self.cfg.grid_resolution[0] == 16:
                dim_height = 128
            else:
                raise NotImplementedError
            self.conv5 = nn.Sequential(convbn(dim_height, 64, 3, 1, 1, 1, gn=self.cfg.gn),
                                       nn.ReLU(inplace=True)
                                       )                         
            # self.hm = get_feat_extraction(cfg=getattr(self.cfg, self.cfg.backbone),
            #                                is_train=self.is_train,
            #                                head_type="heatmap_regression"
            #                                )
            if self.cfg.n_sample_w <= 16:
                self.hm1 = hourglass2d(64, gn=self.cfg.gn)
            else:
                self.hm1 = hourglass2d_downsample_16(64, gn=self.cfg.gn)
            self.hm2 = nn.Conv2d(64, 
                                 num_parts, 
                                 3, 
                                 1, 
                                 1,
                                 bias=False
                                 )
        return
    
    def _get_2d_predctions(self):
        """
        Get 2D predictions from projected 3D RoI for calculating the consistency
        loss.
        """            
        return
    
    def _sample_2d_feat(self, left, right, l_pts, r_pts, aggregate="concat"):
        """
        Aggregate 2D features for 3D points by projecting to left/right images
        and sample.
        """
        # TODO: do not color points that lie outside of the image
        nh, nw, nl = self.cfg.n_sample_h, self.cfg.n_sample_w, self.cfg.n_sample_l
        N_F = left.shape[1]
        batch_size = len(l_pts)
        l_pts = l_pts.permute(0, 2, 1).reshape(batch_size, nh, nw*nl, 2)
        r_pts = r_pts.permute(0, 2, 1).reshape(batch_size, nh, nw*nl, 2)
        # normalize the screen coordinates to [-1, 1]
        l_pts[:,:,:,0] = l_pts[:,:,:,0] / self.cfg.resolution[1] * 2 -1
        l_pts[:,:,:,1] = l_pts[:,:,:,1] / self.cfg.resolution[0] * 2 -1
        r_pts[:,:,:,0] = r_pts[:,:,:,0] / self.cfg.resolution[1] * 2 -1
        r_pts[:,:,:,1] = r_pts[:,:,:,1] / self.cfg.resolution[0] * 2 -1        
        feat_samp_left = grid_sample(left, l_pts).reshape(batch_size, N_F, nh, nw, nl)
        feat_samp_right = grid_sample(right, r_pts).reshape(batch_size, N_F, nh, nw, nl)
        if aggregate == "concat-atten":
            feat_voxel = torch.cat([feat_samp_left, feat_samp_right], dim=1)
            attention = cosine_similarity(feat_samp_left, feat_samp_right, dim=1).unsqueeze(1)
            feat_voxel = feat_voxel * torch.clamp(attention, 0.)
        elif aggregate == "concat":
            feat_voxel = torch.cat([feat_samp_left, feat_samp_right], dim=1)
        else:
            raise NotImplementedError        
        return feat_voxel
    
    def construct_voxel(self, left, right, grid_proj_left, grid_proj_right):
        """
        Construct a voxel representation based on left/right roi features.
        """        
        if self.estimate_depth:
            # build intermediate depth representation
            cv = self.build_cost_volume(left, right)
            # sample the cost volume feature to form a voxel representation
            voxel = None
            # depth prediction
            depth = None
        else:
            # directly build voxel without intermediate depth estimation
            depth = None        
            voxels = self._sample_2d_feat(left, 
                                          right, 
                                          grid_proj_left, 
                                          grid_proj_right
                                          )
        return voxels, depth
    
    def predict_3d_heatmaps(self, voxel, depth=None):
        """
        3D heatmap regression to represent displacement.
        """        
        if depth is None and self.cfg.vernier_type == '3D':
            voxel = self.conv1(voxel)
            voxel = self.conv2(voxel)
            voxel1, pre_voxel, post_voxel = self.hg_conv3d(voxel, None, None)
            voxel = voxel + voxel1
            heatmaps = self.classifier(voxel)
        elif depth is None and self.cfg.vernier_type == 'BEV':
            voxel = self.conv1(voxel)
            voxel = self.conv2(voxel)
            voxel = self.pool_3d(voxel)
            N, F, H, W, L = voxel.shape
            voxel_BEV = voxel.reshape(N, -1, W, L)
            voxel_BEV = self.conv3(voxel_BEV)
            voxel1, pre_voxel, post_voxel = self.hg_conv2d(voxel_BEV, None, None)
            voxel_BEV = voxel_BEV + voxel1
            # occupancy
            occupancy = self.occu_conv1(voxel_BEV)
            occupancy = self.occu_conv2(occupancy)
            # heatmaps
            # voxel2, _, _ = self.hm1(voxel_BEV, None, None)
            # voxel_heatmap = voxel_BEV + voxel2
            # heatmaps = self.classifier(voxel_heatmap).permute(0, 1, 3, 2)
            heatmaps = self.hm1(voxel_BEV)
            heatmaps = self.hm2(heatmaps).permute(0, 1, 3, 2)
        elif depth is None and self.cfg.vernier_type == 'BEV_type2':
            voxel_img_feat = self.vimg_feat(voxel)
            # 3D convolution
            voxel = self.conv1(voxel)
            voxel = self.conv2(voxel) + voxel
            voxel = self.conv3(voxel) + voxel
            voxel = self.hg_conv3d(voxel) + voxel
            # foreground classification and part regression
            # occupancy, part_reg = self.fg_cls_head(voxel), self.part_reg_head(voxel)
            # occupancy, part_reg = occupancy.squeeze(1), part_reg.squeeze(1)
            occupancy = self.fg_cls_head(voxel)
            # offset = self.part_reg_head(voxel)
            offset, coordinates, bbox = None, None, None
            # concatenate image feature
            voxel = torch.cat([voxel, voxel_img_feat * occupancy], dim=1)
            # convert to BEV representation
            voxel = self.conv4(voxel)
            voxel = self.pool_3d(voxel)
            N, F, H, W, L = voxel.shape
            voxel_BEV = voxel.reshape(N, -1, W, L)
            # processing BEV features
            voxel_BEV = self.conv5(voxel_BEV)
            heatmap_feats = self.hm1(voxel_BEV).permute(0, 1, 3, 2)
            heatmaps = self.hm2(heatmap_feats)
        elif depth is None and self.cfg.vernier_type == 'BEV_type3':
            voxel_img_feat = self.vimg_feat(voxel)
            # 3D convolution
            voxel = self.conv1(voxel)
            voxel = self.conv2(voxel) + voxel
            voxel = self.conv3(voxel) + voxel
            if self.cfg.n_sample_w <= 16:
                voxel = self.hg_conv3d(voxel, None, None)[0] + voxel
            else:
                voxel = self.hg_conv3d(voxel) + voxel
            # foreground classification and part regression
            # occupancy, part_reg = self.fg_cls_head(voxel), self.part_reg_head(voxel)
            # occupancy, part_reg = occupancy.squeeze(1), part_reg.squeeze(1)
            occupancy = self.fg_cls_head(voxel)
            if hasattr(self, 'part_reg_head'):
                offset = self.part_reg_head(voxel)
            else:
                offset = None
            # concatenate image feature
            voxel = torch.cat([voxel, voxel_img_feat * occupancy], dim=1)
            # convert to BEV representation
            voxel = self.conv4(voxel)
            voxel = self.pool_3d(voxel)
            N, F, H, W, L = voxel.shape
            voxel_BEV = voxel.reshape(N, -1, W, L)
            # processing BEV features
            voxel_BEV = self.conv5(voxel_BEV)
            if self.cfg.n_sample_w <= 16:
                heatmap_feats = self.hm1(voxel_BEV, None, None)[0].permute(0, 1, 3, 2)
            else:
                heatmap_feats = self.hm1(voxel_BEV).permute(0, 1, 3, 2)
            heatmaps = self.hm2(heatmap_feats)
            # map heatmaps to coordinates
            num_sample = len(heatmaps)
            coor_maps = self.coor_maps.repeat(num_sample, 1, 1, 1).to(heatmaps.device)
            augmented_maps = torch.cat([heatmaps, coor_maps], dim=1)
            coordinates = self.coord_head(augmented_maps).view(num_sample, -1, 2)
            # TODO add coordinate regression and IoU loss head 
            if hasattr(self, 'bbox_head'):
                bbox = self.bbox_head(coordinates.reshape(num_sample, -1))
            else:
                bbox = None
        else:
            raise NotImplementedError
        return heatmaps, occupancy.squeeze(1), offset, coordinates, bbox
    
    def forward(self, 
                left_roi, 
                right_roi,
                grid_proj_left,
                grid_proj_right,
                meta_data=None,
                test=False
                ):
        """
        Forward pass with left/right region of interest.
        """        
        left_feat = self.feat_net(left_roi)
        right_feat = self.feat_net(right_roi)
        if self.use_2d:
            raise NotImplementedError
        voxels, depths = self.construct_voxel(left_feat, 
                                              right_feat, 
                                              grid_proj_left,
                                              grid_proj_right
                                              )
        # a test of the feature aggregation process
        if test:
            idx = 0
            down_samp = 4.
            N = len(left_feat)
            nh, nw, nl = self.cfg.n_sample_h, self.cfg.n_sample_w, self.cfg.n_sample_l
            i, j, k = (np.random.randint(0, nh),
                       np.random.randint(0, nw),
                       np.random.randint(0, nl)
                       )
            p_3d_cam = meta_data['grid_3d'][idx].reshape(nh, nw, nl, 3)
            p_3d_cam = p_3d_cam[i,j,k].reshape(1,3)
            pts_2d_l = meta_data['calib_left'][idx].project_rect_to_image(p_3d_cam)
            pts_2d_r = meta_data['calib_right'][idx].project_rect_to_image(p_3d_cam)
            coord_l = affine_transform(pts_2d_l, meta_data['trans_l'][idx]) / down_samp
            coord_r = affine_transform(pts_2d_r, meta_data['trans_r'][idx]) / down_samp
            # test projection of 3D coordinates
            gpl = meta_data['grid_proj_left'].reshape(N, 2, nh, nw, nl).data.cpu().numpy()
            gpr = meta_data['grid_proj_right'].reshape(N, 2, nh, nw, nl).data.cpu().numpy()
            print(coord_l[:,0]*down_samp - gpl[idx,:,i,j,k])
            print(coord_r[:,0]*down_samp - gpr[idx,:,i,j,k])
            # 2D feature at this location
            x_l, y_l = int(np.round(coord_l[0,0])), int(np.round(coord_l[1,0]))
            x_r, y_r = int(np.round(coord_r[0,0])), int(np.round(coord_r[1,0]))
            lf_samp, rf_samp = left_feat[idx, :, y_l, x_l], right_feat[idx, :, y_r, x_r]
            # sampled 3D feature
            f3d_samp = voxels[idx, :, i, j, k]
            # difference
            dif = torch.abs(f3d_samp - torch.cat((lf_samp, rf_samp)))
            print(dif)
            print(f3d_samp)
            # plotting
            test_visualization(left_roi, 
                               right_roi,
                               meta_data,
                               p_3d_cam,
                               (i,j,k),
                               [x_l*down_samp, y_l*down_samp],
                               [x_r*down_samp, y_r*down_samp],
                               self.cfg,
                               idx=0
                               )            
        ncf, occupancy, part_offsets, coordinates, bboxes = self.predict_3d_heatmaps(voxels, depths)
        if test:
            # plot occupancy
            plt.figure()
            ax = plt.subplot(111) 
            occupancy_vis = occupancy[0].data.cpu().numpy()
            i_select = 16
            ax.imshow(occupancy_vis[i_select,:,:])
            ax.set_xlabel("z index")
            ax.set_ylabel("x index")
            ax.set_title("Occupancy map for y={:d}".format(i_select))
            if meta_data is None or meta_data['gt'][0] is None:
                pass
            else:
                spa = np.array(self.cfg.spacing)
                re = np.array(self.cfg.grid_resolution)
                # calculate and plot ground truth box
                num_parts = self.cfg.num_parts
                kpts_3d_gt = get_cam_cord(meta_data['gt'][0]).T
                kpts_3d_sample = get_cam_cord(meta_data['samples'][0]).T
                offset = kpts_3d_gt[:num_parts, :] - kpts_3d_sample[[0]]
                # rotation -> inner product with basis vectors defined by the sample
                basis = self._get_basis(meta_data['samples'][0])
                x, y, z = np.split(offset @ basis, 3, axis=1)
                # convert location to field index        
                (ny, nx, nz) = 0.5 * (np.array(re) - 1)
                indices = (np.floor((y + ny*spa[0])/spa[0]), # h direction
                           np.floor((x + nx*spa[1])/spa[1]), # w direction
                           np.floor((z + nz*spa[2])/spa[2])  # l direction
                           )
                ax.plot(indices[2], indices[1], 'ro')
        outputs = {#'depth':depths, 
                   'ncf':ncf,
                   'occupancy':occupancy,
                   # 'offset':part_offsets,
                   'coordinates':coordinates,
                   # 'bbox':bboxes
                   }
        return outputs
    
    def ncf_to_offset(self, 
                      ncf,
                      grid_3d,
                      arg_max='hard'
                      ):
        """
        Convert neural confidence field to coordinate-based offeset in the 
        camera coordinate system.
        """           
        # obtain local coordinates with a arg-max function
        num_instance, num_parts  = len(ncf), ncf.shape[1]
        if arg_max == 'hard':
            ncf_np = ncf.data.cpu().numpy()
            ncf_shape = ncf_np.shape[2:]
            # the indices for the flattend array
            flat_indices = np.argmax(ncf_np.reshape(num_instance, num_parts, -1), axis=2)
            i, j, k = np.unravel_index(flat_indices, ncf_shape)
        else:
            raise NotImplementedError
        offset = grid_3d[:, i, j, k]        
        return offset

    def ncf_to_loc(self, 
                   ncf, 
                   current_pred, 
                   grid_3d,
                   arg_max='hard'
                   ):
        """
        Convert neural confidence field to coordinate-based location in the 
        camera coordinate system.
        """           
        offset = self.ncf_to_offset(ncf, grid_3d)
        # convert to global coordinates in the camera coordinate system
        pred_center = current_pred[:, 3:6] # this is the bottom center
        pred_center[:, 1] -= current_pred[:, 0] * 0.5
        loc = pred_center + offset[:,:,0].T # first part is the center        
        return loc, offset

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
    
    def get_canonical(self, w, l):
        """
        Compute 2D part coordinates in the camera coordinate system for a box
        with canonical pose.
        """        
        x_corners = [0, 0.5*l, 0.5*l, 0.5*l, 0.5*l, -0.5*l, -0.5*l, -0.5*l, -0.5*l]
        z_corners = [0, 0.5*w, 0.5*w, -0.5*w, -0.5*w, 0.5*w, 0.5*w, -0.5*w, -0.5*w]
        corners_2d = np.array([x_corners, z_corners])  
        return corners_2d
    
    def get_euler_2d(self, canonical, pred):
        R, T = compute_rigid_transform(canonical, pred)
        theta = np.arctan2(R[1,0], R[0,0]) # arctan(y, x)
        return theta
    
    def register_BEV(self, src, dst, sample, conf=None, debug=False):
        """
        Estimate the transformation (rotation, translation) from a 
        set of source coordinates to destination coordinates.
        
        Return the transformed sample given the estimated transformation.
        """
        R, T = compute_rigid_transform(src, dst, W=conf)
        transformed_src = R @ src + T
        # convert the transformed points to KITTI prediction format
        final_pred = sample.copy()
        canonical_coord = self.get_canonical(sample[1], sample[2])
        # orientaiton and translation in the camera coordinate system
        angle = self.get_euler_2d(canonical_coord, transformed_src)
        # the orientation annotation in KITTI is positive for clockwise rotation       
        final_pred[6] = -angle
        final_pred[[3,5]] = transformed_src[:, 0]        
        if debug:
            # visualization
            print('sample', sample)
            plt.figure()
            ax = plt.subplot(111)
            ax.scatter(src[0,:], src[1,:], c='r')
            vp.annotate_points(src.T, ax, 'r')
            ax.text(src[0,0] + 0.03, src[1,0] + 0.03, '{:.3f}'.format(sample[6]), c='r')
            ax.scatter(dst[0,:], dst[1,:], c='k')
            vp.annotate_points(dst.T, ax, 'k')
            ax.scatter(transformed_src[0,:] + 0.03, transformed_src[1,:] + 0.03, c='m')
            ax.text(transformed_src[0,0], transformed_src[1,0], '{:.3f}'.format(angle), c='m')
            vp.annotate_points(transformed_src.T, ax, 'm')
            ax.set_aspect('equal')
            ax.set_title('Src:Red Dst:Black Pred:Magenta')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('z (m)')
        return final_pred
    
    def ncf_to_update_2d(self,
                         ncf,
                         samples,
                         grid,
                         filter_3d,
                         arg_max='hard',
                         coordinates=None
                         ):
        """
        Convert neural confidence field to coordinate-based update in the camera
        coordinate system.
        
        ncf is of shape [N_z, N_x]. Increasing z points to car head.
        filter_3d decides whether to keep a refined prediction given its
        confidence and other parameters.
        """          
        # convert ncf to coordinates and confidence
        # obtain local coordinates with a arg-max function
        num_instance, num_parts = len(ncf), ncf.shape[1]
        ncf_np = ncf.data.cpu().numpy()
        ncf_reshape = ncf_np.reshape(num_instance, num_parts, -1)
        confidences = np.max(ncf_reshape, axis=2)
        keep_flags = filter_3d.query(ncf_reshape)
        if coordinates is not None:
            offset = np.zeros((len(coordinates), num_parts, 3))
            offset[:,:,[0,2]] = coordinates
            offset[:,:,0] = self.cfg.x_range[0] + offset[:,:,0]*self.xrange
            offset[:,:,2] = self.cfg.z_range[0] + offset[:,:,2]*self.zrange
        else:    
            # the indices for the flattend array
            flat_indices = np.argmax(ncf_reshape, axis=2)
            # offset in the object coordinate system
            offset = grid[flat_indices, :]
            offset[:,:,1] = 0.         
        # if confidence[0,0] > 1:
        #     # over confident
        #     offset[0,0,:] = 0.
        ret = {'pred':{'one_part':[]
                       }, 
               'confidence':confidences,
               'keep_flags':keep_flags
               }             
        if num_parts > 1:
            ret['pred']['five_part'] = []
        # prediction for each sample
        # a filter is used to decide whether to keep a refined
        # prediction or not.
        for idx, sample in enumerate(samples):
            if not keep_flags[idx] and num_parts == 1:
                ret['pred']['one_part'].append(sample.copy())
                continue
            if not keep_flags[idx] and num_parts > 1:
                ret['pred']['five_part'].append(sample.copy())
                continue                
            # convert to the camera coordinate system
            basis = self._get_basis(sample)
            # offset[idx,:,:] = (basis @ (offset[idx,:,:].T)).T
            offset[idx,:,:] = offset[idx,:,:] @ basis.T
            current_center = samples[idx, 3:6].copy() # this is the bottom center
            current_center[1] -= samples[idx, 0] * 0.5
            # part destination coordinates
            dst_part_coord = current_center[None,:] + offset[idx,:,:] #  
            # just use predicted center to move the current prediction
            pred_one_part = samples[idx].copy()
            pred_one_part[3:6] = dst_part_coord[0,:]
            pred_one_part[4] += samples[idx][0] * 0.5
            ret['pred']['one_part'].append(pred_one_part)
            if num_parts > 1:
                # solve a 2D least square problem to find the updated position
                src = (get_cam_cord(samples[idx]).T)[:, [0,2]]
                pred_five_part = self.register_BEV(src.T, 
                                                   dst_part_coord[:, [0,2]].T, 
                                                   sample,
                                                   conf=confidences[idx]
                                                   )
                ret['pred']['five_part'].append(pred_five_part)
        return ret
    
def construct_box_3d(l, h, w):
    """
    Temporary code.
    """ 
    x_corners = [0.5*l, l, l, l, l, 0, 0, 0, 0]
    y_corners = [0.5*h, 0, h, 0, h, 0, h, 0, h]
    z_corners = [0.5*w, w, w, 0, 0, w, w, 0, 0]
    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2
    corners_3d = np.array([x_corners, y_corners, z_corners])     
    return corners_3d

def get_cam_cord(box_3d):
    h, w, l = box_3d[0], box_3d[1], box_3d[2]
    corners_3d_fixed = construct_box_3d(l, h, w)
    # translation
    x, y, z = box_3d[3:6]
    ry = box_3d[6]
    rot_maty = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_maty, corners_3d_fixed)
    # translation
    corners_3d += np.array([x, y, z]).reshape([3, 1])
    return  corners_3d

def unnormalize(tensor, cfg):
    tensor_np = tensor.permute(1,2,0).data.cpu().numpy()
    mean = np.array(cfg.img_mean).reshape(1,1,3)
    std = np.array(cfg.img_std).reshape(1,1,3)
    return tensor_np * std + mean

def test_visualization(left_rois, 
                       right_rois,
                       meta_data,
                       p3d_cam,
                       indices,
                       p3d_proj_l,
                       p3d_proj_r,
                       cfg,
                       idx=0
                       ):
    left, right = left_rois[idx], right_rois[idx]
    plt.figure()
    ax = plt.subplot(131)
    # left image and projected corners of a sampled 3D box
    ax.set_title("Left ROI")
    ax.imshow(unnormalize(left, cfg))
    ax.plot([p3d_proj_l[0]], [p3d_proj_l[1]], 'mx', markersize=20)
    kpts_l_sample = meta_data['kpts_2d_l_local'][idx]   
    ax.plot(kpts_l_sample[:, 0], kpts_l_sample[:, 1], 'ro')
    vp.plot_3d_bbox(ax, kpts_l_sample[1:, :], color='r')                
    ax = plt.subplot(132)
    ax.set_title("Right ROI")
    # right image and projected corners of a sampled 3D box
    ax.imshow(unnormalize(right, cfg))
    ax.plot([p3d_proj_r[0]], [p3d_proj_r[1]], 'mx', markersize=20)
    kpts_r_sample = meta_data['kpts_2d_r_local'][idx]
    ax.plot(kpts_r_sample[:, 0], kpts_r_sample[:, 1], 'ro')
    vp.plot_3d_bbox(ax, kpts_r_sample[1:, :], color='r')             
    ax = plt.subplot(133, projection='3d')
    ax.set_title("3D Plot: point index " + str(indices))
    sample = meta_data['samples'][idx]
    sample_kpts_3d = get_cam_cord(sample).T
    box_3d = sample.copy()
    # the refinement space is represented as a 3D box, centered at the 
    # current sample and has the same orientation as the sample
    old_center_y = box_3d[4] - box_3d[0] * 0.5
    box_3d[:3] = cfg.grid_range # range of the refinement space
    box_3d[4] = old_center_y + box_3d[0] * 0.5 # update the bottom center y-coordinate
    # project to image and get 2D boxes
    field_kpts_3d = get_cam_cord(box_3d).T   
    if meta_data['gt'][idx] is not None:    
        gt = meta_data['gt'][idx]
        gt_kpts_3d = get_cam_cord(gt).T
    else:
        gt_kpts_3d = np.zeros((0,3))
    n_h, n_w, n_l = cfg.grid_resolution
    grid_kpts_3d = meta_data['grid_3d'][idx].reshape(n_h, n_w, n_l, 3) # the points of the 3D grid
    # down-sample for display
    grid_kpts_3d_ds = grid_kpts_3d[::5, ::5, ::5].reshape(-1, 3)
    limit_points = np.vstack([sample_kpts_3d, gt_kpts_3d, field_kpts_3d])
    vp.plot_3d_points(ax, p3d_cam, color='m', size=25)
    vp.plot_3d_points(ax, grid_kpts_3d_ds, color='b', size=1)
    vp.plot_3d_points(ax, sample_kpts_3d, color='r', size=15)
    vp.plot_lines(ax, sample_kpts_3d[1:,], vp.plot_3d_bbox.connections, dimension=3, c='r')
    vp.plot_3d_points(ax, field_kpts_3d, color='y', size=15)
    vp.plot_lines(ax, field_kpts_3d[1:,], vp.plot_3d_bbox.connections, dimension=3, c='y')
    if meta_data['gt'][idx] is not None:
        vp.plot_3d_points(ax, gt_kpts_3d, color='k', size=15)
        vp.plot_lines(ax, gt_kpts_3d[1:,], vp.plot_3d_bbox.connections, dimension=3, c='k')         
    vp.set_3d_axe_limits(ax, limit_points)
    return

def get_feat_extraction(cfg, is_train=False, **kwargs):
    if cfg.name in ['hrnet-w48', 'hrnet-w32']:
        return hrnet.get_model(cfg, is_train, **kwargs)
    else:
        raise NotImplementedError
        
def get_model(cfgs, is_train=False):
    return VernierScale(cfgs, is_train=is_train)