import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from functools import partial

from snvc.utils.torch_utils import compute_locations_bev
# from snvc.layers.sigmoid_focal_loss import SigmoidFocalLoss
# from snvc.layers.iou_loss import IOULoss
#from snvc.utils.bounding_box import compute_corners_sc
#from snvc.thirdparty.oriented_iou_loss import cal_diou, oriented_box_intersection_2d, enclosing_box

INF = 100000000
CFG_NAMES = ['CV_X_MIN', 'CV_Y_MIN', 'CV_Z_MIN', 
             'CV_X_MAX', 'CV_Y_MAX', 'CV_Z_MAX',
             'X_MIN', 'Y_MIN', 'Z_MIN',
             'X_MAX', 'Y_MAX', 'Z_MAX',
             'VOXEL_X_SIZE', 'VOXEL_Y_SIZE', 'VOXEL_Z_SIZE',
             ]

def sigmoid_focal_loss_multi_target(logits, 
                                    targets, 
                                    weights=None, 
                                    gamma=2., 
                                    alpha=0.25
                                    ):
    assert torch.all((targets == 1) | (targets == 0)), \
        'labels should be 0 or 1 in multitargetloss.'
    # N_locations * batchsize, N_angles * N_classes
    assert logits.shape == targets.shape 
    t = targets
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p + 1e-7)
    term2 = p ** gamma * torch.log(1 - p + 1e-7)
    loss = -(t == 1).float() * term1 * alpha - (t == 0).float() * term2 * (1 - alpha)
    if weights is None:
        return loss.sum()
    else:
        return (loss * weights).sum()

def smooth_l1_loss(input, target, weight, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return (loss.mean(dim=1) * weight).sum() / weight.sum()

def map2corners(pred):
    """map from a 7-tuple parametrization to 8 3D corners"""
    hwl = pred[:, 3:6]
    sin_d = torch.sin(pred[:, 6])
    cos_d = torch.cos(pred[:, 6])
    dxyz = pred[:, :3]
    box_corners = compute_corners_sc(hwl, sin_d, cos_d).reshape(-1, 3, 8)
    box_corners[:, 1, :] += hwl[:, 0:1] / 2.
    bbox_reg = box_corners + dxyz[:, :, None]            
    return bbox_reg

def disentangled_loss(pred, target, weight):
    """disentangled loss function where input prediction and target are 7-tuple"""
    group1 = torch.cat([pred[:, :3], target[:, 3:]], dim=1)
    group2 = torch.cat([target[:, [0,1,2]], pred[:, [3,4,5]], target[:, [6]]], dim=1)
    group3 = torch.cat([target[:, :6], pred[:, [6]]], dim=1)
    N = len(target)
    gt_corners = map2corners(target).reshape(N, -1)
    group1_corners = map2corners(group1).reshape(N, -1)
    group2_corners = map2corners(group2).reshape(N, -1)
    group3_corners = map2corners(group3).reshape(N, -1)
    loss = smooth_l1_loss(group1_corners, gt_corners, weight)
    loss += smooth_l1_loss(group2_corners, gt_corners, weight)
    loss += smooth_l1_loss(group3_corners, gt_corners, weight)
    return loss/3

class RPN3DLoss(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.cls_loss_func = partial(
            sigmoid_focal_loss_multi_target,
            gamma=self.cfg.RPN3D.FOCAL_GAMMA,
            alpha=self.cfg.RPN3D.FOCAL_ALPHA
            )
        self.box_reg_loss_func = smooth_l1_loss #IOULoss()
        # self.box_reg_loss_func = disentangled_loss
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.num_angles = self.cfg.num_angles
        self.num_classes = self.cfg.num_classes
        self.anchors_y = torch.as_tensor(self.cfg.RPN3D.ANCHORS_Y)
        self.anchor_angles = torch.as_tensor(self.cfg.ANCHOR_ANGLES)
        self.centerness4class = getattr(self.cfg, 'centerness4class', False)
        self.norm_expdist = getattr(self.cfg, 'norm_expdist', False)
        self.valid_classes = getattr(self.cfg, 'valid_classes', None)
        self.class4angles = getattr(self.cfg, 'class4angles', True)
        self.norm_factor = getattr(self.cfg, 'norm_factor', 1.)
        self.norm_max = getattr(self.cfg, 'norm_max', False)
        self.box_corner_parameters = getattr(self.cfg, 'box_corner_parameters', True)
        self.pred_reg_dim = 24 if self.box_corner_parameters else 7
        self.target_reg_dim = (4 + 24) if self.box_corner_parameters else (4 + 7)
        for name in CFG_NAMES:
            setattr(self, name, getattr(cfg, name))
            
    def prepare_targets(self, locations, targets, ious=None, labels_map=None):
        labels, reg_targets = [], [] # each target is a Box3DList (list of 3D bounding box)
        labels_centerness = []

        xs, zs = locations[:, 0], locations[:, 1]

        ys_cls = torch.zeros_like(xs)[:, None] + self.anchors_y.cuda()[None] # TODO: initialize the 3D location
        xs_cls = xs[:,None].repeat(1, self.num_classes)
        zs_cls = zs[:,None].repeat(1, self.num_classes)
        for i in range(len(targets)):
            target = targets[i]
            labels_per_im = target.get_field('labels')
            # note it should not be labels_per_im < 3.5
            non_ign_inds = labels_per_im.float() < 3.5 

            if non_ign_inds.sum() > 0:
                iou = torch.as_tensor(ious[i].toarray()).cuda() # the pre-computed distance
                labels_precomputed = torch.as_tensor(labels_map[i].toarray())

                box3ds = target.box3d[non_ign_inds].cuda()
                box3ds_corners = (target.box_corners() + target.box3d[:, None, 3:6])[non_ign_inds].cuda()
                labels_per_im = labels_per_im[non_ign_inds].cuda()
                # the bev bounding box of a 3D box 
                box3ds_rect_bev = box3ds_corners[:, :4, [0,2]] 
                box3ds_rect_bev = torch.cat([box3ds_rect_bev.min(1)[0], 
                                             box3ds_rect_bev.max(1)[0]], 
                                            dim=1
                                            ) # [N, 4] xmin, zmin, xmax, zmax

                box3ds_centers = box3ds_corners.mean(dim=1)

                l = xs[:, None] - box3ds_rect_bev[:, 0][None] # the l, r, t, b used in FCOS ICCV 19
                t = zs[:, None] - box3ds_rect_bev[:, 1][None]
                r = box3ds_rect_bev[:, 2][None] - xs[:, None]
                b = box3ds_rect_bev[:, 3][None] - zs[:, None]

                reg_targets2d_per_im = torch.stack([l, t, r, b], dim=2) # N_location, N_gt_bbox, 4 (l,r,t,b)
                reg_targets2d_per_im = reg_targets2d_per_im[:, None, :, :] 
                # 2d offset targets are shared among different classes
                reg_targets2d_per_im = reg_targets2d_per_im.repeat(1, self.num_classes, 1, 1) 
                reg_targets2d_per_im = reg_targets2d_per_im.reshape(len(locations), 
                                                                    self.num_classes, 
                                                                    len(box3ds), 
                                                                    4
                                                                    ) # TODO: this is not used

                locations3d = torch.stack([xs_cls, ys_cls, zs_cls], dim=2).reshape(-1, 3)

                if not self.box_corner_parameters:
                    reg_targets_per_im = (box3ds_centers[None] - locations3d[:, None])
                    reg_targets_per_im = reg_targets_per_im.reshape(len(locations), 
                                                                    self.num_classes, 
                                                                    len(box3ds), 
                                                                    3
                                                                    ) # the regression is class-agnostic
                    box3ds_parameters = torch.cat([box3ds[:, 0:3], box3ds[:, 6:]], dim=1)
                    # 3D (xyz) offset, dimenstion and rotation for each location
                    box3ds_p =  box3ds_parameters[None, None, :].expand(len(locations), 
                                                                        self.num_classes,
                                                                        len(box3ds),
                                                                        4
                                                                        )
                    reg_targets_per_im = torch.cat([reg_targets_per_im, 
                                                    box3ds_p
                                                    ], 
                                                   dim=3
                                                   ) 
                else:
                    reg_targets_per_im = (box3ds_corners[None] - locations3d[:,None,None])
                    reg_targets_per_im = reg_targets_per_im.reshape(len(locations), 
                                                                    self.num_classes, 
                                                                    len(box3ds), 
                                                                    24
                                                                    )
                reg_targets_per_im = torch.cat([reg_targets2d_per_im, 
                                                reg_targets_per_im
                                                ], 
                                               dim=-1
                                               )

                assert iou.shape[1] == len(box3ds), 'Number of Pre computed iou does not match current gts.'
                locations_min_dist, locations_gt_inds = iou.min(dim=1)

                labels_precomputed_inverse = -1 + torch.zeros((len(labels_precomputed), 1 + self.num_classes), dtype=torch.int32, device='cuda')
                labels_precomputed_inverse.scatter_(1, labels_precomputed.long().cuda(), 
                    torch.arange(len(box3ds))[None].expand(len(labels_precomputed), len(box3ds)).int().cuda() ) # N_anchor, 4 each row gives for each class, which instance is scattered to the location
                labels_precomputed_inverse = labels_precomputed_inverse[:, 1:]
                labels_precomputed_inverse = labels_precomputed_inverse.reshape(-1, self.num_angles, self.num_classes)

                labels_per_im = (labels_precomputed_inverse >= 0).int() # >=0 means the anchor is assigned to a target with index labels_precomputed_inverse[anchor_idx, class_idx]

                if self.norm_expdist:
                    min_dists = []
                    max_dists = []
                    for i in range(iou.shape[1]):
                        if labels_precomputed[:, i].sum() == 0: # not assigned to any anchor
                            max_dist = 1.
                            max_dists.append(max_dist)
                            min_dist = 0.
                            min_dists.append(min_dist)
                        else:
                            min_dist = iou[labels_precomputed[:, i] > 0, i].min().clamp(max=5.)
                            min_dists.append(min_dist)
                            if self.norm_max:
                                max_dist = iou[labels_precomputed[:, i] > 0, i].max().clamp(min=0.)
                                max_dists.append(max_dist)

                    min_dists = torch.as_tensor(min_dists, device='cuda')
                    if self.norm_max:
                        max_dists = torch.as_tensor(max_dists, device='cuda')
                        locations_norm_min_dist = (iou - min_dists[None]) / (max_dists[None] - min_dists[None]) 
                    else:
                        locations_norm_min_dist = iou - min_dists[None]
                    
                    if not self.centerness4class:
                        # range(len(iou)) iterate each row :, this line just select for each anchor the normalized distance
                        labels_centerness_per_im = locations_norm_min_dist[range(len(iou)), locations_gt_inds]
                        labels_centerness_per_im = labels_centerness_per_im.reshape(-1, self.num_angles)
                    else:
                        labels_centerness_per_im = locations_norm_min_dist.gather(1, 
                            (labels_precomputed_inverse * (labels_precomputed_inverse > 0).int()).reshape(-1, self.num_classes).long())
                        labels_centerness_per_im = labels_centerness_per_im.reshape(-1, self.num_angles, self.num_classes)
                    labels_centerness_per_im = torch.exp(-labels_centerness_per_im * self.norm_factor)
                else:
                    labels_centerness_per_im = torch.exp(-locations_min_dist)

                reg_targets_per_im = reg_targets_per_im[:, None].repeat(1, self.num_angles, 1, 1, 1).reshape(-1, len(box3ds), self.target_reg_dim)
                reg_targets_per_im = reg_targets_per_im[torch.arange(len(reg_targets_per_im)), labels_precomputed_inverse.reshape(-1).long()]
                reg_targets_per_im = reg_targets_per_im.reshape(-1, self.num_angles, self.num_classes, self.target_reg_dim)
            else:
                labels_per_im = torch.zeros(len(locations), self.num_angles, self.num_classes, dtype=torch.int32).cuda() # no target
                reg_targets_per_im = torch.zeros(len(locations), self.num_angles, self.num_classes, self.target_reg_dim, dtype=torch.float32).cuda()
                if not self.centerness4class:
                    labels_centerness_per_im = torch.zeros(len(locations), self.num_angles, dtype=torch.float32).cuda()
                else:
                    labels_centerness_per_im = torch.zeros(len(locations), self.num_angles, self.num_classes, dtype=torch.float32).cuda()

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            labels_centerness.append(labels_centerness_per_im)

        return labels, reg_targets, labels_centerness

    def __call__(self, bbox_cls, bbox_reg, bbox_centerness, targets, calib, calib_R, 
        ious=None, labels_map=None):
        N = bbox_cls.shape[0]

        locations_bev = compute_locations_bev(self.Z_MIN, 
                                              self.Z_MAX, 
                                              self.VOXEL_Z_SIZE, 
                                              self.X_MIN, 
                                              self.X_MAX, 
                                              self.VOXEL_X_SIZE, 
                                              bbox_cls.device
                                              )

        labels, reg_targets, labels_centerness = self.prepare_targets(locations_bev, 
                                                                      targets, 
                                                                      ious=ious, 
                                                                      labels_map=labels_map
                                                                      )

        labels = torch.stack(labels, dim=0)
        reg_targets = torch.stack(reg_targets, dim=0)
        labels_centerness = torch.stack(labels_centerness, dim=0)

        if self.class4angles:
            bbox_cls = bbox_cls.reshape(N, self.num_angles * self.num_classes, -1).transpose(1, 2).reshape(N, -1, self.num_angles, self.num_classes)
        else:
            bbox_cls = bbox_cls.reshape(N, self.num_classes, -1).transpose(1, 2).reshape(N, -1, self.num_classes)

        if not self.centerness4class:
            bbox_centerness = bbox_centerness.reshape(N, self.num_angles, -1).transpose(1, 2)
        else:
            bbox_centerness = bbox_centerness.reshape(N, self.num_angles * self.num_classes, -1).transpose(1, 2)

        bbox_reg = bbox_reg.reshape(N, 
                                    self.num_angles * self.num_classes * self.pred_reg_dim, 
                                    -1)
        bbox_reg = bbox_reg.transpose(1, 2).reshape(N, 
                                                    -1, 
                                                    self.num_angles, 
                                                    self.num_classes, 
                                                    self.pred_reg_dim
                                                    )

        loss = 0.
        cls_loss = 0.
        reg_loss = 0.
        centerness_loss = 0.

        if self.class4angles:
            bbox_cls = bbox_cls.reshape(-1, self.num_angles * self.num_classes)
        else:
            bbox_cls = bbox_cls.reshape(-1, self.num_classes)
        labels = labels.reshape(-1, self.num_angles * self.num_classes)
        bbox_reg = bbox_reg.reshape(-1, self.num_angles * self.num_classes, self.pred_reg_dim)
        reg_targets = reg_targets.reshape(-1, self.num_angles * self.num_classes, self.target_reg_dim)
        if not self.centerness4class:
            bbox_centerness = bbox_centerness.reshape(-1, self.num_angles)
            labels_centerness = labels_centerness.reshape(-1, self.num_angles)
        else:
            bbox_centerness = bbox_centerness.reshape(-1, self.num_angles * self.num_classes)
            labels_centerness = labels_centerness.reshape(-1, self.num_angles * self.num_classes)

        # cls loss
        pos_inds = torch.nonzero(labels > 0)

        if self.class4angles:
            labels_class = labels
        else:
            labels_class = labels.reshape(-1, self.num_angles, self.num_classes).sum(dim=1) > 0

        cls_loss += self.cls_loss_func( 
            bbox_cls,
            labels_class.int()
        ) / (pos_inds.shape[0] + 10)  # add N to avoid dividing by a zero

        bbox_reg = bbox_reg[pos_inds[:, 0], pos_inds[:, 1]]
        reg_targets = reg_targets[pos_inds[:, 0], pos_inds[:, 1]]
        if not self.centerness4class:
            labels_centerness = labels_centerness[pos_inds[:, 0], pos_inds[:, 1] // self.num_classes]
        else:
            labels_centerness = labels_centerness[pos_inds[:, 0], pos_inds[:, 1]]

        reg_targets_theta = reg_targets[:, -1:]
        bbox_reg_theta = bbox_reg[:, -1:]

        reg_targets = torch.cat([reg_targets[:, :-1], 
                                 torch.sin(reg_targets_theta * 0.5) * torch.cos(bbox_reg_theta * 0.5)
                                 ], 
                                dim=1
                                )
        bbox_reg = torch.cat([bbox_reg[:, :-1], 
                              torch.cos(reg_targets_theta * 0.5) * torch.sin(bbox_reg_theta * 0.5)
                              ], 
                             dim=1
                             )

        if not self.centerness4class:
            bbox_centerness = bbox_centerness[pos_inds[:, 0], pos_inds[:, 1] // self.num_classes]
        else:
            bbox_centerness = bbox_centerness[pos_inds[:, 0], pos_inds[:, 1]]

        # reg loss
        if pos_inds.shape[0] > 0:
            box2d_targets, box3d_corners_targets = torch.split(reg_targets, [4, self.pred_reg_dim], dim=1) # 2d offset and 3d regression targets
            centerness_targets = labels_centerness

            reg_loss += self.box_reg_loss_func(
                bbox_reg,
                box3d_corners_targets,
                centerness_targets
            )
            centerness_loss += self.centerness_loss_func(
                bbox_centerness,
                centerness_targets
            )
        else:
            reg_loss += bbox_reg.sum()
            centerness_loss += bbox_centerness.sum()

        loss = cls_loss + reg_loss + centerness_loss

        return loss, cls_loss, reg_loss, centerness_loss

# class VoxelMSELoss(nn.Module):
#     def __init__(self, use_target_weight=False):
#         super(VoxelMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(reduction='mean')
#         self.use_target_weight = use_target_weight

#     def forward(self, outputs, targets, target_weight=None, meta_data=None):
#         pred_heatmaps = outputs['ncf'] # [N, K, H, W, L]
#         batch_size = pred_heatmaps.size(0)
#         num_parts = pred_heatmaps.size(1)
#         heatmaps_pred = pred_heatmaps.reshape((batch_size, num_parts, -1)).split(1, 1)
#         heatmaps_gt = targets.reshape((batch_size, num_parts, -1)).split(1, 1)
#         loss = 0

#         for idx in range(num_parts):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             if self.use_target_weight:
#                 loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[idx]),
#                                              heatmap_gt.mul(target_weight[idx])
#                                              )
#             else:
#                 loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

#         return loss / num_parts

# test a different weight
def W_loss(prob, 
           target, 
           off, 
           mask,
           depth_levels,
           reduction='mean', 
           p=1
           ):
    # B,D,H,W to B,H,W,D
    off = off.permute(0, 2, 3, 1)
    prob = prob.permute(0, 2, 3, 1)
    grid = depth_levels[None, None, None, :]
    depth = grid + off
    target = target.unsqueeze(3)
    if p == 1:
        out = torch.abs(depth[mask] - target[mask])
    else:
        out = (depth[mask] - target[mask]) ** p

    loss = torch.sum(prob[mask] * out, 1)

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    
def calc_disp_loss(outputs, mask, gt_disp, loss_type='sl1'):
    """
    compute loss for disparity/depth prediction
    loss_type: 
        sl1: pixel-wise smooth l1 loss with mean prediction
        W1: Wasserstein-1 loss
    """
    if loss_type == 'sl1':
        depth_preds = [torch.squeeze(o, 1) for o in outputs['depth_preds']]
        disp_loss = 0.
        weight = [0.5, 0.7, 1.0]
        for i, o in enumerate(depth_preds):
            disp_loss += weight[3 - len(depth_preds) + i]  * \
                F.smooth_l1_loss(o[mask], gt_disp[mask], reduction='mean')
    elif loss_type == 'W1':
        disp_loss = W_loss(outputs['prob'],
                           gt_disp,
                           outputs['offset'],
                           mask,
                           outputs['depth_levels'],
                           reduction='mean',
                           p=1
                           )
    else:
        raise NotImplementedError
    return disp_loss

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, outputs, meta_data=None):
        pred = outputs['depth'] # [N, H, W]
        gt = meta_data['gt_depth'].to(pred.device)
        mask = ((gt != -1) & (gt < 60.)).detach()
        # debug
        # error = torch.ones(pred.shape, device=pred.device) * -1
        # error[mask] = torch.abs(pred[mask] - gt[mask])
        # import matplotlib.pyplot as plt
        # plt.figure()
        # ax = plt.subplot(131)
        # gt_vis = gt[0].data.cpu().numpy()
        # ax.imshow(gt_vis)
        # ax = plt.subplot(132)
        # pred_vis = pred[0].data.cpu().numpy()
        # ax.imshow(pred_vis)
        # ax = plt.subplot(133)
        # ax.imshow(error[0].data.cpu().numpy())        
        if mask.sum() > 0:
            return F.smooth_l1_loss(pred[mask], gt[mask], reduction='mean')
        else:
            return 0.
    
class VoxelMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(VoxelMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, outputs, targets, target_weight=None, meta_data=None):
        pred_heatmaps = outputs['ncf'] # [N, K, H, W, L]
        targets = targets.to(pred_heatmaps.device)
        batch_size = pred_heatmaps.size(0)
        num_parts = pred_heatmaps.size(1)
        heatmaps_pred = pred_heatmaps.reshape((batch_size, num_parts, -1)).split(1, 1)
        heatmaps_gt = targets.reshape((batch_size, num_parts, -1)).split(1, 1)
        loss = 0

        for idx in range(num_parts):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[idx]),
                                             heatmap_gt.mul(target_weight[idx])
                                             )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_parts
    
class OccupancyLoss(nn.Module):
    def __init__(self, 
                 use_target_weight=False, 
                 gamma=2., 
                 alpha=0.25
                 ):
        super(OccupancyLoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, targets, target_weight=None, meta_data=None):
        pred = outputs['occupancy'] # [N, H, W, L]
        gt = targets.to(pred.device)
        t, p = gt, pred
        term1 = (1 - p) ** self.gamma * torch.log(p + 1e-7)
        term2 = p ** self.gamma * torch.log(1 - p + 1e-7)
        loss = -(t == 1).float() * term1 * self.alpha - (t == 0).float() * term2 * (1 - self.alpha)      
        mask = gt != -1
        if mask.sum() > 0:
            return loss[mask].mean()
        else:
            return 0.

class OffsetLoss(nn.Module):
    def __init__(self):
        super(OffsetLoss, self).__init__()

    def forward(self, outputs, meta_data):
        pred = outputs['offset']
        occupancy = meta_data['occupancy'][:, None, None]
        gt = meta_data['offset'].to(pred.device)
        N_part, H, W, L = gt.shape[2], gt.shape[3], gt.shape[4], gt.shape[5]
        pred = pred.reshape(len(pred), 3, N_part, H, W, L)
        loss = F.l1_loss(pred, gt, reduction='none')
        mask = (occupancy == 1).repeat(1, 3, N_part, 1, 1, 1)
        if mask.sum() > 0:
            return loss[mask].mean()
        else:
            return 0.

SELECT_IND1 = [1, 3, 7, 5]
SELECT_IND2 = [2, 4, 8, 6]
def compute_area_4pts(pts, method='cross-product'):
    """
    pts: [1, N, 4, 2]
    
    Edge product approximates area by multiplying edge lengths.
    Cross product gives the exact area of a quatrilateral.
    """
    if method == 'edge-product':
        edge1 = (pts[:,:,1,:] - pts[:,:,0,:])**2
        edge2 = (pts[:,:,3,:] - pts[:,:,0,:])**2
        return torch.sqrt(edge1.sum(dim=-1)) * torch.sqrt(edge2.sum(dim=-1))
    elif method == 'cross-product':
        zero = torch.zeros(1, pts.shape[1], 4, 1, device=pts.device, dtype=pts.dtype)
        pts = torch.cat([pts, zero], dim=-1)
        edge1 = pts[:,:,1,:] - pts[:,:,0,:]
        edge2 = pts[:,:,3,:] - pts[:,:,0,:]
        edge3 = pts[:,:,1,:] - pts[:,:,2,:]
        edge4 = pts[:,:,3,:] - pts[:,:,2,:]  
        area1 = torch.linalg.norm(torch.cross(edge1, edge2), dim=-1)
        area2 = torch.linalg.norm(torch.cross(edge3, edge4), dim=-1) 
        return (area1 + area2) * 0.5
        raise NotImplementedError
    return

def compute_IoU_loss_corner(pred, gt):
    """
    Compute DIoU loss using noisy box corners as input which may deviate 
    slightly from a rectangle.
    
    pred: [1, N, 9, 2]
    gt: [1, N, 9, 2]
    """
    # iou, corners1, corners2, u = cal_iou(box1, box2)
    pred_corners = 0.5 * (pred[:, :, SELECT_IND1, :] + pred[:, :, SELECT_IND2, :])
    gt_corners = gt[:, :, SELECT_IND1, :]
    # calculate IoU
    inter_area, _ = oriented_box_intersection_2d(pred_corners, gt_corners)        #(B, N)
    # area1 = box1[:, :, 2] * box1[:, :, 3]
    # area2 = box2[:, :, 2] * box2[:, :, 3]
    area1 = compute_area_4pts(pred_corners)
    area2 = compute_area_4pts(gt_corners)
    u = area1 + area2 - inter_area
    iou = inter_area / u    
    # calculate displacement 
    w, h = enclosing_box(pred_corners, gt_corners, enclosing_type="smallest")
    c2 = w*w + h*h      # (B, N)
    # x_offset = box1[...,0] - box2[..., 0]
    # y_offset = box1[...,1] - box2[..., 1]
    # d2 = x_offset*x_offset + y_offset*y_offset
    offset = (pred[:, :, 0, :] - gt[:, :, 0, :])**2
    d2 = torch.sum(offset, dim=-1)
    diou_loss = 1. - iou + d2/c2
    return diou_loss, iou

class ShapeLoss(nn.Module):
    def __init__(self, scaling=1e4):
        super(ShapeLoss, self).__init__()
        self.scaling = scaling
        
    def forward(self, outputs, data_dict):
        pred = outputs['shape'] # [N, 512] 
        gt = data_dict['shape'].to(pred.device) 
        loss = F.l1_loss(pred, gt / self.scaling)
        return loss

def approximated_3d_iou_pt(pred, gt, bev_indices):
    """
    Approximate 3D overlap as bev overlap * height overlap
    
    pred: N, 7 [center_x/y/z, H, W, L, alpha]
    """
    pred_bbox = pred[None, :, bev_indices]           
    gt_bbox = gt[None, :, bev_indices]           
    _, iou_bev, overlap_bev = cal_diou(pred_bbox, gt_bbox)
    # height overlap
    pred_height_max = (pred[:, 1] + pred[:, 3] / 2)
    pred_height_min = (pred[:, 1] - pred[:, 3] / 2)
    gt_height_max = (gt[:, 1] + gt[:, 3] / 2)
    gt_height_min = (gt[:, 1] - gt[:, 3] / 2)        
    max_of_min = torch.max(pred_height_min, gt_height_min)
    min_of_max = torch.min(pred_height_max, gt_height_max)
    overlap_h = torch.clamp(min_of_max - max_of_min, min=0)
    # 3d iou
    overlaps_3d = overlap_bev[0] * overlap_h
    vol_pred = (pred[:, 3] * pred[:, 4] * pred[:, 5])
    vol_gt = (gt[:, 3] * gt[:, 4] * gt[:, 5])
    iou3d = overlaps_3d / torch.clamp(vol_pred + vol_gt - overlaps_3d, min=1e-6)     
    return iou3d
    
class BboxLoss(nn.Module):
    def __init__(self, cfg):
        super(BboxLoss, self).__init__()
        self.bbox_type = '3D' if cfg.head_reg_type == 'vector3d' else '2D'
        self.beta = 0.2
        self.indices = [0,2,5,4,6]
        self.cls_iou3d_min = 0.45
        self.cls_iou3d_max = 0.6
        self.reg_iou3d_min = 0.55
        self.use_cls_loss = False
        self.use_reg_mask = False
        
    def forward(self, outputs, data_dict):
        if self.bbox_type == '2D':
            pred = outputs['bbox'] # [N, 5] 
            gt = data_dict['gt_box_local'].to(pred.device) # ignore y coordinate
            loss = F.l1_loss(pred, gt)
            return {'l1': loss}
        elif self.bbox_type == '3D':
            pred = outputs['bbox'][:,:7] # [N, 9]  x,y,z,deltaH,deltaW,deltaL,alpha + confidence classification
            gt = data_dict['gt_box_local'].to(pred.device)    
            HWL_sample = np.zeros(pred.shape, dtype=np.float32)
            if isinstance(data_dict['samples'], list):
                # inference
                sample_array = np.vstack(data_dict['samples'])
            else:
                # training
                sample_array = data_dict['samples']
                
            # prepare 3D box
            HWL_sample[:, 3:6] = sample_array[:, :3]
            HWL_sample = torch.from_numpy(HWL_sample).to(pred.device)
            pred += HWL_sample
            gt += HWL_sample
            
            if not self.use_reg_mask:
                iou3d_pred_gt = approximated_3d_iou_pt(pred, gt, self.indices)
                loss_iou3d = (1 - iou3d_pred_gt).mean()
                # translation and dimension loss
                loss_loc_dim = F.smooth_l1_loss(pred, gt, beta=self.beta)                
                return {'sl1': loss_loc_dim * 0.5, 
                        'IoU3D':loss_iou3d
                        } 
        
            # classify easy and difficult boxes and only penalize the predictions
            # that have 3D IoU within a specified range
            samples_tensor = self._3dbox_to_pth(sample_array, pred.device)
            gt_tensor = self._3dbox_to_pth(np.vstack(data_dict['gt']), pred.device)
            iou3d_sample_gt = approximated_3d_iou_pt(samples_tensor, gt_tensor, self.indices)
            reg_mask, cls_mask, cls_label = self._get_mask(iou3d_sample_gt)
                
            # box regression loss
            reg_loss_cnt = reg_mask.sum()
            if  reg_loss_cnt > 0:
                iou3d_pred_gt = approximated_3d_iou_pt(pred, gt, self.indices)
                loss_iou3d = (1 - iou3d_pred_gt[reg_mask]).sum() / reg_loss_cnt
                # translation and dimension loss
                loss_loc_dim = F.smooth_l1_loss(pred[reg_mask, :], 
                                                gt[reg_mask, :], 
                                                beta=self.beta,
                                                reduction='sum'
                                                ) / reg_loss_cnt
            else:
                loss_iou3d = 0.
                loss_loc_dim = 0.
            
            # difficulty classification loss
            cls_loss_cnt = cls_mask.sum()
            if self.use_cls_loss and cls_loss_cnt > 0:
                confidence_pred = outputs['bbox'][:,7:]
                loss_cls = F.cross_entropy(confidence_pred[cls_mask, :], 
                                           cls_label[cls_mask],
                                           reduction='sum'
                                           ) / cls_loss_cnt
            else:
                loss_cls = 0.
            return {'sl1': loss_loc_dim * 0.5, 
                    'confidence': loss_cls,
                    'IoU3D':loss_iou3d
                    }                
        else:
            raise NotImplementedError
    
    def _3dbox_to_pth(self,
                      nparray,
                      device
                      ):
        array_copy = nparray[:, [3,4,5,0,1,2,6]].copy()
        # from bottom center to center
        array_copy[:, 1] = array_copy[:, 1] - 0.5 * array_copy[:, 3]
        # in the iou calculation the rotation is counter-clockwise while in kitti
        # the rotation is clockwise
        array_copy[:, 6] *= -1.
        return torch.from_numpy(array_copy).to(device)

    def _get_mask(self, iou3d):
        reg_mask = iou3d > self.reg_iou3d_min
        cls_mask = torch.logical_or(iou3d > self.cls_iou3d_max, iou3d < self.cls_iou3d_min)
        cls_label = -torch.ones(cls_mask.shape, 
                                dtype=torch.int64, 
                                device=cls_mask.device
                                )
        cls_label[iou3d > self.cls_iou3d_max] = 1
        cls_label[iou3d < self.cls_iou3d_min] = 0
        return reg_mask, cls_mask, cls_label
            
class CoordinateLoss(nn.Module):
    def __init__(self, cfg, enable_IoU=False, IoU_type='corner', normalize_gt=False):
        super(CoordinateLoss, self).__init__()
        self.enable_IoU = enable_IoU
        self.IoU_type = IoU_type
        # self.x_range = cfg.x_range
        self.xmin = cfg.x_range[0]
        self.xrange = cfg.x_range[1] - cfg.x_range[0]
        self.zmin = cfg.z_range[0]
        self.zrange = cfg.z_range[1] - cfg.z_range[0]
        self.normalize_gt = normalize_gt
        if self.enable_IoU:
            self.weight_l1 = 0.1
        else:
            self.weight_l1 = 1.
            
    def forward(self, outputs, meta_data):
        pred = outputs['coordinates'][None] # [1, N, 9, 2] 
        gt = (meta_data['gt_corners_local'][None, :, :, [0,2]]).to(pred.device) # ignore y coordinate
        if self.normalize_gt:
            # compute loss as descrepancy for 9 corner points
            gt[...,0] = (gt[...,0] - self.xmin) / self.xrange
            gt[...,1] = (gt[...,1] - self.zmin) / self.zrange
        loss = self.weight_l1 * F.l1_loss(pred, gt)
        if self.enable_IoU and self.IoU_type == 'bbox':
            pred_bbox = outputs['bbox']
            pred_bbox[:,:2] = torch.sigmoid(pred_bbox[:,:2]) * 2 - 1
            pred_bbox[:,2:4] = torch.sigmoid(pred_bbox[:,2:4])
            pred_bbox[:,[0,2]] *= self.xrange
            pred_bbox[:,[1,3]] *= self.zrange
            pred_bbox[:,4] = torch.sigmoid(pred_bbox[:,4]) * np.pi * 2             
            gt_bbox = meta_data['gt_box_local'].to(pred_bbox.device)
            iou_loss, iou = cal_diou(pred_bbox[None], gt_bbox[None])
            loss += iou_loss.mean()
        if self.enable_IoU and self.IoU_type == 'corner':
            # debug
            # noise = torch.randn(gt.shape, dtype=gt.dtype, device=gt.device)
            # loss, iou = compute_IoU_loss_corner(gt + noise, gt)
            diou_loss, iou = compute_IoU_loss_corner(pred, gt)
            loss += diou_loss.mean()
        # some temporary code for testing
        # print('sample', meta_data['samples'][0])
        # print('gt', meta_data['gt'][0])
        # print('pred_bbox', pred_bbox[0])
        # print('gt_bbox', gt_bbox[0])
        # print('iou_loss, iou', iou_loss, iou)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # ax = plt.subplot(111)
        # from snvc.thirdparty.utiles import box2corners
        # pred_corners = box2corners(*pred_bbox[0].data.cpu().numpy().tolist())
        # gt_corners = box2corners(*gt_bbox[0].data.cpu().numpy().tolist())
        # ax.plot(pred_corners[:,0], pred_corners[:,1], 'ro')
        # ax.plot(gt_corners[:,0], gt_corners[:,1], 'go')
        # ax.set_aspect('equal')    
        return loss
        
class VoxelMSELossWeighted(nn.Module):
    def __init__(self, use_target_weight=False):
        super(VoxelMSELossWeighted, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, outputs, targets, target_weight=None, meta_data=None):
        pred_heatmaps = outputs['ncf'] # [N, K, H, W, L]
        batch_size = pred_heatmaps.size(0)
        num_parts = pred_heatmaps.size(1)
        heatmaps_pred = pred_heatmaps.reshape((batch_size, num_parts, -1)).split(1, 1)
        heatmaps_gt = targets.reshape((batch_size, num_parts, -1)).split(1, 1)
        loss = 0

        for idx in range(num_parts):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[idx]),
                                             heatmap_gt.mul(target_weight[idx])
                                             )
            else:
                mask_posi = heatmap_gt > 0
                mask_zero = heatmap_gt <= 0
                assert mask_posi.sum() > 0
                loss += 0.5 * (self.criterion(heatmap_pred[mask_posi], heatmap_gt[mask_posi]) + 
                               self.criterion(heatmap_pred[mask_zero], heatmap_gt[mask_zero]))

        return loss / num_parts