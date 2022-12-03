"""
Implementation of the model-agnostic inference of a Vernier scale network. 
It corresponds to the V-A in the paper.
"""

import sys
sys.path.append('../')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.utils.data
torch.backends.cudnn.benchmark = True

# custom libraries
import snvc.visualization.points as vp

from snvc.dataset import KITTILoader3D as ls
from snvc.dataset import KITTIRefinement_dataset as DR

from snvc.utils.mis_utils import mem_info
from snvc.utils.exp_utils import Experimenter
from snvc.utils.torch_utils import box2corners_th
from snvc.models.vernier import get_model
from snvc.models.loss3d import VoxelMSELoss, OccupancyLoss, OffsetLoss, CoordinateLoss

def get_parser():
    parser = argparse.ArgumentParser(description='model-agnostic refinement')
    # path to a specified configuration file
    parser.add_argument('-cfg', '--cfg', '--config', default=None)
    # path to the KITTI dataset
    parser.add_argument('--data_path', default='../data/kitti/training/', help='data_path')
    # path to the model checkpoint
    parser.add_argument('--loadmodel', default=None, help='load model')
    # path to save the predictions
    parser.add_argument('--output_dir', default=None, help='path to the output directory')
    # switch to the debugging mode
    parser.add_argument('--debug', action='store_true', default=False, help='debugging mode')
    # use the training split for sanity check
    parser.add_argument('--train_split', action='store_true', default=False, help='train split')
    # specify the random seed value
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # fix the random seed
    parser.add_argument('--fix_seed', default=False, help='reproduce experiments')
    # specify the used Nvidia GPUs
    parser.add_argument('--devices', '-d', type=str, default=None)
    # path to the data split file
    parser.add_argument('--split_file', default='../data/kitti/val.txt', help='split file')
    # specify the number of workers used in data loading
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--btest', type=int, default=1)
    # customized experiment tag
    parser.add_argument('--tag', '-t', type=str, default='')
    # parser.add_argument('--debugnum', default=1, type=int, help='debug mode')    
    args = parser.parse_args()
    
    if not args.devices:
        args.devices = str(np.argmin(mem_info()))
    
    if args.devices is not None and '-' in args.devices:
        gpus = args.devices.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.devices = ','.join(map(lambda x: str(x), list(range(*gpus))))

    if args.debug:
        args.btest = len(args.devices.split(','))
        args.workers = 0
        args.tag += 'debug{}'.format(args.debugnum)
    
    if args.train_split:
        # inference with the training split
        args.split_file = '../data/kitti/train.txt'
        args.tag += '_train'
    
    print('Using GPU:{}'.format(args.devices))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    if args.fix_seed:    
        # fix random state generator
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
    return args

class Filter(object):
    def __init__(self, min_val=-1, max_val=2):
        self.min_val = min_val
        self.max_val = max_val
        return
    
    def query(self, ncf):
        ncf_reshape = ncf.reshape(len(ncf), -1)
        flags = np.logical_and(np.all(ncf_reshape >= self.min_val, axis=1),
                               np.all(ncf_reshape <= self.max_val, axis=1)
                               )
        return flags
    
def process_meta_data(meta_dict):
    meta_dict['grid_proj_left'] = torch.from_numpy(meta_dict['grid_proj_left'])
    meta_dict['grid_proj_right'] = torch.from_numpy(meta_dict['grid_proj_right'])
    for key in ['occupancy', 'offset', 'gt_corners_local', 'gt_box_local']:
        if key in meta_dict:
            meta_dict[key] = torch.from_numpy(meta_dict[key])
    return meta_dict

def num_params(model):
    return sum([p.data.nelement() for p in model.parameters()])

def calculate_loss(outputs, targets, loss_funcs, meta_data=None, weights=None):
    losses = {}
    total_loss = 0.
    
    losses['ncf'] = loss_funcs['ncf'](outputs, targets)
    total_loss += losses['ncf']
    
    if 'occupancy' in loss_funcs:
        # occupancy
        losses['occupancy'] = loss_funcs['occupancy'](outputs, meta_data['occupancy'])
        occup_weight = 1. if weights is None else weights.occupancy
        total_loss += losses['occupancy'] * occup_weight
        
    if 'offset' in loss_funcs:
        losses['offset'] = loss_funcs['offset'](outputs, meta_data)
        total_loss += losses['offset']
        
    losses['coordinates'] = loss_funcs['coordinates'](outputs, meta_data)
    coord_weight = 0.1 if weights is None else weights.coordinates
    total_loss = total_loss + losses['coordinates'] * coord_weight
    
    losses['total_loss'] = total_loss
    return losses

def show_ncf(ncf, string=''):
    # optional prediction parametrized by coordinates and bbox 
    fig = plt.figure()
    ax = plt.subplot(131)
    ax.imshow(ncf.mean(axis=0))
    ax.set_title(string + ': ZX plane')
    ax = plt.subplot(132)
    ax.imshow(ncf.mean(axis=1))
    ax.set_title(string + ': ZY plane')
    ax = plt.subplot(133)
    ax.imshow(ncf.mean(axis=2))
    ax.set_title(string + ': XY plane')  
    return fig

def normalize(inp):
    return (inp - inp.min()) / (inp.max() - inp.min())

def show_ncf_2d(ncf, string='', coordinates=None, bbox=None):
    fig = plt.figure()
    num_cols = 3 if len(ncf) > 1 else 1
    num_rows = int(np.round(len(ncf) / num_cols))
    for part_id in range(len(ncf)): # for each part
        ax = plt.subplot(num_rows, num_cols, part_id + 1)
        ax.imshow(ncf[part_id])
        # ax.imshow(normalize(ncf[part_id]))
        ax.invert_yaxis()
        ax.set_title('XZ plane, Part {:d}, {:s}'.format(part_id + 1, string))
        if coordinates is not None:
            ax.plot(coordinates[part_id][0], coordinates[part_id][1], 'rx')
        if bbox is not None:
            ax.plot(bbox[:,0],bbox[:,1],'yx')
    return fig

def plot_box_3d(ax, box, color, size):
    vp.plot_3d_points(ax, box, color=color, size=size)
    vp.plot_lines(ax, 
                  box[1:,], 
                  vp.plot_3d_bbox.connections, 
                  dimension=3, 
                  c=color
                  )
    return

def show_update(samples, pred_dict, dataset, gts=None, img_paths=None):
    figs = []
    num_styles = len(pred_dict)
    for ins_idx, sample in enumerate(samples):
        fig = plt.figure()
        if gts[0] is not None:
            gt = gts[ins_idx]
            gt_kpts_3d = dataset._get_cam_cord(gt).T       
        elif not "test" in dataset.split:
            # if the ground truth is not specified
            # find the closest gt 3D box with respect to the initial prediction
            neighbor = dataset.get_neighbor(img_paths[ins_idx], sample)
            gt_kpts_3d = dataset._get_cam_cord(neighbor).T 
        else:
            gt_kpts_3d = np.zeros((0,3))
        for style_idx, style_name in enumerate(pred_dict):           
            ax = plt.subplot(1, num_styles, style_idx + 1, projection='3d')
            ax.set_title(style_name)
            sample_kpts_3d = dataset._get_cam_cord(sample).T
            roi_3d = sample.copy()
            # the refinement space is represented as a 3D box, centered at the 
            # current sample and has the same orientation as the sample
            old_center_y = roi_3d[4] - roi_3d[0] * 0.5
            roi_3d[:3] = dataset.df_params['range'] # range of the refinement space
            roi_3d[4] = old_center_y + roi_3d[0] * 0.5 # update the bottom center y-coordinate
            field_kpts_3d = dataset._get_cam_cord(roi_3d).T       
            update_kpts_3d = dataset._get_cam_cord(pred_dict[style_name][ins_idx]).T
            limit_points = np.vstack([sample_kpts_3d, 
                                      gt_kpts_3d,
                                      field_kpts_3d,
                                      update_kpts_3d]
                                     )
            plot_box_3d(ax, sample_kpts_3d, 'r', 15)
            plot_box_3d(ax, field_kpts_3d, 'y', 15)
            plot_box_3d(ax, update_kpts_3d, 'm', 15)
            if len(gt_kpts_3d) != 0:
                plot_box_3d(ax, gt_kpts_3d, 'k', 15)
            vp.set_3d_axe_limits(ax, limit_points)            
        figs.append(fig)
    return figs

def visualize_outputs(outputs, 
                      targets=None, 
                      meta_data=None, 
                      output_type='BEV', 
                      dataset=None,
                      cfg=None
                      ):
    if output_type == '3D':
        # plot the projection of 3D displacement field
        ncf = outputs['ncf'][0][0].data.cpu().numpy()
        show_ncf(ncf, 'pred')
        ncf_gt = targets[0][0].data.cpu().numpy()
        show_ncf(ncf_gt, 'gt')
    elif output_type in ['BEV', 'BEV_type2', 'BEV_type3']:
        ncf = outputs['ncf'][0].data.cpu().numpy()
        # show_ncf_2d(ncf, 'pred')
        if 'coordinates' in outputs:
            coordinates_pred = outputs['coordinates'][0].data.cpu().numpy()
            coordinates_pred[:,0] *= cfg.grid_resolution[1]
            coordinates_pred[:,1] *= cfg.grid_resolution[2]
        else:
            coordinates_pred = None
        if 'bbox' in outputs:
            bbox_pred = outputs['bbox'] 
            corners = box2corners_th(bbox_pred[None])
            # map to heatmap indices
            bbox_pred = corners[0,0].data.cpu().numpy()
            bbox_pred[:,0] = (bbox_pred[:,0] - cfg.x_range[0]) / cfg.spacing[1]
            bbox_pred[:,1] = (bbox_pred[:,1] - cfg.z_range[0]) / cfg.spacing[2]
        else: 
            bbox_pred = None
        show_ncf_2d(ncf, 'pred', coordinates_pred, bbox_pred)
        if targets is not None:
            ncf_gt = targets[0].data.cpu().numpy()
            show_ncf_2d(ncf_gt, 'gt', coordinates_pred, bbox_pred)   
        # show updated 3D box
        if 'gt' in meta_data:
            gts = meta_data['gt']
        else:
            gts = None
        show_update(meta_data['samples'],
                    outputs['update']['pred'], 
                    dataset,
                    gts=gts,
                    img_paths=meta_data['lp']
                    )
    else:
        raise NotImplementedError
    return

def update_record(record, updates, meta_data):
    paths = meta_data['lp']
    
    for idx, img_path in enumerate(paths):
        img_path = paths[idx]
        save_name = img_path.split(os.sep)[-1][:-4] + ".txt"
        if save_name not in record:
            # record[save_name] = {'one_part':[], 'all_parts':[]}
            record[save_name] = {'all_parts':[]}
        # record[save_name]['one_part'].append(get_instance_str(updates, 
        #                                                      meta_data, 
        #                                                      idx, 
        #                                                      'one_part')
        #                                     )
        if 'all_parts' in updates['pred']:
            record[save_name]['all_parts'].append(get_instance_str(updates, 
                                                                 meta_data, 
                                                                 idx, 
                                                                 'all_parts')
                                                )
    return

def write_txt(pred, save_path):
    with open(save_path, 'w') as f:
        for idx, line in enumerate(pred):
            if idx != len(pred) - 1:
                f.write(line + '\n')
            else:
                f.write(line)
    print('Wrote prediction file at {:s}'.format(save_path))
    return

def generate_empty_file(output_dir, calib_dir):
    """
    Generate empty files for images without any predictions.
    """    
    all_files = os.listdir(calib_dir)
    detected = os.listdir(output_dir)
    for file_name in all_files:
        if not file_name.endswith(".txt"):
            continue
        if file_name not in detected:
            file = open(os.path.join(output_dir, file_name[:-4] + '.txt'), 'w')
            file.close()
    return

def generate_output(record, cfg, args):
    for pred_type in cfg.pred_type:
        save_folder = os.path.join(cfg.output_dir, pred_type, 'data')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    for file_name in record:
        for pred_type in cfg.pred_type:
            save_file_path = os.path.join(cfg.output_dir, pred_type, 'data', file_name)
            write_txt(record[file_name][pred_type], save_file_path)
    if "test" in args.split_file:
        calib_dir = os.path.join(args.data_path, 'calib')
        generate_empty_file(save_folder, calib_dir)
    return

def roty2alpha(x3d, z3d, ry3d):
    """
    Convert egocentric pose to allocentric pose in [-pi, pi].
    """   
    alpha = ry3d - np.arctan2(-z3d, x3d) - 0.5 * np.pi
    while alpha > np.pi: alpha -= np.pi * 2
    while alpha < (-np.pi): alpha += np.pi * 2    
    return alpha

def get_instance_str(updates, meta_data, idx, method):
    """
    Format KITTI style prediction string for one instance.
    """     
    box_2d = meta_data['box2d'][idx]
    box_3d = updates['pred'][method][idx]
    score = meta_data['score'][idx]
    string = ""
    string += "Car " # only for Car class for now
    string += "{:.1f} ".format(-1.) # truncation
    string += "{:.1f} ".format(-1.) # occlusion
    alpha = roty2alpha(box_3d[3], box_3d[5], box_3d[6])
    string += "{:.6f} ".format(alpha)
    string += "{:.6f} {:.6f} {:.6f} {:.6f} ".format(box_2d[0], box_2d[1], box_2d[2], box_2d[3])
    string += "{:.6f} {:.6f} {:.6f} ".format(box_3d[0], box_3d[1], box_3d[2])
    string += "{:.6f} {:.6f} {:.6f} ".format(box_3d[3], box_3d[4], box_3d[5])
    string += "{:.6f} ".format(box_3d[6])
    string += "{:.8f}".format(score)
    return string

@torch.no_grad()
def inference(nvs, dataset, loss_funcs, args, cfg, visualize=False):
    """
    Inference function.
    """
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=args.btest, 
                                         shuffle=args.debug, 
                                         num_workers=args.workers,
                                         collate_fn=BatchCollator(cfg)
                                         )
    save_record = {}
    filter_3d = Filter()
    
    for batch_idx, data_batch in enumerate(loader):
        print("Batch {:d}, progress {:.2f}".format(batch_idx, batch_idx/len(loader)))
        if batch_idx == 1 and args.debug:
            break
        left_rois, right_rois, targets, meta_data = data_batch
        # process the meta_data dictionary
        meta_data = process_meta_data(meta_data)
        # the inputs are PyTorch Tensor objects, which will be distributed
        # to multiple GPUs
        outputs = nvs(left_rois, 
                      right_rois, 
                      meta_data['grid_proj_left'],
                      meta_data['grid_proj_right'],
                      meta_data,
                      test=False
                      )
        coordinates_pred = outputs['coordinates'].data.cpu().numpy() if 'coordinates' in outputs else None
        outputs['update'] = nvs.module.ncf_to_update_2d(outputs['ncf'],
                                                        meta_data['samples'], 
                                                        dataset.grid_bev_flat,
                                                        filter_3d,
                                                        coordinates=coordinates_pred
                                                        )
        
        # record updated prediction for saving
        if cfg.save:
            update_record(save_record, outputs['update'], meta_data)

        if targets is not None:
            targets = targets.to(outputs['ncf'].device)
            if 'occupancy' in meta_data:
                meta_data['occupancy'] = meta_data['occupancy'].to(targets.device)
            loss_dict = calculate_loss(outputs, targets, loss_funcs, meta_data)
            for loss_component in loss_dict:
                print('{:s}: {:.6f}'.format(loss_component, 
                                            loss_dict[loss_component].item()
                                            ))
        if visualize:
            visualize_outputs(outputs, 
                              targets, 
                              meta_data, 
                              output_type=cfg.vernier_type,
                              dataset=dataset,
                              cfg=cfg
                              )
        del left_rois, right_rois, targets, meta_data

    if cfg.save and not cfg.debug:
        generate_output(save_record, cfg, args)
    return

def main():
    args = get_parser()
    assert args.loadmodel is not None and args.loadmodel.endswith('tar')
    
    if args.debug:
        args.savemodel = './outputs/debug/'
        args.workers = 0
    
    exp = Experimenter(os.path.dirname(args.loadmodel))
    cfg = exp.config
    cfg.debug = args.debug

    # directory that saves the outputs 
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    # initialize a model
    nvs = get_model(cfg, is_train=False)
    
    ckpt = torch.load(args.loadmodel)
    nvs.load_state_dict(ckpt['state_dict'], strict=True)
    print('Loaded {}'.format(args.loadmodel))    
    print('Number of model parameters: {}'.format(num_params(nvs)))

    # initializa dataset
    is_train = False if "test" in args.split_file else True
    all_left_img, all_right_img, all_left_disp, = ls.get_img_paths(args.data_path,
                                                                   args.split_file,
                                                                   depth_disp=True,
                                                                   cfg=cfg,
                                                                   is_train=is_train # used to load 3D box
                                                                   ) # get the list of paths for the training files

    dataset = DR.refinementDataset(all_left_img, 
                                   all_right_img, 
                                   all_left_disp, 
                                   False, 
                                   split=args.split_file, 
                                   cfg=cfg
                                   ) 
    # return
    # multi-gpu with DataParallel
    nvs.eval()
    nvs = torch.nn.DataParallel(nvs).cuda()
    loss_funcs = {'ncf':VoxelMSELoss(),
                  'occupancy':OccupancyLoss(),
                  'offset':OffsetLoss(),
                  'coordinates':CoordinateLoss(cfg)
                  }
    inference(nvs, dataset, loss_funcs, args, cfg, visualize=cfg.debug)
    return

def collate_dict(dict_list):
    ret = {}
    for key in dict_list[0]:
        if key in ['gt', 'lp', 'calib_left', 'calib_right', 'score']:
            ret[key] = [item[key] for item in dict_list]
        else:
            ret[key] = np.concatenate([d[key] for d in dict_list], axis=0)
    return ret

class BatchCollator(object):
    def __init__(self, cfg):
        super(BatchCollator, self).__init__()
        self.cfg = cfg

    def __call__(self, batch):
        batch = list(zip(*batch))
        meta_data = {}
        left_rois = torch.cat(batch[0], dim=0)
        right_rois = torch.cat(batch[1], dim=0)
        if batch[2][0] is not None:
            targets = torch.cat(batch[2], dim=0)
        else:
            targets = None
        meta_data = collate_dict(batch[3])
        return left_rois, right_rois, targets, meta_data

if __name__ == '__main__':
    main()