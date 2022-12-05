import numpy as np

from snvc.dataset.kitti_dataset import kitti_dataset as kittidataset

def get_kitti_annos(labels,
    # ignore_van_and_personsitting=False,
    # ignore_smaller=True,
    # ignore_occlusion=True,
    ignore_van_and_personsitting=False,
    ignore_smaller=False,
    ignore_occlusion=False,
    ignore_truncation=True,
    valid_classes=[1,2,3,4],
    depth_range=None,
    truncation_threshold=0.98,
    ret_scores=False,
    ret_indices=False
    ):

    assert not ignore_occlusion # 6150 occlusion should be induced

    boxes = []
    box3ds = []
    ori_classes = []
    scores = []
    indices = []
    for i, label in enumerate(labels):
        # 4 will be ignored.
        if label.type == 'Pedestrian' or label.type == 'Person_sitting': typ = 1
        elif label.type == 'Car' or label.type == 'Van': typ = 2
        elif label.type == 'Cyclist': typ = 3
        elif label.type == 'DontCare': typ = 4
        elif label.type in ['Misc', 'Tram', 'Truck']: continue
        else:
            raise ValueError('Invalid Label.')

        # only train Car or Person
        if typ != 4 and typ not in set(valid_classes) - set([4]):
            continue

        if ignore_van_and_personsitting and (label.type == 'Van' or label.type == 'Person_sitting'):
            typ = 4

        if ignore_smaller and label.box2d[3] - label.box2d[1] <= 10.:
            typ = 4

        if ignore_occlusion and label.occlusion >= 3:
            typ = 4

        if ignore_truncation and label.truncation >= truncation_threshold:
            typ = 4

        if typ not in valid_classes:
            continue 
        if depth_range is not None and (label.box3d[2] < depth_range[0] or label.box3d[2] > depth_range[1]):
            # depth < z_min or depth > z_max
            continue
        boxes.append( np.array(label.box2d) )
        box3ds.append( np.array(label.box3d[[3,4,5, 0,1,2, 6]]) ) # size, location, orientation
        ori_classes.append(typ)
        indices.append(i)
        if hasattr(label, 'score'):
            scores.append(label.score)
        # boxes[-1][2:4] = boxes[-1][2:4] - boxes[-1][0:2]

        # if typ == 4:
        #     box3ds[-1] = np.zeros((7,))

    boxes = np.asarray(boxes, dtype=np.float32)
    box3ds = np.asarray(box3ds, dtype=np.float32)
    ori_classes = np.asarray(ori_classes, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    # inds = ori_classes.argsort()
    # boxes = boxes[inds]
    # box3ds = box3ds[inds]
    # ori_classes = ori_classes[inds]
    ret = boxes, box3ds, ori_classes
    if ret_scores:
        ret += (scores,)
    if ret_indices:
        ret += (indices,)
    return ret

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_filter_flag(cfg):
    flag = False
    if hasattr(cfg, 'RPN3D_ENABLE') and cfg.RPN3D_ENABLE:
        flag = True
    if hasattr(cfg, 'REFINE') and cfg.REFINE:
        flag = True
    return flag

def get_img_paths(root, 
                  split_file, 
                  depth_disp=False, 
                  cfg=None, 
                  is_train=False, 
                  generate_target=False
                  ):
    if "test.txt" in split_file:
        kitti_dataset = kittidataset('trainval').val_dataset # just for function re-use
    else:
        kitti_dataset = kittidataset('trainval').train_dataset # just for function re-use
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    if depth_disp:
        disp_L = cfg.depth_folder

    with open(split_file, 'r') as f:
        all_idx = [x.strip() for x in f.readlines()]

    if is_train or generate_target:
        filter_idx = []
        if get_filter_flag(cfg):
            for image_index in all_idx:
                labels = kitti_dataset.get_label_objects(int(image_index))
                # filter labels within get_kitti_annos
                boxes, box3ds, ori_classes = get_kitti_annos(labels,
                                                             valid_classes=cfg.valid_classes
                                                             )
                if len(box3ds) > 0:
                    filter_idx.append(image_index)
            all_idx = filter_idx

    left_paths = [root + '/' + left_fold + img + '.png' for img in all_idx]
    right_paths = [root + '/' + right_fold + img + '.png' for img in all_idx]
    disp_L_paths = [root + '/' + disp_L + img + '.npy' for img in all_idx]
    return left_paths, right_paths, disp_L_paths