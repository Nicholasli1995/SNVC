"""
A self-contained python script that visualizes a set of cuboids considering
vertex visibility.
 
It can be used to genereate qualitative results for the predictions of a 
3D object detection model. The KITTI annotation style is used for example, 
but you can easily adapt it to other annotation styles.
"""

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import csv 
import cv2
import torch
import torchvision

# annotation style of KITTI dataset
FIELDNAMES = ['type', 
              'truncated', 
              'occluded', 
              'alpha', 
              'xmin', 
              'ymin', 
              'xmax', 
              'ymax', 
              'dh', 
              'dw',
              'dl', 
              'lx', 
              'ly', 
              'lz', 
              'ry',
              'score'
              ]

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}

def parse_args():
    parser = argparse.ArgumentParser(description='a general parser')
    # path to the configuration file
    parser.add_argument('--pred_dir',
                        help='path to the predicted file',
                        type=str
                        )
    parser.add_argument('--data_dir',
                        default="../data/kitti",
                        type=str
                        )    
    parser.add_argument('--split',
                        default="training",
                        type=str
                        )
    parser.add_argument('--save_dir',
                        default="../data/kitti/visualization",
                        type=str
                        )      
    parser.add_argument('--num_show',
                        default=1,
                        type=int
                        )    
    args, unknown = parser.parse_known_args()
    return args

def csv_read_annot(file_path, fieldnames):
    """
    Read instance attributes in the KITTI format. Instances not in the 
    selected class will be ignored. 
    
    A list of python dictionary is returned where each dictionary 
    represents one instsance.
    """        
    annotations = []
    with open(file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            annot_dict = {
                "class": row["type"],
                "label": TYPE_ID_CONVERSION[row["type"]],
                "truncation": float(row["truncated"]),
                "occlusion": float(row["occluded"]),
                "alpha": float(row["alpha"]),
                "dimensions": [float(row['dl']), 
                               float(row['dh']), 
                               float(row['dw'])
                               ],
                "locations": [float(row['lx']), 
                              float(row['ly']), 
                              float(row['lz'])
                              ],
                "rot_y": float(row["ry"]),
                "bbox": [float(row["xmin"]),
                         float(row["ymin"]),
                         float(row["xmax"]),
                         float(row["ymax"])
                         ]
            }
            if "score" in fieldnames:
                annot_dict["score"] = float(row["score"])
            annotations.append(annot_dict)        
    return annotations

def csv_read_calib(file_path):
    """
    Read camera projection matrix in the KITTI format.
    """  
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P = row[1:]
                P = [float(i) for i in P]
                P = np.array(P, dtype=np.float32).reshape(3, 4)
                break        
    return P
    
def load_annotations(label_path, calib_path, fieldnames=FIELDNAMES): 
    """
    Read 3D annotation and camera parameters.
    """          
    annotations = csv_read_annot(label_path, fieldnames)
    # get camera intrinsic matrix K
    P = csv_read_calib(calib_path)
    return annotations, P

def construct_box_3d(l, h, w):
    """
    Construct 3D bounding box corners in the canonical pose.
    """        
    x_corners = [0.5*l, l, l, l, l, 0, 0, 0, 0]
    y_corners = [0.5*h, 0, h, 0, h, 0, h, 0, h]
    z_corners = [0.5*w, w, w, 0, 0, w, w, 0, 0]
    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2
    corners_3d = np.array([x_corners, y_corners, z_corners])     
    return corners_3d
    
def get_cam_cord(shift, ry, dimension, locs):
    """
    Construct 3D bounding box corners in the camera coordinate system.
    """         
    l, h, w = dimension
    corners_3d_fixed = construct_box_3d(l, h, w)
    x, y, z = locs[0], locs[1], locs[2] # bottom center of the labeled 3D box
    rot_maty = np.array([[np.cos(ry), 0, np.sin(ry)],
                        [0, 1, 0],
                        [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = np.matmul(rot_maty, corners_3d_fixed)
    # translation
    corners_3d += np.array([x, y, z]).reshape([3, 1])
    camera_coordinates = corners_3d + shift
    return camera_coordinates

def project_3d_to_2d(points, K):
    """ 
    Get 2D projection of 3D points in the camera coordinate system. 
    """          
    projected = K @ points
    projected[:2, :] /= projected[2, :]
    return projected

def plot_lines(ax, 
               points, 
               connections, 
               dimension, 
               lw=4, 
               c='k', 
               linestyle='-', 
               alpha=1, 
               add_index=False,
               visibility=None
               ):
    """
    Plot 2D/3D lines given points and connection.
    
    connections are of shape [n_lines, 2]
    """
    if add_index:
        for idx in range(len(points)):
            if dimension == 2:
                x, y = points[idx][0], points[idx][1]
                ax.text(x, y, str(idx))
            elif dimension == 3:
                x, y, z = points[idx][0], points[idx][1], points[idx][2]
                ax.text(x, y, z, str(idx))                
    connections = connections.reshape(-1, 2)
    original_lstyle = linestyle
    # original_color = c
    original_alpha = alpha
    for connection in connections:
        x = [points[connection[0]][0], points[connection[1]][0]]
        y = [points[connection[0]][1], points[connection[1]][1]]
        if visibility is not None:
            vis1, vis2 = visibility[connection[0]], visibility[connection[1]]
            linestyle = '--' if ((not vis1) or (not vis2)) else original_lstyle
            # c = 'y' if ((not vis1) or (not vis2)) else original_color
            alpha = 0.5 if ((not vis1) or (not vis2)) else original_alpha
        if dimension == 3:
            z = [points[connection[0]][2], points[connection[1]][2]]
            line, = ax.plot(x, y, z, lw=lw, c=c, linestyle=linestyle, alpha=alpha)
        else:
            line, = ax.plot(x, y, lw=lw, c=c, linestyle=linestyle, alpha=alpha)
    plt.show()
    return line

def plot_3d_bbox(ax, 
                 bbox_3d_projected, 
                 color=None, 
                 linestyle='-', 
                 add_index=False,
                 visibility=None
                 ):
    """
    Draw the projected edges of a 3D cuboid.
    """ 
    c = np.random.rand(3) if color is None else color
    plot_lines(ax, 
               bbox_3d_projected, 
               plot_3d_bbox.connections, 
               dimension=2, 
               c=c, 
               linestyle=linestyle, 
               add_index=add_index,
               visibility=visibility
               )
    return
plot_3d_bbox.connections = np.array([[0, 1],
                                     [0, 2],
                                     [1, 3],
                                     [2, 3],
                                     [4, 5],
                                     [5, 7],
                                     [4, 6],
                                     [6, 7],
                                     [0, 4],
                                     [1, 5],
                                     [2, 6],
                                     [3, 7]])

def ray_intersect_triangle(p0, p1, triangle):
    """
    Tests if a ray starting at point p0, in the direction
    p1 - p0, will intersect with the triangle.
    
    arguments:
    p0, p1: numpy.ndarray, both with shape (3,) for x, y, z.
    triangle: numpy.ndarray, shaped (3,3), with each row
        representing a vertex and three columns for x, y, z.
    
    returns: 
        0.0 if ray does not intersect triangle, 
        1.0 if it will intersect the triangle,
        2.0 if starting point lies in the triangle.
    """
    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)
    if (b == 0.0):
        # ray is parallel to the plane
        if a != 0.0:
            # ray is outside but parallel to the plane
            return 0
        else:
            # ray is parallel and lies in the plane
            rI = 0.0
    else:
        rI = a / b
    if rI < 0.0:
        return 0
    w = p0 + rI * (p1 - p0) - v0
    denom = np.inner(u, v) * np.inner(u, v) - \
        np.inner(u, u) * np.inner(v, v)
    si = (np.inner(u, v) * np.inner(w, v) - \
        np.inner(v, v) * np.inner(w, u)) / denom
    if (si < 0.0) | (si > 1.0):
        return 0
    ti = (np.inner(u, v) * np.inner(w, u) - \
        np.inner(u, u) * np.inner(w, v)) / denom
    if (ti < 0.0) | (si + ti > 1.0):
        return 0
    if (rI == 0.0):
        return 2
    return 1

def get_visibility(box3d, triangles):
    """
    Get visibility for each vertex of a 3D bounding box given all the triangles
    in a scene.
    
    box3d: [8, 3] The vertex coordinates in the camera coordinate system.
    triangles: [N, 3, 3]
    """      
    visibility = np.ones(8, dtype=np.bool)
    p1 = np.zeros(3)
    for idx, p0 in enumerate(box3d):
        intersects = set()
        for triangle in triangles:
            intersects.add(ray_intersect_triangle(p0, p1, triangle))
        if 1 in intersects:
            visibility[idx] = False
    return visibility

def get_color(style='random', z=None, MAXIMUM_DEPTH=45.):
    if style=='random':
        NUM_COLORS = 10
        CMAP = get_cmap(NUM_COLORS)
        fillcolor = CMAP(np.random.randint(0, NUM_COLORS))
    elif style == 'depth':
        assert z is not None
        cmap = cm.get_cmap('autumn')
        rgba = cmap(z/MAXIMUM_DEPTH)
        fillcolor = rgba[:-1]
    return fillcolor

def render(ax, K, cam_cord, visibility):
    fillcolor = get_color(z=cam_cord[0,2])
    # get 2D projections 
    projected = project_3d_to_2d(cam_cord, K)
    scatter_c = np.array([[fillcolor[0], fillcolor[1], fillcolor[2]]])
    ax.scatter(projected[0, 1:], projected[1, 1:], marker='o', c=scatter_c, s=10)
    ax.set_xlim([0, 1150])
    ax.set_ylim([0, 350])
    ax.invert_yaxis()
    plot_3d_bbox(ax, projected[:2, 1:].T, color=fillcolor, visibility=visibility)
    faces = [projected[:2, [2, 1, 3, 4]],
             projected[:2, [8, 7, 5, 6]],
             projected[:2, [6, 5, 1, 2]],
             projected[:2, [4, 3, 7, 8]],
             projected[:2, [1, 5, 7, 3]],
             projected[:2, [8, 6, 2, 4]]]
    
    for idx, face in enumerate(faces):
        if idx == 5:
            ax.fill(face[0,:], face[1,:], c=fillcolor, alpha=0.15, hatch='//')
        else:
            ax.fill(face[0,:], face[1,:], c=fillcolor, alpha=0.15)
    return

def get_triangles(box3d):
    v1s = box3d[get_triangles.connections[:,0]][:, None, :]
    v2s = box3d[get_triangles.connections[:,1]][:, None, :]
    v3s = box3d[get_triangles.connections[:,2]][:, None, :]
    return np.concatenate((v1s, v2s, v3s), axis=1)

get_triangles.connections = np.array([[1,3,4],
                                      [1,2,4],
                                      [5,7,8],
                                      [5,6,8],
                                      [1,5,6],
                                      [1,2,6],
                                      [3,7,8],
                                      [3,4,8],
                                      [1,5,7],
                                      [1,3,7],
                                      [2,6,8],
                                      [2,4,8]])

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def show(image_path, 
         pred_path, 
         calib_path, 
         save_dir=None
         ):
    """
    Show the annotation of an image with visibility considered.
    """      
    image_name = image_path.split(os.path.sep)[-1]
    anns, P = load_annotations(pred_path, calib_path)
    K = P[:, :3]
    shift = np.linalg.inv(K) @ P[:, 3].reshape(3,1)        
    image = imageio.imread(image_path)
    fig = plt.figure(figsize=(11.3, 9))
    ax = plt.subplot(111)
    ax.imshow(image)     
    all_boxes_3d = []
    all_triangles = []
    for i, a in enumerate(anns):
        a = a.copy()
        obj_class = a["label"]
        dimension = a["dimensions"]
        locs = np.array(a["locations"])
        rot_y = np.array(a["rot_y"])
        cam_cord = get_cam_cord(shift, rot_y, dimension, locs)
        all_boxes_3d.append(cam_cord)
        all_triangles.append(get_triangles(cam_cord.T))
    all_triangles = np.concatenate(all_triangles, axis=0)
    for box_3d in all_boxes_3d:
        render(ax, K, box_3d, get_visibility(box_3d.T[1:,:], all_triangles)) 
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        output_path =  os.path.join(save_dir, image_name)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, 
                            bottom = 0, 
                            right = 1, 
                            left = 0, 
                            hspace = 0, 
                            wspace = 0
                            )
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(output_path, dpi=100, bbox_inches = 'tight', pad_inches = 0)
    return fig
    
def visualize(args):       
    name_list = os.listdir(args.pred_dir)
    cnt = 0
    for name in name_list:
        # Example: name = "000002.txt"
        img_path = os.path.join(args.data_dir, args.split, "image_2", name[:-3] + "png")
        pred_path = os.path.join(args.pred_dir, name)
        calib_path = os.path.join(args.data_dir, args.split, "calib", name)
        fig = show(img_path, pred_path, calib_path, args.save_dir)
        cnt += 1
        if cnt > args.num_show:
            break
        plt.close(fig)
    return

def save_batch_image(image_dir,
                     file_name='./batch.png', 
                     ncol=3, 
                     padding=1
                     ):
    '''
    batch_image: [batch_size, channel, height, width]
    }
    '''
    all_images = []
    names = os.listdir(image_dir)
    for name in names:
        img_path = os.path.join(image_dir, name)
        img = imageio.imread(img_path)
        img = img.transpose(2, 0, 1)
        all_images.append(img[None,:,:,:])
    batch_image = torch.from_numpy(np.concatenate(all_images, axis=0))
    grid = torchvision.utils.make_grid(batch_image, ncol, padding)
    ndarr = grid.permute(1, 2, 0).data.numpy()
    ndarr = ndarr.copy()
    cv2.imwrite(file_name, cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR))    
    return

def main():
    args = parse_args()
    visualize(args)
    return 

if __name__ == "__main__":
    main()