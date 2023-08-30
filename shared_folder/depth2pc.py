from PIL import Image, ImageDraw
import requests
import cv2
import os
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms
import sys
from numpy import asarray
import imageio.v3 as iio
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
sys.path.insert(0, '/app/vlmaps/')
from tqdm import tqdm
from utils.clip_mapping_utils import load_semantic, load_obj2cls_dict, save_map, cvt_obj_id_2_cls_id, depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point
from utils.clip_mapping_utils import load_map, get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats

depth_sample_rate = 100
camera_height = 1.2
gs = 4000
cs = 0.05

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def load_pose(pose_filepath):
    with open(pose_filepath, "r") as f:
        line = f.readline()
        row = [float(x) for x in line.split()]
        pos = np.array(row[:3], dtype=float).reshape((3, 1))
        quat = row[3:]
        #print("quat1:",quat)
        quat2 = quat
        quat[1] = quat2[2]
        quat[2] = 0
        rot_eul = euler_from_quaternion(quat[0],quat[1],quat[2],quat[3])
        print("Euler angle:",rot_eul[1])
        print("quat2:",quat)
        r = R.from_quat(quat)
        rot = r.as_matrix()

        return pos, rot, rot_eul

#img_save_dir = "./5LpN3gDmAk7_1/"
img_save_dir = "../image/data_tsvet_test/"

rgb_dir = os.path.join(img_save_dir, "rgb")
depth_dir = os.path.join(img_save_dir, "depth")
pose_dir = os.path.join(img_save_dir, "pose")

rgb_list = sorted(os.listdir(rgb_dir))
depth_list = sorted(os.listdir(depth_dir))
pose_list = sorted(os.listdir(pose_dir))
pose_list = sorted(os.listdir(pose_dir))

#rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
#    x.split("_")[-1].split(".")[0]))
#depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
#    x.split("_")[-1].split(".")[0]))
#pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(
#    x.split("_")[-1].split(".")[0]))
#pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(
#    x.split("_")[-1].split(".")[0]))

rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
depth_list = [os.path.join(depth_dir, x) for x in depth_list]
pose_list = [os.path.join(pose_dir, x) for x in pose_list]

map_save_dir = os.path.join(img_save_dir, "map")
obstacles_save_path = os.path.join(map_save_dir, "obstacles3.npy")
obstacles = np.ones((gs, gs), dtype=np.uint8)

#bgr = cv2.imread("./5LpN3gDmAk7_1/rgb/5LpN3gDmAk7_139.png")
#bgr = cv2.imread("../image/dataset/rgb/frame000068.png")
save_map(obstacles_save_path, obstacles)

count = 0
tf_list = []
#data_iter = zip(rgb_list, depth_list, pose_list)
data_iter = zip( depth_list, pose_list)
pbar = tqdm(total=len(depth_list))
xpos = []
ypos = []

ang = []
for data_sample in data_iter:
    count = count +1
    if count < 0:
        continue
    if count>1500:
          break
    
    #rgb_path, depth_path, pose_path = data_sample
    depth_path, pose_path = data_sample
    print(pose_path)
    #bgr = cv2.imread(rgb_path)

    #rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # read pose
    #pos, rot = load_pose("./5LpN3gDmAk7_1/pose/5LpN3gDmAk7_139.txt")  # z backward, y upward, x to the right
    #pos, rot = load_pose("../image/dataset/pose/frame000067.txt")  # z backward, y upward, x to the right
    pos, rot, rot_eul = load_pose(pose_path)  # z backward, y upward, x to the right
    pos2 = pos

    xpos = [xpos, pos[0]]
    ypos = [ypos, pos[1]]
    ang = [ang, rot_eul[1]]

    #putting the frames in same direction as the code wants
    pos[0] = pos2[1]
    pos[1] = pos2[2]
    pos[2] = pos2[0]
    
    print("rot:",rot)

    rot_ro_cam = np.eye(3)
    rot_ro_cam[1, 1] = -1
    rot_ro_cam[2, 2] = -1
    rot = rot @ rot_ro_cam
    pos[1] += camera_height

    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos.reshape(-1)
    
    tf_list.append(pose)
    if len(tf_list) == 1:
        init_tf_inv = np.linalg.inv(tf_list[0])

    tf = init_tf_inv @ pose
    print("tf:",tf)
    def load_depth(depth_filepath):
        with open(depth_filepath, 'rb') as f:
            depth = np.load(f)
        return depth

    depth = load_depth(depth_path)
    #depth = load_depth("./5LpN3gDmAk7_1/depth/5LpN3gDmAk7_139.npy")
    x_indices, y_indices = np.where(depth >= 2.5)

    filled = pd.DataFrame(depth).fillna(0)
    depth = filled.to_numpy()

    print("max:",depth.max())
    print("min:",depth.min())

    depth[x_indices,y_indices] = 0

    print("max:",depth.max())
    print("min:",depth.min())

    print(depth)
    # transform all points to the global frame
    pc, mask = depth2pc(depth)
    shuffle_mask = np.arange(pc.shape[1])
    np.random.shuffle(shuffle_mask)
    shuffle_mask = shuffle_mask[::depth_sample_rate]
    mask = mask[shuffle_mask]
    pc = pc[:, shuffle_mask]
    pc = pc[:, mask]
    pc_global = transform_pc(pc, tf)

    #rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])

    #print("Pc_global:",pc_global)

    #feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

    #Plot PointCloud
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')

                 
    # Data for a three-dimensional line
    #x = pc_global[0,:]
    #y = pc_global[1,:]
    #z = pc_global[2,:]
    #xpc = xpc + x
    #ax.scatter(x, y, z, c=y, cmap='viridis', linewidth=0.1)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')

    #fig.show()

    #input("Press the Enter key to continue: ")

    
    for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
        x, y = pos2grid_id(gs, cs, p[0], p[2])
        #print("x:",x)
        #print("y:",y)
        # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
        if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
            x < 0 or y < 0 or p_local[1] < -0.5:
            continue
        #print(p_local[2])    
        # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
        if p_local[1] > camera_height:
            continue
        obstacles[y, x] = 0
    pbar.update(1)
    #x_indices, y_indices = np.where(obstacles == 0)
    #xmin = np.min(x_indices)
    #xmax = np.max(x_indices)
    #ymin = np.min(y_indices)
    #ymax = np.max(y_indices)
    #print(np.unique(obstacles))
    #obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
    #plt.figure(figsize=(8, 6), dpi=120)
    #plt.imshow(obstacles_pil, cmap='gray')
    #plt.show()

print("went out")
save_map(obstacles_save_path, obstacles)
obstacles = load_map(obstacles_save_path)
x_indices, y_indices = np.where(obstacles == 0)
xmin = np.min(x_indices)
xmax = np.max(x_indices)
ymin = np.min(y_indices)
ymax = np.max(y_indices)
print(np.unique(obstacles))
obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
plt.figure(figsize=(8, 6), dpi=120)
plt.imshow(obstacles_pil, cmap='gray')
plt.show()

input("Press the Enter key to continue: ")
