#!/usr/bin/env python3

import numpy as np
import os
from string import Template
import cv2
import argparse
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description='Convert KITTI calibration to basalt and save it int the dataset folder as basalt_calib.json.')
parser.add_argument('-d', '--dataset-path', required=True, help="Path to the dataset in KITTI format")
args = parser.parse_args()

dataset_path = args.dataset_path
dataset_pathUp = os.path.abspath(os.path.dirname(dataset_path))   #the same day calib store

kitti_calib_c2c = dataset_pathUp + '/calib_cam_to_cam.txt'
kitti_calib_i2v = dataset_pathUp + '/calib_imu_to_velo.txt'
kitti_calib_v2c = dataset_pathUp + '/calib_velo_to_cam.txt'
values = {}
T_imu2velo = []
T_velo2cam = []



calib_template = Template('''{
    "value0": {
        "T_imu_cam": [
            {
                "px": $px,
                "py": $py,
                "pz": $pz,
                "qx": $qx,
                "qy": $qy,
                "qz": $qz,
                "qw": $qw
            },
            {
                "px": $px,
                "py": $py1,
                "pz": $pz,
                "qx": $qx,
                "qy": $qy,
                "qz": $qz,
                "qw": $qw
            }
        ],
        "intrinsics": [
            {
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": $fx0,
                    "fy": $fy0,
                    "cx": $cx0,
                    "cy": $cy0
                }
            },
            {
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": $fx1,
                    "fy": $fy1,
                    "cx": $cx1,
                    "cy": $cy1
                }
            }
        ],
        "resolution": [
            [
                $rx,
                $ry
            ],
            [
                $rx,
                $ry
            ]
        ],
        "vignette": [],
        "calib_accel_bias": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "calib_gyro_bias": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "imu_update_rate": 100,
        "accel_noise_std": [0.2, 0.2, 0.2],
        "gyro_noise_std": [0.0005, 0.0005, 0.0005],
        "accel_bias_std": [0.01, 0.01, 0.01],
        "gyro_bias_std": [0.00001, 0.00001, 0.00001],

        "cam_time_offset_ns": 0
    }
}
''')


with open(kitti_calib_c2c, 'r') as stream:
    lines = (' '.join([x.strip('\n ') for x in stream.readlines() if x.strip('\n ') ])).split(' ')

    if len(lines) != 241:
        print('Issues loading c2c calibration')
        print(lines)
    
    P0 = np.array([float(x) for x in lines[52:64]]).reshape(3,4)
    P1 = np.array([float(x) for x in lines[111:123]]).reshape(3,4)
    print('P0\n', P0)
    print('P1\n', P1)

    baseline = -P1[0,3]/P1[0,0]

    img = cv2.imread(dataset_path + '/image_00/data/0000000000.png')
    rx = img.shape[1]
    ry = img.shape[0]

    values = {'fx0': P0[0,0], 'fy0': P0[1,1], 'cx0': P0[0,2], 'cy0': P0[1,2], 'fx1': P1[0,0], 'fy1': P1[1,1], 'cx1': P1[0,2], 'cy1': P1[1,2], 'baseline': baseline, 'rx': rx, 'ry': ry}


with open(kitti_calib_i2v, 'r') as stream:
    lines = (' '.join([x.strip('\n ') for x in stream.readlines() if x.strip('\n ') ])).split(' ')

    if len(lines) != 17:
        print('Issues loading i2v calibration')
        print(lines)

    R = np.array([float(x) for x in lines[4:13]]).reshape(3,3)
    T = np.array([float(x) for x in lines[14:17]]).reshape(3,1)
    T_imu2velo = np.vstack((np.hstack((R,T)),[0,0,0,1]))


with open(kitti_calib_v2c, 'r') as stream:
    lines = (' '.join([x.strip('\n ') for x in stream.readlines() if x.strip('\n ') ])).split(' ')

    if len(lines) != 23:
        print('Issues loading v2c calibration')
        print(lines)

    R = np.array([float(x) for x in lines[4:13]]).reshape(3,3)
    T = np.array([float(x) for x in lines[14:17]]).reshape(3,1)
    T_velo2cam = np.vstack((np.hstack((R,T)),[0,0,0,1]))


#to fact(display) imu axis
T_imu_k2e2fact = np.array([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
T_0 = np.dot(np.linalg.inv(T_imu2velo),np.linalg.inv(T_velo2cam))
T_0 = np.dot(T_imu_k2e2fact,T_0)
p0 = T_0[:3,3]
q0 = (Rotation.from_matrix(T_0[:3,:3])).as_quat()
valuesT = {'px':p0[0],'py':p0[1],'pz':p0[2],
            'qx':q0[0],'qy':q0[1],'qz':q0[2],'qw':q0[3],'py1':p0[1]+values['baseline']}
values.update(valuesT)

calib = calib_template.substitute(values)
print(calib)


with open(dataset_pathUp + '/basalt_calib.json', 'w') as stream2:
    stream2.write(calib)