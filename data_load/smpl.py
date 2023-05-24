import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
import pathlib
from data_load.common import Mesh
import random
from natsort import natsorted


class SMPLDataset(Dataset):
    def __init__(self,
                 path='../../datasets/smpl-data/train',
                 identity_num=30,
                 pose_num=800,
                 identity_range=(0, 15),
                 pose_range=(0, 399),
                 type='obj',
                 shuffle_points=False,
                 filp_yz_axis=False,
                 vertices_normalized=True):
        super(SMPLDataset, self).__init__()
        self.data_dir = pathlib.Path(path)
        self.identity_num = identity_num
        self.pose_num = pose_num
        self.identity_range = identity_range
        self.pose_range = pose_range
        self.shuffle_points = shuffle_points
        self.filp_yz_axis = filp_yz_axis
        self.vertices_normalized = vertices_normalized
        f_list = [str(f) for f in list(self.data_dir.glob('*.' + type))]
        f_list = natsorted(f_list)
        if len(f_list) == 0:
            raise Exception('The destination folder is empty or does not exist: ' + os.path.abspath(path))
        self.file_list = []
        for i in range(identity_num):
            temp = []
            for j in range(pose_num):
                temp.append(f_list[i * pose_num + j])
            self.file_list.append(temp)
        self.file_list_enabled = self.file_list[identity_range[0]:identity_range[1]+1]
        for i in range(identity_range[1]-identity_range[0]+1):
            self.file_list_enabled[i] = self.file_list_enabled[i][pose_range[0]:pose_range[1]+1]

    def __len__(self):
        return (self.identity_range[1] - self.identity_range[0] + 1) * (self.pose_range[1] - self.pose_range[0] + 1)

    def __getitem__(self, item):
        id_max = self.identity_range[1] - self.identity_range[0]
        po_max = self.pose_range[1] - self.pose_range[0]
        id_id = random.randint(0, id_max)
        id_po = random.randint(0, po_max)
        po_id = random.randint(0, id_max)
        po_po = random.randint(0, po_max)
        po_id = po_id if po_id != id_id else random.randint(0, id_max)
        po_po = po_po if po_po != id_po else random.randint(0, po_max)
        gt_id = id_id
        gt_po = po_po
        id = Mesh(self.file_list[id_id][id_po], self.vertices_normalized)
        po = Mesh(self.file_list[po_id][po_po], self.vertices_normalized)
        gt = Mesh(self.file_list[gt_id][gt_po], self.vertices_normalized)
        if self.filp_yz_axis:
            id.filp_yz_axis()
            po.filp_yz_axis()
            gt.filp_yz_axis()
        if self.shuffle_points:
            random_sample_id = np.random.choice(id.vertices.shape[0], size=id.vertices.shape[0],
                                                replace=False)
            random_sample_po = np.random.choice(po.vertices.shape[0], size=po.vertices.shape[0],
                                                replace=False)
            id.shuffle_points(random_sample_id)
            po.shuffle_points(random_sample_po)
            gt.shuffle_points(random_sample_id)
        # return id.vertices, po.vertices, gt.vertices
        return Data(v=id.vertices,
                    n=id.normals,
                    face_index=id.face_index.t(),
                    edge_index=id.edge_index,
                    num_nodes=id.vertices.shape[0]), \
               Data(v=po.vertices,
                    n=po.normals,
                    face_index=po.face_index.t(),
                    edge_index=po.edge_index,
                    num_nodes=po.vertices.shape[0]), \
               Data(v=gt.vertices,
                    n=gt.normals,
                    face_index=gt.face_index.t(),
                    edge_index=gt.edge_index,
                    num_nodes=gt.vertices.shape[0])