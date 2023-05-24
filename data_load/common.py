from typing import overload
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import datetime

import trimesh
class Mesh:
    @overload
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, vertices_normalized: bool) -> None:
        ...

    @overload
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, vertices_normalized: bool) -> None:
        ...

    @overload
    def __init__(self, mesh: str, vertices_normalized: bool) -> None:
        ...

    @overload
    def __init__(self, mesh: trimesh.Trimesh, vertices_normalized: bool) -> None:
        ...

    def __init__(self, *args):
        if isinstance(args[0], str):
            try:
                self.mesh: trimesh.Trimesh = trimesh.load(args[0])
            except:
                raise Exception('Can not read file: '+args[0])
        elif isinstance(args[0], trimesh.Trimesh):
            self.mesh: trimesh.Trimesh = args[0]
        elif isinstance(args[0], np.ndarray):
            try:
                self.mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=args[0], faces=args[1])
            except:
                raise Exception('Input parameter error')
        elif isinstance(args[0], torch.Tensor):
            try:
                self.mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=args[0].numpy(), faces=args[1].numpy())
            except:
                raise Exception('Input parameter error')
        else:
            raise Exception('Input parameters are not satisfied')
        # self.vertices = torch.Tensor(self.mesh.vertices)  # N,3
        if (len(args) == 2 and isinstance(args[1], bool)):
            if not args[1]:
                self.normalize = False
                self.vertices = torch.Tensor(self.vertices_centered())
            else:
                self.normalize = True
                self.vertices = torch.Tensor(self.vertices_normalized())  # N,3
        elif (len(args) == 3 and isinstance(args[2], bool)):
            if not args[2]:
                self.normalize = False
                self.vertices = torch.Tensor(self.vertices_centered())
            else:
                self.normalize = True
                self.vertices = torch.Tensor(self.vertices_normalized())  # N,3
        else:
            self.normalize = True
            self.vertices = torch.Tensor(self.vertices_normalized())  # N,3
        self.normals = torch.Tensor(np.copy(self.mesh.vertex_normals))
        self.face_index = torch.LongTensor(self.mesh.faces)                           # F,3
        self.edge_index = torch.LongTensor(self.get_tpl_edges_trimesh()).t()          # 2,E

    def shuffle_points(self, random_sample=None):
        if random_sample is None:
            random_sample = np.random.choice(self.vertices.shape[0], size=self.vertices.shape[0], replace=False)
        new_vertices = self.vertices[random_sample]
        new_normals = self.normals[random_sample]
        face_dict = {}
        for tar_idx, src_idx in enumerate(random_sample):
            face_dict[src_idx] = tar_idx
        new_face = []
        for i in range(self.face_index.shape[0]):
            new_face.append([face_dict[int(self.face_index[i][0])],
                             face_dict[int(self.face_index[i][1])],
                             face_dict[int(self.face_index[i][2])]])
        new_face = torch.LongTensor(new_face)
        self.vertices = new_vertices
        self.normals = new_normals
        self.face_index = new_face
        self.mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=self.vertices.numpy(), faces=self.face_index.numpy())
        self.edge_index = torch.LongTensor(self.get_tpl_edges_trimesh()).t()

    def get_tpl_edges_trimesh(self):
        vertex_neighbors = self.mesh.vertex_neighbors
        tpl_edges = []
        for i, v_n in enumerate(vertex_neighbors):
            v_n = np.expand_dims(np.array(vertex_neighbors[i]), axis=1)
            v_n = np.concatenate((i * np.ones_like(v_n), v_n), axis=1)
            tpl_edges.append(v_n)
        tpl_edges = np.concatenate(tpl_edges, axis=0)
        return tpl_edges

    def vertices_centered(self):
        bbox_min, bbox_max = self.mesh.bounds
        bbox_center = (bbox_min + bbox_max) / 2
        return (self.mesh.vertices - bbox_center)

    def vertices_normalized(self):
        bbox_min, bbox_max = self.mesh.bounds
        bbox_center = (bbox_min + bbox_max)/2
        scale = np.max(bbox_max-bbox_min) / 2
        return (self.mesh.vertices - bbox_center) / scale
        # return (self.mesh.vertices - bbox_min) / scale

    def add_noise(self, noise=0.05):
        noise = np.random.uniform(-noise, noise, self.mesh.vertices.shape)
        self.mesh.vertices = self.mesh.vertices + noise
        if self.normalize:
            self.vertices = torch.Tensor(self.vertices_centered())
        else:
            self.vertices = torch.Tensor(self.vertices_normalized())  # N,3
        self.normals = torch.Tensor(np.copy(self.mesh.vertex_normals))
        self.face_index = torch.LongTensor(self.mesh.faces)                           # F,3
        self.edge_index = torch.LongTensor(self.get_tpl_edges_trimesh()).t()          # 2,E

    def filp_yz_axis(self):
        vertices = self.mesh.vertices
        vertices[:, 1] = self.mesh.vertices[:, 2]
        vertices[:, 2] = self.mesh.vertices[:, 1]
        self.mesh.vertices = vertices
        if self.normalize:
            self.vertices = torch.Tensor(self.vertices_centered())
        else:
            self.vertices = torch.Tensor(self.vertices_normalized())  # N,3
        self.normals = torch.Tensor(np.copy(self.mesh.vertex_normals))
        self.face_index = torch.LongTensor(self.mesh.faces)                           # F,3
        self.edge_index = torch.LongTensor(self.get_tpl_edges_trimesh()).t()          # 2,E

    def view(self, save_obj=False, save_path=None):
        if save_obj:
            if save_path is None:
                save_path = os.path.join('../saved_obj',
                                         'Mesh' + datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S') + '.obj')
            self.mesh.export(save_path)
            print('The obj has been saved: ' + save_path)
        self.mesh.show()

    def save_obj(self, save_path):
        self.mesh.export(save_path)
        print('The obj has been saved: ' + save_path)
        return True

    def view_snapshot(self, resolution=(480, 640), save_img=False, save_path=None):
        scene = trimesh.Scene()
        scene.add_geometry(self.mesh)
        r_e = trimesh.transformations.euler_matrix(0, 0, 0, "ryxz",)
        t_r = scene.camera.look_at(self.mesh.bounds, rotation=r_e)
        scene.camera_transform = t_r
        png = scene.save_image(resolution=resolution)
        file = io.BytesIO(png)
        img = plt.imread(file)
        plt.figure()
        plt.imshow(img)
        if save_img:
            if save_path is None:
                save_path = os.path.join('../saved_images',
                                         'Mesh' + datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S') + '.pdf')
            plt.savefig(save_path)
            print('The image has been saved: '+save_path)
        plt.show()
        return img
