import taichi as ti
import numpy as np
import math

from global_variabel import *
from script.helper_function import *

## declare field
mesh_position = None
mesh_rotation = None


## initialize bodies
def init_collision_bodies():
    # init rotation
    for b_idx in range(num_collision_bodies):
        mesh_rotation[b_idx] = ti.Vector([0., math.pi - 0.5*b_idx, 0.])
    # create mesh particles
    idx = 0
    for b_idx in range(num_collision_bodies):
        mesh = meshio.read(mesh_names[b_idx])
        mesh_points = mesh.points
        R = rotation_mat(mesh_rotation[b_idx][0],mesh_rotation[b_idx][1],mesh_rotation[b_idx][2])
        print_mesh_info(b_idx, mesh_points.shape[0], R)
        for i in range(mesh_points.shape[0]):
            mesh_position[idx] = ti.Vector(R @ np.array(mesh_points[i]))
            mesh_position[idx] *= 10
            mesh_position[idx] += ti.math.vec3([15., 1., 30.]) + b_idx*ti.math.vec3([0., 0., 15.])
            idx += 1

def init_collision_bodies_lighthouse_scene():
    # init rotation
    for b_idx in range(num_collision_bodies):
         mesh_rotation[b_idx] = ti.Vector([0., (math.pi/4)*b_idx, 0.])
    # create mesh particles
    idx = 0
    for b_idx in range(num_collision_bodies):
        mesh = meshio.read(mesh_names[b_idx])
        mesh_points = mesh.points
        R = rotation_mat(mesh_rotation[b_idx][0],mesh_rotation[b_idx][1],mesh_rotation[b_idx][2])
        print_mesh_info(b_idx, mesh_points.shape[0], R)
        for i in range(mesh_points.shape[0]):
            mesh_position[idx] = ti.Vector(R @ np.array(mesh_points[i]))
            mesh_position[idx] *= 10* (1-b_idx*0.65)
            mesh_position[idx] += ti.math.vec3([20., 1., 14.]) + b_idx*ti.math.vec3([-6., 0., 15.])
            idx += 1            

def init_collision_bodies_bathroom_scene():
    # init rotation
    for b_idx in range(num_collision_bodies):
         mesh_rotation[b_idx] = ti.Vector([0., math.pi/2.- (0.3+math.pi)*b_idx, 0.])
    # create mesh particles
    idx = 0
    for b_idx in range(num_collision_bodies):
        mesh = meshio.read(mesh_names[b_idx])
        mesh_points = mesh.points
        R = rotation_mat(mesh_rotation[b_idx][0],mesh_rotation[b_idx][1],mesh_rotation[b_idx][2])
        print_mesh_info(b_idx, mesh_points.shape[0], R)
        for i in range(mesh_points.shape[0]):
            mesh_position[idx] = ti.Vector(R @ np.array(mesh_points[i]))
            mesh_position[idx] *= 8 * (1-b_idx/2*0.3)
            mesh_position[idx] += ti.math.vec3([22., 0., 40.]) + b_idx*ti.math.vec3([-7., 0., -20./(b_idx+1)])
            idx += 1                        

## print info
def print_mesh_info(b_idx, num, R):
    print("Insert Mesh vtk-file: "+mesh_names[b_idx]+"\n"
            +" number of particles: "+str(num)+"\n"
            +" rotation angels:     "+str(mesh_rotation[b_idx])+"\n"
            +" rotation matrix:     ["+str(R[0])+"\n"
            +"                       "+str(R[1])+"\n"
            +"                       "+str(R[2])+"]")
    print('-'*term_size.columns)