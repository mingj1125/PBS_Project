import taichi as ti
import numpy as np
from math import *

from global_variabel import *

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_, h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1] and 0 <= c[2] and c[2] < grid_size[2]

@ti.func
def confine_position_to_boundary_no_ti(p, boundary):
    bmin = particle_radius
    bmax = boundary - particle_radius
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

def rotation_mat(a, b, c):
    R_x = np.array([[1., 0., 0.], [0., cos(a), -sin(a)], [0., sin(a), cos(a)]])
    R_y = np.array([[cos(b), 0., sin(b)], [0., 1., 0.], [-sin(b), 0., cos(b)]])
    R_z = np.array([[cos(c), -sin(c), 0.], [sin(c), cos(c), 0.], [0., 0., 1.]])

    R = R_z @ R_y @ R_x
    return R

@ti.func
def rotation_mat_ti(a, b, c):
    cos_a = ti.cos(a)
    sin_a = ti.sin(a)
    cos_b = ti.cos(b)
    sin_b = ti.sin(b)
    cos_c = ti.cos(c)
    sin_c = ti.sin(c)

    R_x = ti.Vector([1., 0., 0., 0., cos_a, -sin_a, 0., sin_a, cos_a])
    R_y = ti.Vector([cos_b, 0., sin_b, 0., 1., 0., -sin_b, 0., cos_b])
    R_z = ti.Vector([cos_c, -sin_c, 0., sin_c, cos_c, 0., 0., 0., 1.])
    
    R_xy = add_rotation(R_x,R_y)
    R = add_rotation(R_xy,R_z)
    return R

@ti.func
def rotate_vertex(R,x):
    x_coord = R[0]*x[0] + R[1]*x[1] + R[2]*x[2]
    y_coord = R[3]*x[0] + R[4]*x[1] + R[5]*x[2]
    z_coord = R[6]*x[0] + R[7]*x[1] + R[8]*x[2]
    return ti.Vector([x_coord, y_coord, z_coord])

@ti.func
def rotate_vertex_inv(R,x):
    x_coord = R[0]*x[0] + R[3]*x[1] + R[6]*x[2]
    y_coord = R[1]*x[0] + R[4]*x[1] + R[7]*x[2]
    z_coord = R[2]*x[0] + R[5]*x[1] + R[8]*x[2]
    return ti.Vector([x_coord, y_coord, z_coord])

@ti.func
def add_rotation(R1,R2):
    R3 = ti.Vector([0.,0.,0.,0.,0.,0.,0.,0.,0.])
    for i in range(dim):
        a = ti.Vector([R2[i+0],R2[i+3],R2[i+6]])
        b = rotate_vertex(R1,a)
        R3[i+0] = b[0] 
        R3[i+3] = b[1] 
        R3[i+6] = b[2]
    return R3 

@ti.func
def dotP(x,y):
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]

def set_params_body_bools():
    bool_box = True
    if  num_collision_boxes == 0:
        bool_box = False
        num_collision_boxes = 1

    bool_sphere = True
    if  num_collision_spheres == 0:
        bool_sphere = False
        num_collision_spheres = 1

    bool_mesh = True
    if  num_collision_bodies == 0:
        bool_mesh = False
        num_collision_bodies = 1
        num_bodies_particles = 1

@ti.func
def get_particle_phase(p_idx):
    # solid
    ph_particle = 1
    if p_idx < particle_numbers[0]:
        # fluid
        ph_particle = 0
    return ph_particle

@ti.func
def get_particle_info(p_idx):
    particle_phase = get_particle_phase(p_idx)
    particle_id = get_particle_obj
    return particle_phase, particle_id

def set_camera_position(camera, camera_position, camera_lookat):
    camera.position(camera_lookat[0]+cos(camera_position[0])*cos(camera_position[1])*camera_position[2],
                    camera_lookat[1]+                        sin(camera_position[1])*camera_position[2],
                    camera_lookat[2]+sin(camera_position[0])*cos(camera_position[1])*camera_position[2])
    camera.lookat(camera_lookat[0],camera_lookat[1],camera_lookat[2])