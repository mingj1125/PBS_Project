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

def rotation_mat(a, b, c):
    R_x = np.array([[1., 0., 0.], [0., cos(a), -sin(a)], [0., sin(a), cos(a)]])
    R_y = np.array([[cos(b), 0., sin(b)], [0., 1., 0.], [-sin(b), 0., cos(b)]])
    R_z = np.array([[cos(c), -sin(c), 0.], [sin(c), cos(c), 0.], [0., 0., 1.]])

    R = R_z @ R_y @ R_x
    return R

@ti.func
def get_particle_phase(p_idx):
    # phase
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