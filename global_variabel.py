import math
import os
import meshio
import taichi as ti

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

## screen params
screen_res = (1566, 1500)
screen_to_world_ratio = 10.0
bound = (400, 300, 400)
boundary = (bound[0] / screen_to_world_ratio,
            bound[1] / screen_to_world_ratio,
            bound[2] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size

# terminal params
term_size = os.get_terminal_size()
# window params
bool_record = False
bool_pause = False

## simulation params
# dimensions
dim = 3
# tolerance
tol = 1e-6
# num of particles
num_particles_x = 20
num_particles_y = 30
num_particles_z = 30
num_particles = num_particles_x * num_particles_y * num_particles_z
num_fluid_particles = num_particles
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
# particle radius
particle_radius = 0.2
particle_radius_in_world = particle_radius / screen_to_world_ratio
# grid size
grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

## particle numbers vector
particle_numbers = [num_fluid_particles]

## spheres params
num_collision_spheres = 0
collision_sphere_radius = 4.
collision_contact_offset = 0.9*particle_radius
collision_velocity_damping = 0.001

## box params
num_collision_boxes = 0
num_lines_per_box = 12

## body params
num_collision_bodies = 2
num_bodies_particles = 0
mesh_names = ["meshes/bunny_dense.vtk","meshes/bunny_dense.vtk"]
mesh_sizes = []
# mesh input
mesh_names = ["meshes/bunny_dense.vtk","meshes/bunny_dense.vtk"]
mesh_sizes = []
for i in range(num_collision_bodies):
    mesh_obj = meshio.read(mesh_names[i])
    mesh_points = mesh_obj.points
    mesh_sizes.append(mesh_points.shape[0])
    particle_numbers.append(mesh_points.shape[0])
    num_bodies_particles += mesh_points.shape[0]
num_particles += num_bodies_particles

## object bools
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

## PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

## vector fields
old_positions = None
positions = None
velocities = None
#mass = None