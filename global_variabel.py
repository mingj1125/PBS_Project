import math
import os
import meshio
import taichi as ti

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

## screen params
screen_res = (1566, 1500)
screen_to_world_ratio = 10.0
# boundary
bound = (500, 300, 400)
boundary = (bound[0] / screen_to_world_ratio,
            bound[1] / screen_to_world_ratio,
            bound[2] / screen_to_world_ratio)
# cell
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
num_particles_x = 10
num_particles_y = 35
num_particles_z = 50
# bathroom scene
# num_particles_x = 30
# num_particles_y = 15
# num_particles_z = 30
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

## spheres params
num_collision_spheres = 0
collision_sphere_radius = 4.
collision_contact_offset = 0.9*particle_radius
collision_velocity_damping = 0.001

## box params
num_collision_boxes = 0
num_lines_per_box = 12

## mesh params
# static
num_static_meshes = 0
num_static_mesh_particles = 0
# dynamic
num_dynamic_meshes = 0
num_dynamic_mesh_particles = 0

## particle numbers vector
particle_numbers = [num_fluid_particles]

mesh_input = ["NULL", "meshes/bunny_dense.vtk"]
num_mesh_particles = 0

## body params
num_collision_bodies = 2
num_bodies_particles = 0
mesh_sizes = []
# mesh input for bunnies
# mesh_names = ["meshes/bunny_dense.vtk","meshes/bunny_dense.vtk"]
# mesh input for bathroom scene
mesh_names = ["meshes/bathtub_volume.vtk","meshes/bunny.vtk","meshes/bunny.vtk"]
# mesh input for lighthouse scene
# mesh_names = ["meshes/lighthouse.vtk","meshes/sea_arch.vtk"]
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
# if  num_static_meshes == 0 and num_dynamic_meshes == 0:
if  num_collision_bodies == 0:
    bool_mesh = False
    num_static_meshes = 1
    num_dynamic_meshes = 1

## PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
stablization_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h_ * 1.05
# solid density scaling eq.27 unified particle system
s = 0.5

# dynamic balls params
num_collision_balls = 4
collision_ball_radius = 1.5
stablization_iters = 8

max_num_balls_per_cell = 50
max_num_virtual_neighbors = 50
virtual_particle_neighbors_dist = (particle_radius*4*h_)
neighborhood_particle_off = math.floor(neighbor_radius/virtual_particle_neighbors_dist)

bool_ball = True
if  num_collision_balls == 0:
    bool_ball = False
    num_collision_balls = 1

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

## vector fields
old_positions = None
positions = None
velocities = None
#mass = None
