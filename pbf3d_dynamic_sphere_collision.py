# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Macklin, M., Müller, M., Chentanez, N., & Kim, T. Y. (2014). Unified particle physics for real-time applications. ACM Transactions on Graphics (TOG), 33(4), 1-12.

import math

import numpy as np

import taichi as ti
from script.helper_function import get_particle_phase

ti.init(arch=ti.cuda)

screen_res = (1500, 1500)
screen_to_world_ratio = 10.0
bound = (400, 300, 200)
boundary = (bound[0] / screen_to_world_ratio,
            bound[1] / screen_to_world_ratio,
            bound[2] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

dim = 3
num_particles_x = 20
num_particles_y = 30
num_particles_z = 20
num_particles = num_particles_x * num_particles_y * num_particles_z
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 0.1
particle_radius_in_world = particle_radius / screen_to_world_ratio
num_collision_balls = 5
collision_ball_radius = 1.
collision_contact_offset = 1.5*particle_radius

# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 8
stablization_iters = 8
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

max_num_balls_per_cell = 50
max_num_virtual_neighbors = 50
virtual_particle_neighbors_dist = (particle_radius*4*h_)
neighborhood_particle_off = math.floor(neighbor_radius/virtual_particle_neighbors_dist)

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
old_collision_ball_positions = ti.Vector.field(dim, float)
collision_ball_positions = ti.Vector.field(dim, float)
collision_ball_velocities = ti.Vector.field(dim, float)
collision_ball_weights = ti.Vector.field(1, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(ti.i32)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
grid_num_balls = ti.field(int)
grid2balls = ti.field(ti.i32)
particle_num_virtual_neighbors = ti.field(int)
virtual_particle_neighbors = ti.Vector.field(dim, float)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
ti.root.dense(ti.i, num_collision_balls).place(old_collision_ball_positions, \
                collision_ball_positions, collision_ball_velocities, collision_ball_weights)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles, grid_num_balls)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
grid_snode.dense(ti.l, max_num_balls_per_cell).place(grid2balls)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
vnb_node = ti.root.dense(ti.i, num_particles)
vnb_node.place(particle_num_virtual_neighbors)
vnb_node.dense(ti.j, max_num_virtual_neighbors).place(virtual_particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.place(board_states)


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
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_,
                                                     h_)
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
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.func
def particle_collide_collision_ball(p):
    for i in range(num_collision_balls):
        sdf_value = (p-collision_ball_positions[i]).norm()- \
                        (collision_ball_radius+particle_radius_in_world+collision_contact_offset)
        factor = (1.+1./collision_ball_weights[i][0])                
        if sdf_value <= 0.:
            sdf_normal = (p-collision_ball_positions[i])/(p-collision_ball_positions[i]).norm()
            p -= sdf_normal *(sdf_value + particle_radius_in_world + collision_contact_offset+ epsilon * ti.random())*1./factor
            collision_ball_positions[i] += sdf_normal * (sdf_value + particle_radius_in_world + collision_contact_offset) \
                                                * 1./collision_ball_weights[i][0]/factor
    return p

@ti.func
def static_mesh_particle_collide_collision_ball(p):
    for i in range(num_collision_balls):
        sdf_value = (p-collision_ball_positions[i]).norm()- \
                        (1.2*collision_ball_radius+particle_radius_in_world+collision_contact_offset)                                  
        if sdf_value <= 0.:
            sdf_normal = (p-collision_ball_positions[i])/(p-collision_ball_positions[i]).norm()
            collision_ball_positions[i] -= sdf_normal * (sdf_value + particle_radius_in_world + collision_contact_offset)

@ti.func
def balls_collision(p,j):
    for i in range(num_collision_balls):
        if (i == j): 
            continue
        sdf_value = (p-collision_ball_positions[i]).norm()- \
                        (2.9*collision_ball_radius+collision_contact_offset*5)
        factor = (1./collision_ball_weights[j][0]+1./collision_ball_weights[i][0])                
        if sdf_value <= 0.:
            sdf_normal = (p-collision_ball_positions[i])/(p-collision_ball_positions[i]).norm()
            p -= sdf_normal * (sdf_value + collision_contact_offset*2)*1./collision_ball_weights[j][0]/factor
            collision_ball_positions[i] += sdf_normal * (sdf_value + collision_contact_offset*2) \
                                            * 1./collision_ball_weights[i][0]/factor
    return p

@ti.func
def ball_collision_response_boundary(p):
    bmin = collision_ball_radius*1.5
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]
                      ]) - collision_ball_radius
    for i in ti.static(range(dim)):
        if p[i] <= bmin:
            p[i] = bmin 
        elif bmax[i] <= p[i]:
            p[i] = bmax[i]    
    return p   

@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 70
    vel_strength = 5.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    for k in collision_ball_positions:
        old_collision_ball_positions[k] = collision_ball_positions[k]    
    # apply gravity within boundary
    for k in collision_ball_positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        pos_ball, vel_ball = collision_ball_positions[k], collision_ball_velocities[k]    
        vel_ball += g * time_delta
        pos_ball += vel_ball * time_delta
        pos_ball = balls_collision(pos_ball, k)
        collision_ball_positions[k] = ball_collision_response_boundary(pos_ball)
    for i in positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)
        positions[i] = particle_collide_collision_ball(positions[i])

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(grid_num_balls):
        grid_num_balls[I] = 0    
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1
    for I in ti.grouped(virtual_particle_neighbors):
        virtual_particle_neighbors[I] = [0., 0., 0.]

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    for p_s in collision_ball_positions:
        cell = get_cell(collision_ball_positions[p_s])
        offs = ti.atomic_add(grid_num_balls[cell], 1)
        grid2balls[cell, offs] = p_s
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        nb_i_v = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
                for j in range(grid_num_balls[cell_to_check]):
                    p_j = grid2balls[cell_to_check, j]
                    pos_j = collision_ball_positions[p_j]
                    dist = (pos_i- pos_j).norm() 
                    if dist < collision_ball_radius + neighbor_radius:
                        for particle_off_in_box in ti.grouped(ti.ndrange((-neighborhood_particle_off, neighborhood_particle_off), (-neighborhood_particle_off, neighborhood_particle_off), (-neighborhood_particle_off, neighborhood_particle_off))):
                            p_insert = pos_i + particle_off_in_box*virtual_particle_neighbors_dist
                            if nb_i < max_num_virtual_neighbors and (p_insert - pos_i).norm() > epsilon and (
                                    pos_j - p_insert).norm() < collision_ball_radius:
                                virtual_particle_neighbors[p_i, nb_i_v] = p_insert
                                nb_i_v += 1
        particle_num_neighbors[p_i] = nb_i
        particle_num_virtual_neighbors[p_i] = nb_i_v      

@ti.kernel
def resolve_contact():
    for k in collision_ball_positions:
        pos_ball = collision_ball_positions[k]  
        pos_ball = balls_collision(pos_ball, k)
    for i in positions:    
        positions[i] = particle_collide_collision_ball(positions[i])    

@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)
        
        if particle_num_virtual_neighbors[p_i] < particle_num_neighbors[p_i]*15:
            for j in range(particle_num_virtual_neighbors[p_i]) :
                pos_j = virtual_particle_neighbors[p_i, j]
                norm = (pos_j-[0.,0.,0.]).norm()
                if norm < epsilon :
                    break
                pos_ji = pos_i - pos_j
                grad_j = spiky_gradient(pos_ji, h_)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i

    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    for k in collision_ball_positions:
        pos_ball = collision_ball_positions[k]  
        pos_ball = balls_collision(pos_ball, k)
        collision_ball_positions[k] = ball_collision_response_boundary(pos_ball)    
        collision_ball_velocities[k] = (collision_ball_positions[k] - old_collision_ball_positions[k]) / time_delta    
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
        positions[i] = particle_collide_collision_ball(positions[i])
    # vorticity/xsph
    c = 0.01
    for p_i in positions:
        K = ti.Vector([0.0, 0.0, 0.0])
        pos_i = positions[p_i]
        density_constraint = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_j = positions[p_j]
            pos_ji = pos_i - pos_j
            density_constraint += poly6_value(pos_ji.norm(), h_)
            K += (mass / rho0) * (velocities[p_j] - velocities[p_i]) * density_constraint
        velocities[p_i] = velocities[p_i] + c * K


def run_pbf():
    prologue()
    for _ in range(stablization_iters):
        resolve_contact()    
    for _ in range(pbf_num_iters):
        substep()    
    epilogue()


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02, 0.0])
        pos_y = i // (num_particles_x * num_particles_z)
        positions[i] = ti.Vector([i % num_particles_x, pos_y, (i - pos_y * (num_particles_x * num_particles_z)) // num_particles_x
                                  ]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
        p = positions[i] 
        positions[i] = particle_collide_collision_ball(p)    
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

@ti.kernel
def init_collision_balls():
    for i in range(num_collision_balls):
        delta = h_ * 1.8
        offs = ti.Vector([boundary[0]*0.15, boundary[1] * 0.2,  boundary[2] * 0.3])
        collision_ball_positions[i] = ti.Vector([2*i*collision_ball_radius,i%2,i%2*collision_ball_radius/2.])*delta + offs
        collision_ball_weights[i][0] = (i*6+20.)


def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max_}')
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max_}')

init_collision_balls()
init_particles()
window = ti.ui.Window("PBF_3D", screen_res)
canvas = window.get_canvas()
canvas.set_background_color((0.9,0.7,0.6))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(boundary[0]/2+1,40, -50)
camera.lookat(boundary[0]/2+1 ,7, 0)
while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.particles(positions, color = (0.18, 0.26, 0.79), radius = particle_radius)
    # scene.particles(collision_ball_positions, color = (0.7, 0.4, 0.4), radius = collision_ball_radius)
    # draw a smaller ball to avoid visual penetration if you don't like using contact offset
    scene.particles(collision_ball_positions, color = (0.7, 0.4, 0.4), radius = collision_ball_radius+particle_radius*4)
    move_board()
    run_pbf()
    canvas.scene(scene)
    window.show()
    for event in window.get_events(ti.ui.PRESS):
        if event.key in [ti.ui.ESCAPE]:
            window.running = False