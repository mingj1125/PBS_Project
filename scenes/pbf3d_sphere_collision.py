# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# 3D extension from Taichi 2d implementation by Ye Kuang (k-ye)

## SPHERE

import math
import numpy as np
import taichi as ti

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
num_collision_spheres = 3
collision_sphere_radius = 4.
collision_contact_offset = 0.9*particle_radius
collision_velocity_damping = 0.001

# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 8
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
collision_sphere_positions = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(ti.i32)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
ti.root.dense(ti.i, num_collision_spheres).place(collision_sphere_positions)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
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

def set_camera_position(camera, camera_position, camera_lookat):
    camera.position(camera_lookat[0]+np.cos(camera_position[0])*np.cos(camera_position[1])*camera_position[2],
                    camera_lookat[1]+                           np.sin(camera_position[1])*camera_position[2],
                    camera_lookat[2]+np.sin(camera_position[0])*np.cos(camera_position[1])*camera_position[2])
    camera.lookat(camera_lookat[0],camera_lookat[1],camera_lookat[2])

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
def particle_collide_collision_sphere(p):
    for i in range(num_collision_spheres):
        sdf_value = (p-collision_sphere_positions[i]).norm()- \
                        (collision_sphere_radius+particle_radius_in_world+collision_contact_offset)
        if sdf_value <= 0.:
            sdf_normal = (p-collision_sphere_positions[i])/(p-collision_sphere_positions[i]).norm()
            closest_p_on_sphere = p - sdf_value*sdf_normal
            p = closest_p_on_sphere + sdf_normal * (particle_radius_in_world + collision_contact_offset + epsilon * ti.random())
            # v -= v.dot(sdf_normal)*sdf_normal*2.0
            # v *= collision_velocity_damping
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
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)
        positions[i] = particle_collide_collision_sphere(positions[i])

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


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
        positions[i] = particle_collide_collision_sphere(positions[i])
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
        positions[i] = particle_collide_collision_sphere(positions[i])
    # no vorticity/xsph because we cannot do cross product in 2D...
    # TO DO for 3D
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
        v = velocities[i]    
        particle_collide_collision_sphere(p)    
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

@ti.kernel
def init_collision_spheres():
    for i in range(num_collision_spheres):
        delta = h_ * 1.8
        offs = ti.Vector([boundary[0]*0.15, boundary[1] * 0.2,  boundary[2] * 0.3])
        collision_sphere_positions[i] = ti.Vector([i*collision_sphere_radius,i%2,i%2*collision_sphere_radius/2.])*delta + offs


def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max_}')
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max_}')

init_collision_spheres()
init_particles()
window = ti.ui.Window("PBF_3D_Spheres", screen_res)
canvas = window.get_canvas()
canvas.set_background_color((0.9,0.7,0.6))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
# setup camera
camera_position = ti.Vector([(-1./2.)*math.pi,(1./4.)*math.pi, 60.])  
camera_lookat = ti.Vector([boundary[0]/2+1 ,7, 0])
set_camera_position(camera, camera_position, camera_lookat)
while window.running and not window.is_pressed(ti.GUI.ESCAPE):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.particles(positions, color = (0.18, 0.26, 0.79), radius = particle_radius)
    scene.particles(collision_sphere_positions, color = (0.7, 0.4, 0.4), radius = collision_sphere_radius)
    # draw a smaller ball to avoid visual penetration if you don't like using contact offset
    # scene.particles(collision_sphere_positions, color = (0.7, 0.4, 0.4), radius = collision_sphere_radius*0.95)
    move_board()
    run_pbf()
    canvas.scene(scene)
    window.show()
    
    # handel input
    if (window.is_pressed('r')):
        time.sleep(0.1)
        bool_record = not bool_record
        if bool_record:
            print("recording now ...")
        else:
            print("stop record")
    if (window.is_pressed('p')):
        time.sleep(1)
        bool_pause = not bool_pause
        if bool_pause:
            print("pause")
        else:
            print("continue")
        
    # move camera position
    if (window.is_pressed(ti.GUI.LEFT, ti.GUI.RIGHT, ti.GUI.UP, ti.GUI.DOWN, 'i', 'o')):
        if (window.is_pressed(ti.GUI.LEFT)):
            camera_position[0] = camera_position[0] + 0.01 % 2*math.pi
        elif (window.is_pressed(ti.GUI.RIGHT)):
            camera_position[0] = camera_position[0] - 0.01 % 2*math.pi
        if (window.is_pressed(ti.GUI.UP)):
            camera_position[1] = max(min(camera_position[1] + 0.01, math.pi/2.), -math.pi/2.)
        elif (window.is_pressed(ti.GUI.DOWN)):
            camera_position[1] = max(min(camera_position[1] - 0.01, math.pi/2.), -math.pi/2.)
        if (window.is_pressed('i')):
            camera_position[2] = max(camera_position[2] - 1., 0.)
        elif (window.is_pressed('o')):
            camera_position[2] = max(camera_position[2] + 1., 0.)
        set_camera_position(camera, camera_position, camera_lookat)
    
    # move lookat position
    if (window.is_pressed('w','a','s','d','f','g')):
        if (window.is_pressed('a')):
            camera_lookat[0] = camera_lookat[0] + 1.
        elif (window.is_pressed('d')):
            camera_lookat[0] = camera_lookat[0] - 1.
        if (window.is_pressed('f')):
            camera_lookat[1] = camera_lookat[1] + 1.
        elif (window.is_pressed('g')):
            camera_lookat[1] = camera_lookat[1] - 1.
        if (window.is_pressed('w')):
            camera_lookat[2] = camera_lookat[2] + 1.
        elif (window.is_pressed('s')):
            camera_lookat[2] = camera_lookat[2] - 1.
        set_camera_position(camera, camera_position, camera_lookat)