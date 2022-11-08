# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# 3D extension from Taichi 2d implementation by Ye Kuang (k-ye)

import math

import numpy as np

import taichi as ti

import time

ti.init(arch=ti.cuda)

screen_res = (1566, 1500)
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
particle_radius = 0.2
particle_radius_in_world = particle_radius / screen_to_world_ratio

# spheres params
#num_collision_spheres = 1
#collision_sphere_radius = 4

# box params
num_collision_boxes = 1
collision_box_size = (5,15,10)

# tolerance
tol = 1e-6

# PBF params
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

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
#collision_sphere_positions = ti.Vector.field(dim, float)
collision_boxes_positions = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(ti.i32)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
#ti.root.dense(ti.i, num_collision_spheres).place(collision_sphere_positions)
ti.root.dense(ti.i, num_collision_boxes).place(collision_boxes_positions)
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
def particle_collide_collision_sphere(p,v):
    for i in range(num_collision_spheres):
        sdf_value = (p-collision_sphere_positions[i]).norm()-collision_sphere_radius
        if sdf_value <= 0.:
            sdf_normal = (p-collision_sphere_positions[i])/(p-collision_sphere_positions[i]).norm()
            closest_p_on_sphere = p - sdf_value*sdf_normal
            p = closest_p_on_sphere
            v -= v.dot(sdf_normal)*sdf_normal*1.7
    return p,v

@ti.func
def particle_collide_collision_box(p,v):
    for i in range(num_collision_boxes):
        # signed distance
        dist = collision_boxes_positions[i] - p
        d = dist
        d_max = d
        for j in ti.static(range(dim)):
            d[j] = abs(d[j]) - 0.5*collision_box_size[j]
            d_max[j] = max(d[j], 0.0)
        d_norm = d_max.norm()
        max_d = d[0]
        for j in ti.static(range(dim)):
            if (d[j]>max_d):
                max_d = d[j]
        sdf_value = min(max_d,0.0) + d_norm - tol

        # collision
        if sdf_value < 0.:
            # surface normal vector
            n = ti.Vector([0,0,0])
            for j in ti.static(range(dim)):
                if (d[j] >= max_d):
                    max_d = d[j]
                    if dist[j] >= 0:
                        n[j] = -1
                    else:
                        n[j] = 1
            
            closest_p_on_box = p - sdf_value*n
            #print("move out:",p,n,sdf_value,closest_p_on_box,d_norm)
            p = closest_p_on_box
            v -= v.dot(n)*n*1.7
    return p,v

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
        #positions[i], velocities[i] = particle_collide_collision_sphere(positions[i], velocities[i])
        positions[i], velocities[i] = particle_collide_collision_box(positions[i], velocities[i])
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
    # no vorticity/xsph because we cannot do cross product in 2D...
    # TO DO for 3D


def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()

##
# Init functions
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
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

@ti.kernel
def init_collision_spheres():
    for i in range(num_collision_spheres):
        delta = h_ * 1.8
        offs = ti.Vector([boundary[0]*0.3, boundary[1] * 0.1,  boundary[2] * 0.7])
        collision_sphere_positions[i] = ti.Vector([i*collision_sphere_radius,i,i])*delta + offs

@ti.kernel
def init_collision_boxes():
    for i in range(num_collision_boxes):
        collision_boxes_positions[i] = ti.Vector([10,6.5,4])

##
# print func
def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max_}')
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max_}')

#if "__init__" == __name__:
print("Running PBF ...")
init_particles()
#init_collision_spheres()
init_collision_boxes()
window = ti.ui.Window("PBF_3D", screen_res)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(boundary[0]/2+1,40, -50)
camera.lookat(boundary[0]/2+1 ,7, 0)

counter = 0
bool_record = False
bool_pause = False

vertices = ti.Vector.field(dim, float)
ti.root.dense(ti.i, 8).place(vertices)
tmp = ti.Vector(collision_box_size)*0.5
vertices[0] = collision_boxes_positions[0]+tmp
vertices[6] = collision_boxes_positions[0]-tmp
tmp[0] *= -1
vertices[1] = collision_boxes_positions[0]+tmp
vertices[7] = collision_boxes_positions[0]-tmp
tmp[1] *= -1
vertices[2] = collision_boxes_positions[0]+tmp
vertices[4] = collision_boxes_positions[0]-tmp
tmp[0] *= -1
vertices[3] = collision_boxes_positions[0]+tmp
vertices[5] = collision_boxes_positions[0]-tmp

lines = ti.Vector.field(dim, float)
ti.root.dense(ti.i, 2*28).place(lines)
num_lines = 0
for i in range(8):
    for j in range(i+1, 8):
        lines[num_lines] = vertices[i]
        num_lines += 1
        lines[num_lines] = vertices[j]
        num_lines += 1


while window.running and not window.is_pressed(ti.GUI.ESCAPE):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    scene.particles(positions, color = (0.18, 0.26, 0.79), radius = particle_radius)
    scene.particles(vertices, color = (0.79, 0.26, 0.18), radius = particle_radius)

    scene.lines(lines, width=1, color = (0.79, 0.26, 0.18))

    if not bool_pause:
        move_board()
        run_pbf()
    canvas.scene(scene) 
    if bool_record:
        window.save_image("images/pbf-"+str(counter)+".png")
        counter += 1
    window.show()

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