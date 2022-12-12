import math
import numpy as np
import taichi as ti
import time
import meshio

## include own code
from script.helper_function import *
from global_variabel import *
import script.box_collision as bc
import script.sphere_collision as sc
import script.mesh_collision as mc
from script.particle_bunny import Particle_Bunny

ti.init(arch=ti.cuda, dynamic_index=True)


## change params (override global params)
num_collision_boxes = 1
bool_box = False

bunny = Particle_Bunny()
bunny.initialize()

num_dynamic_meshes = 1
num_bunny_particles = bunny.num_particles
particle_offset_CoM = bunny.particle_offset_CoM
num_dynamic_mesh_particles = num_dynamic_meshes * bunny.num_particles
particle_numbers.append(num_dynamic_mesh_particles)


mesh_names = ["meshes/bunny_dense.vtk"]
# mesh_sizes = []
mesh_obj = meshio.read(mesh_names[0])
mesh_points = mesh_obj.points
particle_numbers.append(mesh_points.shape[0])
num_static_mesh_particles += mesh_points.shape[0]

num_mesh_particles = num_dynamic_mesh_particles + num_static_mesh_particles

num_particles += num_mesh_particles

particle_numbers = ti.Vector(particle_numbers)

## initialize vectors
# mesh vectors
center_of_mass = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_dynamic_meshes).place(center_of_mass)
# sphere positions
sc.collision_sphere_positions = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_collision_spheres).place(sc.collision_sphere_positions)
# box size
bc.collision_box_size = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_collision_boxes).place(bc.collision_box_size)
bc.collision_box_size[0] = ti.Vector([5.,16.,10.])
bc.collision_box_size[1] = ti.Vector([5.,5.,5.])
# box vertices
bc.vertices = ti.Vector.field(dim, float)
ti.root.dense(ti.i, 8*num_collision_boxes).place(bc.vertices)
# box edges
bc.lines = ti.Vector.field(dim, float)
ti.root.dense(ti.i, 2*num_lines_per_box*num_collision_boxes).place(bc.lines)
# box midpoints
bc.collision_boxes_positions = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_collision_boxes).place(bc.collision_boxes_positions)
# box velocites
bc.collision_boxes_velocities = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_collision_boxes).place(bc.collision_boxes_velocities)
# box rotation
bc.collision_boxes_rotations = ti.Vector.field(dim*dim, float)
ti.root.dense(ti.i, num_collision_boxes).place(bc.collision_boxes_rotations)
# particle position
old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
#mass = ti.field(int)
ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)#, mass)
# grid
grid_num_particles = ti.field(int)
grid2particles = ti.field(ti.i32)
particle_num_neighbors = ti.field(int)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
# neighbor
particle_neighbors = ti.field(int)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# boarder
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.place(board_states)
# mesh
mc.mesh_position = ti.Vector.field(dim, dtype=ti.f32, shape = num_mesh_particles)
mc.mesh_rotation = ti.Vector.field(dim, dtype=ti.f32, shape = num_mesh_particles)

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]
                      ]) - particle_radius
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.func
def shape_matching(idx, particle_offset, particle_index):
    A = ti.Matrix([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]], ti.f32)
    #Paper: unified particle system Eq. 16
    for i in range(particle_index): 
        v3 = ti.math.vec3(positions[particle_offset+i]-center_of_mass[idx])
        v4 = ti.math.vec3(particle_offset_CoM[i])
        A += ti.math.mat3([[v3[0]*v4[0],v3[0]*v4[1],v3[0]*v4[2]],\
                           [v3[1]*v4[0],v3[1]*v4[1],v3[1]*v4[2]],\
                           [v3[2]*v4[0],v3[2]*v4[1],v3[2]*v4[2]]])

    Q, _ = ti.polar_decompose(A, dt=ti.f32)   
    for i in range(particle_index):
        #Paper: unified particle system Eq. 15
        v4 = ti.math.vec3(particle_offset_CoM[i])
        e1 = ti.math.dot(ti.math.vec3([Q[0,0],Q[0,1],Q[0,2]]),v4)
        e2 = ti.math.dot(ti.math.vec3([Q[1,0],Q[1,1],Q[1,2]]),v4)
        e3 = ti.math.dot(ti.math.vec3([Q[2,0],Q[2,1],Q[2,2]]),v4)
        v1 = ti.math.vec3([e1, e2, e3])
        position_deltas[particle_offset+i] = (v1+center_of_mass[idx]) - positions[particle_offset+i]

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

@ti.func
def update_center_of_mass():
    center_of_mass.fill(0.)  
    for i in ti.static(range(1, 1+num_dynamic_meshes)):
        num_of_1_mesh_particles = particle_numbers[i]
        for j in range(num_of_1_mesh_particles):  
            center_of_mass[i] += positions[num_fluid_particles + i*num_of_1_mesh_particles+j]/num_of_1_mesh_particles

@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in range(num_fluid_particles+num_dynamic_mesh_particles):
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
    for p_i in range(num_fluid_particles):
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
    for p_i in range(num_particles):
        if get_particle_phase(p_i) == 0:
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
        else:
            position_deltas[p_i] = ti.Vector([0.0,0.0,0.0])
    # update dynamic mesh
    update_center_of_mass()
    tmp_num_particles = num_fluid_particles
    for b_idx in ti.static(range(1, 1+num_dynamic_meshes)):    
        shape_matching(b_idx, tmp_num_particles, particle_numbers[b_idx])
        tmp_num_particles += particle_numbers[b_idx]
    # apply position deltas
    for i in range(num_particles):
        if bool_sphere:
            positions[i], velocities[i] = sc.particle_collide_collision_sphere(positions[i], velocities[i])
        if bool_box:
            positions[i], velocities[i] = bc.particle_collide_collision_box(positions[i], velocities[i])
        positions[i] += position_deltas[i]

@ti.kernel
def epilogue():
    # confine to boundary
    for i in range(num_fluid_particles):
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 3D
    c = 0.01
    for p_i in range(num_fluid_particles):
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

## Init functions
@ti.kernel
def init_particles():
    for i in range(num_fluid_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02, 0.0])
        pos_y = i // (num_particles_x * num_particles_z)
        positions[i] = ti.Vector([i % num_particles_x, pos_y, (i - pos_y * (num_particles_x * num_particles_z)) // num_particles_x
                                  ]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])
    # add mesh particles to all particles
    if bool_mesh:
        # static
        for i in range(num_static_mesh_particles):
            positions[i+num_fluid_particles+num_dynamic_mesh_particles] = mc.mesh_position[i]
            velocities[i+num_fluid_particles+num_dynamic_mesh_particles] = ti.math.vec3([0., 0., 0.])
        # dynamic
        for j in range(num_dynamic_meshes):
            for i in range(num_bunny_particles):
                positions[num_fluid_particles+j*num_bunny_particles+i] = bunny.particle_pos[i] + ti.math.vec3([0.,9.*j,0.])
                velocities[num_fluid_particles+j*num_bunny_particles+i] = ti.math.vec3([0.,0.,0.])

@ti.kernel
def replace_init_particle():
    for p_idx in range(num_fluid_particles):
        p = positions[p_idx]
        v = velocities[p_idx]
        if bool_box:
            p,v = bc.particle_collide_collision_box(p,v)
        # if True:

        positions[p_idx] = p

## print func
def print_stats():
    print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #particles per cell: avg={avg:.2f} max={max_}')
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f'  #neighbors per particle: avg={avg:.2f} max={max_}')

def print_init():
    print('='*term_size.columns)
    print("Position-Based Fluid 3D Simulation \n"
            + " screen resolution:       "+str(screen_res[0])+"/"+str(screen_res[1])+"\n"
            + " tolerance:               "+str(tol)+"\n"
            + " particle radius:         "+str(particle_radius))
    if bool_sphere:
        print(" number of spheres:       "+str(num_collision_spheres))
    else:
        print(" number of spheres:       0")
    if bool_box:
        print(" number of boxes:         "+str(num_collision_boxes))
    else:
        print(" number of boxes:         0")
    if bool_mesh:
        print(" number of meshes(stat):  "+str(num_static_meshes)+"\n"
            + " number of meshes(dyn):   "+str(num_dynamic_meshes))
    else:
        print(" number of meshes:        0")
    print(    " number of particles:     "+str(num_particles)+"\n"
            + " number of fluid part:    "+str(num_fluid_particles)+"\n"
            + " num. of (stat)mesh part: "+str(num_static_mesh_particles)+"\n"
            + " num. of (dyn)mesh part:  "+str(num_dynamic_mesh_particles))
    print('-'*term_size.columns)

def print_particle_num():
    print("Particle Numbers:")
    for n in particle_numbers:
        print(" "+str(n))
    print('-'*term_size.columns)

def main():
    print_init()
    print_particle_num()
    if bool_mesh:
        mc.init_collision_bodies()
    init_particles()
    if bool_sphere:
        sc.init_collision_spheres()
    if bool_box:
        bc.init_collision_boxes()
    replace_init_particle()
    window = ti.ui.Window("PBF_3D", screen_res)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(boundary[0]/2+1,40, -50)
    camera.lookat(boundary[0]/2+1 ,7, 0)

    counter = 0
    global bool_pause
    global bool_record

    while window.running and not window.is_pressed(ti.GUI.ESCAPE):
        # set camera
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        # set light
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        # draw particles and collision obj
        scene.particles(positions, index_count = num_fluid_particles, color = (0.18, 0.36, 0.79), radius = particle_radius)
        if bool_sphere:
            scene.particles(sc.collision_sphere_positions, color = (0.7, 0.4, 0.4), radius = collision_sphere_radius)
        if bool_box:
            scene.particles(bc.vertices, color = (0.79, 0.26, 0.18), radius = particle_radius)
            scene.lines(bc.lines, width=1, color = (0.79, 0.26, 0.18))
        if bool_mesh:
            p_idx = num_fluid_particles
            for b_idx in range(1,1+num_dynamic_meshes+num_static_meshes):
                scene.particles(positions, index_count=particle_numbers[b_idx], index_offset=p_idx, color=(0.78, 0.36+b_idx*0.2, 0.79), radius = particle_radius)
                p_idx += particle_numbers[b_idx]
        scene.particles(positions, index_offset = num_fluid_particles, index_count = num_bunny_particles, color = (0.38, 0.26, 0.49), radius = particle_radius)

        # step
        if not bool_pause:
            move_board()
            run_pbf()
        canvas.scene(scene) 
        # save image
        if bool_record:
            window.save_image("images/pbf-"+str(counter)+".png")
            counter += 1
        
        # display window
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

if __name__ == "__main__":
    main()