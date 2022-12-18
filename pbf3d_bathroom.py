# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# 3D extension from Taichi 2d implementation by Ye Kuang (k-ye)

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
import script.ball_collision as dc

ti.init(arch=ti.cuda, dynamic_index=True)


## change params (override global params)


## initialize vectors
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
bc.lines_idx = ti.field(float)
ti.root.dense(ti.i, 2*num_lines_per_box*num_collision_boxes).place(bc.lines_idx)
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
old_collision_ball_positions = ti.Vector.field(dim, float)
dc.collision_ball_positions = ti.Vector.field(dim, float)
collision_ball_velocities = ti.Vector.field(dim, float)
dc.collision_ball_weights = ti.Vector.field(1, float)
#mass = ti.field(int)
ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)#, mass)
ti.root.dense(ti.i, num_collision_balls).place(old_collision_ball_positions, \
                dc.collision_ball_positions, collision_ball_velocities, dc.collision_ball_weights)
# grid
grid_num_particles = ti.field(int)
grid2particles = ti.field(ti.i32)
particle_num_neighbors = ti.field(int)
grid_num_balls = ti.field(int)
grid2balls = ti.field(ti.i32)
particle_num_virtual_neighbors = ti.field(int)
virtual_particle_neighbors = ti.Vector.field(dim, float)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles, grid_num_balls)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
grid_snode.dense(ti.l, max_num_balls_per_cell).place(grid2balls)
# neighbor
particle_neighbors = ti.field(int)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
vnb_node = ti.root.dense(ti.i, num_fluid_particles)
vnb_node.place(particle_num_virtual_neighbors)
vnb_node.dense(ti.j, max_num_virtual_neighbors).place(virtual_particle_neighbors)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# boarder
# 0: x-pos, 1: timestep in sin()
dc.board_states = ti.Vector.field(2, float)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.place(dc.board_states)
# mesh
mc.mesh_position = ti.Vector.field(dim, dtype=ti.f32, shape = num_bodies_particles)
mc.mesh_rotation = ti.Vector.field(dim, dtype=ti.f32, shape = num_collision_bodies)

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius
    bmax = ti.Vector([dc.board_states[None][0], boundary[1], boundary[2]
                      ]) - particle_radius
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = dc.board_states[None]
    b[1] += 1.0
    period = 70
    vel_strength = 5.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    dc.board_states[None] = b


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    if bool_ball:    
        for k in dc.collision_ball_positions:
            old_collision_ball_positions[k] = dc.collision_ball_positions[k]    
    # apply gravity within boundary
    if bool_ball:
        for k in dc.collision_ball_positions:
            g = ti.Vector([0.0, -9.8, 0.0])
            pos_ball, vel_ball = dc.collision_ball_positions[k], collision_ball_velocities[k]    
            vel_ball += g * time_delta
            pos_ball += vel_ball * time_delta
            pos_ball = dc.balls_collision(pos_ball, k)
            dc.collision_ball_positions[k] = dc.ball_collision_response_boundary(pos_ball)
    for i in range(num_fluid_particles):
        g = ti.Vector([0.0, -9.8, 0.0])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)
        if bool_ball:
            positions[i] = dc.particle_collide_collision_ball(positions[i])

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
    if bool_ball:    
        for p_s in dc.collision_ball_positions:
            cell = get_cell(dc.collision_ball_positions[p_s])
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
                if bool_ball and get_particle_phase(p_i) == 0:   
                    for j in range(grid_num_balls[cell_to_check]):
                        p_j = grid2balls[cell_to_check, j]
                        pos_j = dc.collision_ball_positions[p_j]
                        dist = (pos_i- pos_j).norm() 
                        if dist < collision_ball_radius + neighbor_radius:
                            for particle_off_in_box in ti.grouped(ti.ndrange((-neighborhood_particle_off, neighborhood_particle_off), (-neighborhood_particle_off, neighborhood_particle_off), (-neighborhood_particle_off, neighborhood_particle_off))):
                                p_insert = pos_i + particle_off_in_box*virtual_particle_neighbors_dist
                                if nb_i < max_num_virtual_neighbors and (p_insert - pos_i).norm() > epsilon and (
                                        pos_j - p_insert).norm() < collision_ball_radius:
                                    virtual_particle_neighbors[p_i, nb_i_v] = p_insert
                                    nb_i_v += 1
        particle_num_neighbors[p_i] = nb_i
        if bool_ball:
            particle_num_virtual_neighbors[p_i] = nb_i_v   

@ti.kernel
def resolve_contact():
    for k in dc.collision_ball_positions:
        pos_ball = dc.collision_ball_positions[k]  
        pos_ball = dc.balls_collision(pos_ball, k)
    for i in positions:    
        if get_particle_phase(i) == 0:
            positions[i] = dc.particle_collide_collision_ball(positions[i])   
        else:
            dc.static_mesh_particle_collide_collision_ball(positions[i])

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

    # apply position deltas
    for i in positions:
        if bool_sphere:
            positions[i], velocities[i] = sc.particle_collide_collision_sphere(positions[i], velocities[i])
        if bool_box:
            positions[i], velocities[i] = bc.particle_collide_collision_box(positions[i], velocities[i])
        positions[i] += position_deltas[i]    

@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    if bool_ball:    
        for k in dc.collision_ball_positions:
            pos_ball = dc.collision_ball_positions[k]  
            pos_ball = dc.balls_collision(pos_ball, k)
            dc.collision_ball_positions[k] = dc.ball_collision_response_boundary(pos_ball)    
            collision_ball_velocities[k] = (dc.collision_ball_positions[k] - old_collision_ball_positions[k]) / time_delta    
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    if bool_ball:    
        for i in range(num_fluid_particles):
            positions[i] = dc.particle_collide_collision_ball(positions[i])
    # no vorticity/xsph because we cannot do cross product in 3D
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
    if bool_ball:
        for _ in range(stablization_iters):
            resolve_contact()    
    for _ in range(pbf_num_iters):
        substep()
    epilogue()

## Init functions
@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.5, 10.0])
        pos_y = i // (num_particles_x * num_particles_z)
        positions[i] = ti.Vector([i % num_particles_x, pos_y, (i - pos_y * (num_particles_x * num_particles_z)) // num_particles_x
                                  ]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
        if bool_ball:
            p = positions[i] 
            positions[i] = dc.particle_collide_collision_ball(p)        
    dc.board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])
    # add mesh particles to all particles
    if bool_mesh:
        for i in range(num_bodies_particles):
            positions[i+num_fluid_particles] = mc.mesh_position[i] + ti.math.vec3([0., 0., -15.])
            velocities[i+num_fluid_particles] = ti.math.vec3([0., 0., 0.])

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
            + " screen resolution:   "+str(screen_res[0])+"/"+str(screen_res[1])+"\n"
            + " tolerance:           "+str(tol)+"\n"
            + " particle radius:     "+str(particle_radius))
    if bool_sphere:
        print(" number of spheres:   "+str(num_collision_spheres))
    else:
        print(" number of spheres:   0")
    if bool_box:
        print(" number of boxes:     "+str(num_collision_boxes))
    else:
        print(" number of boxes:     0")
    if bool_mesh:
        print(" number of meshes:    "+str(num_collision_bodies))
    else:
        print(" number of meshes:    0")
    print(    " number of particles: "+str(num_particles)+"\n"
            + " number of fluid part:"+str(num_fluid_particles)+"\n"
            + " number of mesh part: "+str(num_bodies_particles))
    print('-'*term_size.columns)

def print_particle_num():
    print("Particle Numbers:")
    for n in particle_numbers:
        print(" "+str(n))
    print('-'*term_size.columns)

def main():
    print_init()
    print_particle_num()
    if bool_ball:
        dc.init_collision_balls()    
    if bool_mesh:
        mc.init_collision_bodies_bathroom_scene()
    init_particles()
    if bool_sphere:
        sc.init_collision_spheres()
    if bool_box:
        bc.init_collision_boxes()
    window = ti.ui.Window("PBF_3D", screen_res)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(boundary[0]/2+1,40, -50)
    camera.lookat(boundary[0]/2+1 ,7, 0)

    counter = 0
    global bool_pause
    global bool_record

    FRAME = 160
    frame_counter = 0
    series_prefix = "ply_fluids/fluids.ply"
    series_prefix_balls = "ply_balls/balls.ply"

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
            for b_idx in range(num_collision_bodies):
                scene.particles(positions, index_count=mesh_sizes[b_idx], index_offset=p_idx, color=(0.78, 0.36+b_idx*0.2, 0.79), radius = particle_radius)
                p_idx += mesh_sizes[b_idx]
        if bool_ball:
                scene.particles(dc.collision_ball_positions, color = (0.7, 0.5, 0.4), radius = collision_ball_radius+particle_radius*4)

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

        # scene bath
        np_pos_fluids = np.reshape(positions.to_numpy(), (27570, 3))
        # np_pos_balls = np.reshape(dc.collision_ball_positions.to_numpy(), (4, 3))
        # scene lighthouse
        # print(np_pos_fluids.shape)
        if frame_counter < FRAME:
            # writer = ti.tools.PLYWriter(num_vertices=num_fluid_particles)
            # writer.add_vertex_pos(np_pos_fluids[0:17500, 0], np_pos_fluids[0:17500, 1], np_pos_fluids[0:17500, 2])
            # writer.export_frame_ascii(frame_counter, series_prefix)

            # writer_balls = ti.tools.PLYWriter(num_vertices=4)
            # writer_balls.add_vertex_pos(np_pos_balls[:, 0], np_pos_balls[:, 1], np_pos_balls[:, 2])
            # writer_balls.export_frame_ascii(frame_counter, series_prefix_balls)
            # print(np_pos_balls[:, 0], np_pos_balls[:, 1], np_pos_balls[:, 2])
            frame_counter += 1
            print('Exporting frame: {}', frame_counter)

if __name__ == "__main__":
    main()