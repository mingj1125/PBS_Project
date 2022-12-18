import numpy as np
import taichi as ti
import time

from script.helper_function import *
from global_variabel import *
from script.particle_bunny import Particle_Bunny

ti.init(arch=ti.cuda, dynamic_index=True)

bunny = Particle_Bunny()
bunny.initialize()
num_bunnies = 1
num_bunny_particles = bunny.num_particles
particle_offset_CoM = bunny.particle_offset_CoM
total_num_bunny_particles = num_bunnies * bunny.num_particles

# particle position
old_bunny_positions = ti.Vector.field(dim, float)
bunny_positions = ti.Vector.field(dim, float)
bunny_velocities = ti.Vector.field(dim, float)
ti.root.dense(ti.i, total_num_bunny_particles).place(old_bunny_positions, bunny_positions, bunny_velocities)
center_of_mass = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_bunnies).place(center_of_mass)
# grid
grid_num_particles = ti.field(int)
grid2particles = ti.field(ti.i32)
particle_num_neighbors = ti.field(int)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
# neighbor
particle_neighbors = ti.field(int)
bunny_nb_node = ti.root.dense(ti.i, total_num_bunny_particles)
bunny_nb_node.place(particle_num_neighbors)
bunny_nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)

bunny_position_deltas = ti.Vector.field(dim, float)
ti.root.dense(ti.i, total_num_bunny_particles).place(bunny_position_deltas)

particle_numbers = ti.field(int)
ti.root.dense(ti.i, num_bunnies+1).place(particle_numbers)
particle_numbers[0] = 0
particle_numbers[1] = num_bunny_particles
particle_numbers[2] = num_bunny_particles
particle_numbers[3] = num_bunny_particles

@ti.func
def get_particle_obj(p_idx):
    # obj id
    particle_id = 0
    tmp_sum = 0
    while p_idx >= tmp_sum and tmp_sum < total_num_bunny_particles:
        tmp_sum += particle_numbers[particle_id]
        particle_id += 1
    return particle_id

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius
    bmax = ti.Vector([boundary[0], boundary[1], boundary[2]
                      ]) - particle_radius
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin 
        elif bmax[i] <= p[i]:
            p[i] = bmax[i]
    return p

@ti.func
def update_center_of_mass():
    center_of_mass.fill(0.)  
    for i in range(num_bunnies):
        for j in range(bunny.num_particles):  
            center_of_mass[i] += bunny_positions[i*num_bunny_particles+j]/bunny.num_particles  

@ti.func
def shape_matching(bunny):
    A = ti.Matrix([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]], ti.f32)
    #Paper: unified particle system Eq. 16
    for i in range(num_bunny_particles): 
        v3 = ti.math.vec3(bunny_positions[bunny*num_bunny_particles+i]-center_of_mass[bunny])
        v4 = ti.math.vec3(particle_offset_CoM[i])
        A += ti.math.mat3([[v3[0]*v4[0],v3[0]*v4[1],v3[0]*v4[2]],\
                           [v3[1]*v4[0],v3[1]*v4[1],v3[1]*v4[2]],\
                           [v3[2]*v4[0],v3[2]*v4[1],v3[2]*v4[2]]])

    Q, _ = ti.polar_decompose(A, dt=ti.f32)   
    for i in range(num_bunny_particles):
        #Paper: unified particle system Eq. 15
        v4 = ti.math.vec3(particle_offset_CoM[i])
        e1 = ti.math.dot(ti.math.vec3([Q[0,0],Q[0,1],Q[0,2]]),v4)
        e2 = ti.math.dot(ti.math.vec3([Q[1,0],Q[1,1],Q[1,2]]),v4)
        e3 = ti.math.dot(ti.math.vec3([Q[2,0],Q[2,1],Q[2,2]]),v4)
        v1 = ti.math.vec3([e1, e2, e3])
        bunny_position_deltas[bunny*num_bunny_particles+i] = (v1+center_of_mass[bunny]) - bunny_positions[bunny*num_bunny_particles+i]
 

@ti.kernel
def prologue():
    # save old positions
    for i in bunny_positions:
        old_bunny_positions[i] = bunny_positions[i]
    # apply gravity within boundary
    for i in bunny_positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        pos, vel = bunny_positions[i], bunny_velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        bunny_positions[i] = pos

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in bunny_positions:
        cell = get_cell(bunny_positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in bunny_positions:
        pos_i = bunny_positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - bunny_positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i        
        

@ti.kernel
def solve_solid_contact():
    #shape matching for rigid bodies 
    for i in bunny_positions:
        pos = bunny_positions[i]
        bunny_positions[i]= confine_position_to_boundary(pos)     
    # for bunny in range(num_bunnies):    
    #     shape_matching(bunny)
    # for i in bunny_positions:
    #     bunny_positions[i] += bunny_position_deltas[i]
    #     old_bunny_positions[i] += bunny_position_deltas[i]

@ti.kernel
def solve_collision():
    #shape matching for rigid bodies 
    for p_i in bunny_positions:
        pos_i = bunny_positions[p_i] 
        ph_i = get_particle_obj(p_i)

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_j = bunny_positions[p_j]
            ph_j = get_particle_obj(p_j)

            pos_ji = pos_i - pos_j

            if pos_ji.norm() < 2*particle_radius and not ph_i == ph_j:
                pos_delta_i, pos_delta_j = mesh_particle_collision(pos_i, pos_j)
                bunny_positions[p_i] = pos_i + pos_delta_i
                bunny_positions[p_j] = pos_j + pos_delta_j   

@ti.func
def mesh_particle_collision(pos_i,pos_j):
    pos_ji = pos_i - pos_j
    sdf_value = (pos_ji).norm() - (2*particle_radius)
    
    sdf_normal = (pos_ji)/(pos_ji.norm())
    
    delta = sdf_value*sdf_normal * 0.5 * (sdf_value + particle_radius + epsilon * ti.random())
        
    return -delta, delta

@ti.kernel
def epilogue():
    for i in bunny_positions:
        bunny_velocities[i] = (bunny_positions[i] - old_bunny_positions[i]) / time_delta
        #particle sleeping
        if((bunny_positions[i] - old_bunny_positions[i]).norm() < 0.05):
            bunny_positions[i] = old_bunny_positions[i]

@ti.kernel
def solve_shape():
    update_center_of_mass()
    for bunny in range(num_bunnies):    
        shape_matching(bunny)
    for i in bunny_positions:
        bunny_positions[i] += bunny_position_deltas[i]
        old_bunny_positions[i] += bunny_position_deltas[i]

def run_pbf():
    prologue()
    for _ in range(10):
        solve_solid_contact()
        solve_collision()
        solve_shape()
    epilogue()

@ti.kernel
def init_particles():
    for j in range(num_bunnies):
        for i in range(num_bunny_particles):
            bunny_positions[j*num_bunny_particles+i] = bunny.particle_pos[i] + ti.math.vec3([0.,9.*j,0.])
    bunny_velocities.fill(0.)

def main():
    init_particles()
    window = ti.ui.Window("Bunny Simulator", screen_res)
    canvas = window.get_canvas()
    canvas.set_background_color((0.1,0.1,0.1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(boundary[0]/2+1,35,50)
    camera.lookat(boundary[0]/2+1 ,7, 0)    

    global bool_pause        

    while window.running and not window.is_pressed(ti.GUI.ESCAPE):
        # set camera
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        # set light
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        # draw particles and collision obj
        scene.particles(bunny_positions, index_count = num_bunny_particles, color = (0.38, 0.26, 0.49), radius = particle_radius)
        
        # step
        if not bool_pause:
            run_pbf()
        canvas.scene(scene) 

         # display window
        window.show()

        if (window.is_pressed('p')):
            time.sleep(1)
            bool_pause = not bool_pause
            if bool_pause:
                print("pause")
            else:
                print("continue")

if __name__ == "__main__":
    main()