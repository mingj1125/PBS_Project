import numpy as np
import taichi as ti

from global_variabel import *
from script.helper_function import get_particle_phase

## declare field
collision_ball_positions = None
collision_ball_weights = None
old_collision_ball_positions = None
collision_ball_velocities = None
board_states = None

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
def init_collision_balls():
    for i in range(num_collision_balls):
        delta = h_ * 1.8
        offs = ti.Vector([boundary[0]*0.35, boundary[1] * 0.2,  boundary[2] * 0.4])
        collision_ball_positions[i] = ti.Vector([2*i*collision_ball_radius,i%2,i%2*collision_ball_radius/2.])*delta + offs
        collision_ball_weights[i][0] = (i*3+17.5)