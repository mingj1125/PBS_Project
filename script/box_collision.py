import numpy as np
import taichi as ti

from global_variabel import *
from script.helper_function import *

## declare fields
# box size
collision_box_size = None
# box vertices
vertices = None
# box edges
lines = None
# box midpoints
collision_boxes_positions  = None
collision_boxes_old_positions  = None
# box velocites
collision_boxes_velocities = None
collision_boxes_angular_velocities = None
# box rotation
collision_boxes_rotations  = None
# box edges index
lines_idx = None
# box mass
mass = None
# box init rotation
rotations = [[0.,0.,0.] for i in range(num_collision_boxes)]

## confine box to boundary
@ti.func
def confine_box_to_boundary(idx, pos, bound):
    # rotation
    R = collision_boxes_rotations[idx]
    # box size
    box_half_diag = 0.5*collision_box_size[idx]
    # calculate vertices values and confine to boundary
    for x_sign in range(2):
        box_half_diag[0] *= -1.
        for y_sign in range(2):
            box_half_diag[1] *= -1.
            for z_sign in range(2):
                box_half_diag[2] *= -1.
                vertex_coord = pos + rotate_vertex(R, box_half_diag)
                new_pos = confine_position_to_boundary_no_ti(vertex_coord, bound)
                pos += new_pos - vertex_coord
    # update box position
    collision_boxes_positions[idx] = pos

## box collision
@ti.func
def box_box_collision(idx):
    # rotation
    R = collision_boxes_rotations[idx]
    # box size
    half_diag = 0.5*collision_box_size[idx]
    # box position
    pos = collision_boxes_positions[idx]
    # box velocity
    vel = collision_boxes_velocities[idx]

    for x_sign in range(2):
        half_diag[0] *= -1.
        for y_sign in range(2):
            half_diag[1] *= -1.
            for z_sign in range(2):
                half_diag[2] *= -1.

                vertex_coord = pos + rotate_vertex(R, half_diag)
                new_pos, new_vel = particle_collide_dynamic_collision_box(vertex_coord, vel, idx)
                pos += new_pos - vertex_coord

    collision_boxes_positions[idx] = pos

@ti.func
def particle_collide_collision_box(p,v):
    for i in range(num_collision_boxes):
        # rotation
        R = collision_boxes_rotations[i]
        rot_p = rotate_vertex_inv(R, p - collision_boxes_positions[i]) + collision_boxes_positions[i]
        # signed distance
        dist = collision_boxes_positions[i] - rot_p
        d = d_max = dist
        box_size = collision_box_size[i]
        for j in ti.static(range(dim)):
            d[j] = abs(d[j]) - 0.5*box_size[j]
            d_max[j] = max(d[j], 0.0)
        d_norm = d_max.norm()
        max_d = d[0]
        for j in ti.static(range(dim)):
            if (d[j]>max_d):
                max_d = d[j]
        sdf_value = min(max_d,0.0) + d_norm

        # collision
        if sdf_value <= particle_radius:
            # surface normal vector
            n = ti.Vector([0,0,0])
            for j in ti.static(range(dim)):
                if (d[j] >= max_d):
                    max_d = d[j]
                    if dist[j] >= 0:
                        n[j] = -1
                    else:
                        n[j] = 1
            closest_p_on_box = rot_p - (sdf_value - particle_radius)*n + epsilon*ti.random()*ti.Vector([1.,1.,1.])
            #print("move out:",p,n,sdf_value,closest_p_on_box,d_norm)
            p = rotate_vertex(R, closest_p_on_box - collision_boxes_positions[i]) + collision_boxes_positions[i]
            v -= v.dot(n)*n*1.7
    return p,v

@ti.func
def particle_collide_dynamic_collision_box(p,v,jdx=-1):
    for i in range(num_collision_boxes):
        if i == jdx: continue
        # rotation
        R = collision_boxes_rotations[i]
        rot_p = rotate_vertex_inv(R, p - collision_boxes_positions[i]) + collision_boxes_positions[i]
        rot_v = rotate_vertex_inv(R, v)
        # signed distance
        dist = collision_boxes_positions[i] - rot_p
        d = d_max = dist
        box_size = collision_box_size[i]
        for j in ti.static(range(dim)):
            d[j] = abs(d[j]) - 0.5*box_size[j]
            d_max[j] = max(d[j], 0.0)
        d_norm = d_max.norm()
        max_d = d[0]
        for j in ti.static(range(dim)):
            if (d[j]>max_d):
                max_d = d[j]
        sdf_value = min(max_d,0.0) + d_norm

        # collision
        if sdf_value <= particle_radius:
            # surface normal vector
            n = ti.Vector([0,0,0])
            for j in ti.static(range(dim)):
                if (d[j] >= max_d):
                    max_d = d[j]
                    if dist[j] >= 0:
                        n[j] = -1
                    else:
                        n[j] = 1
            # calculate new positions
            delta_x = (sdf_value - particle_radius)*n
            box_mass = mass[i]
            particle_mass = 1.
            closest_p_on_box = rot_p - (box_mass/(particle_mass+box_mass))*delta_x + epsilon*ti.random()*ti.Vector([1.,1.,1.])
            collision_boxes_positions[i] += (particle_mass/(particle_mass+box_mass))*delta_x 
            p = rotate_vertex(R, closest_p_on_box - collision_boxes_positions[i]) + collision_boxes_positions[i]
            # calculate new rotation
            impuls_p = rot_v.dot(n)*particle_mass
            dist = dist - dist.dot(n)*n
            avg_size = 0.33*float(box_size[0]+box_size[1]+box_size[2])
            J_b = (1./12.)*box_mass*(avg_size*avg_size + avg_size*avg_size)
            dreh = impuls_p/J_b * dist * time_delta
            R_ = rotation_mat_ti(dreh[0], dreh[1], dreh[2])
            Rv = collision_boxes_angular_velocities[i]
            # R_v = add_rotation(R_,Rv)
            # collision_boxes_angular_velocities[i] = ti.Vector([R_v[0],R_v[1],R_v[2],R_v[3],R_v[4],R_v[5],R_v[6],R_v[7],R_v[8]])
            R_v = add_rotation(R_,R)
            collision_boxes_rotations[i] = ti.Vector([R_v[0],R_v[1],R_v[2],R_v[3],R_v[4],R_v[5],R_v[6],R_v[7],R_v[8]])
            # update velocity
            v -= v.dot(n)*n*1.7
    return p,v

@ti.func
def box_ball_collision(p,v,radius):
    for i in range(num_collision_boxes):
        # rotation
        R = collision_boxes_rotations[i]
        rot_p = rotate_vertex_inv(R, p - collision_boxes_positions[i]) + collision_boxes_positions[i]
        rot_v = rotate_vertex_inv(R, v)
        # signed distance
        dist = collision_boxes_positions[i] - rot_p
        d = d_max = dist
        box_size = collision_box_size[i]
        for j in ti.static(range(dim)):
            d[j] = abs(d[j]) - 0.5*box_size[j]
            d_max[j] = max(d[j], 0.0)
        d_norm = d_max.norm()
        max_d = d[0]
        for j in ti.static(range(dim)):
            if (d[j]>max_d):
                max_d = d[j]
        sdf_value = min(max_d,0.0) + d_norm

        # collision
        if sdf_value <= radius:
            # surface normal vector
            n = ti.Vector([0,0,0])
            for j in ti.static(range(dim)):
                if (d[j] >= max_d):
                    max_d = d[j]
                    if dist[j] >= 0:
                        n[j] = -1
                    else:
                        n[j] = 1
            # calculate new positions
            delta_x = (sdf_value - radius)*n
            box_mass = mass[i]
            particle_mass = 1.
            closest_p_on_box = rot_p - (box_mass/(particle_mass+box_mass))*delta_x + epsilon*ti.random()*ti.Vector([1.,1.,1.])
            collision_boxes_positions[i] += (particle_mass/(particle_mass+box_mass))*delta_x 
            p = rotate_vertex(R, closest_p_on_box - collision_boxes_positions[i]) + collision_boxes_positions[i]
            # calculate new rotation
            impuls_p = rot_v.dot(n)*particle_mass
            dist = dist - dist.dot(n)*n
            avg_size = 0.33*float(box_size[0]+box_size[1]+box_size[2])
            J_b = (1./12.)*box_mass*(avg_size*avg_size + avg_size*avg_size)
            dreh = impuls_p/J_b * dist * time_delta
            R_ = rotation_mat_ti(dreh[0], dreh[1], dreh[2])
            Rv = collision_boxes_angular_velocities[i]
            R_v = add_rotation(R_,Rv)
            collision_boxes_angular_velocities[i] = R_v
            # update velocity
            v -= v.dot(n)*n*1.7
    return p,v

# box initialization
@ti.func
def calculate_box_vertices(idx):
    # rotation
    R = collision_boxes_rotations[idx]
    # box size
    half_diag = 0.5*collision_box_size[idx]
    # box position
    midpoint = collision_boxes_positions[idx]
    # calculate vertices values
    vertex_idx = 0
    for x_sign in range(2):
        half_diag[0] *= -1.
        for y_sign in range(2):
            half_diag[1] *= -1.
            for z_sign in range(2):
                half_diag[2] *= -1.
                vertices[8*idx+vertex_idx] = midpoint + rotate_vertex(R, half_diag)
                vertex_idx += 1

@ti.func
def calculate_box_edges(num_lines=2*num_lines_per_box*num_collision_boxes):
    for line_idx in range(num_lines):
        lines[line_idx] = vertices[int(lines_idx[line_idx])]

def init_collision_boxes_rotation():
    for box_idx in range(num_collision_boxes):
        # angel
        w = rotations[box_idx]
        # rotation
        R = rotation_mat(w[0],w[1],w[2])
        collision_boxes_rotations[box_idx] = ti.Vector([R[0][0],R[0][1],R[0][2],R[1][0],R[1][1],R[1][2],R[2][0],R[2][1],R[2][2]])
        # angular velocity
        collision_boxes_angular_velocities[box_idx] = ti.Vector([1.,0.,0.,0.,1.,0.,0.,0.,1.])

@ti.kernel
def init_collision_boxes_outline():
    bmax = ti.Vector([boundary[0], boundary[1], boundary[2]])  
    num_lines = 0
    for box_idx in range(num_collision_boxes):
        # rotation
        R = collision_boxes_rotations[box_idx]
        # midpoint
        box_half_diag = collision_box_size[box_idx]*0.5
        midpoint = collision_boxes_positions[box_idx]
        # boundary artefact
        new_midpoint = midpoint
        new_size = box_half_diag
        if all(R == [1.,0.,0.,0.,1.,0.,0.,0.,1.]):
            for d in range(dim):
                if abs(midpoint[d] - box_half_diag[d]) <= tol:
                    new_size[d] = 2*box_half_diag[d]
                    new_midpoint[d] = 0.
                elif abs(midpoint[d] + box_half_diag[d] - bmax[d]) <= tol:
                    new_size[d] = 2*box_half_diag[d]
                    new_midpoint[d] = bmax[d]
        collision_box_size[box_idx] = 2*new_size
        collision_boxes_positions[box_idx] = new_midpoint
        # vertices
        vertex_idx = 0
        for x_sign in range(2):
            box_half_diag[0] *= -1.
            for y_sign in range(2):
                box_half_diag[1] *= -1.
                for z_sign in range(2):
                    box_half_diag[2] *= -1.
                    vertices[8*box_idx+vertex_idx] = box_half_diag
                    vertex_idx += 1
        
        # edges
        for i in range(8):
            for j in range(i+1, 8):
                d = (vertices[8*box_idx+i]-vertices[8*box_idx+j])
                counter_empty_dim = 0
                for k in ti.static(range(dim)):
                    if abs(d[k]) < tol:
                        counter_empty_dim += 1
                # if counter_empty_dim == 2:
                #     lines[num_lines] = vertices[8*box_idx+i]
                #     num_lines += 1
                #     lines[num_lines] = vertices[8*box_idx+j]
                #     num_lines += 1
                if counter_empty_dim == 2:
                    lines_idx[num_lines] = 8*box_idx+i
                    num_lines += 1
                    lines_idx[num_lines] = 8*box_idx+j
                    num_lines += 1
        
        # rotate vertices
        for i in range(8):
            vertices[8*box_idx+i] = midpoint + rotate_vertex(R, vertices[8*box_idx+i])
    
    calculate_box_edges(num_lines)

def init_collision_boxes():
    init_collision_boxes_rotation()
    init_collision_boxes_outline()

def init_boxes_scene_default():
    # rotation
    for i in range(num_collision_boxes):
        rotations[i] = [0.5*i,0.3*i,0.1*i]

    # sizes
    for i in range(num_collision_boxes):
        collision_box_size[i] = ti.Vector([5.,5.,5.])

    # velocities
    for i in range(num_collision_boxes):
        collision_boxes_velocities[i] = ti.Vector([0.,0.,0.])
    
    # positions
    offs = ti.Vector([4.,4.,4.])
    diff = ti.Vector([10.,0.,2.])
    for i in range(num_collision_boxes):
        box_half_diag = 0.5*collision_box_size[i]
        collision_boxes_positions[i] = box_half_diag + i*diff + offs
    
    # mass
    for i in range(num_collision_boxes):
        mass[i] = 100. + 25*i

def init_boxes_scene_static():
    # rotation
    for i in range(num_collision_boxes):
        rotations[i] = [0.5*i,0.3*i,0.1*i]

    # sizes
    collision_box_size[0] = ti.Vector([5.,10.,10.])
    collision_box_size[1] = ti.Vector([8.,8.,8.])
    collision_box_size[2] = ti.Vector([5.,15.,3.])


    # velocities
    for i in range(num_collision_boxes):
        collision_boxes_velocities[i] = ti.Vector([0.,0.,0.])
    
    # positions
    offs = ti.Vector([5.,0.,0.])
    diff = ti.Vector([10.,0.,2.])
    for i in range(num_collision_boxes):
        box_half_diag = 0.5*collision_box_size[i]
        collision_boxes_positions[i] = box_half_diag + i*diff + offs
    
    # mass
    for i in range(num_collision_boxes):
        mass[i] = 100. + 25*i