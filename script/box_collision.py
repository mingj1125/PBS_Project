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
# box velocites
collision_boxes_velocities = None
# box rotation
collision_boxes_rotations  = None
# box edges index
lines_idx = None

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
            box_mass = 100.
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
            R = add_rotation(R_,R)
            collision_boxes_rotations[i] = ti.Vector([R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7],R[8]])
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
        # rotation
        R = rotation_mat(0.5, 0., 0.)
        collision_boxes_rotations[box_idx] = ti.Vector([R[0][0],R[0][1],R[0][2],R[1][0],R[1][1],R[1][2],R[2][0],R[2][1],R[2][2]])

@ti.kernel
def init_collision_boxes():
    bmax = ti.Vector([boundary[0], boundary[1], boundary[2]])  
    num_lines = 0
    for box_idx in range(num_collision_boxes):
        # rotation
        R = collision_boxes_rotations[box_idx]
        # midpoint
        offs = ti.Vector([5.,0.,5.])
        diff = ti.Vector([10.,0.,0.])
        box_half_diag = collision_box_size[box_idx]*0.5
        midpoint = box_half_diag + (box_idx)*diff + offs
        # boundary artefact
        new_midpoint = midpoint
        new_size = box_half_diag
        # for d in range(dim):
        #     if abs(midpoint[d] - box_half_diag[d]) <= tol:
        #         new_size[d] = 2*box_half_diag[d]
        #         new_midpoint[d] = 0.
        #     elif abs(midpoint[d] + box_half_diag[d] - bmax[d]) <= tol:
        #         new_size[d] = 2*box_half_diag[d]
        #         new_midpoint[d] = bmax[d]
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
        # vertices[8*box_idx+0] =  box_half_diag
        # vertices[8*box_idx+6] = -box_half_diag
        # box_half_diag[0] *= -1
        # vertices[8*box_idx+7] = -box_half_diag
        # vertices[8*box_idx+1] =  box_half_diag
        # box_half_diag[1] *= -1
        # vertices[8*box_idx+2] =  box_half_diag
        # vertices[8*box_idx+4] = -box_half_diag
        # box_half_diag[0] *= -1
        # vertices[8*box_idx+3] =  box_half_diag
        # vertices[8*box_idx+5] = -box_half_diag
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
        vertices[8*box_idx+6] = midpoint + rotate_vertex(R, vertices[8*box_idx+6])
        vertices[8*box_idx+7] = midpoint + rotate_vertex(R, vertices[8*box_idx+7])
        vertices[8*box_idx+0] = midpoint + rotate_vertex(R, vertices[8*box_idx+0])
        vertices[8*box_idx+1] = midpoint + rotate_vertex(R, vertices[8*box_idx+1])
        vertices[8*box_idx+2] = midpoint + rotate_vertex(R, vertices[8*box_idx+2])
        vertices[8*box_idx+4] = midpoint + rotate_vertex(R, vertices[8*box_idx+4])
        vertices[8*box_idx+3] = midpoint + rotate_vertex(R, vertices[8*box_idx+3])
        vertices[8*box_idx+5] = midpoint + rotate_vertex(R, vertices[8*box_idx+5])
    
    calculate_box_edges(num_lines)
    # for line_idx in range(num_lines):
    #     lines[line_idx] = vertices[int(lines_idx[line_idx])]