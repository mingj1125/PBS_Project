import numpy as np
import taichi as ti

from global_variabel import *

## declare fields
# box size
collision_box_size = None
# box vertices
vertices = None
# box edges
lines = None
# box midpoints
collision_boxes_positions = None

## box collision
@ti.func
def particle_collide_collision_box(p,v):
    for i in range(num_collision_boxes):
        # signed distance
        dist = collision_boxes_positions[i] - p
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
            closest_p_on_box = p - (sdf_value - particle_radius - epsilon * ti.random())*n
            #print("move out:",p,n,sdf_value,closest_p_on_box,d_norm)
            p = closest_p_on_box
            v -= v.dot(n)*n*1.7
    return p,v

# box initialization
@ti.kernel
def init_collision_boxes():    
    num_lines = 0
    for box_idx in range(num_collision_boxes):
        # midpoint
        offs = ti.Vector([0.,0.,0.])
        diff = ti.Vector([10.,0.,0.])
        box_half_diag = collision_box_size[box_idx]*0.5
        midpoint = box_half_diag + (box_idx+1)*diff + offs
        # boundary artefact
        new_midpoint = midpoint
        new_size = box_half_diag
        for d in range(dim):
            if abs(midpoint[d] - box_half_diag[d]) <= tol:
                new_size[d] = 2*box_half_diag[d]
                new_midpoint[d] = 0.
        collision_box_size[box_idx] = 2*new_size
        collision_boxes_positions[box_idx] = new_midpoint
        # vertices
        vertices[8*box_idx+0] = midpoint+box_half_diag
        vertices[8*box_idx+6] = midpoint-box_half_diag
        box_half_diag[0] *= -1
        vertices[8*box_idx+1] = midpoint+box_half_diag
        vertices[8*box_idx+7] = midpoint-box_half_diag
        box_half_diag[1] *= -1
        vertices[8*box_idx+2] = midpoint+box_half_diag
        vertices[8*box_idx+4] = midpoint-box_half_diag
        box_half_diag[0] *= -1
        vertices[8*box_idx+3] = midpoint+box_half_diag
        vertices[8*box_idx+5] = midpoint-box_half_diag
        # edges
        for i in range(8):
            for j in range(i+1, 8):
                d = (vertices[8*box_idx+i]-vertices[8*box_idx+j])
                counter_empty_dim = 0
                for k in ti.static(range(dim)):
                    if abs(d[k]) < tol:
                        counter_empty_dim += 1
                if counter_empty_dim == 2:
                    lines[num_lines] = vertices[8*box_idx+i]
                    num_lines += 1
                    lines[num_lines] = vertices[8*box_idx+j]
                    num_lines += 1
