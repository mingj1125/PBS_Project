import numpy as np
import taichi as ti

from global_variabel import *

## declare field
collision_sphere_positions = None

## sphere collision
@ti.func
def particle_collide_collision_sphere(p,v):
    for i in range(num_collision_spheres):
        sdf_value = (p-collision_sphere_positions[i]).norm()- \
                        (collision_sphere_radius+particle_radius_in_world+collision_contact_offset)
        if sdf_value <= 0.:
            sdf_normal = (p-collision_sphere_positions[i])/(p-collision_sphere_positions[i]).norm()
            closest_p_on_sphere = p - sdf_value*sdf_normal
            p = closest_p_on_sphere + sdf_normal * (particle_radius_in_world + collision_contact_offset + epsilon * ti.random())
            v -= v.dot(sdf_normal)*sdf_normal*2.0
            v *= collision_velocity_damping
    return p,v

## sphere collision
@ti.kernel
def init_collision_spheres():
    for i in range(num_collision_spheres):
        delta = h_ * 1.8
        offs = ti.Vector([boundary[0]*0.15, boundary[1] * 0.2,  boundary[2] * 0.3])
        collision_sphere_positions[i] = ti.Vector([i*collision_sphere_radius,i%2,i%2*collision_sphere_radius/2.])*delta + offs