import numpy as np

import taichi as ti

import meshio
import os
 
# get current directory
path = os.getcwd()

@ti.data_oriented
class Particle_Bunny:

    def __init__(self, solid=True, filename = path+"/meshes/bunny_data/bunny_volume_0.1_m4.vtk" ) -> None:
        self.solid = solid
        if(self.solid == False):
            filename = path+"/meshes/bunny_data/bunny_volume.vtk" 
        volume_bunny = meshio.read(filename)
        bunnyv_pos = volume_bunny.points
        self.num_particles = bunnyv_pos.shape[0]
        if(self.solid):
            self.num_particles_volume = self.num_particles
            filename = path+"/meshes/bunny_data/bunny_surface_0.3_m1.vtk"
            surface_bunny = meshio.read(filename)
            bunnys_pos = surface_bunny.points
            self.num_particles_surface = bunnys_pos.shape[0]
            self.num_particles += bunnys_pos.shape[0]
        self.particle_pos = ti.Vector.field(3, dtype=ti.f32, shape = self.num_particles)
        if(self.solid):
            for i in range(self.num_particles_volume):
                self.particle_pos[i] = bunnyv_pos[i] * 2.3 
                self.particle_pos[i] += ti.math.vec3([17., 18., 12.])
            for i in range(self.num_particles_surface):
                self.particle_pos[i+self.num_particles_volume] = bunnys_pos[i] * 2.3
                self.particle_pos[i+self.num_particles_volume] += ti.math.vec3([17., 18., 12.])    
        else:
            for i in range(self.num_particles):
                self.particle_pos[i] = bunnyv_pos[i] * 2.3 
                self.particle_pos[i] += ti.math.vec3([17., 18., 12.]) 

        self.center_of_mass = ti.Vector.field(3, dtype=ti.f32, shape = 1)    
        self.particle_offset_CoM = ti.Vector.field(3, dtype=ti.f32, shape = self.num_particles) 
    
    @ti.kernel       
    def initialize(self):
        self.center_of_mass.fill(0.)
        for i in range(self.num_particles):
                self.center_of_mass[0] += self.particle_pos[i]/self.num_particles
        for i in range(self.num_particles):    
                self.particle_offset_CoM[i] = self.particle_pos[i] - self.center_of_mass[0]