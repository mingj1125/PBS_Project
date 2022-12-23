<p float="middle">
  <img src="/images/example_1.png" width="32%" />
  <img src="/images/example_3.png" width="32%" /> 
  <img src="/images/example_2.png" width="32%" />
</p>

# PBS Project: Position-Based Fluid Simulation

## Run Simulation
The main functions for each PBF-Simulation are mostly in one file (pbf3d*.py). Some additionally function for different features are in other files from the 'script' folder. Important global variables are in defined in global_variabel.py, to be available in the different files. \
To run a simulation, it is necessayr to execute right Python-file and use the global_variabel.py with the correct variabels.
Some scenes are already prepared, with it's one python-file and global\_variabel-file (gl\_var\_*.py). Their is a bash-script, run.sh, to run these files.

### Prepared Scenes
To run a prepared scene, call run.sh with one of the following arguments:
- **drop**: A bunny drop in PBF-Simulation.
- **sphere_dynamic**: PBF-Simulation with dynamic spheres.
- **sphere_static**: PBF-Simulation with static spheres.
- **lighthouse**: Scene with lighthouse and arch in particle representation.
- **bathroom**: Scene with bathtube, bunny and dynamic spheres.
- **box_static**: PBF-Simulation with static boxes.
- **box_dynamic**: PBF-Simulation with dynamic boxes.
- **bunny**: Stanford-Bunny simulated with shape matching in particle representation.
- **bunny_collision**: Collision handling of objects in particle representations.
- **unified**: PBF-simulation with unified particel representation of complex meshes.
- *default*: Simple PBF-Simulation without anything.

## Move Camera
The camera is moveable (in most newer files). We can move the position of the camera and the position of the lookat-Point. The camera is sat at the camera position and looks towards the other point. This point is marked with a white particle.

### Camera position
The camera position is in polar coordinates (alpha, beta, radius).
- **UP**: move the camera up
- **DOWN**: move the camera down
- **LEFT**: rotate the camera around the lookat in the left direction
- **RIGHT**: rotate the camera around the lookat in the right direction
- **i**: (in), decrease the radius/distance from lookat
- **o**: (out), increase the radius/distance from lookat

### lookat position
The position of the lookat-Point is in cartesian-coordinates (x,y,z).
- **w**: in positive z-direction
- **s**: in negative z-direction
- **a**: in positive x-direction
- **d**: in negative x-direction
- **f**: in positive y-direction
- **g**: in negative y-direction

![Alt](/images/camera_pos.png "Explaination for the moves of the camera")

<p float="middle">
  <img src="/images/example_4.png" width="32%" />
  <img src="/images/example_5.png" width="32%" /> 
  <img src="/images/example_6.png" width="32%" />
</p>

## Implementation details
The whole project contains implementations of position based fluids, collision functions of both static and dynamic spheres and boxes, rigid body under unified particle system using Taichi Lang as framework.

### position based fluids
We took the code example for 2D PBF as reference and extended the system in 3D. The implementation uses spatial hashing for neighborhood search and solve PBD in the particle centric style. The artificial pressure, vorticity confinement and viscosity according to the paper is added to our 3D implementation as well. This implementation can be found in the pbf3d.py file and is shown by default scene.

### collision functions of static and dynamic objects
The collision functions for objects are implemented using signed distance field and can be found in the script directory. This will be implemented with a constraint centric style. The density estimation correction for non-particle representative dynamic spheres and boxes is implemented directly in the simulation file. For visualization the scenes box_dynamic, sphere_dynamic and sphere_static are prepared.

### bunny rigid body collision under unified particle representation
For rigid Stanford Bunny we provide an implementation of the shape matching and sparse signed distance field collision method which is mentioned in the paper https://matthias-research.github.io/pages/publications/flex.pdf. You can find this in the bunny_collision_simulator.py file. The reference coordinate system for the particles group of bunny is precaculated in the bunny constructor which can be found in the particle_bunny.py file in the script directory.

### unified particle representation
The particle representation of the complex objects like bathtub, lighthouse, sea architecture and Stanford Bunny are downloaded from open source as obj files and sampled with the Sampling tool of SPlisHSPlasH https://github.com/InteractiveComputerGraphics/SPlisHSPlasH in vtk files in the mesh directory. We extracted the position information of the meshes using meshio python site-package. 

## Rendering for Bathroom scene
To make our simulation more realistic, we decided to render one of the scenes which includes a bathtub, a Stanford bunny and several dynamic rigid balls.
	
We exported 200 frames of our particle system as point cloud,  reconstruct mesh for each frame in Houdini, then we render those generated meshes in Blender with the help of cycles (Render engine). You can find the rendered result in bathroom.mp4.

<p float="middle">
  <img src="/images/bath.gif" width="100%" /> 
</p>
