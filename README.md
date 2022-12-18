![](/images/example_1.png "unified pbf") | ![](/images/example_1.png "...")
# PBS_Project: Position-Based Fluid Simulation

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
- **box_dynamic**: PBF-Simulation with dynamic boxes.
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
