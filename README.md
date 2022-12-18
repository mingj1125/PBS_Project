# PBS_Project: Position-Based Fluid Simulation


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
