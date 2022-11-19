import taichi as ti
import meshio

ti.init(arch=ti.cuda)

N = 10

particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)

@ti.kernel
def init_points_pos(points : ti.template()):
    for i in range(points.shape[0]):
        points[i] = [i for j in ti.static(range(3))]

init_points_pos(particles_pos)
init_points_pos(points_pos)

mesh = meshio.read("torus.vtk")
mesh_bunny = meshio.read("bunny_scale.vtk")

torusm_pos = mesh.points
bunnym_pos = mesh_bunny.points
print(torusm_pos.shape)
torus_pos = ti.Vector.field(3, dtype=ti.f32, shape = torusm_pos.shape[0])
bunny_pos = ti.Vector.field(3, dtype=ti.f32, shape = bunnym_pos.shape[0])

for i in range(torus_pos.shape[0]):
    torus_pos[i] = torusm_pos[i]
for i in range(bunny_pos.shape[0]):
    bunny_pos[i] = bunnym_pos[i]
    bunny_pos[i] += ti.math.vec3([0., 1., 0.])

window = ti.ui.Window("Test for Drawing 3d-lines", (1200, 1200))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(15, 2, 2)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    scene.particles(particles_pos, color = (0.68, 0.26, 0.19), radius = 0.1)
    scene.particles(torus_pos, color = (0.08, 0.26, 0.19), radius = 0.1)
    scene.particles(bunny_pos, color = (0.08, 0.36, 0.79), radius = 0.05)
    # Draw 3d-lines in the scene
    # Draw 3d-lines in the scene
    scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0)
    canvas.scene(scene)
    window.show()