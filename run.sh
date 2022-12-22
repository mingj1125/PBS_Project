echo "run PBF-Simulation"

echo "arg: " $1

if   [ "$1" = "drop" ]
then
    echo "let's go"
    python3 ./scenes/pbf3d_bunny_drop.py
elif [ "$1" = "sphere_dynamic" ]
then
    python3 ./scenes/pbf3d_dynamic_sphere_collision.py
elif [ "$1" = "sphere_static" ]
then
    python3 ./scenes/pbf3d_sphere_collision.py
elif [ "$1" = "lighthouse" ]
then
    cp scenes/gl_var_lighthouse.py global_variabel.py
    python3 ./scenes/pbf3d_lighthouse.py
elif [ "$1" = "bathroom" ]
then
    cp scenes/gl_var_bathroom.py global_variabel.py
    python3 ./scenes/pbf3d_bathroom.py
elif [ "$1" = "box_static" ]
then
    cp scenes/gl_var_boxes_static.py global_variabel.py
    python3 ./scenes/pbf3d_box_collision.py
elif [ "$1" = "box_dynamic" ]
then
    cp scenes/gl_var_boxes_dynamic.py global_variabel.py
    python3 pbf3d.py
elif [ "$1" = "unified" ]
then
    cp scenes/gl_var_unified.py global_variabel.py
    python3 ./scenes/pbf3d_bunny_simulator.py
elif [ "$1" = "dynamics" ]
then
    cp scenes/gl_var_dynamics.py global_variabel.py
    python3 pbf3d.py
else
    echo " run default"
    cp scenes/gl_var_default.py global_variabel.py
    python3 pbf3d.py
fi