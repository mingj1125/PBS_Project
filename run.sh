echo "run PBF-Simulation"

echo "arg: " $1

if   [ "$1" = "drop" ]
then
    echo "let's go"
    python3 ./scenes/pbf3d_bunny_drop.py
elif [ "$1" = "sphere_dynamic" ]
then
    python3 ./scenes/pbf3d_dynamic_sphere_collision.py
elif [ "$1" = "sphere" ]
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
elif [ "$1" = "box_dynamic" ]
then
    cp scenes/gl_var_boxes_dynamic.py global_variabel.py
    python3 ./scenes/pbf3d_box_collision.py
elif [ "$1" = "unified" ]
then
    cp scenes/gl_var_unified.py global_variabel.py
    python3 pbf3d_bunny_simulator.py
else
    cp scenes/gl_var_default.py global_variabel.py
    python3 pbf3d_bunny_simulator.py
fi