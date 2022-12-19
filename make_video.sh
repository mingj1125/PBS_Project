cd images
ls -v --hide=camera_pos.png --hide=example_*.png --hide=*.mp4 | cat -n | while read n f; do mv -n "${f}" "$(printf "pbf-%04d" $n).${f#*.}"; done

ffmpeg -framerate 30 -pattern_type glob -i 'pbf-*.png' -c:v libx264 -pix_fmt yuv420p out_pbf.mp4
cd ..