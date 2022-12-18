import bpy
import os

scn = bpy.context.scene
start_frame = 1
end_frame = 200
base_path = 'C:/Users/shaw/onedrive/Desktop/out_mesh/' # replace by your own file location
for f in range(start_frame, end_frame+1):
    
    fpath = base_path + f'om{f}.obj'
    bpy.ops.import_scene.obj(filepath=fpath)
    obj = bpy.context.selected_objects[0]
    mat = bpy.data.materials.get("Fluid") # assign pre-defined material
    obj.data.materials[0] = mat
    obj.hide_viewport = False
    obj.hide_render = False
    # key as visible on the current frame
    obj.keyframe_insert(data_path="hide_render",frame=f)
    obj.keyframe_insert(data_path="hide_viewport",frame=f)
    
    # hide it
    obj.hide_render = True
    obj.hide_viewport = True
    
    # key as hidden on the previous frame
    obj.keyframe_insert(data_path='hide_render',frame=f-1)
    obj.keyframe_insert(data_path='hide_viewport',frame=f-1)
    
    # key as hidden on the next frame
    obj.keyframe_insert(data_path='hide_render',frame=f+1)
    obj.keyframe_insert(data_path='hide_viewport',frame=f+1)
    