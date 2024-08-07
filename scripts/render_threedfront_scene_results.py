# 
# Modified from: 
#   https://github.com/nv-tlabs/ATISS.
#
"""Script used for visualizing 3D-FRONT room specified by its scene_id."""
import argparse
import os
import sys
import pickle
import numpy as np
import seaborn as sns

from simple_3dviz.utils import render
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveGif
from threed_front.evaluation import ThreedFrontResults
 
from threed_front.simple_3dviz_setup import SIDEVIEW_SCENE
from threed_front.rendering import get_floor_plan, export_scene,scene_from_args,get_textured_objects
from utils import PROJ_DIR, PATH_TO_PICKLED_3D_FUTURE_MODEL, PATH_TO_FLOOR_PLAN_TEXTURES
from threed_front.datasets import ThreedFutureDataset

def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate synthetic layout images from predicted results"
    )  
    parser.add_argument(
        "result_file",
        help="Path to a pickled result (ThreedFrontResults object)"
    )
    #chen
    parser.add_argument(
        "--index",
        default=None,
        help="The scene index of MiDiffusion result"
    )
    parser.add_argument(
        "--path_to_pickled_3d_future_model",
        default=PATH_TO_PICKLED_3D_FUTURE_MODEL,
        help="Path to pickled 3d future model"
        "(default: output/threed_future_model_<room_type>.pkl)"
    )
    parser.add_argument(
        "--output_directory",
        default=PROJ_DIR + "/output/scene/",
        help="Path to output directory if needed (default: output//)"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering and save result gif to output directory"
    )
    parser.add_argument(
        "--floor_color",
        type=lambda x: tuple(map(float, x.split(","))) if x!= None else None,
        default=None,
        help="Set floor color of generated images, e.g. 0.8,0.8,0.8 "
        "(Note: this overrides path_to_floor_plan_textures)"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default=PATH_TO_FLOOR_PLAN_TEXTURES,
        help="Path to floor texture image directory or a single image file "
        "(default: demo/floor_plan_texture_images)"
    )
    parser.add_argument(
        "--without_texture",
        action="store_true",
        help="Visualize without texture "
        "(object color: (0.5,0.5,0.5), floor color: (0.8,0.8,0.8) or specified)"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="Visualize without the room's floor"
    )
    parser.add_argument(
        "--with_walls",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--with_door_and_windows",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--export_mesh",
        action="store_true",
        help="Export scene to output_directory/<scene_id> using trimesh "
        "(trimesh rendering style is only affected by floor_color and path_to_floor_plan_textures)"
    )

    args = parser.parse_args(argv)


    # Load results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    room_type = next((
        type for type in ["diningroom", "livingroom", "bedroom", "library"] \
        if type in os.path.basename(threed_front_results.config["data"]["dataset_directory"])
        ), None)
    assert room_type is not None
    print("Room type:", room_type)
    if not threed_front_results.config["network"].get("room_mask_condition", True):
        args.without_floor = True

    # Default output directory
    if args.output_directory is None:
        args.output_directory = os.path.dirname(args.result_file)
    print("Saving rendered results to: {}.".format(args.output_directory))

    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_future_model.format(room_type)
    )
    print("Loaded {} 3D-FUTURE models from: {}.".format(
        len(objects_dataset), args.path_to_pickled_3d_future_model.format(room_type)
    ))
    
    # Set floor texture or color
    if args.without_floor:
        args.floor_color = None
        floor_textures = [None]
    elif args.without_texture:
        # set floor to specified color if given, or white
        if args.floor_color is None:
            args.floor_color = (1, 1, 1)
        floor_textures = [None]
    else:
        # set floor to specified color if given, or sampled textures
        if args.floor_color is None:
            floor_textures = \
                [os.path.join(args.path_to_floor_plan_textures, fi)
                    for fi in os.listdir(args.path_to_floor_plan_textures)]
        else:
            floor_textures = [None]
    
    # Set color palette if args.no_texture
    if args.without_texture:
        color_palette = \
            sns.color_palette('hls', threed_front_results.test_dataset.n_object_types)
    else:
        color_palette = None
    
    # get the scene from index
   
    idx = int(args.index)
    if idx>=len(threed_front_results._test_dataset)-1:
        idx=0
        print("input index out of range")    
   
    scene_idx = threed_front_results._scene_indices[idx]
    ss = threed_front_results._test_dataset[scene_idx]
    # Get renderables
    renderables, _ = get_textured_objects(
        threed_front_results._predicted_layouts[idx], objects_dataset,threed_front_results._test_dataset.object_types, retrieve_mode="size", 
        color_palette=color_palette, with_trimesh=False
    )

    if not args.without_floor:
        # use a single floor color for rendering without texture
        if args.without_texture:    
            floor_texture = None
            if args.floor_color is None:
                args.floor_color = (0.8, 0.8, 0.8, 1.0)
        # use input floor color if available
        elif args.floor_color:
            floor_texture = None
        # use floor texture files otherwise
        else:
            if os.path.isdir(args.path_to_floor_plan_textures):
                floor_textures = \
                    [os.path.join(args.path_to_floor_plan_textures, fi)
                        for fi in os.listdir(args.path_to_floor_plan_textures)]
                floor_texture = np.random.choice(floor_textures)
            else:
                floor_texture = args.path_to_floor_plan_textures            
        
        floor_plan, _, _ = get_floor_plan(
            ss, floor_texture, args.floor_color, with_room_mask=False
        )
        renderables.append(floor_plan)

    if args.with_walls:
        for ei in ss.extras:
            if "WallInner" in ei.model_type:
                renderables.append(
                    ei.mesh_renderable(
                        offset=-ss.centroid, colors=(0.8, 0.8, 0.8, 0.6)
                    )
                )

    if args.with_door_and_windows:
        for ei in ss.extras:
            if "Window" in ei.model_type or "Door" in ei.model_type:
                renderables.append(
                    ei.mesh_renderable(
                        offset=-ss.centroid, colors=(0.8, 0.8, 0.8, 0.6)
                    )
                )

    # Visualize scene
    if args.without_screen:
        os.makedirs(args.output_directory, exist_ok=True)
        path_to_gif = "{}/{}.gif".format(
            args.output_directory, 
            args.index + "_notexture" if args.without_texture else args.index
        )
        behaviours = [
            LightToCamera(),
            CameraTrajectory(Circle(
                [0, SIDEVIEW_SCENE["camera_position"][1], 0],
                SIDEVIEW_SCENE["camera_position"],
                SIDEVIEW_SCENE["up_vector"]
            ), speed=1/360),
            SaveGif(path_to_gif, 1, duration=36)
        ]
        render(renderables, behaviours, 360, **SIDEVIEW_SCENE)
        print("Saved scene to {}.".format(path_to_gif))
    else:
        behaviours = [LightToCamera(), SnapshotOnKey()]
        show(
            renderables, behaviours=behaviours, **SIDEVIEW_SCENE
        )

    # Create a trimesh scene and export it
    if args.export_mesh:
        path_to_objs = os.path.join(args.output_directory, args.scene_id)
        os.makedirs(path_to_objs, exist_ok=True)

        trimesh_meshes = ss.furniture_meshes()
        _, tr_floor, _ = get_floor_plan(
            ss, floor_texture, args.floor_color, with_room_mask=False,
            with_trimesh=True
        )
        trimesh_meshes.append(tr_floor)
        export_scene(path_to_objs, trimesh_meshes)
        print("Saved meshes to {}.".format(path_to_objs))


if __name__ == "__main__":
    main(sys.argv[1:])
