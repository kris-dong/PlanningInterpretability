import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as tf
import utils
import contextlib
import open3d as o3d
from viplanner.viplanner.config import VIPlannerSemMetaHandler
import viplanner_wrapper
from visualize_pc import visualize_semantic_top_down, get_points_in_fov_with_intrinsics
import os
import cv2

# TODO this is taken from the semantic handler code, not sure if there's another way to import it
TRAVERSABLE_INTENDED_LOSS = 0
TRAVERSABLE_UNINTENDED_LOSS = 0.5
ROAD_LOSS = 1.5
def generate_traversable_goal(fov_point_cloud, cam_pos, sem_handler=None, min_distance=2.0, max_distance=10.0):
    """
    Generate a random goal point from traversable areas in the field of view
    
    Args:
        fov_point_cloud: Point cloud filtered to camera's field of view
        cam_pos: Camera position as numpy array [x, y, z]
        sem_handler: VIPlannerSemMetaHandler instance
        min_distance: Minimum distance from camera (meters)
        max_distance: Maximum distance from camera (meters)
        
    Returns:
        goal_point: 3D numpy array with [x, y, z] coordinates of goal in world frame
               or None if no suitable points found
    """
    if sem_handler is None:
        sem_handler = VIPlannerSemMetaHandler()
        
    # Extract points and colors
    points = np.asarray(fov_point_cloud.points)
    colors = np.asarray(fov_point_cloud.colors)
    
    if len(points) == 0:
        print("No points in field of view")
        return None
    
    # Get traversable intended classes
    traversable_classes = []
    for name, loss in sem_handler.class_loss.items():
        if loss == TRAVERSABLE_INTENDED_LOSS or loss == TRAVERSABLE_UNINTENDED_LOSS or loss == ROAD_LOSS:
            traversable_classes.append(name)
    
    print(f"Identified traversable classes: {traversable_classes}")
    
    # Get normalized colors for traversable classes
    traversable_colors = []
    for class_name in traversable_classes:
        rgb_color = sem_handler.class_color[class_name]
        normalized_color = tuple(np.array(rgb_color) / 255.0)
        traversable_colors.append(normalized_color)
    
    # Find points that match traversable classes
    traversable_mask = np.zeros(len(points), dtype=bool)
    
    for i, point_color in enumerate(colors):
        # Check if point color is close to any traversable color
        for trav_color in traversable_colors:
            color_diff = np.sum(np.abs(point_color - np.array(trav_color)))
            if color_diff < 0.1:  # Allow small tolerance for floating point differences
                traversable_mask[i] = True
                break
    
    traversable_points = points[traversable_mask]
    
    if len(traversable_points) == 0:
        print("No traversable points found in field of view")
        return None
    
    print(f"Found {len(traversable_points)} traversable points")
    
    # Calculate distances from camera
    distances = np.linalg.norm(traversable_points - cam_pos, axis=1)
    
    # Filter points by distance
    valid_mask = (distances >= min_distance) & (distances <= max_distance)
    valid_points = traversable_points[valid_mask]
    valid_distances = distances[valid_mask]
    
    if len(valid_points) == 0:
        print(f"No traversable points found within distance range [{min_distance}, {max_distance}]")
        return None
    
    print(f"Found {len(valid_points)} valid traversable points within distance range")
    
    # Randomly select a point, with probability weighted by distance
    # This encourages picking points that are farther away
    weights = valid_distances / np.sum(valid_distances)
    selected_idx = np.random.choice(len(valid_points), p=weights)
    goal_point = valid_points[selected_idx]
    
    print(f"Selected goal at {goal_point} (distance: {valid_distances[selected_idx]:.2f}m)")
    
    # Return the goal point in world frame
    return goal_point

def generate_predefined_goal(cfg: DictConfig, sem_handler=None):
    # data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device
    pc_path = cfg.viplanner.point_cloud_path
    image_path = os.path.join(cfg.viplanner.image_path, "0059.png") 
    img_num = 59
    if sem_handler is None:
        sem_handler = VIPlannerSemMetaHandler()
    goal_point_wf = torch.tensor([-1.0, -0.4, 0], device=device).repeat(1, 1)
    goal_point_bf = viplanner_wrapper.transform_goal(camera_cfg_path, goal_point_wf, img_num, device=device)

    print(f"Selected goal at {goal_point_bf}")
    # print(f"Selected goal at {goal_point_bf} (distance: {valid_distances[selected_idx]:.2f}m)")

    # Return the goal point in world frame
    return goal_point_bf    # Extract points and colors
    # points = np.asarray(fov_point_cloud.points)
    # colors = np.asarray(fov_point_cloud.colors)
    
    # if len(points) == 0:
    #     print("No points in field of view")
    #     return None
    
    # # Get traversable intended classes
    # traversable_classes = []
    # for name, loss in sem_handler.class_loss.items():
    #     if loss == TRAVERSABLE_INTENDED_LOSS or loss == TRAVERSABLE_UNINTENDED_LOSS or loss == ROAD_LOSS:
    #         traversable_classes.append(name)
    
    # print(f"Identified traversable classes: {traversable_classes}")
    
    # # Get normalized colors for traversable classes
    # traversable_colors = []
    # for class_name in traversable_classes:
    #     rgb_color = sem_handler.class_color[class_name]
    #     normalized_color = tuple(np.array(rgb_color) / 255.0)
    #     traversable_colors.append(normalized_color)
    
    # # Find points that match traversable classes
    # traversable_mask = np.zeros(len(points), dtype=bool)
    
    # for i, point_color in enumerate(colors):
    #     # Check if point color is close to any traversable color
    #     for trav_color in traversable_colors:
    #         color_diff = np.sum(np.abs(point_color - np.array(trav_color)))
    #         if color_diff < 0.1:  # Allow small tolerance for floating point differences
    #             traversable_mask[i] = True
    #             break
    
    # traversable_points = points[traversable_mask]
    
    # if len(traversable_points) == 0:
    #     print("No traversable points found in field of view")
    #     return None
    
    # print(f"Found {len(traversable_points)} traversable points")
    
    # # Calculate distances from camera
    # distances = np.linalg.norm(traversable_points - cam_pos, axis=1)
    
    # # Filter points by distance
    # valid_mask = (distances >= min_distance) & (distances <= max_distance)
    # valid_points = traversable_points[valid_mask]
    # valid_distances = distances[valid_mask]
    
    # if len(valid_points) == 0:
    #     print(f"No traversable points found within distance range [{min_distance}, {max_distance}]")
    #     return None
    
    # print(f"Found {len(valid_points)} valid traversable points within distance range")
    
    # Randomly select a point, with probability weighted by distance
    # This encourages picking points that are farther away
    # weights = valid_distances / np.sum(valid_distances)
    # selected_idx = np.random.choice(len(valid_points), p=weights)
    # goal_point = valid_points[selected_idx]


def save_segmentation_output(image_path, yolo_model, output_dir):
    """
    Save YOLO-seg segmentation output for a given image.

    Args:
        image_path (str): Path to the input image.
        yolo_model (YOLO): YOLO model instance.
        output_dir (str): Directory to save the segmentation output.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO-seg segmentation
    results = yolo_model.predict(img_rgb, conf=0.5, verbose=False)

    # Create an empty segmentation image
    seg_image = np.zeros_like(img_rgb, dtype=np.uint8)

    for result in results:
        if result.masks is None or result.boxes is None:
            continue

        for mask, cls_idx in zip(result.masks.xy, result.boxes.cls.int()):
            points = np.int32([mask])
            color = tuple(np.random.randint(0, 255, size=3).tolist())  # Random color
            cv2.fillPoly(seg_image, points, color)

    # Save the segmentation output
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
    print(f"Saved segmentation output to {output_path}")


def save_fastdepth_output(image_path, depth_model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Depth estimation
    depth_input, (h, w) = depth_model.image2tensor(img_rgb, 518)
    print(f'depth input shape: {depth_input.shape}')
    depth_image = depth_model.forward(depth_input)
    print(f'depth image shape: {depth_image.shape}')
    print(f'h, w: {h}, {w}')
    depth_image = cv2.resize(
        depth_image[0, 0].detach().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Normalize depth image for visualization
    depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    depth_image_normalized = (depth_image_normalized * 255).astype(np.uint8)

    # Save the depth output
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".png", "_fast_depth.png"))
    cv2.imwrite(output_path, depth_image_normalized)
    print(f"Saved depth output to {output_path}")


def save_depthAnything_output(image_path, depth_model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Depth estimation
    depth_input, (h, w) = depth_model.image2tensor(img_rgb, 518)
    depth_image = depth_model.forward(depth_input)
    depth_image = torch.nn.functional.interpolate(
        depth_image[:, None], (h, w), mode="bilinear", align_corners=True
    )[0, 0].detach().cpu().numpy()
    # Normalize depth image for visualization
    depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    depth_image_normalized = (depth_image_normalized * 255).astype(np.uint8)

    # Save the depth output
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace(".png", "_depth.png"))
    cv2.imwrite(output_path, depth_image_normalized)
    print(f"Saved depth output to {output_path}")

# preprocess_images(image_path, yolo_model_m, depth_model_m, device=device)
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def generate_goal_from_file(cfg: DictConfig):
    """
    Generate a goal using depth images from precomputed outputs.
    """
    yolo_model_m, depth_model_m = viplanner_wrapper.load_models("medium")
    model_path = cfg.viplanner.model_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    image_path = os.path.join(cfg.viplanner.image_path, "0053.png") 
    point_cloud_path = cfg.viplanner.point_cloud_path
    device = cfg.viplanner.device

    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480
    depth_image_dir = "/scratch/kris/RoboRT_Experiment/correct_plots"
    output_dir = "/scratch/kris/RoboRT_Experiment/output"

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    sem_handler = VIPlannerSemMetaHandler()

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Iterate through images in the specified range
    for img_num in range(45, 63):
        # Determine the depth image file name based on the flag
        if cfg.viplanner.use_fast_depth:
            depth_image_path = os.path.join(depth_image_dir, f"{img_num:04d}_fast_depth.png")
        else:
            depth_image_path = os.path.join(depth_image_dir, f"{img_num:04d}_accurate_depth.png")

        # Load the depth image
        if not os.path.exists(depth_image_path):
            print(f"Depth image not found at {depth_image_path}, skipping...")
            continue

        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Failed to load depth image from {depth_image_path}, skipping...")
            continue
    # Convert depth image to tensor
        depth_image = depth_image.to(device, dtype=torch.float32)

        # Reshape to add batch dimension and permute to match expected format
        depth_image = depth_image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

        # # Normalize and preprocess the depth image
        # depth_image = np.array(depth_image, dtype=np.float32)
        # depth_image = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).to(cfg.viplanner.device)  # Add batch and channel dimensions

        print(f"Loaded depth image from {depth_image_path} with shape {depth_image.shape}")
        goal_point_wf = torch.tensor([3.0, -1.4, 0], device=device).repeat(1, 1)
        goal_point_bf = viplanner_wrapper.transform_goal(camera_cfg_path, goal_point_wf, img_num, device=device)
        print (f"goal point in world frame: {goal_point_wf}")
        # forward/inference
        depth_image = viplanner.input_transformer(depth_image)
        sem_image= viplanner_wrapper.preprocess_seg_images(image_path, yolo_model_m, device=device)
        _, paths, fear = viplanner.plan_dual(depth_image, sem_image, goal_point_bf, no_grad=True)
        print(f"Generated path with fear: {fear}")
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
        path = viplanner.path_transformer(paths, cam_pos, cam_quat)
        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        # Get only the points in the camera's field of view
        fov_point_cloud = get_points_in_fov_with_intrinsics(
            point_cloud, 
            cam_pos, 
            cam_quat, 
            K,
            img_width, 
            img_height,
            forward_axis="X+",  # Use the detected best axis
            max_distance=15
        )

        # Generate goal visualization
        fig, ax = visualize_semantic_top_down(
            fov_point_cloud,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            resolution=0.1,
            height_range=[-1.5, 2.0],
            sem_handler=VIPlannerSemMetaHandler(),
            forward_axis="X+",
            fig_name=f"goal_{img_num}",
            file_name=os.path.join(output_dir, f"goal_{img_num:04d}.png"),
        )

        print(f"Saved goal visualization to {os.path.join(output_dir, f'goal_{img_num:04d}.png')}")

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def generate_depth_img(cfg: DictConfig):
    # yolo_model_s, depth_model_s = viplanner_wrapper.load_models("small")
    yolo_model_n, fast_depth = viplanner_wrapper.load_models("fast")
    yolo_model_m, depth_anything_m = viplanner_wrapper.load_models("medium")
    model_path = cfg.viplanner.model_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    image_path = os.path.join(cfg.viplanner.image_path, "0053.png") 
    point_cloud_path = cfg.viplanner.point_cloud_path
    device = cfg.viplanner.device

    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    sem_handler = VIPlannerSemMetaHandler()

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    output_dir = "/scratch/kris/RoboRT_Experiment/correct_plots"

    # Save segmentation output for each image
    for img_num in range(45, 63):
        image_path = os.path.join(cfg.viplanner.image_path, f"{img_num:04d}.png")
        save_segmentation_output(image_path, yolo_model_m, output_dir)

    # Save depth output for each image
    for img_num in range(45, 63):
        image_path = os.path.join(cfg.viplanner.image_path, f"{img_num:04d}.png")
        save_depthAnything_output(image_path, depth_anything_m, output_dir)
        save_fastdepth_output(image_path, fast_depth, output_dir)

# @hydra.main(version_base="1.3", config_path="configs", config_name="config")
# def generate_goal(cfg: DictConfig):
#     # yolo_model_n, depth_model_n = viplanner_wrapper.load_models("nano")
#     # yolo_model_s, depth_model_s = viplanner_wrapper.load_models("small")
#     yolo_model_m, depth_model_m = viplanner_wrapper.load_models("medium")
#     model_path = cfg.viplanner.model_path
#     camera_cfg_path = cfg.viplanner.camera_cfg_path
#     image_path =  image_path = os.path.join(cfg.viplanner.image_path, "0053.png") 
#     point_cloud_path = cfg.viplanner.point_cloud_path
#     device = cfg.viplanner.device

#     # Define camera intrinsics from the camera_intrinsics.txt file in carla folder
#     # TODO are these right intrinsics?
#     K = np.array([
#         [430.69473, 0,        424.0],
#         [0,         430.69476, 240.0],
#         [0,         0,          1.0]
#     ])
#     img_width, img_height = 848, 480

#     point_cloud = o3d.io.read_point_cloud(point_cloud_path)
#     sem_handler = VIPlannerSemMetaHandler()

#     viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

#     output_dir = "/scratch/kris/RoboRT_Experiment/plots"

#     # Save segmentation output for each image
#     for img_num in range(45, 63):
#         image_path = os.path.join(cfg.viplanner.image_path, f"{img_num:04d}.png")
#         save_segmentation_output(image_path, yolo_model_m, output_dir)

#     # Save depth output for each image
#     for img_num in range(45, 63):
#         image_path = os.path.join(cfg.viplanner.image_path, f"{img_num:04d}.png")
#         save_depth_output(image_path, depth_model_m, output_dir)
#         save_fastdepth_output(image_path, depth_model_m, output_dir)
#     # Get camera parameters
#     for img_num in range(45, 63):
#         cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device="cpu")
#         cam_pos = cam_pos.cpu().numpy().squeeze(0)
#         cam_quat = cam_quat.cpu().numpy().squeeze(0)

#         # Get only the points in the camera's field of view
#         fov_point_cloud = get_points_in_fov_with_intrinsics(
#             point_cloud, 
#             cam_pos, 
#             cam_quat, 
#             K,
#             img_width, 
#             img_height,
#             forward_axis="X+",  # Use the detected best axis
#             max_distance=15
#         )

#         # goal_point = generate_predefined_goal(
#         #     cfg,
#         #     sem_handler=sem_handler
#         # )

#         # print(goal_point)

#         # Load and process images from training data. Need to reshape to add batch dimension in front
#         depth_image_n, sem_image_n = viplanner_wrapper.preprocess_images(image_path, yolo_model_m, depth_model_m, device=device)
#         # depth_image_s, sem_image_s = viplanner_wrapper.preprocess_images(image_path, yolo_model_s, depth_model_s, device=device)
#         # depth_image_m, sem_image_m = viplanner_wrapper.preprocess_images(image_path, yolo_model_m, depth_model_m, device=device)
#         # setup goal, also needs to have batch dimension in front
#         goal_point_wf = torch.tensor([3.0, -1.4, 0], device=device).repeat(1, 1)
#         goal_point_bf = viplanner_wrapper.transform_goal(camera_cfg_path, goal_point_wf, img_num, device=device)
#         print (f"goal point in world frame: {goal_point_wf}")
#         # forward/inference
#         depth_image_n = viplanner.input_transformer(depth_image_n)
#         _, paths, fear = viplanner.plan_dual(depth_image_n, sem_image_n, goal_point_bf, no_grad=True)
#         print(f"Generated path with fear: {fear}")
#         cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
#         path = viplanner.path_transformer(paths, cam_pos, cam_quat)

#         cam_pos = cam_pos.cpu().numpy().squeeze(0)
#         cam_quat = cam_quat.cpu().numpy().squeeze(0)

#         # Visualize the results
#         fig, ax = visualize_semantic_top_down(
#             fov_point_cloud,
#             cam_pos=cam_pos,
#             cam_quat=cam_quat,
#             resolution=0.1,
#             height_range=[-1.5, 2.0],
#             sem_handler=sem_handler,
#             forward_axis="X+",
#             path=path.cpu().numpy()[0],
#             fig_name=f"goal_{img_num} with fear {fear.item():.2f}",
#             file_name=f"plots/goal_{img_num}.png",
#         )

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def new_generate_goal(cfg: DictConfig):
    # yolo_model_n, depth_model_n = viplanner_wrapper.load_models("nano")
    # yolo_model_s, depth_model_s = viplanner_wrapper.load_models("small")
    yolo_model_m, depth_model_m = viplanner_wrapper.load_models("medium")
    model_path = cfg.viplanner.model_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    image_path =  image_path = os.path.join(cfg.viplanner.image_path, "0053.png") 
    point_cloud_path = cfg.viplanner.point_cloud_path
    device = cfg.viplanner.device

    # Define camera intrinsics from the camera_intrinsics.txt file in carla folder
    # TODO are these right intrinsics?
    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    sem_handler = VIPlannerSemMetaHandler()

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    output_dir = "/scratch/kris/RoboRT_Experiment/plots"

    for img_num in range(45, 63):
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device="cpu")
        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        # Get only the points in the camera's field of view
        fov_point_cloud = get_points_in_fov_with_intrinsics(
            point_cloud, 
            cam_pos, 
            cam_quat, 
            K,
            img_width, 
            img_height,
            forward_axis="X+",  # Use the detected best axis
            max_distance=15
        )

        # goal_point = generate_predefined_goal(
        #     cfg,
        #     sem_handler=sem_handler
        # )

        # print(goal_point)

        # Load and process images from training data. Need to reshape to add batch dimension in front
        depth_image_n, sem_image_n = viplanner_wrapper.preprocess_images(image_path, yolo_model_m, depth_model_m, device=device)
        # depth_image_s, sem_image_s = viplanner_wrapper.preprocess_images(image_path, yolo_model_s, depth_model_s, device=device)
        # depth_image_m, sem_image_m = viplanner_wrapper.preprocess_images(image_path, yolo_model_m, depth_model_m, device=device)
        # setup goal, also needs to have batch dimension in front
        goal_point_wf = torch.tensor([3.0, -1.4, 0], device=device).repeat(1, 1)
        goal_point_bf = viplanner_wrapper.transform_goal(camera_cfg_path, goal_point_wf, img_num, device=device)
        print (f"goal point in world frame: {goal_point_wf}")
        # forward/inference
        depth_image_n = viplanner.input_transformer(depth_image_n)
        _, paths, fear = viplanner.plan_dual(depth_image_n, sem_image_n, goal_point_bf, no_grad=True)
        print(f"Generated path with fear: {fear}")
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device=device)
        path = viplanner.path_transformer(paths, cam_pos, cam_quat)

        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        # Visualize the results
        fig, ax = visualize_semantic_top_down(
            fov_point_cloud,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            resolution=0.1,
            height_range=[-1.5, 2.0],
            sem_handler=sem_handler,
            forward_axis="X+",
            path=path.cpu().numpy()[0],
            fig_name=f"goal_{img_num} with fear {fear.item():.2f}",
            file_name=f"plots/goal_{img_num}.png",
        )


def generate_all_goals_tensor(cfg: DictConfig, image_count=1000):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    point_cloud_path = cfg.viplanner.point_cloud_path
    device = cfg.viplanner.device

    K = np.array([
        [430.69473, 0,        424.0],
        [0,         430.69476, 240.0],
        [0,         0,          1.0]
    ])
    img_width, img_height = 848, 480

    point_cloud = o3d.io.read_point_cloud(point_cloud_path)
    sem_handler = VIPlannerSemMetaHandler()

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device, eval=True)

    # Get camera parameters
    goals = []
    for img_num in range(image_count):
        # if img_num == 42:
        #     img_num = 100 # TEMP: Image 42 is low quality, so swap it for a better image
        cam_pos, cam_quat = utils.load_camera_extrinsics(camera_cfg_path, img_num, device="cpu")
        cam_pos = cam_pos.cpu().numpy().squeeze(0)
        cam_quat = cam_quat.cpu().numpy().squeeze(0)

        with contextlib.redirect_stdout(None):
            # Get only the points in the camera's field of view
            fov_point_cloud = get_points_in_fov_with_intrinsics(
                point_cloud, 
                cam_pos, 
                cam_quat, 
                K,
                img_width, 
                img_height,
                forward_axis="X+",  # Use the detected best axis
                max_distance=15
            )

            goal_point = generate_predefined_goal(
                cfg,
                sem_handler=sem_handler
            )

        # # setup goal, also needs to have batch dimension in front
        # if goal_point is None:
        #     goal_point = [5.3700, 0.2616, 0.1474] # Hardcoded average of images 1-100
        #     print(f"No traversable terrain in image {img_num}, skipping...")
        goal = torch.tensor(goal_point, device=device, dtype=torch.float32).repeat(1, 1)
        goal = viplanner_wrapper.transform_goal(camera_cfg_path, goal, img_num, device=device)
        goals.append(goal)

    goals = torch.cat(goals, axis=0)
    return goals


if __name__ == '__main__':
    # generate_depth_img()
    new_generate_goal()
    # generate_goal_from_file()