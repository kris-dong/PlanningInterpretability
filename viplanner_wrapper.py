import os
import torch
torch.cuda.empty_cache()
import torchvision.transforms as transforms
import numpy as np
from viplanner.viplanner.config import TrainCfg
from viplanner.viplanner.plannernet import AutoEncoder, DualAutoEncoder
from viplanner.viplanner.traj_cost_opt.traj_opt import TrajOpt
import sys
sys.path.append('/scratch/kris/RoboRT_Experiment')
import utils
from PIL import Image
from ultralytics import YOLO
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from shared.depth.scripts.convert_model import MobileNetSkipAdd
import cv2
import random
import torch
import numpy as np
from ultralytics.utils.plotting import colors


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Invert a quaternion."""
    q_conj = q.clone()
    q_conj[..., :3] *= -1  # Negate the vector part
    return q_conj

def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector."""
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]

    # Compute cross products
    t = 2 * torch.cross(q_xyz, v, dim=-1)
    v_rotated = v + q_w * t + torch.cross(q_xyz, t, dim=-1)
    return v_rotated

# def preprocess_training_images(DATA_PATH: str, img_number: int, device):
#     """
#     Preprocess images taken from training script to be passed into the model
#     """
#     # Load and process images from training data. Need to reshape to add batch dimension in front
#     depth_image_file = f"{DATA_PATH}/depth/{img_number:04d}.npy"
#     sem_image_file = f"{DATA_PATH}/semantics/{img_number:04d}.png"
#     depth_image = np.load(depth_image_file)
#     depth_image = np.array(depth_image, dtype=np.float32) / 1000.0
#     sem_image = np.array(Image.open(sem_image_file), dtype=np.float32)
    
#     # turn into torch tensors
#     # depth_image = torch.tensor(depth_image, device=device)
#     # sem_image = torch.tensor(sem_image, device=device)
#     depth_image = torch.from_numpy(depth_image).to(device)
#     sem_image = torch.from_numpy(sem_image).to(device)
    
#     # reshape to add batch dimension
#     depth_image = torch.reshape(depth_image, (1, depth_image.shape[0], depth_image.shape[1], depth_image.shape[2]))
#     sem_image = torch.reshape(sem_image, (1, sem_image.shape[0], sem_image.shape[1], sem_image.shape[2]))
#     depth_image = depth_image.permute(0, 3, 1, 2)
#     sem_image = sem_image.permute(0, 3, 1, 2)

#     return depth_image, sem_image

def transform_goal(CAMERA_CFG_PATH: str, goals: torch.Tensor, image_number: int, device):
    """
    Transform goasl into camera frame
    """
    # Load camera extrinsics collected during training
    cam_pos, cam_quat = utils.load_camera_extrinsics(CAMERA_CFG_PATH, image_number, device=device)
    # transform goal to camera frame
    goal_cam_frame = goals - cam_pos
    goal_cam_frame[:, 2] = 0  # trained with z difference of 0
    goal_cam_frame = quat_apply(quat_inv(cam_quat), goal_cam_frame)
    return goal_cam_frame

def load_models(model_size: str = "nano"):
    """
    Load YOLO-seg and Depth models based on the specified model size.

    Args:
        model_size (str): Model size to load. Options are "nano", "small", "medium", or "big".

    Returns:
        tuple: YOLO model and Depth model.
    """
    if model_size == "small":
        yolo_model = YOLO("yolo11s-seg.pt")  # YOLO nano model
        depth_model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            max_depth=10
        )
        depth_model.load_state_dict(torch.load(
            "/scratch/kris/RoboRT_Experiment/depth_model/checkpoints/depth_anything_v2_metric_hypersim_vits.pth",
            map_location="cuda:0"
        ))
    elif model_size == "fast":
        yolo_model = YOLO("yolo11n-seg.pt")  # YOLO small model
        depth_model = MobileNetSkipAdd() 
        depth_model.load_state_dict(torch.load(
            "/scratch/kris/RoboRT_Experiment/shared/depth/checkpoints/mobilenet_skip_add.pth",
            map_location="cuda:0"
        ))
        depth_model = depth_model.to("cuda").eval()
        
    elif model_size == "medium":
        yolo_model = YOLO("yolo11m-seg.pt")  # YOLO medium model
        depth_model = DepthAnythingV2(
            encoder="vitb",
            features=128,
            out_channels=[96, 192, 384, 768],
            max_depth=10.0
        )
        depth_model.load_state_dict(torch.load(
            "/scratch/kris/RoboRT_Experiment/depth_model/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth",
            map_location="cuda:0"
        ))
    elif model_size == "big": 
        yolo_model = YOLO("yolo11l-seg.pt")  # YOLO large model
        depth_model = DepthAnythingV2(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
            max_depth=10.0
        )
        depth_model.load_state_dict(torch.load(
            "/scratch/kris/RoboRT_Experiment/depth_model/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
            map_location="cuda:0"
        ))
    else:
        raise ValueError(f"Invalid model size '{model_size}'. Choose from 'nano', 'small', 'medium', or 'big'.")

    depth_model = depth_model.to("cuda").eval()
    return yolo_model, depth_model

def preprocess_seg_images(image_path, yolo_model, device):
    """
    Preprocess image using YOLO-seg for semantic segmentation only.
    Returns a semantic segmentation tensor formatted for downstream use.
    """
    YOLO_TO_VIPLANNER = {
        "person": "person", "bicycle": "bicycle", "car": "vehicle", "motorbike": "motorcycle",
        "bus": "vehicle", "train": "on_rails", "truck": "vehicle", "dog": "anymal", "cat": "anymal",
        "horse": "anymal", "sheep": "anymal", "cow": "anymal", "elephant": "anymal", "bear": "anymal",
        "zebra": "anymal", "giraffe": "anymal", "traffic light": "traffic_light", "stop sign": "traffic_sign",
        "fire hydrant": "traffic_sign", "parking meter": "traffic_sign", "bench": "bench", "road": "road",
        "sidewalk": "sidewalk", "stairs": "stairs", "carpet": "gravel", "mat": "indoor_soft",
        "rug": "indoor_soft", "tile": "floor", "floor marking": "floor", "building": "building",
        "wall": "wall", "fence": "fence", "bridge": "bridge", "tunnel": "tunnel", "pole": "pole",
        "forklift": "truck", "hand truck": "truck", "cart": "truck", "trolley": "truck", "box": "furniture",
        "rack": "furniture", "shelf": "furniture", "fire extinguisher": "fire hydrant", "suitcase": "static",
        "tape": "floor", "chair": "furniture", "sofa": "furniture", "pottedplant": "vegetation",
        "bed": "furniture", "dining table": "furniture", "toilet": "furniture", "tvmonitor": "static",
        "laptop": "static", "mouse": "static", "remote": "static", "keyboard": "static", "cell phone": "static",
        "microwave": "static", "oven": "static", "toaster": "static", "sink": "static", "refrigerator": "static",
        "book": "static", "clock": "static", "vase": "static", "scissors": "static", "teddy bear": "furniture",
        "hair drier": "static", "toothbrush": "static", "airplane": "sky", "kite": "sky", "surfboard": "water_surface",
        "boat": "water_surface", "background": "background"
    }

    VIPLANNER_SEM_META = [
        {"name": "sidewalk", "color": [0, 255, 0]}, {"name": "crosswalk", "color": [0, 102, 0]},
        {"name": "floor", "color": [0, 204, 0]}, {"name": "stairs", "color": [0, 153, 0]},
        {"name": "gravel", "color": [204, 255, 0]}, {"name": "sand", "color": [153, 204, 0]},
        {"name": "snow", "color": [204, 102, 0]}, {"name": "indoor_soft", "color": [102, 153, 0]},
        {"name": "terrain", "color": [255, 255, 0]}, {"name": "road", "color": [255, 128, 0]},
        {"name": "person", "color": [255, 0, 0]}, {"name": "anymal", "color": [204, 0, 0]},
        {"name": "vehicle", "color": [153, 0, 0]}, {"name": "on_rails", "color": [51, 0, 0]},
        {"name": "motorcycle", "color": [102, 0, 0]}, {"name": "bicycle", "color": [102, 0, 0]},
        {"name": "building", "color": [127, 0, 255]}, {"name": "wall", "color": [102, 0, 204]},
        {"name": "fence", "color": [76, 0, 153]}, {"name": "bridge", "color": [51, 0, 102]},
        {"name": "tunnel", "color": [51, 0, 102]}, {"name": "pole", "color": [0, 0, 255]},
        {"name": "traffic_sign", "color": [0, 0, 153]}, {"name": "traffic_light", "color": [0, 0, 204]},
        {"name": "bench", "color": [0, 0, 102]}, {"name": "vegetation", "color": [153, 0, 153]},
        {"name": "water_surface", "color": [204, 0, 204]}, {"name": "sky", "color": [102, 0, 51]},
        {"name": "background", "color": [102, 0, 51]}, {"name": "dynamic", "color": [32, 0, 32]},
        {"name": "static", "color": [0, 0, 0]}, {"name": "furniture", "color": [0, 0, 51]},
        {"name": "door", "color": [153, 153, 0]}, {"name": "ceiling", "color": [25, 0, 51]},
    ]
    VIPLANNER_CLASS_TO_COLOR = {entry["name"]: entry["color"] for entry in VIPLANNER_SEM_META}

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    yolo_results = yolo_model.predict(img_rgb, conf=0.5, verbose=False)

    sem_image = np.zeros_like(img_rgb, dtype=np.uint8)
    for result in yolo_results:
        if result.masks is None or result.boxes is None:
            continue

        names = result.names  # class index -> class name
        for mask, cls_idx in zip(result.masks.xy, result.boxes.cls.int()):
            yolo_class_name = names.get(cls_idx, "background")
            vpl_class_name = YOLO_TO_VIPLANNER.get(yolo_class_name, "background")
            vpl_color = VIPLANNER_CLASS_TO_COLOR.get(vpl_class_name, [102, 0, 51])

            points = np.int32([mask])
            cv2.fillPoly(sem_image, points, vpl_color)

    # Convert to tensor and return
    sem_image = torch.from_numpy(sem_image).to(device, dtype=torch.float32)
    sem_image = sem_image.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
    return sem_image

def preprocess_images(image_path, yolo_model, depth_model, device):
    """
    Preprocess images using YOLO-seg for segmentation and DepthAnything for depth estimation.
    Ensures output dimensions match preprocess_training_images.
    """
    ###
    YOLO_TO_VIPLANNER = {
        # People and animals
        "person": "person",
        "bicycle": "bicycle",
        "car": "vehicle",
        "motorbike": "motorcycle",
        "bus": "vehicle",
        "train": "on_rails",
        "truck": "vehicle",
        "dog": "anymal",
        "cat": "anymal",
        "horse": "anymal",
        "sheep": "anymal",
        "cow": "anymal",
        "elephant": "anymal",
        "bear": "anymal",
        "zebra": "anymal",
        "giraffe": "anymal",

        # Traffic-related
        "traffic light": "traffic_light",
        "stop sign": "traffic_sign",
        "fire hydrant": "traffic_sign",
        "parking meter": "traffic_sign",
        "bench": "bench",

        # Road and walkable areas (approximate)
        "road": "road",
        "sidewalk": "sidewalk",
        "stairs": "stairs",
        "carpet": "gravel",
        "mat": "indoor_soft",
        "rug": "indoor_soft",
        "tile": "floor",
        "floor marking": "floor",

        # Obstacles / Structures
        "building": "building",
        "wall": "wall",
        "fence": "fence",
        "bridge": "bridge",
        "tunnel": "tunnel",
        "pole": "pole",
        "forklift":        "truck",         # ✔ vehicle class
        "hand truck":      "truck",         # falls back to a generic vehicle
        "cart":            "truck",         # likewise
        "trolley":         "truck",         # likewise
        "box":             "furniture",        # no direct COCO box class → treat as background/static
        "rack":            "furniture",        # rigid structure → static
        "shelf":           "furniture",        # rigid structure → static
        "fire extinguisher": "fire hydrant", # closest COCO safety‐device class
        "truck": "vehicle",  # used for forklift
        "suitcase": "static",  # used as visual proxy for box
        "floor marking": "floor",
        "tape": "floor",


        # Indoor static
        "chair": "furniture",
        "sofa": "furniture",
        "pottedplant": "vegetation",
        "bed": "furniture",
        "dining table": "furniture",
        "toilet": "furniture",
        "tvmonitor": "static",
        "laptop": "static",
        "mouse": "static",
        "remote": "static",
        "keyboard": "static",
        "cell phone": "static",
        "microwave": "static",
        "oven": "static",
        "toaster": "static",
        "sink": "static",
        "refrigerator": "static",
        "book": "static",
        "clock": "static",
        "vase": "static",
        "scissors": "static",
        "teddy bear": "furniture",
        "hair drier": "static",
        "toothbrush": "static",

        # Sky/Unknown
        "airplane": "sky",
        "kite": "sky",
        "surfboard": "water_surface",
        "boat": "water_surface",

        # Catch-all / fallback
        "background": "background",
    }
    VIPLANNER_SEM_META = [
        {"name": "sidewalk", "color": [0, 255, 0]},
        {"name": "crosswalk", "color": [0, 102, 0]},
        {"name": "floor", "color": [0, 204, 0]},
        {"name": "stairs", "color": [0, 153, 0]},
        {"name": "gravel", "color": [204, 255, 0]},
        {"name": "sand", "color": [153, 204, 0]},
        {"name": "snow", "color": [204, 102, 0]},
        {"name": "indoor_soft", "color": [102, 153, 0]},
        {"name": "terrain", "color": [255, 255, 0]},
        {"name": "road", "color": [255, 128, 0]},
        {"name": "person", "color": [255, 0, 0]},
        {"name": "anymal", "color": [204, 0, 0]},
        {"name": "vehicle", "color": [153, 0, 0]},
        {"name": "on_rails", "color": [51, 0, 0]},
        {"name": "motorcycle", "color": [102, 0, 0]},
        {"name": "bicycle", "color": [102, 0, 0]},
        {"name": "building", "color": [127, 0, 255]},
        {"name": "wall", "color": [102, 0, 204]},
        {"name": "fence", "color": [76, 0, 153]},
        {"name": "bridge", "color": [51, 0, 102]},
        {"name": "tunnel", "color": [51, 0, 102]},
        {"name": "pole", "color": [0, 0, 255]},
        {"name": "traffic_sign", "color": [0, 0, 153]},
        {"name": "traffic_light", "color": [0, 0, 204]},
        {"name": "bench", "color": [0, 0, 102]},
        {"name": "vegetation", "color": [153, 0, 153]},
        {"name": "water_surface", "color": [204, 0, 204]},
        {"name": "sky", "color": [102, 0, 51]},
        {"name": "background", "color": [102, 0, 51]},
        {"name": "dynamic", "color": [32, 0, 32]},
        {"name": "static", "color": [0, 0, 0]},
        {"name": "furniture", "color": [0, 0, 51]},
        {"name": "door", "color": [153, 153, 0]},
        {"name": "ceiling", "color": [25, 0, 51]},
    ]
    VIPLANNER_CLASS_TO_COLOR = {entry["name"]: entry["color"] for entry in VIPLANNER_SEM_META}
    # VIPLANNER_ID_TO_COLOR = np.array([entry["color"] for entry in VIPLANNER_SEM_META], dtype=np.uint8)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO-seg segmentation
    yolo_results = yolo_model.predict(img_rgb, conf=0.5, verbose=False)

    sem_image = np.zeros_like(img_rgb, dtype=np.uint8)
    for result in yolo_results:
        if result.masks is None or result.boxes is None:
            continue  # nothing to process

        names = result.names  # map from index to class name

        for mask, cls_idx in zip(result.masks.xy, result.boxes.cls.int()):
            # Handle unmapped class indices by assigning them to "background"
            if cls_idx not in names:
                print(f"Warning: Unmapped class index {cls_idx}. Assigning to 'background'.")
                yolo_class_name = "background"
            else:
                yolo_class_name = names[cls_idx]

            vpl_class_name = YOLO_TO_VIPLANNER.get(yolo_class_name, "background")
            vpl_color = VIPLANNER_CLASS_TO_COLOR.get(vpl_class_name, [102, 0, 51])  # default to 'background'

            points = np.int32([mask])
            cv2.fillPoly(sem_image, points, vpl_color)

            print(f"names: {names}")
            print(f"cls_idx: {cls_idx}")


    # sem_image = np.zeros_like(img, dtype=np.uint8)
    # for result in yolo_results:
    #     for mask, box in zip(result.masks.xy, result.boxes):
    #         points = np.int32([mask])
    #         color = random.choices(range(256), k=3)
    #         cv2.fillPoly(sem_image, points, color)

    # Depth estimation
    depth_input, (h, w) = depth_model.image2tensor(img_rgb, 518)
    depth_image = depth_model.forward(depth_input)
    depth_image = torch.nn.functional.interpolate(
        depth_image[:, None], (h, w), mode="bilinear", align_corners=True
    )[0, 0]

    # Normalize depth image
    depth_image = depth_image # Match the scale in preprocess_training_images

    # Convert semantic image to tensor
    sem_image = torch.from_numpy(sem_image).to(device, dtype=torch.float32)

    # Convert depth image to tensor
    depth_image = depth_image.to(device, dtype=torch.float32)

    # Reshape to add batch dimension and permute to match expected format
    depth_image = depth_image.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    sem_image = sem_image.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)

    return depth_image, sem_image

class VIPlannerAlgo:
    def __init__(self, model_dir: str, fear_threshold: float = 0.5, device: str = "cuda:0", eval=True):
        """Apply VIPlanner Algorithm

        Args:
            model_dir (str): Directory that includes model.pt and model.yaml
        """
        super().__init__()

        assert os.path.exists(model_dir), "Model directory does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.pt")), "Model file does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.yaml")), "Model config file does not exist"

        # Parameters
        self.fear_threshold = fear_threshold
        self.device = device

        # Load model
        self.train_config: TrainCfg = None
        self.load_model(model_dir, eval=eval)

        # Get transforms for images
        self.transform = transforms.Resize(self.train_config.img_input_size, antialias=None)

        # Initialize trajectory optimizer
        self.traj_generate = TrajOpt()

        # Visualization colors and sizes
        self.color_fear = (1.0, 0.4, 0.1)  # red
        self.color_path = (0.4, 1.0, 0.1)  # green
        self.size = 5.0

    def load_model(self, model_dir: str, eval=True):
        """Load the model and its configuration."""
        # Load training configuration
        self.train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
        print(
            f"Model loaded using sem: {self.train_config.sem}, rgb: {self.train_config.rgb}, "
            f"knodes: {self.train_config.knodes}, in_channel: {self.train_config.in_channel}"
        )

        if isinstance(self.train_config.data_cfg, list):
            self.max_goal_distance = self.train_config.data_cfg[0].max_goal_distance
            self.max_depth = self.train_config.data_cfg[0].max_depth
        else:
            self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
            self.max_depth = self.train_config.data_cfg.max_depth

        # Initialize the appropriate model
        if self.train_config.sem:
            self.net = DualAutoEncoder(self.train_config)
        else:
            self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)

        # Load model weights
        try:
            model_state_dict, _ = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        except ValueError:
            model_state_dict = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        self.net.load_state_dict(model_state_dict)

        # Set model to evaluation mode
        if eval:
            self.net.eval()

        # Move model to the appropriate device
        if self.device.lower() == "cpu":
            print("CUDA not available, VIPlanner will run on CPU")
            self.cuda_avail = False
        else:
            self.net = self.net.to(torch.device("cuda:0"))  # Explicitly move to cuda:0
            self.cuda_avail = True

    ###
    # Transformations
    ###

    def goal_transformer(self, goal: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
        """Transform goal into camera frame."""
        goal_cam_frame = goal - cam_pos
        goal_cam_frame[:, 2] = 0  # trained with z difference of 0
        goal_cam_frame = self.quat_apply(self.quat_inv(cam_quat), goal_cam_frame)
        return goal_cam_frame

    def path_transformer(
        self, path_cam_frame: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor
    ) -> torch.Tensor:
        """Transform path from camera frame to world frame."""
        return quat_apply(
            cam_quat.unsqueeze(1).repeat(1, path_cam_frame.shape[1], 1), path_cam_frame
        ) + cam_pos.unsqueeze(1)

    def input_transformer(self, image: torch.Tensor) -> torch.Tensor:
        """Transform input images."""
        image = self.transform(image)
        image[image > self.max_depth] = 0.0
        image[~torch.isfinite(image)] = 0  # set all inf or nan values to 0
        return image

    ###
    # Planning
    ###

    def plan(self, image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        """Plan a trajectory using a single input image."""
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(image), goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    def plan_dual(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor, no_grad=True, ablate=None) -> tuple:
        """Plan a trajectory using depth and semantic images."""
        # Transform input
        sem_image = self.transform(sem_image) / 255
        if no_grad:
            with torch.no_grad():
                keypoints, fear = self.net(dep_image, sem_image, goal_robot_frame, ablate=ablate)
        else:
            keypoints, fear = self.net(dep_image, sem_image, goal_robot_frame, ablate=ablate)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    ###
    # Debugging
    ###

    def debug_draw(self, paths: torch.Tensor, fear: torch.Tensor, goal: torch.Tensor):
        """Debugging utility to print paths and fear levels."""
        for idx, curr_path in enumerate(paths):
            if fear[idx] > self.fear_threshold:
                print(f"[FEAR] Path {idx}: {curr_path.cpu().numpy()}, Goal: {goal.cpu().numpy()}")
            else:
                print(f"[SAFE] Path {idx}: {curr_path.cpu().numpy()}, Goal: {goal.cpu().numpy()}")