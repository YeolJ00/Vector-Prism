import base64

import json

import cv2
import numpy as np
import torch
import yaml
from dover.datasets import UnifiedFrameSampler
from dover.models import DOVER
from dover.datasets import spatial_temporal_view_decomposition
from langchain_core.messages import HumanMessage
from transformers import pipeline
from transformers.image_utils import load_image

from prompts.llm_templates import VIDEO_BINARY_PROMPT, VIDEO_QUALITY_PROMPT
from utils.viclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.viclip.viclip import ViCLIP
import logging

logger = logging.getLogger(__name__)

# DINO_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_NAME = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
# DINO_NAME = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

# Video sampling parameters
NUM_FRAMES = 8
TARGET_SIZE = (224, 224)
V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

MAX_RETRIES = 5

def preload_dino_model():
    """
    Preload DINO model for image feature extraction
    Returns the preloaded pipeline
    """
    feature_extractor = pipeline(
        model=DINO_NAME,
        task="image-feature-extraction", 
    )
    return feature_extractor

def preload_viclip_model(ckpt_path=None):
    """
    Preload ViCLIP model for video-text alignment
    Returns the preloaded model
    """
    # Placeholder for actual ViCLIP model loading
    tokenizer = _Tokenizer()
    model = ViCLIP(tokenizer=tokenizer, pretrain=ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, tokenizer

def preload_dover_model(opt_path="./dover.yml"):
    """
    Preload DOVER model for video quality assessment
    Returns the preloaded model, config options, and temporal samplers
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)
    
    # Load DOVER model
    evaluator = DOVER(**opt["model"]["args"]).to(device)
    if "plus" in opt["test_load_path"]:
        evaluator.load_state_dict(
            torch.load(opt["test_load_path"], map_location=device, weights_only=False)['state_dict']
        )

    else:
        evaluator.load_state_dict(
            torch.load(opt["test_load_path"], map_location=device)
        )
    evaluator.eval()
    
    # Setup temporal samplers
    dopt = opt["data"]["val-l1080p"]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            # resized temporal sampling for TQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            # temporal sampling for AQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )
    
    return evaluator, opt, temporal_samplers

@torch.no_grad()
def extract_dino_features(image_paths, preloaded_pipeline=None):
    """
    image_paths: list, list of paths to the image file
    model_name: str, name of the pre-trained model
    Returns a torch.Tensor of shape (feature_dim,)
    """

    # We use pipeline for simplicity
    if preloaded_pipeline is None:
        feature_extractor = pipeline(
            model=DINO_NAME,
            task="image-feature-extraction", 
        )
        image_processor = feature_extractor.image_processor
        model = feature_extractor.model
        device = model.device
    else:
        image_processor = preloaded_pipeline.image_processor
        model = preloaded_pipeline.model
        device = model.device

    images = [load_image(image_path) for image_path in image_paths]
    
    # Use the pipeline's model and image_processor directly
    inputs = image_processor(images, return_tensors='pt').to(device)
    outputs = model(**inputs)

    features = outputs.pooler_output  # Shape: [batch_size, 1024]
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    
    return features

def davies_bouldin_index(X, labels):
    """
    X: torch.Tensor of shape (n_samples, n_features)
    labels: list of strings of length n_samples
    Returns the Davies-Bouldin index (float)
    """
    n_samples, n_features = X.shape
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)

    # Compute centroids and scatter for each cluster
    centroids = []
    scatter = []
    for label in unique_labels:
        cluster_points = X[torch.tensor([i for i, l in enumerate(labels) if l == label])]
        centroid = cluster_points.mean(dim=0)
        centroids.append(centroid)
        scatter.append(torch.sqrt(((cluster_points - centroid) ** 2).sum(dim=1).mean()))

    centroids = torch.stack(centroids)
    scatter = torch.tensor(scatter)
    centroid_distances = torch.cdist(centroids, centroids, p=2)

    # Compute the Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(n_clusters):
            if i != j:
                ratio = (scatter[i] + scatter[j]) / (centroid_distances[i, j] + 1e-10)
                max_ratio = max(max_ratio, ratio.item())
        db_index += max_ratio

    db_index /= n_clusters

    return db_index

def normalize_video(data):
    """Normalize image data"""
    return (data / 255.0 - V_MEAN) / V_STD

def sample_video_frames(video_path, num_samples):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        video.release()
        return []

    frames = []
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frames.append(frame)

    video.release()
    return frames

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    """Convert list of frames to tensor format"""
    assert len(vid_list) >= fnum, f"Need at least {fnum} frames, got {len(vid_list)}"
    
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    
    # Resize and convert BGR to RGB
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    
    # Normalize and add batch/time dimensions
    vid_tube = [np.expand_dims(normalize_video(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    
    # Transpose to (batch, time, channels, height, width)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    
    return vid_tube

@torch.no_grad()
def dover_video_quality(video_path, preloaded=None):
    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])
    
    if preloaded is None:
        logger.warning("Loading DOVER model for each call is inefficient. Consider preloading the model.")
        evaluator, opt, temporal_samplers = preload_dover_model()
    else:
        evaluator, opt, temporal_samplers = preloaded
    
    device = next(evaluator.parameters()).device
    mean = mean.to(device)
    std = std.to(device)
    
    dopt = opt["data"]["val-l1080p"]["args"]
    
    # View Decomposition
    views, _ = spatial_temporal_view_decomposition(
        video_path, dopt["sample_types"], temporal_samplers
    )

    # Normalize views
    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        v = v.to(device)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
            .to(device)
        )
    
    # Get quality scores
    results = [r.mean().item() for r in evaluator(views)]
    
    # Fuse technical and aesthetic quality scores
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    fused_score = 1 / (1 + np.exp(-x))
    return float(fused_score)

@torch.no_grad()
def clip_t2v_alignment(video_path, text_prompt, preloaded=None):
    if preloaded is None:
        logger.warning("Loading ViCLIP model for each call is inefficient. Consider preloading the model.")
        model, tokenizer = preload_viclip_model()
    else:
        model, tokenizer = preloaded
    device = next(model.parameters()).device
    
    # Sample frames from video
    frames = sample_video_frames(video_path, num_samples=NUM_FRAMES)
    if len(frames) == 0:
        logger.warning(f"Could not extract frames from {video_path}")
        return float('nan')
    
    # Inference model
    vid_tensor = frames2tensor(frames, fnum=NUM_FRAMES, target_size=TARGET_SIZE, device=device)
    vid_feat = model.get_vid_features(vid_tensor)
    text_feat = model.get_text_features(text_prompt, tokenizer, {})
    
    score = 100 * (vid_feat * text_feat).sum(axis=-1)
    
    return float(score.item())


def llm_t2v_alignment(video_path, text_prompt, llm, response_format="score", retries=0):
    frames = sample_video_frames(video_path, num_samples=24)
    if len(frames) == 0:
        logger.warning(f"Could not extract frames from {video_path}")
        return 0 if response_format == "score" else False
    
    # Encode frames as base64
    encoded_frames = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame_rgb)
        encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))
    
    if response_format == "binary":
        evaluation_prompt = VIDEO_BINARY_PROMPT.format(text_prompt=text_prompt)
    else:  # score format
        evaluation_prompt = VIDEO_QUALITY_PROMPT.format(text_prompt=text_prompt)
    
    # Create message with frames
    content = [{"type": "text", "text": evaluation_prompt}]
    for encoded_frame in encoded_frames:
        content.append({"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_frame}"}
        })
    
    logger.info(f"[Langchain] Sending LLM request with {len(encoded_frames)} frames.")
    message = HumanMessage(content=content)
    response = llm.invoke([message])
    response_text = response.content if hasattr(response, 'content') else str(response)

    while retries < MAX_RETRIES:
        try:
            json_response = json.loads(response_text)
            break
        except json.JSONDecodeError:
            logger.warning("LLM response is not valid JSON.")
            return llm_t2v_alignment(video_path, text_prompt, llm, response_format, retries=retries+1)
    logger.debug(f"[Langchain] LLM response:\n{json_response}")
    
    if response_format == "binary":
        if json_response["answer"] in ["yes", "Yes", "true", "True"]:
            return True
        else:
            return False
    else:  # score format
        return float(json_response.get("score", float('nan')))


# TODO: Determine what is better, Qwen-3 T2V or ViCLIP T2V alignment