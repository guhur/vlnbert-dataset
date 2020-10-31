#!/usr/bin/env python3

"""
Script to precompute image features using bottom-up attention (i.e., 
Faster R-CNN pretrained on Visual Genome) 
"""
from dataclasses import dataclass, field
from typing import List, Dict, Union
from pathlib import Path
import cv2
import json
import argtyped
import math
import base64
import csv
import os
import sys
import random
from multiprocessing import Pool
from tqdm.auto import trange, tqdm
from sklearn.metrics import pairwise_distances
import numpy as np
import requests
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch import nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from torchvision.ops import nms
import MatterSim  # MatterSim needs to be on the Python path

bottomup_root = "./bottom-up-attention.pytorch/"
sys.path.append(bottomup_root)

from models.bua import _C
from models.bua.box_regression import BUABoxes
from models import add_config
from utils.progress_bar import ProgressBar
from extract_features import setup
from utils.utils import mkdir, save_features
from utils.extract_utils import (
    save_roi_features,
)

mpl.use("Agg")
random.seed(1)
csv.field_size_limit(sys.maxsize)


TSV_FIELDNAMES = [
    "scanId",
    "viewpointId",
    "image_w",
    "image_h",
    "vfov",
    "features",
    "boxes",
    "cls_prob",
    "attr_prob",
    "featureViewIndex",
    "featureHeading",
    "featureElevation",
    "viewHeading",
    "viewElevation",
]
DRY_RUN = False  # Just run a few images and save the visualization output
NUM_GPUS = 0

# Camera sweep parameters
NUM_SWEEPS = 3
VIEWS_PER_SWEEP = 12
VIEWPOINT_SIZE = NUM_SWEEPS * VIEWS_PER_SWEEP  # Number of total views from one pano
HEADING_INC = 360 / VIEWS_PER_SWEEP  # in degrees
ANGLE_MARGIN = 5  # margin of error for deciding if an object is closer to the centre of another view
ELEVATION_START = -30  # Elevation on first sweep
ELEVATION_INC = 30  # How much elevation increases each sweep

# Filesystem etc
FEATURE_SIZE = 2048
# UPDOWN_DATA = caffe_root + "/data/genome/1600-400-20"
OUTFILE = "ResNet-101-faster-rcnn-genome.tsv"
GRAPHS = "connectivity/"
MODEL_PATH = "bua-caffe-frcn-r101_with_attributes.pth"
MODEL_URL = f"https://vln-r2r.s3.amazonaws.com/{MODEL_PATH}"

# Simulator image parameters
WIDTH = 600  # Max size handled by Faster R-CNN model
HEIGHT = 600
VFOV = 80
ASPECT = WIDTH / HEIGHT
HFOV = math.degrees(2 * math.atan(math.tan(math.radians(VFOV / 2)) * ASPECT))
FOC = (HEIGHT / 2) / math.tan(math.radians(VFOV / 2))  # focal length

# Settings for the number of features per image
MIN_LOCAL_BOXES = 5
MAX_LOCAL_BOXES = 20
MAX_TOTAL_BOXES = 100
NMS_THRESH = 0.3  # same as bottom-up
CONF_THRESH = 0.4  # increased from 0.2 in bottom-up paper

TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000


def get_image_blob(im: np.ndarray, pixel_means) -> Dict:
    """Converts an image into a network input."""
    pixel_means = np.array([[pixel_means]])
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(
            im_orig,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR,
        )

    dataset_dict: Dict[str, Union[float, torch.Tensor]] = {}
    dataset_dict["image"] = torch.from_numpy(im).permute(2, 0, 1)
    dataset_dict["im_scale"] = im_scale

    return dataset_dict


def load_viewpointids() -> List:
    viewpoint_ids = []
    with open(GRAPHS + "scans.txt") as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS + scan + "_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpoint_ids.append((scan, item["image_id"]))
    random.shuffle(viewpoint_ids)
    print("Loaded %d viewpoints" % (len(viewpoint_ids)))
    return viewpoint_ids


def transform_img(im) -> np.ndarray:
    """ Prep opencv BGR 3 channel image for the network """
    blob = np.array(im, copy=True)
    return blob


def visual_overlay(im, dets, ix, classes, attributes):
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    valid = np.where(dets["featureViewIndex"] == ix)[0]
    objects = np.argmax(dets["cls_prob"][valid, 1:], axis=1)
    obj_conf = np.max(dets["cls_prob"][valid, 1:], axis=1)
    attr_thresh = 0.1
    attr = np.argmax(dets["attr_prob"][valid, 1:], axis=1)
    attr_conf = np.max(dets["attr_prob"][valid, 1:], axis=1)
    boxes = dets["boxes"][valid]

    for i in range(len(valid)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        cls = classes[objects[i] + 1]
        if attr_conf[i] > attr_thresh:
            cls = attributes[attr[i] + 1] + " " + cls
        cls += " %.2f" % obj_conf[i]
        plt.gca().add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="red",
                linewidth=2,
                alpha=0.5,
            )
        )
        plt.gca().text(
            bbox[0],
            bbox[1] - 2,
            "%s" % (cls),
            bbox=dict(facecolor="blue", alpha=0.5),
            fontsize=10,
            color="white",
        )
    return fig


@torch.no_grad()
def get_detections_from_im(
    record, model: nn.Module, im: np.ndarray, cfg, conf_thresh=CONF_THRESH
):
    #     im = cv2.imread(os.path.join(args.image_dir, im_file))
    dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
    # extract roi features
    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
    rois = boxes[0].tensor.cpu()
    # unscale back to raw image space
    cls_boxes = rois / dataset_dict["im_scale"]
    scores = scores[0].cpu()
    pool5 = features_pooled[0].cpu()
    attr_prob = attr_scores[0].cpu()

    if "features" not in record:
        ix = 0  # First view in the pano
    elif record["featureViewIndex"].shape[0] == 0:
        ix = 0  # No detections in pano so far
    else:
        ix = int(record["featureViewIndex"][-1]) + 1

    # Keep only the best detections
    max_conf = torch.zeros(rois.shape[0])
    for cls_ind in range(1, scores.shape[1]):
        cls_scores = scores[:, cls_ind]
        keep = nms(cls_boxes, cls_scores, NMS_THRESH)
        max_conf[keep] = torch.tensor(
            np.where(
                cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
            )
        )

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_LOCAL_BOXES:
        keep_boxes = np.argsort(max_conf.numpy())[::-1][:MIN_LOCAL_BOXES]
    elif len(keep_boxes) > MAX_LOCAL_BOXES:
        keep_boxes = np.argsort(max_conf.numpy())[::-1][:MAX_LOCAL_BOXES]

    # Discard any box that would be better centered in another image
    # threshold for pixel distance from center of image
    hor_thresh = FOC * math.tan(math.radians(HEADING_INC / 2 + ANGLE_MARGIN))
    vert_thresh = FOC * math.tan(math.radians(ELEVATION_INC / 2 + ANGLE_MARGIN))
    center_x = 0.5 * (cls_boxes[:, 0] + cls_boxes[:, 2])
    center_y = 0.5 * (cls_boxes[:, 1] + cls_boxes[:, 3])
    reject = (center_x < WIDTH / 2 - hor_thresh) | (center_x > WIDTH / 2 + hor_thresh)
    heading = record["viewHeading"][ix]
    elevation = record["viewElevation"][ix]
    if ix >= VIEWS_PER_SWEEP:  # Not lowest sweep
        reject |= center_y > HEIGHT / 2 + vert_thresh
    if ix < VIEWPOINT_SIZE - VIEWS_PER_SWEEP:  # Not highest sweep
        reject |= center_y < HEIGHT / 2 - vert_thresh
    keep_boxes = np.setdiff1d(keep_boxes, np.argwhere(reject))

    # Calculate the heading and elevation of the center of each observation
    featureHeading = heading + np.arctan2(center_x[keep_boxes] - WIDTH / 2, FOC)
    # normalize featureHeading
    featureHeading = np.mod(featureHeading, math.pi * 2)
    # force it to be the positive remainder, so that 0 <= angle < 360
    featureHeading = np.expand_dims(
        np.mod(featureHeading + math.pi * 2, math.pi * 2), axis=1
    )
    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    featureHeading = np.where(
        featureHeading > math.pi, featureHeading - math.pi * 2, featureHeading
    )
    featureElevation = np.expand_dims(
        elevation + np.arctan2(-center_y[keep_boxes] + HEIGHT / 2, FOC), axis=1
    )

    # Save features, etc
    if "features" not in record:
        record["boxes"] = cls_boxes[keep_boxes]
        record["cls_prob"] = scores[keep_boxes]
        record["attr_prob"] = attr_prob[keep_boxes]
        record["features"] = pool5[keep_boxes]
        record["featureViewIndex"] = (
            np.ones((len(keep_boxes), 1), dtype=np.float32) * ix
        )
        record["featureHeading"] = featureHeading
        record["featureElevation"] = featureElevation
    else:
        record["boxes"] = np.vstack([record["boxes"], cls_boxes[keep_boxes]])
        record["cls_prob"] = np.vstack([record["cls_prob"], scores[keep_boxes]])
        record["attr_prob"] = np.vstack([record["attr_prob"], attr_prob[keep_boxes]])
        record["features"] = np.vstack([record["features"], pool5[keep_boxes]])
        record["featureViewIndex"] = np.vstack(
            [
                record["featureViewIndex"],
                np.ones((len(keep_boxes), 1), dtype=np.float32) * ix,
            ]
        )
        record["featureHeading"] = np.vstack([record["featureHeading"], featureHeading])
        record["featureElevation"] = np.vstack(
            [record["featureElevation"], featureElevation]
        )
    return


def filter(record, max_boxes):
    # Remove the most redundant features (that have similar heading, elevation and
    # are close together to an existing feature in cosine distance)
    feat_dist = pairwise_distances(record["features"], metric="cosine")
    # Heading and elevation diff
    heading_diff = pairwise_distances(record["featureHeading"], metric="euclidean")
    heading_diff = np.minimum(heading_diff, 2 * math.pi - heading_diff)
    elevation_diff = pairwise_distances(record["featureElevation"], metric="euclidean")
    feat_dist = feat_dist + heading_diff + elevation_diff  # Could add weights
    # Discard diagonal and upper triangle by setting large distance
    feat_dist += 10 * np.identity(feat_dist.shape[0], dtype=np.float32)
    feat_dist[np.triu_indices(feat_dist.shape[0])] = 10.0
    ind = np.unravel_index(np.argsort(feat_dist, axis=None), feat_dist.shape)
    # Remove indices of the most similar features (in appearance and orientation)
    keep = set(range(feat_dist.shape[0]))
    ix = 0
    while len(keep) > max_boxes:
        i = ind[0][ix]
        j = ind[1][ix]
        if i not in keep or j not in keep:
            ix += 1
            continue
        if record["cls_prob"][i, 1:].max() > record["cls_prob"][j, 1:].max():
            keep.remove(j)
        else:
            keep.remove(i)
        ix += 1
    # Discard redundant features
    for k, v in record.items():
        if k in [
            "boxes",
            "cls_prob",
            "attr_prob",
            "features",
            "featureViewIndex",
            "featureHeading",
            "featureElevation",
        ]:
            record[k] = v[sorted(keep)]


# def load_classes():
#     # Load updown object classes
#     classes = ["__background__"]
#     with open(os.path.join(UPDOWN_DATA, "objects_vocab.txt")) as f:
#         for object in f.readlines():
#             classes.append(object.split(",")[0].lower().strip())
#
#     # Load updown attributes
#     attributes = ["__no_attribute__"]
#     with open(os.path.join(UPDOWN_DATA, "attributes_vocab.txt")) as f:
#         for att in f.readlines():
#             attributes.append(att.split(",")[0].lower().strip())
#     return classes, attributes


def build_tsv(gpu_id: int = 0):
    args = Argument()
    cfg = setup(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=False,
    )
    model.eval()

    # Set up the simulator
    print("Starting sim")
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(False)
    sim.setBatchSize(1)
    sim.setPreloadingEnabled(False)
    sim.initialize()

    print("Sim has start")
    # import pdb

    # pdb.set_trace()

    print("STARTING")
    with open(OUTFILE, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)

        # Loop all the viewpoints in the simulator
        print("loading viewpoints")
        viewpoint_ids = load_viewpointids()

        for scanId, viewpointId in tqdm(viewpoint_ids):
            # Loop all discretized views from this location
            ims = []
            sim.newEpisode(
                [scanId], [viewpointId], [0], [math.radians(ELEVATION_START)]
            )
            for ix in range(VIEWPOINT_SIZE):
                state = sim.getState()[0]

                # Transform and save generated image
                ims.append(transform_img(state.rgb))
                # Build state
                if ix == 0:
                    record = {
                        "scanId": state.scanId,
                        "viewpointId": state.location.viewpointId,
                        "viewHeading": np.zeros(VIEWPOINT_SIZE, dtype=np.float32),
                        "viewElevation": np.zeros(VIEWPOINT_SIZE, dtype=np.float32),
                        "image_h": HEIGHT,
                        "image_w": WIDTH,
                        "vfov": VFOV,
                    }
                record["viewHeading"][ix] = state.heading
                record["viewElevation"][ix] = state.elevation

                # Move the sim viewpoint so it ends in the same place
                elev = 0.0
                heading_chg = math.pi * 2 / VIEWS_PER_SWEEP
                view = ix % VIEWS_PER_SWEEP
                sweep = ix // VIEWS_PER_SWEEP
                if view + 1 == VIEWS_PER_SWEEP:  # Last viewpoint in sweep
                    elev = math.radians(ELEVATION_INC)
                sim.makeAction([0], [heading_chg], [elev])

            # Run detection
            for ix in range(VIEWPOINT_SIZE):
                get_detections_from_im(record, model, ims[ix], cfg)
            filter(record, MAX_TOTAL_BOXES)

            for k, v in record.items():
                if isinstance(v, np.ndarray):
                    record[k] = str(base64.b64encode(v), "utf-8")
            writer.writerow(record)


def read_tsv(infile):
    # Verify we can read a tsv
    in_data = []
    with open(infile, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for item in reader:
            item["scanId"] = item["scanId"]
            item["image_h"] = int(item["image_h"])  # pixels
            item["image_w"] = int(item["image_w"])  # pixels
            item["vfov"] = int(item["vfov"])  # degrees
            item["features"] = np.frombuffer(
                base64.b64decode(item["features"]), dtype=np.float32
            ).reshape(
                (-1, FEATURE_SIZE)
            )  # variable number of features per pano, say K
            item["boxes"] = np.frombuffer(
                base64.b64decode(item["boxes"]), dtype=np.float32
            ).reshape(
                (-1, 4)
            )  # x1, y1, x2, y2 coords in the original image
            item["viewHeading"] = np.frombuffer(
                base64.b64decode(item["viewHeading"]), dtype=np.float32
            )  # 36 values (heading of each image)
            item["viewElevation"] = np.frombuffer(
                base64.b64decode(item["viewElevation"]), dtype=np.float32
            )  # 36 values (elevation of each image)
            item["featureHeading"] = np.frombuffer(
                base64.b64decode(item["featureHeading"]), dtype=np.float32
            )  # heading of each K features
            item["featureElevation"] = np.frombuffer(
                base64.b64decode(item["featureElevation"]), dtype=np.float32
            )  # elevation of each K features
            item["cls_prob"] = np.frombuffer(
                base64.b64decode(item["cls_prob"]), dtype=np.float32
            ).reshape(
                (-1, 1601)
            )  # K,1601 object class probabilities
            item["attr_prob"] = np.frombuffer(
                base64.b64decode(item["attr_prob"]), dtype=np.float32
            ).reshape(
                (-1, 401)
            )  # K,401 attribute class probabilities
            item["featureViewIndex"] = np.frombuffer(
                base64.b64decode(item["featureViewIndex"]), dtype=np.float32
            )  # K indices mapping each feature to one of the 36 images
            in_data.append(item)
    return in_data


@dataclass
class Argument(argtyped.Arguments):
    min_max_boxes: str = "10,100"
    extract_mode: str = "roi_feats"
    mode: str = "caffe"
    num_cpus: int = 4
    num_gpus: int = 1
    config_file: str = bottomup_root + "configs/bua-caffe/extract-bua-caffe-r101.yaml"
    opts: List = field(default_factory=list)


if __name__ == "__main__":
    args = Argument()

    if not Path(MODEL_PATH).is_file():
        print("Downloading model")
        response = requests.get(MODEL_URL, stream=True)

        with open(MODEL_PATH, "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

    gpu_ids = range(args.num_gpus)
    p = Pool(args.num_gpus)
    p.map(build_tsv, gpu_ids)
