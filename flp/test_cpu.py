
import sys
sys.path.append("../flp")

import numpy as np
import cv2
import yaml
import types
import os
import urllib.request
import onnxruntime as ort

# Force CPU
ort.set_default_logger_severity(3)

def dict_to_object(d):
    if not isinstance(d, dict):
        return d
    obj = types.SimpleNamespace()
    for k, v in d.items():
        setattr(obj, k, dict_to_object(v))
    return obj

with open("configs/trt_infer.yaml", "r") as f:
    raw_cfg = yaml.safe_load(f)

# Modify config to use CPU
# raw_cfg['models']['warping_spade']['predict_type'] = 'ort'
# raw_cfg['models']['warping_spade']['provider'] = ['CPUExecutionProvider']

cfg = dict_to_object(raw_cfg)
cfg.models = raw_cfg['models']

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

print("Creating CPU pipeline...")
pipeline = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

# Download test image
# url = "https://raw.githubusercontent.com/warmshao/FasterLivePortrait/master/assets/examples/source/s9.jpg"
# urllib.request.urlretrieve(url, "/tmp/s9.jpg")

src_path = "assets/examples/source/s9.jpg"
drv_path = "assets/examples/driving/d14.mp4"

print("Preparing source...")
success = pipeline.prepare_source(src_path)
print(f"Success: {success}")

if success and len(pipeline.src_infos) > 0:
    # driving = cv2.imread(drv_path)
    cap = cv2.VideoCapture(drv_path)
    ret, driving = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read driving video frame")

    print("Running GPU inference...")
    try:
        img_crop, out_crop, I_p_pstbk, dri_motion_info = pipeline.run(
            driving,
            pipeline.src_imgs[0],
            pipeline.src_infos[0],
            first_frame=True
        )
        print(f"✅ GPU Success! I_p_pstbk shape: {I_p_pstbk.shape if I_p_pstbk is not None else 'None'}")
    except Exception as e:
        print(f"❌ GPU failed: {e}")
