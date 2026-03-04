from stereo_matcher import save_disparity_map, DEFOMStereoInference, inference_stereo, RAFTStereoInference, UniMatchStereo, BridgeDepthStereo, inference_stereo
import cv2
from pathlib import Path
import time
from loguru import logger

# model = BridgeDepthStereo(weight_path="../data/stereo_weights/BridgeDepth/bridge_rvc_pretrain.pth")
# model = UniMatchStereo(weight_path="../data/stereo_weights/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth")
model = RAFTStereoInference(weight_path="../data/stereo_weights/RAFT_Stereo/raftstereo-middlebury.pth")
# model = DEFOMStereoInference(weight_path="../data/stereo_weights/DEFOMStereo/defomstereo_vits_rvc.pth")
IMG_ROOT = Path("../working_data")  
left_right_pairs = [
    (cv2.cvtColor(cv2.imread(str(l), cv2.IMREAD_ANYDEPTH), cv2.COLOR_GRAY2RGB),
     cv2.cvtColor(cv2.imread(str(r), cv2.IMREAD_ANYDEPTH), cv2.COLOR_GRAY2RGB))
    for l in sorted(IMG_ROOT.rglob("captured_left_ir.png"))
    for r in [l.parent / "captured_right_ir.png"]
    if r.exists()
]
idx = 0
# while True:                              
left_ir, right_ir = left_right_pairs[idx % len(left_right_pairs)]
t0 = time.time()
inference_stereo(model, left_ir, right_ir, pred_mode='bidir', bidir_verify_th=1)
t1 = time.time()
logger.info(f"[ARM disk-read  cost {(t1-t0)*1000:.0f} ms")
idx += 1    