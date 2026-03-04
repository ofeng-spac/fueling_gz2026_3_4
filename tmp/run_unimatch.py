from stereo_matcher import inference_stereo
from fueling.stereo_matcher.unimatch.UniMatchStereo import UniMatchStereo
from fueling.stereo_matcher import save_disparity_map
import cv2
from fueling.stereo_matcher import save_disparity_map

# stereo_matcher = StereoMatcherService('unimatch', '/data/stereo_weights/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth', 1)
stereo_matcher = UniMatchStereo('/data/stereo_weights/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth')
left = cv2.imread('left.png')
right = cv2.imread('right.png')
result = inference_stereo(stereo_matcher, left, right, pred_mode='bidir', bidir_verify_th=1.0)
save_disparity_map(result, './')