local default = import "../default_config.jsonnet";
local eye_hand_matrix = import "eye_hand_matrix.json";
local camera_intrinsics = import "camera_intrinsics.json";

{
  device: "arm1",
  camera: default.camera{
    exposure: 2000,
    gain: 4000,
    camera_serial : "AYZT35300CJ"
  },

  stereo_matcher: default.stereo_matcher,

  point_cloud: default.point_cloud,

  robot: default.robot {
    delay: 0,
    ip: "192.168.0.109",
    target_pot: "pot5",

    // 基础姿态
    init_pose: [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    capture_pose: [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    fueling_pose: [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
  } + eye_hand_matrix + camera_intrinsics,

  minima: default.minima + {
      left_ir_path: 'fueling/initialization/orbbec_output/Flood_light/arm1/' + $.robot.target_pot + '/captured_left_ir.png',
      right_ir_path: 'fueling/initialization/orbbec_output/Flood_light/arm1/' + $.robot.target_pot + '/captured_right_ir.png',
  },
}