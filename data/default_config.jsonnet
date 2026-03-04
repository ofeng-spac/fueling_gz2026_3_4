{
  camera: {
    exposure: 1700,
    gain: 1000
  },

  stereo_matcher: {
    max_parallel: 2,
    method: 'unimatch',
    pred_mode: 'left',
    bidir_verify_th: 1,
    unimatch: {
      model_path: '/data/stereo_weights/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth',
    },
    raft:{
      model_path: '/data/stereo_weights/RAFT_Stereo/raftstereo-middlebury.pth',
    },
    defom:{
      model_path: '/data/stereo_weights/DEFOMStereo/defomstereo_vits_rvc.pth',
    },
    bridgedepth:{
      model_path: '/data/stereo_weights/BridgeDepth/bridge_rvc_pretrain.pth',
    },
  },
  minima : {
    model_path: '/data/minima_weights/minima_roma.pth',
    projection_target: 'left', // 'left' or 'right'
    left_ir_path: '',
    right_ir_path: '',
    point_arr_left_path: '',
    indices_left_path: '',
    point_arr_right_path: '',
    indices_right_path: '',
  },
  point_cloud: {
    voxel_size: 0.8,
    radius: 1,
    min_neighbors: 10,
    remove_outliers: true,
    cut_box: [
      350,
      350,
      200
    ]
  },

  robot: {
    strategy: "method2",
    debug_mode: false,
    move_command: "movel",
    control_port: 30001,
    req_port: 40011,
    movel_params: {
      "a": 1.0,
      "r": 0.0,
      "t": 0.0,
      "v": 0.2
    },
    movej_params: {
      "a": 1.0,
      "r": 0.0,
      "t": 0.0,
      "v": 0.2
    },
    pos_tol: 0.01,
    rot_tol: 0.01,
    check_interval: 0.03
  }
}
