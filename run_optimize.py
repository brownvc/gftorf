import os
from datetime import datetime

# FToRF Scenes (Trained on quads)
for scene_type, scene, total_num_views, zfar, lambda_tof, quad_scale, iterations in [
    # ("ftorf_synthetic_scenes", "sliding_cube",          60, 0.45, 1.0,  1.0, 60000), 
    # ("ftorf_synthetic_scenes", "arcing_cube",           60, 0.45, 1.0,  5.0, 60000), # This is dark
    # ("ftorf_synthetic_scenes", "acute_z_speed_test",    60, 0.45, 1.0,  1.0, 60000), 
    # ("ftorf_synthetic_scenes", "speed_test_texture",    60, 0.45, 1.0,  1.0, 60000), 
    # ("ftorf_synthetic_scenes", "speed_test_chair",      60, 0.45, 1.0,  1.0, 60000), 
    # ("ftorf_synthetic_scenes", "occlusion",             60, 0.45, 1.0,  1.0, 60000), 
    # ("ftorf_synthetic_scenes", "z_motion_speed_test",   60, 0.45, 1.0,  1.0, 60000), 
    # ("ftorf_real_scenes",      "pillow",                64, 0.45, 5.0,  1.0, 60000),
    ("ftorf_real_scenes",      "baseball",              60, 0.45, 5.0,  1.0, 60000), 
    # ("ftorf_real_scenes",      "fan",                   60, 0.45, 5.0,  1.0, 60000),
    # ("ftorf_real_scenes",      "jacks1",                68, 0.45, 1.0,  1.0, 20000), 
    # ("ftorf_real_scenes",      "target1",               68, 0.65, 1.0, 10.0, 20000), # dark
]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    iterations = 20000 
    base_args = [
        "--config", "configs/ftorf.json",
        "--seed", "42",
        "--debug",
        
        "-s", f"data/{scene_type}/{scene}",
        "--total_num_views", f"{total_num_views}",

        "--min_depth_fac", "0.01",
        "--max_depth_fac", f"{zfar}",

        "--iterations", f"{iterations}",
        "--position_lr_max_steps", f"{iterations}",
        "--densify_until_iter", f"{int(iterations*0.6)}",

        "--lambda_tof", f"{lambda_tof}",
        "--densify_grad_threshold", f"{0.0002*lambda_tof}",

        "--lambda_flow", "0.0008",
        "--quad_scale", f"{quad_scale}",
    ]

    if scene in ["target1"]:
        amp_div = 1000.0
        initial_amp = 0.5
    elif scene in ["jacks1"]:
        amp_div = 1000.0
        initial_amp = 0.1
    else:
        initial_amp = 0.02
        amp_div = 100.0
    lambda_dd = 0.0
    output_folder = f"output/{scene}_{timestamp}"
    control_args = [
        "-m", f"{output_folder}",
        
        "--initial_amplitude", f"{initial_amp}",

        "--feature_amp_lr_init", f"{0.0016/amp_div}",
        "--feature_amp_lr_final", f"{0.0016/amp_div}",
    ]
    args_str = ' '.join(base_args+control_args)
    # os.system(f'python train.py {args_str}')
    # os.system(f'python render.py -m {output_folder} --iteration {iterations}')

# ToRF Scenes (Trained on phasors)
for scene, total_num_views in [
    # ("cupboard",    30),
    # ("deskbox",     30),
    # ("studybook",   30),
    # ("copier",      30),
    # ("phonebooth",  30),
]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_args = [
        "--config", "configs/torf.json",
        "--seed", "42",
        
        "-s", f"data/torf_scenes/{scene}",
        "--total_num_views", f"{total_num_views}",

        "--min_depth_fac", "0.01",
        "--max_depth_fac", "0.55",

        "--debug",
        "--debug_interval", "100",
    ]

    iterations = 20000 
    lambda_flow = 0.0

    if scene in ["copier", "phonebooth"]:
        initial_amp = 0.5
        lambda_mlp_reg = 0.0
    else:
        initial_amp = 0.1
        lambda_mlp_reg = 0.05
    amp_div = 10.0

    output_folder = f"output/{scene}_{timestamp}"
    control_args = [
        "-m", f"{output_folder}",
        "--lambda_mlp_reg", f"{lambda_mlp_reg}",
        "--lambda_flow", f"{lambda_flow}",

        "--iterations", f"{iterations}",
        "--position_lr_max_steps", f"{iterations}",
        "--densify_until_iter", f"{iterations}",
        
        "--initial_amplitude", f"{initial_amp}",

        "--feature_amp_lr_init", f"{0.0016/amp_div}",
        "--feature_amp_lr_final", f"{0.0016/amp_div}",

        "--lambda_tof", "1.0",
        "--densify_grad_threshold", "0.0004",
    ]
    args_str = ' '.join(base_args+control_args)
    os.system(f'python train.py {args_str}')
    os.system(f'python render.py -m {output_folder} --iteration {iterations}')
