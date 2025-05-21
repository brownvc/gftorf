import os


################################# Synthetic FToRF Scenes
# Sliding_cube
args = [
    "-m", "output/pretrained_models/ftorf_synthetic_scenes/sliding_cube",
    "--big_motion_quantile", "0.9",
    "--z_distr_quantile", "0.45",
    "--opacity_quantile", "0.9",
    "--small_size_quantile", "0.1",
    "--big_size_quantile", "0.9",
    "--iteration", "60000",
    "--scene_name", "sliding_cube",
    "--baseline_start_fid", "8",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# occlusion
args = [
    "-m", "output/pretrained_models/ftorf_synthetic_scenes/occlusion",
    "--big_motion_quantile", "0.9",
    "--z_distr_quantile", "0.45",
    "--opacity_quantile", "0.8",
    "--small_size_quantile", "0.2",
    "--big_size_quantile", "1.0",
    "--iteration", "30000",
    "--scene_name", "occlusion",
    "--baseline_start_fid", "8",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# z_motion_st
args = [
    "-m", "output/pretrained_models/ftorf_synthetic_scenes/z_motion_speed_test",
    "--big_motion_quantile", "0.25",
    "--z_distr_quantile", "0.65",
    "--opacity_quantile", "0.5",
    "--small_size_quantile", "0.1",
    "--big_size_quantile", "0.9",
    "--iteration", "30000",
    "--scene_name", "z_motion_speed_test",
    "--baseline_start_fid", "8",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# st_texture
args = [
    "-m", "output/pretrained_models/ftorf_synthetic_scenes/speed_test_texture",
    "--big_motion_quantile", "0.2",
    "--z_distr_quantile", "0.9",
    "--opacity_quantile", "0.85",
    "--small_size_quantile", "0.3",
    "--big_size_quantile", "1.0",
    "--iteration", "30000",
    "--scene_name", "speed_test_texture",
    "--baseline_start_fid", "8",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# st_chair
args = [
    "-m", "output/pretrained_models/ftorf_synthetic_scenes/speed_test_chair",
    "--big_motion_quantile", "0.1",
    "--z_distr_quantile", "0.75",
    "--opacity_quantile", "0.1",
    "--small_size_quantile", "0.0",
    "--big_size_quantile", "0.5",
    "--iteration", "30000",
    "--scene_name", "speed_test_chair",
    "--baseline_start_fid", "8",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# arcing_cube
args = [
    "-m", "output/pretrained_models/ftorf_synthetic_scenes/arcing_cube",
    "--big_motion_quantile", "0.7",
    "--z_distr_quantile", "0.99",
    "--opacity_quantile", "0.01",
    "--small_size_quantile", "0.01",
    "--big_size_quantile", "0.99",
    "--iteration", "30000",
    "--scene_name", "arcing_cube",
    "--baseline_start_fid", "8",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")


# acute_z_st
args = [
    "-m", "output/pretrained_models/ftorf_synthetic_scenes/acute_z_speed_test",
    "--big_motion_quantile", "0.6",
    "--z_distr_quantile", "0.85",
    "--opacity_quantile", "0.5",
    "--small_size_quantile", "0.01",
    "--big_size_quantile", "0.99",
    "--iteration", "30000",
    "--scene_name", "acute_z_speed_test",
    "--baseline_start_fid", "8",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

################################# Real FToRF Scenes

# Baseball
args = [
    "-m", "output/pretrained_models/ftorf_real_scenes/baseball",
    "--big_motion_quantile", "0.9",
    "--z_distr_quantile", "0.65",
    "--opacity_quantile", "0.1",
    "--small_size_quantile", "0.01",
    "--big_size_quantile", "0.99",
    "--iteration", "60000",
    "--scene_name", "baseball",
    "--baseline_start_fid", "0",
    "--baseline_end_fid", "52",
    "--baseline_duration", "2.0",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# Pillow
args = [
    "-m", "output/pretrained_models/ftorf_real_scenes/pillow",
    "--big_motion_quantile", "0.8", # sparse 0.9
    "--z_distr_quantile", "0.45",
    "--opacity_quantile", "0.6", # sparse 0.9
    "--small_size_quantile", "0.1",
    "--big_size_quantile", "0.9",
    "--iteration", "60000",
    "--scene_name", "pillow",
    "--baseline_start_fid", "0",
    "--baseline_end_fid", "52",
    "--baseline_duration", "2.0",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# target1
args = [
    "-m", "output/pretrained_models/ftorf_real_scenes/target1",
    "--big_motion_quantile", "0.9",
    "--z_distr_quantile", "0.45",
    "--opacity_quantile", "0.6",
    "--small_size_quantile", "0.1",
    "--big_size_quantile", "0.9",
    "--iteration", "20000",
    "--scene_name", "target1",
    "--baseline_start_fid", "0",
    "--baseline_end_fid", "52",
    "--baseline_duration", "2.0",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# jacks1
args = [
    "-m", "output/pretrained_models/ftorf_real_scenes/jacks1",
    "--big_motion_quantile", "0.9",
    "--z_distr_quantile", "0.45",
    "--opacity_quantile", "0.6",
    "--small_size_quantile", "0.01",
    "--big_size_quantile", "0.99",
    "--iteration", "60000",
    "--scene_name", "jacks1",
    "--baseline_start_fid", "0",
    "--baseline_end_fid", "52",
    "--baseline_duration", "2.0",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

# fan
args = [
    "-m", "output/pretrained_models/ftorf_real_scenes/fan",
    "--big_motion_quantile", "0.75",
    "--z_distr_quantile", "0.35",
    "--opacity_quantile", "0.01",
    "--small_size_quantile", "0.01",
    "--big_size_quantile", "0.5",
    "--iteration", "60000",
    "--scene_name", "fan",
    "--baseline_start_fid", "0",
    "--baseline_end_fid", "52",
    "--baseline_duration", "2.0",
]
args_str = ' '.join(args)
os.system(f"python render_ftorf_viz_traj.py {args_str}")

################################# ToRF Scenes

for scene in [
    "copier", 
    "cupboard", 
    "deskbox", 
    "phonebooth",
    "studybook",
]:
    output_folder = "output/pretrained_models/torf_scenes/copier"
    os.system(f'python render.py -m {output_folder} --iteration 30000')
