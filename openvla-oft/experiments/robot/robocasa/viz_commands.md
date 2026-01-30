 python visualize_point_tracks_new_lines_subsample.py   --demo_dir /data/human_demon_action/scratch2/robocasa/temp_v3/demo_1   --save /data/human_demon_action/pointcloud_dump/demo_1_tracks_lines6.mp4   --view_mode camera2d   --overlay_video   --show_trails   --trail_len 16   --trail_alpha 0.9   --background_alpha 0.35   --track_style lines   --line_thickness 2   --alpha 0.8   --radius 2 --subsample_frac 0.4 --subsample_seed 11   --trail_color hotpink   --object_keyword "microwave"  --object_color "110,110,255"  --object_trail_rgb "0,255,255"

 #For demo files from /robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 created using 

 python ./generate_tracking_data_rgb_new.py \
    --dataset "/data/human_demon_action/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5" \
    --point_cloud_dir /data/human_demon_action/scratch2/robocasa/temp_v3 \
    --cube_half 0.6 \
    --anchor_body robot0_link0 \
    --max_track_points 1024 \
    --exclude_wall \
    --skip_mesh_save \
    --save_first_ply \
    --direction_anchor_body robot0_link0 \
    --direction_target_body gripper0_right_right_gripper \
    --direction_offset 0.5 \
    --recenter_points \
    --align_forward_to_neg_x \
    --save_video \
    --video_width 640 --video_height 480 --recenter_points \
    --camera_name robot0_agentview_right \
    --exclude_wall --exclude_table \
    --table_weight 0 \
    --robot_weight 50 --gripper_weight 50 \
    --keyword "microwave" --keyword_weight 20 \
