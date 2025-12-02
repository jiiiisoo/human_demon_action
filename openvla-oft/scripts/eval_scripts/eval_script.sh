#!/bin/bash
#SBATCH --job-name=evpw10-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err



export PYTHONPATH=/home/jisookim/openvla-oft/LIBERO:$PYTHONPATH

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /mnt/data/jisookim/openvla_finetune/track_512_use_input_weight10/openvla-7b+libero_goal_no_noops+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug--track_512_weight10_use_input-parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--proprio_state--40000_chkpt \
  --task_suite_name libero_goal \
  --num_images_in_input 1 \
  --local_log_dir /home/jisookim/openvla-oft/experiments/512_point_track_w10/logs_40000_chkpt \
  --rollout_dir /home/jisookim/openvla-oft/rollouts/512_point_track_w10_40000_chkpt