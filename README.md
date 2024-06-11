# METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

This repository contains the official implementation of **METRA: Scalable Unsupervised RL with Metric-Aware Abstraction**.
The implementation is based on
[Lipschitz-constrained Unsupervised Skill Discovery](https://github.com/seohongpark/LSD).

Visit [our project page](https://seohong.me/projects/metra/) for more results including videos.

## Requirements
- Python 3.8

## Installation

```
conda create --name metra python=3.10.12
conda activate metra
apt install cmake
pip install -r requirements.txt --no-deps
pip install swig
pip install -e .
pip install -e garaged
pip install --upgrade cloudpickle
pip install --upgrade joblib
```

## Examples

```
# METRA on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2

# LSD on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --dual_reg 0 --spectral_normalization 1 --discrete 0 --dim_option 2

# DADS on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo dads --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

# DIAYN on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

# METRA on state-based HalfCheetah (16 skills)
python tests/main.py --run_group Debug --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 1 --dim_option 16

# METRA on pixel-based Quadruped (4-D skills)
python tests/main.py --run_group Debug --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 1 --sample_cpu 0

# METRA on pixel-based Humanoid (2-D skills)
python tests/main.py --run_group Debug --env dmc_humanoid --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 1 --sample_cpu 0

# METRA on pixel-based Kitchen (24 skills)
python tests/main.py --run_group Debug --env kitchen --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0


# TEST time
python tests/main.py --run_group Debug --env ant_nav_prime --max_path_length 8 --dim_option 2 --num_random_trajectories 48 --seed 0 --normalizer_type off --use_gpu 0 --traj_batch_size 8 --n_parallel 1 --algo sac --n_epochs_per_eval 100 --n_thread 1 --model_master_dim 1024 --n_epochs_per_log 25 --n_epochs_per_save 0 --n_epochs_per_pt_save 0 --n_epochs_per_pkl_update 0 --eval_record_video 1 --n_epochs 501 --discrete 0 --sac_discount 0.99 --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50  --sac_max_buffer_size 1000000 --common_lr 0.0001 --trans_minibatch_size 256 --goal_range 7.5 --alpha 0.1 --cp_multi_step 25 --downstream_reward_type esparse --downstream_num_goal_steps 50 --cp_path exp/Debug/sd000_1718017287_ant_metra/option_policy9000.pt --cp_path_idx 0 --cp_unit_length 1