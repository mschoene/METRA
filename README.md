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
python tests/main.py --dual_dist s2_from_s

# METRA (with Euclidean distance) on state-based Ant (2-D skills)
python tests/main.py --dual_dist l2

# LSD on state-based Ant (2-D skills)
python tests/main.py --dual_reg 0 --spectral_normalization 1

# TEST time
python tests/main.py --dual_dist s2_from_s --cp_path exp/Debug/sd000_1718017287_ant_metra/
```

### Extra examples

```
# DADS on state-based Ant (2-D skills)
python tests/main.py --algo dads --inner 0 --unit_length 0 --dual_reg 0 --dim_option 2

# DIAYN on state-based Ant (2-D skills)
python tests/main.py --algo metra --inner 0 --unit_length 0 --dual_reg 0

# METRA on state-based HalfCheetah (16 skills)
python tests/main.py --run_group Debug --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 1 --dim_option 16

# METRA on pixel-based Quadruped (4-D skills)
python tests/main.py --run_group Debug --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 1 --sample_cpu 0

# METRA on pixel-based Humanoid (2-D skills)
python tests/main.py --run_group Debug --env dmc_humanoid --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 1 --sample_cpu 0

# METRA on pixel-based Kitchen (24 skills)
python tests/main.py --run_group Debug --env kitchen --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0
```
