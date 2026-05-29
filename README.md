<div align="center">

# ManiVID-3D: Generalizable View-Invariant Reinforcement Learning for Robotic Manipulation via Disentangled 3D Representations
**Zheng Li<sup>*</sup>**, **Pei Qu<sup>*</sup>**, **Yufei Jia<sup>*</sup>**, **Shihui Zhou**, **Haizhou Ge**, **Jiahang Cao**,  
**Jinni Zhou<sup>†</sup>**, **Guyue Zhou**, **Jun Ma**

<sup>*</sup> Equal Contribution &nbsp;&nbsp;&nbsp;
<sup>†</sup> Corresponding Author

[![arXiv](https://img.shields.io/badge/arXiv-2509.11125-b31b1b.svg)](https://arxiv.org/abs/2509.11125)
[![RAL](https://img.shields.io/badge/RAL-2026-blue.svg)](https://doi.org/10.1109/LRA.2026.3662647)
[![project](https://img.shields.io/badge/Project-Page-green.svg)](https://zheng-joe-lee.github.io/manivid3d/)



</div>

---

## :memo: Paper

This is a novel 3D visual RL architecture designed for strong robustness toward extreme viewpoint changes during manipulation tasks.

This work has been accepted to **IEEE Robotics and Automation Letters (RA-L) 2026**!

---

## :rocket: Getting Started

This release provides the full training and evaluation pipeline. An on-policy efficient parallel version will be released in a future update.

### Installation

Requires an NVIDIA GPU. MuJoCo rendering uses EGL (`MUJOCO_GL=egl`).

```bash
conda env create -f environments.yaml
conda activate mv3
conda install pytorch3d::pytorch3d
```

### Configuration

Before running experiments, update the following paths and settings as needed:

| Purpose | File | Key fields |
|---------|------|------------|
| ViewNet data output | `viewnet/collect_data.py` | `output_dir` |
| ViewNet training | `viewnet/config_viewnet.py` | `workspace_dir`, `task_name` |
| ViewNet testing | `viewnet/test_viewnet.py` | `checkpoint_path` |
| ManiVID-3D evaluation | `mani_eval.py` | `model_path` |
| Enable ViewNet at eval | `cfgs/camera_aug_config_pc.yaml` | `use_viewnet` |
| ViewNet checkpoint | `dmc.py` | `viewnet_path` |
| Task name | `scripts/collect_data.sh`, `scripts/train.sh`, `scripts/eval.sh` | `task_name` |

### Training

ViewNet and ManiVID-3D are modular and can be trained in either of the following orders:

**Option 1: Train independently, enable ViewNet at test time**

Train **ViewNet** and **ManiVID-3D** separately. ManiVID-3D can be trained without ViewNet. At evaluation, plug in a trained ViewNet checkpoint by setting `use_viewnet: true` in `cfgs/camera_aug_config_pc.yaml` and updating `viewnet_path` in `dmc.py`.

**Option 2: Pre-train ViewNet first, then train ManiVID-3D with frozen ViewNet**

First pre-train **ViewNet** to convergence. Then load the frozen ViewNet weights into the ManiVID-3D pipeline. In the current release, frozen ViewNet is applied during **evaluation** (via `use_viewnet` and `viewnet_path`); ManiVID-3D training itself runs without ViewNet by default.

#### ViewNet Pre-training

Before training **ViewNet**, configure the data output path in `viewnet/collect_data.py` and set `workspace_dir` / `task_name` in `viewnet/config_viewnet.py`. Also specify `task_name` in `scripts/collect_data.sh`.

Collect data:

```bash
bash scripts/collect_data.sh
```

After data collection, start training with:

```bash
python3 viewnet/train_viewnet.py
```

#### ManiVID-3D Training

To train the **ManiVID-3D** model, run:

```bash
bash scripts/train.sh
```

### Inference

#### ViewNet Testing

For initial verification of **ViewNet**, modify `checkpoint_path` in `viewnet/test_viewnet.py`, then run:

```bash
python3 viewnet/test_viewnet.py
```

#### ManiVID-3D Testing

To evaluate the **ManiVID-3D** model, modify `model_path` in `mani_eval.py`, then execute:

```bash
bash scripts/eval.sh
```

⚠️ Note: It is recommended to visually inspect the saved videos, as the recorded success rate may occasionally miss some successful trials.

---

## :books: Citation

If you find this work useful, please cite:

```bibtex
@article{li2025manivid3d,
  title={ManiVID-3D: Generalizable View-Invariant Reinforcement Learning for Robotic Manipulation via Disentangled 3D Representations},
  author={Li, Zheng and Qu, Pei and Jia, Yufei and Zhou, Shihui and Ge, Haizhou and Cao, Jiahang and Zhou, Jinni and Zhou, Guyue and Ma, Jun},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  volume={11},
  number={4},
  pages={4235-4242},
  doi={10.1109/LRA.2026.3662647},
}
```

---

## :heart: Acknowledgement

This codebase is built upon [Maniwhere](https://github.com/gemcollector/maniwhere) (CoRL 2024, *Learning to Manipulate Anywhere: A Visual Generalizable Framework For Reinforcement Learning*). We sincerely thank the authors for open-sourcing their code and their great contributions to the community.
