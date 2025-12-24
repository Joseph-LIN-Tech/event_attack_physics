# Event Attack

Official implementation of **“Adversarial Attacks on Event-Based Pedestrian Detectors: A Physical Approach”** (AAAI 2025).

---

## Download Checkpoints

Checkpoints are available via Google Drive:

- Google Drive: https://drive.google.com/drive/folders/1X1fINjuOPxh_sM_1Sv8Tu3xVGbNF_WhO?usp=drive_link

Please place the following files under `checkpoints/`:

- `best.pth`
- `gen_rvt-b.ckpt`

Example directory structure:

```text
event_attack_physics/
└── checkpoints/
    ├── best.pth
    └── gen_rvt-b.ckpt
```
where best.pth is the checkpoint of our model, and gen_rvt-b.ckpt is the checkpoint of the target detector [RVT](https://github.com/uzh-rpg/RVT).

---

## Download Datasets
Download train and test dataset of the digital attack: https://drive.google.com/drive/folders/1wYwMbDzgdrcFD1I3W0i9_qzQo0w0dJnJ?usp=sharing

## Easy Start

```bash
git clone https://github.com/Joseph-LIN-Tech/event_attack_physics.git
cd event_attack_physics
```

### test
```bash
 python test.py
```

1. The testing process uses the default configuration file located under Options/: test.yaml.

2. Please update the data paths 'data_simu_pose' and 'labeled_simu_pose' to the corresponding local directories on your machine.

3. To reproduce the evaluation results reported in the paper, you can set the metric (in line 122) to either 'seq_attack_success_rate' or 'AP'.

---

### train
```bash
 python train.py
```
1. The training process uses the default configuration file located under Options/: train.yaml.

### configuration files
1. The target detector configuration is stored in Options/rvt_config.yaml. If you would like to adjust the confidence threshold used in the paper, please modify the value of 'confidence_threshold' in rvt_config.yaml.




## Acknowledgement
Some of our code is based on the implementation of [LETGAN](https://github.com/iCVTEAM/LETGAN/tree/master), [RVT](https://github.com/uzh-rpg/RVT).

---

## Citation

If you find this repository useful, please cite our paper:

@inproceedings{lin2025adversarial,
  title={Adversarial Attacks on Event-Based Pedestrian Detectors: A Physical Approach},
  author={Lin, Guixu and Niu, Muyao and Zhu, Qingtian and Yin, Zhengwei and Li, Zhuoxiao and He, Shengfeng and Zheng, Yinqiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={5227--5235},
  year={2025}
}

---

## License

Please check the license files in this repository.

