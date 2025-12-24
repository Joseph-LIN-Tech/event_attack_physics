# Event Attack

Official implementation of **“Adversarial Attacks on Event-Based Pedestrian Detectors: A Physical Approach”** (AAAI 2025).

---

## Download Checkpoints and Datasets

#### Checkpoints:

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
---

#### Digital Attack Dataset:

-- Google Drive: https://drive.google.com/drive/folders/1wYwMbDzgdrcFD1I3W0i9_qzQo0w0dJnJ?usp=sharing

---

## Easy Start

```bash
git clone https://github.com/Joseph-LIN-Tech/event_attack_physics.git
cd event_attack_physics
```

---

### Test

```bash
python test.py
```

1. The test process uses the default configuration files located under Options/: test.yaml 

2. Please update the data paths 'data_simu_pose' and 'labeled_simu_pose' in Options/test.yaml to the corresponding local directories on your machine.


3. You can set the metrics as 'seq_attack_success_rate' or 'AP' seperately to get the evaluation value in the paper.

---


### Train
```bash
python train.py
```

1. The training process uses the default configuration files located under Options/: train.yaml 

2. Please update the data paths 'data_simu_pose' and 'labeled_simu_pose' in Options/train.yaml to the corresponding local directories on your machine.

---


## Acknowledgement

Some of our code is based on the implementation of [LETGAN](https://github.com/iCVTEAM/LETGAN/tree/master), and [RVT](https://github.com/uzh-rpg/RVT).


---

## Citation

If you find this repository useful, please cite our paper:

@inproceedings{lin2025adversarial,
  title={Adversarial Attacks on Event-Based Pedestrian Detectors: A Physical Approach},\
  author={Lin, Guixu and Niu, Muyao and Zhu, Qingtian and Yin, Zhengwei and Li, Zhuoxiao and He, Shengfeng and Zheng, Yinqiang}, \
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, \
  year={2025}
}

---

## License

Please check the license files in this repository.
