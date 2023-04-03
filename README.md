# WEAKLY SUPERVISED NUCLEI SEGMENTATION VIA INSTANCE LEARNING

## Description

This page contains the code of "WEAKLY SUPERVISED NUCLEI SEGMENTATION VIA INSTANCE LEARNING". 
This work has been accepted by *IEEE International Symposium on Biomedical Imaging (ISBI), 2022* as **Oral 
Presentation**. For more details, please refer to https://arxiv.org/abs/2202.01564.

## Run the code

### 0. Create environment under CUDA 10.2

```angular2html
conda create -n nucseg python=3.7 h5py
conda activate nucseg
pip install torch==1.4.0 torchvision==0.5.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip install -r requirements.txt
```

### 1. Train SPN

```angular2html
python main.py --id SPN --cfg network/exp/MO/SPN.yaml --gpu 1
```

### 2. Train IEN

```angular2html
python main.py --id IEN --cfg network/exp/MO/IEN.yaml --gpu 1
```

### 3. Model inference

```angular2html
python main.py --id IEN_infer --cfg network/exp/MO/IEN_infer.yaml --gpu 1
```

## Citation 
If you find this code helpful, please cite our work:

```angular2html
@article{liu2022weakly,
  title={Weakly Supervised Nuclei Segmentation via Instance Learning},
  author={Liu, Weizhen and He, Qian and He, Xuming},
  journal={arXiv preprint arXiv:2202.01564},
  year={2022}
}
```