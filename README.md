
# WordLenSpotter

This is the official implementation of Paper: [Word length-aware text spotting: Enhancing detection and recognition in dense text image](https://arxiv.org/abs/2312.15690). 

## Models
[WordLenSpotter-MIXTRAIN [config]](https://github.com/unxiaohao/WordLenSpotter/blob/main/projects/WordLenSpotter/configs/WordLenSpotter-mixtrain.yaml) \| [model_Google Drive](https://drive.google.com/file/d/1oI8fSImCfIJ7g3w1bwscWnsM16mTNhP8/view?usp=drive_link) 

## Installation
- Python=3.8
- PyTorch=1.8.0, torchvision=0.9.0, cudatoolkit=11.1
- OpenCV for visualization

## Steps
1. Install the repository (we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda create -n WordLenSpotter python=3.8 -y
conda activate WordLenSpotter
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python
pip install scipy
pip install shapely
pip install rapidfuzz
pip install timm
pip install Polygon3
git clone https://github.com/unxiaohao/WordLenSpotter.git
cd WordLenSpotter
python setup.py build develop
```

2. dataset path
```
datasets
|_ dstd1500
|  |_ train_images
|  |_ test_images
|  |_ dstd1500_test.json
|  |_ dstd1500_train.json
|  |_ weak_voc_new.txt
|  |_ weak_voc_pair_list.txt
|_ evaluation
|  |_ test_gt.zip
```
Downloaded images
- The dense text spotting dataset (DSTD1500) in real reading scenarios can be downloaded [here](https://drive.google.com/file/d/1qajTH8h7BZaqdeKvzYQeRskRVNvLzldp/view?usp=drive_link).
- Sample dataset images <img src="https://github.com/unxiaohao/WordLenSpotter/blob/main/demo/datasets_sample.png" style="zoom: 50%;" />

You can also prepare your custom dataset following the example scripts.
[[example scripts]](https://drive.google.com/file/d/1eb7g2v0NkjlICYdiKbWRe2bjPj-bxree/view?usp=drive_link)

## Usage
### Preparation
1. To evaluate on DSTD1500, first download the zipped [annotations](https://drive.usercontent.google.com/download?id=1NnFu_x39ZvOc9Yn4ZipSMvZDySIz8jea&export=download&authuser=0&confirm=t&uuid=7f0a35b7-6813-419d-9783-3784e0c791ff&at=APZUnTXpmtOKqW6YtKAwPvW-KUVF:1709107009429).
### Training
1. Pretrain WordLenSpotter

```
python projects/WordLenSpotter/train_net.py \
  --num-gpus 8 \
  --config-file projects/WordLenSpotter/configs/WordLenSpotter-pretrain.yaml
```

2. Fine-tune model on the mixed real dataset

```
python projects/WordLenSpotter/train_net.py \
  --num-gpus 8 \
  --config-file projects/WordLenSpotter/configs/WordLenSpotter-mixtrain.yaml
```
### Fine-tune
1. Fine-tune model

```
python projects/WordLenSpotter/train_net.py \
  --num-gpus 8 \
  --config-file projects/WordLenSpotter/configs/WordLenSpotter-WordLenSpotter-finetune-dstd1500.yaml
```
### Evaluation
1. Evaluate WordLenSpotter

```
python projects/WordLenSpotter/train_net.py \
  --config-file projects/WordLenSpotter/configs/WordLenSpotter-finetune-dstd1500.yaml \
  --eval-only MODEL.WEIGHTS ./output/FINETUNE20K/model_final.pth
```
### Visualize
1. Visualize the detection and recognition results

```
python demo/demo.py \
  --config-file projects/WordLenSpotter/configs/WordLenSpotter-finetune-dstd1500.yaml \
  --input input1.jpg \
  --output ./output \
  --confidence-threshold 0.4 \
  --opts MODEL.WEIGHTS ./output/FINETUNE20K/model_final.pth
```

## Acknowlegement
This project is based on [Adelaidet](https://github.com/aim-uofa/AdelaiDet), [Detectron2](https://github.com/facebookresearch/detectron2) and [SwinTextSpotter](https://github.com/mxin262/SwinTextSpotter).

## Citation

If our paper helps your research, please cite it in your publications:

```BibText
@article{wang2023word,
  title={Word length-aware text spotting: Enhancing detection and recognition in dense text image},
  author={Wang, Hao and Zhou, Huabing and Zhang, Yanduo and Lu, Tao and Ma, Jiayi},
  journal={arXiv preprint arXiv:2312.15690},
  year={2023}
}
```
