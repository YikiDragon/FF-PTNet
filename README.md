# FF-PTNet
This is an implementation of the paper "Dual-Side Feature Fusion 3D Pose Transfer".
Please check our paper for more details.
## Source Code
```bash
git clone https://github.com/YikiDragon/FF-PTNet.git
cd FF-PTNet
```
## Requirements
- python3.7
- numpy
- tqdm
- natsort
- [trimesh](https://trimsh.org/index.html)==3.20.2
- pytorch==1.13.0
- [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
```bash
conda create -n ff_ptnet python=3.7
conda activate ff_ptnet
# install pytorch and pyg
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
# install trimesh
conda install -c conda-forge trimesh
# install other packages
conda install -c conda-forge trimesh
pip install tqdm natsort
```
Our code has been tested with Python 3.7, Pytorch1.13.0, CUDA 11.7 on Ubuntu 20.04.

## Dataset and pre-trained model
We use [SMPL](https://smpl.is.tue.mpg.de/) as the human mesh data, and we use [SMAL](https://smal.is.tue.mpg.de/) as the animal mesh data. please download them and pre-trained model [here](https://drive.google.com/drive/folders/1uP6H0j7mUJ6utgvXxpT-2rn4EYhJ3el5?usp=sharing).

## Demo
Run the following command in the root directory of the code to create the necessary folders:
```bash
mkdir datasets
```
Unzip the downloaded datasets and move the folders `smpl` and `smal` to the datasets directory.
For SMPL(human), run the following command directly:
```bash
python demo.py
```
Then, the generated results will be in the `results` folder.
| Source mesh | Target mesh | Output mesh | Ground truth mesh |
| :----: | :----: | :----: | :----: 
| `po.obj` | `id.obj` | `out.obj` | `gt.obj` |

