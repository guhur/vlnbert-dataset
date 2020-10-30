# vlnbert-dataset

Precompute the dataset of VLN BERT as described in the paper:


Improving Vision-and-Language Navigation with Image-Text Pairs from the Web.
Arjun Majumdar, Ayush Shrivastava, Stefan Lee, Peter Anderson, Devi Parikh, Dhruv Batra

# 0. Clone this repository with its submodule

```bash
git clone --recursive https://github.com/guhur/vlnbert-dataset
```

# 1. Prepare images from Matterport3D
In the folder `data/`, download the folders `undistorted_camera_parameters` and `matter_skybox_imags` for every scan.

# 2. Install Matterport3DSimulator

# 3. Download precomputed model in this folder
[Link here](https://awma1-my.sharepoint.com/personal/yuz_l0_tn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuz%5Fl0%5Ftn%2FDocuments%2Fshare%2Fbua%2Dpytorch%2Fckpts%2Fbua%2Dcaffe%2Dfrcn%2Dr101%5Fwith%5Fattributes%2Epth&parent=%2Fpersonal%2Fyuz%5Fl0%5Ftn%2FDocuments%2Fshare%2Fbua%2Dpytorch%2Fckpts&originalPath=aHR0cHM6Ly9hd21hMS1teS5zaGFyZXBvaW50LmNvbS86dTovZy9wZXJzb25hbC95dXpfbDBfdG4vRWFYdkNDM1dqdGxMdnZFZkxyM29hOFVCTEEyMXRjTGg0TDhZTGJZWGw2amdqZz9ydGltZT1pekxwY05KODJFZw)

# 4. Install dependencies

```bash
pip install -r requirements.txt
cd bottom-up-attention.pytorch/detectron2
pip install -e .
cd ../..
```

# 5. Launch code

It can take a few hours.

```bash
export PYTHONPATH=$PWD/Matterport3DSimulator/build/
python precompute_updown_img_features.py
```
