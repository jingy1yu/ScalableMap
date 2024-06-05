# ScalableMap
The official repository for ScalableMap (CoRL 2023)

## Environment Preparation
```shell
conda create -n smap python=3.8 -y
conda activate smap
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c omgarcia gcc-5
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install timm

cd ./mmdetection3d
python setup.py develop
cd ..

pip install -r requirement.txt
```
## Data Preparation
### Nuscenes
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```
**Folder structure**
```
ScalableMap
├── mmdetection3d/
├── projects/
├── tools/
├── ckpts/
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```

### 2.Eval/FPS Test/Train/Visualization
#### Eval
Evaluate ScalableMap with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/scalablemap/scalablemap_r50_110e_l60.py ./ckpts/epoch_110.pth 1
```

#### FPS test
```
./tools/maptr/benchmark.py ./projects/configs/scalablemap/scalablemap_r50_110e_l60.py --checkpoint ./ckpts/epoch_110.pth
```

#### Train
Train ScalableMap with 8 GPUs
```
./tools/dist_train.sh ./projects/configs/scalablemap/scalablemap_r50_110e_l60.py 8
```

#### Visualization
```
./tools/scalablemap/gt_pred_comparison.py ./projects/configs/scalablemap/scalablemap_r50_110e_l60.py 8
```

## Acknowledgements
ScalableMap is based on mmdetection3d. It's also inspired by the following outstanding contributions to the open-source community: BEVFormer, HDMapNet, VectorMapNet, MapTR.

## Citation
If our code or models help your work, , please consider citing the following bibtex:
```
@inproceedings{yu2023scalablemap,
  title={ScalableMap: Scalable Map Learning for Online Long-Range Vectorized HD Map Construction},
  author={Yu, Jingyi and Zhang, Zizhao and Xia, Shengfu and Sang, Jizhang},
  booktitle={Conference on Robot Learning},
  pages={2429--2443},
  year={2023},
  organization={PMLR}
}
```
