# base1
## MXFont
- https://github.com/clovaai/mxfont.git
- 가져올 network
    - style feature map
    - content feature map
- train
    ```
    python train.py cfgs/train.yaml
    ```
- test
    ```
    python eval.py cfgs/eval.yaml --weight path/to/gen_xxx.pth --vis_n 100
    ```
- inference
    ```
    python mxfont/char_comparison_api.py --weight mxfont/generator.pth --imgA path/to/a.png --imgB path/to/b.png
    ```

## Image Generating
- generating_tar_img.py
```
python "new_api/generating_tar_img.py"
```

## seg+crop+pad
- img_reconstruct.py
```
python "seg+crop+pad/no-time-to-train/img_reconstruct.py"
```
    - input : data.csv(ref_path, tar_path) - pair data
- before start
    - terminal
    ```
    cd seg+crop+pad/no-time-to-train
    mkdir checkpoints
    mkdir checkpoints/dinov2
    
    # linux
    wget -q -P ./checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    wget -q -P ./checkpoints/dinov2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

    # window
    curl -L -o checkpoints/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    curl -L -o checkpoints/dinov2_vitl14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
    ```