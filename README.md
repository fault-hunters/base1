# base1
## MXFont
- https://github.com/clovaai/mxfont.git
- 가져올 network
    - style feature map
    - content feature map
- train
    - 단일 gpu
        ```
        python train.py cfgs/train.yaml
        ```
        - train.yaml에서 use_ddp = False
    - 멀티 gpu
        - train.yaml에서 use_ddp = True
        - 터미널에서 gpu 현황 확인하고 사용하기
            ```
            nvidia-smi # 한번만 보기
            watch -n 1 nvidia-smi # 실시간으로 1초마다 자동 출력
            ```
        - N수 정할때는 사용가능한 gpu개수 확인
            ```
            # python
            import torch

            print("CUDA available:", torch.cuda.is_available()) # gpu사용가능 여부
            print("GPU count:", torch.cuda.device_count()) # 보이는 gpu개수
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