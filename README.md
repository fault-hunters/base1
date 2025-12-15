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
    python eval.py cfgs/train.yaml --weight path/to/gen_xxx.pth
    ```