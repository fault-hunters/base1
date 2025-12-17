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