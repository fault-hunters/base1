from torch.utils.data import DataLoader
from .img_dataset import ImagePairCSVDataset

def get_img_loader(csv_path, trn_transform, batch_size, **kwargs):
    pair_dset = ImagePairCSVDataset(
        csv_path = csv_path,      # csv_path
        root_dir=".",               # 필요 없으면 None
        transform=trn_transform,    # MX-Font에서 쓰는 transform 재사용
    )
    pair_loader = DataLoader(
        pair_dset,
        batch_size=batch_size,
        shuffle=kwargs.get("shuffle", True),
        num_workers=kwargs.get("num_workers", 0),
    )
    return pair_dset, pair_loader