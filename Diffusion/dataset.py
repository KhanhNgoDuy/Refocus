import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import re
from time import perf_counter_ns as pcn


class MyDataset(Dataset):
    def __init__(self, 
                 root="/mnt/HDD1/tuong/workspace/khanh/courses/image_processing/final-project/datasets", 
                 size=512, 
                 train=True):
        if train:
            self.root = Path(root) / "train"
        else:
            self.root = Path(root) / "val"
        self.train = train

        self.metadata = self.load_metadata(self.root / "meta.txt")
        self.indices = self.get_all_indices(self.root)

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.RandomCrop(size),
            ]
        )
    
    def __getitem__(self, index):
        index = self.indices[index]
        src = Image.open(self.root / f"{index}.src.jpg").convert("RGB")         # blurred image
        
        if not self.train:
            cam = self.metadata[index]      # dict, key: ["src", "tgt", "disparity"]
            cam_out = self.process_camera(cam)
            src = self.transforms(src)
            return {"src": src, "cam": cam_out}
        
        tgt = Image.open(self.root / f"{index}.tgt.jpg").convert("RGB")         # clear image
        alpha = Image.open(self.root / f"{index}.alpha.png").convert("L")       # mask
        depth = Image.open(self.root / f"{index}.depth.jpg").convert("RGB")     # synthesized depth

        cam = self.metadata[index]      # dict, key: ["src", "tgt", "disparity"]
        cam_out = self.process_camera(cam)

        src = self.transforms(src)
        tgt = self.transforms(tgt)
        mask = 1 - self.transforms(alpha)
        depth = self.transforms(depth)

        mask, masked = prepare_mask_and_masked_image(src, mask)

        out = {
            "src": src,
            "tgt": tgt,
            "masked": masked,
            "depth": depth,
            "cam": cam_out
        }

        # for key, value in out.items():
        #     print(f"{key}: {value.shape}")
        #     image_data = value.permute(1, 2, 0).numpy()
        #     image_data = (image_data * 255).astype('uint8')
        #     image = Image.fromarray(image_data)
        #     image.save(f'{key}.png')
        # exit()

        return out
        
    def __len__(self):
        return len(self.indices)
    
    def process_camera(self, cam):
        src = cam["src"]
        tgt = cam["tgt"]
        cam_out = torch.tensor([src, tgt])
        cam_out = cam_out / cam_out.max()
        return cam_out
    
    def get_all_indices(self, root):
        indices = []
        for image_path in Path(root).iterdir():
            if image_path.suffix not in [".png", ".jpg"]:
                continue
            indices.append(image_path.name.split('.')[0])
        indices = list(set(indices)) 
        indices.sort(key=lambda x: int(x)) 
        return indices
    
    def load_metadata(self, path):
        df = pd.read_csv(path, header=None, dtype=str)
        metadata = {}

        for row in df.itertuples(index=False):
            idx, cam_src, cam_tgt, disparity = row
            pattern = r"f(.*?)BS"
            f_src = re.search(pattern, cam_src).group(1)
            f_tgt = re.search(pattern, cam_tgt).group(1)
            metadata[idx] = {
                "src": float(f_src),
                "tgt": float(f_tgt),
                "disparity": float(disparity)
            }
        return metadata


def prepare_mask_and_masked_image(image, mask):
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    masked_image = image * (mask < 0.5)

    return mask, masked_image


if __name__ == "__main__":
    # path = "/mnt/HDD1/tuong/workspace/khanh/courses/image_processing/final-project/datasets/train/meta.txt"
    # df = pd.read_csv(path, header=None)
    # meta = {key: value for (key, value) in zip(
    #     df[0],                      # key
    #     zip(df[1], df[2], df[3])    # value
    #     )
    # }
    # meta = {}

    # for row in df.iterrows():
    #     row = row[1]
    #     idx, cam_src, cam_tgt, disparity = row
    #     pattern = r"f(.*?)BS"
    #     f_src = re.search(pattern, cam_src).group(1)
    #     f_tgt = re.search(pattern, cam_tgt).group(1)
    #     meta[idx] = {
    #         "src": f_src,
    #         "tgt": f_tgt,
    #         "disparity": disparity
    #     }

    #     print(type(f_src))

    # for key, value in meta.items():
    #     print(key, value)

    ds = MyDataset()
    for d in ds:
        pass