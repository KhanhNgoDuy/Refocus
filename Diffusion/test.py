import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from pathlib import Path

from pipeline_camera import build_model
from dataset import MyDataset


def load_pretrained(checkpoint, gpu_id):
    model = build_model(rank=gpu_id)
    model.unet.load_state_dict(torch.load(checkpoint, weights_only=True))
    return model

    
def process_camera(cam):
    src = cam["src"]
    tgt = cam["tgt"]
    cam_out = torch.tensor([src, tgt])
    cam_out = cam_out / cam_out.max()
    return cam_out


def prepare_mask_and_masked_image(image, mask):
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def test_transform(size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomCrop(size),
        ]
    )
    return transform
    

if __name__ == "__main__":
    checkpoint = "checkpoints/camera/models/unet_last.pt"
    root = "../datasets"
    save_path = Path("output")
    save_path.mkdir(parents=True, exist_ok=True)
    
    dataset = MyDataset(root)
    
    ######################## example ########################
    # img_idx = "00000"
    # mask = f"{img_idx}.alpha.png"
    # src = f"{img_idx}.src.jpg"
    # tgt = f"{img_idx}.tgt.jpg"
    # depth = f"{img_idx}.depth.jpg"
    
    # transform = test_transform(512)
    batch = dataset[0]
    masked = batch["masked"].unsqueeze(0)
    src = batch["src"].unsqueeze(0)
    tgt = batch["tgt"].unsqueeze(0)
    depth = batch["depth"].unsqueeze(0)
    cam = batch["cam"].unsqueeze(0)
    #########################################################
    cam_scale = (cam[0][0] / cam[0][1]).item()
    cam_scale = round(cam_scale, 4)   
    # if > 1, then the model does blurring
    # if < 1, then the model does super-resolution
    # if = 1, do nothing
    
    ### inference
    model = load_pretrained(checkpoint, 0) 
    out = model(src, masked, depth, cam, use_progress_bar=False)['images'][0]
    
    idx = 0
    TF.to_pil_image(src.squeeze(0)).save(save_path / f"{idx}-src.png")
    TF.to_pil_image(tgt.squeeze(0)).save(save_path / f"{idx}-tgt.png")
    out.save(save_path / f"{idx}-output-{cam_scale}.png")