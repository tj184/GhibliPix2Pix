import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform

input_image_path = "test_image.png"
output_path = "generated_ghibli.png"

sys.argv = [
    "generate_image.py",
    "--dataroot", "./datasets/ghibliify",
    "--name", "ghibli_pix2pix",
    "--model", "pix2pix",
    "--direction", "AtoB",
    "--checkpoints_dir", "./checkpoints",
    "--preprocess", "resize_and_crop",
    "--load_size", "256",
    "--crop_size", "256",
    "--no_dropout",
    "--serial_batches",
    "--gpu_ids", "-1"  # replace with 0 for GPU
]

opt = TestOptions().parse()
opt.isTrain = False

transform = get_transform(opt)
img = Image.open(input_image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)

model = create_model(opt)
model.setup(opt)
model.eval()

data = {"A": input_tensor, "B": input_tensor.clone(), "A_paths": input_image_path}
model.set_input(data)
model.test()
generated = model.get_current_visuals()["fake_B"]

output_image = (generated.squeeze().detach().cpu() + 1) / 2
output_image = transforms.ToPILImage()(output_image)
output_image.save(output_path)

print(f"Ghibli-style image saved at: {output_path}")

plt.imshow(output_image)
plt.axis('off')
plt.title("Ghibli Style Output")
plt.show()
