import os
from PIL import Image
from tqdm import tqdm

input_root = 'dataset'
output_root = 'processed'
image_size = (256, 256)

splits = {
    'training': 'train',
    'testing': 'test'
}

for split in splits.values():
    os.makedirs(os.path.join(output_root, split), exist_ok=True)

for in_folder, out_folder in splits.items():
    split_input = os.path.join(input_root, in_folder)
    split_output = os.path.join(output_root, out_folder)

    folders = [f for f in os.listdir(split_input) if os.path.isdir(os.path.join(split_input, f))]

    for folder in tqdm(folders, desc=f"Processing {in_folder}"):
        folder_path = os.path.join(split_input, folder)
        o_path = os.path.join(folder_path, 'o.png')
        g_path = os.path.join(folder_path, 'g.png')

        if not os.path.exists(o_path) or not os.path.exists(g_path):
            print(f"Skipping {folder} in {in_folder} due to missing images.")
            continue

        try:
            o_img = Image.open(o_path).convert('RGB').resize(image_size)
            g_img = Image.open(g_path).convert('RGB').resize(image_size)

            combined = Image.new('RGB', (image_size[0]*2, image_size[1]))
            combined.paste(o_img, (0, 0))
            combined.paste(g_img, (image_size[0], 0))

            save_path = os.path.join(split_output, f'{folder}.jpg')
            combined.save(save_path)

        except Exception as e:
            print(f"Error processing {folder} in {in_folder}: {e}")
