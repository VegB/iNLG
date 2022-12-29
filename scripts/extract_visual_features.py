import os
import torch
import clip
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(prog='ExtractVisualFeature', description='Extract visual features with CLIP')
parser.add_argument('--input_image_dir', type=str, default='./image/')
parser.add_argument('--output_feature_filename', type=str, default='./feature.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


# load CLIP model
CLIP_TYPE = 'ViT-B/32'
clip_feature_name = 'vitb32'
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load(CLIP_TYPE, device=device)
clip_model = clip_model.to(device)


def _process_image_batch(h, img_path_batch):
    image_batch = torch.tensor(np.stack(
        [preprocess(Image.open(os.path.join(args.input_image_dir, i))) for i in img_path_batch]
        )).to(device)
    image_features = clip_model.encode_image(image_batch).cpu().detach().numpy()
    for i, img_path in enumerate(img_path_batch):
        h.create_dataset(img_path, data=image_features[i].astype(np.float32))


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    # fetch a list of image filenames
    image_filename_list = os.listdir(args.input_image_dir)
    
    # extract CLIP features in batch
    batch_size = args.batch_size
    with h5py.File(args.output_feature_filename, 'w') as h:
        total = (len(image_filename_list) + batch_size - 1) // batch_size
        for img_path_batch in tqdm(chunks(image_filename_list, batch_size), total=total, desc='extract'):
            _process_image_batch(h, img_path_batch)
