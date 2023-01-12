import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Device: {device} ***")
import clip
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

data_dir = '/cs/snapless/roys/yonatanbitton/CLIPEvaluationData'
# data_dir = '/usr/local/google/home/yonatanbitton/CLIPEval/CLIPEvaluationData'
# data_dir = '/Users/yonatanbitton/Documents/CLIPEvaluationData'
_FLICKR_ANNOTATIONS = f'{data_dir}/caption_datasets/dataset_flickr30k.json'
_FLICKER_IMAGES = f'{data_dir}/relevant_images/Flickr'
_FLICKR30 = 'flickr30'

_MSCOCO_ANNOTATIONS = f'{data_dir}/caption_datasets/dataset_coco.json'
_MSCOCO_IMAGES = f"{data_dir}/relevant_images/COCO/val2014"
_MSCOCO = 'mscoco'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--clip_backend', default='RN50', choices=['ViT-B/32', 'RN50'],
                    help='The CLIP backend version')
parser.add_argument('--batch_size', type=int, default=128, help='Text batch size for each image')
parser.add_argument('--dataset', default=_FLICKR30, choices=[_FLICKR30, _MSCOCO], help='The name of the file to process')
args = parser.parse_args()

def main():
    print(f"Dataset: {args.dataset}, backend: {args.clip_backend}")
    # Get Image Retrieval Dataset
    all_captions, all_images = get_ir_dataset()
    print(f"Aggregated {len(all_images)} images and {len(all_captions)} captions")

    # Initialize CLIP model and processor
    clip_model, clip_processor = clip.load(args.clip_backend, device=device)
    print(f"Loaded model CLIP {args.clip_backend}")
    tokenized_txt = clip.tokenize(all_captions, truncate=True).to(device)

    num_batches = len(all_captions) // args.batch_size + (len(all_captions) % args.batch_size != 0)
    print(f"batch_size: {args.batch_size}, num_batches: {num_batches}")
    images_root = _FLICKER_IMAGES if args.dataset == _FLICKR30 else _MSCOCO_IMAGES

    similarities = np.zeros((len(all_captions), len(all_images)))
    for j, imgname in tqdm(enumerate(all_images), desc='calculating similarities', total=len(all_images)):
        image = Image.open(os.path.join(images_root, imgname))
        image_tensor = clip_processor(image).unsqueeze(0).to(device)
        for i in range(0, len(all_captions), args.batch_size):
            start = i
            end = min(i + args.batch_size, len(all_captions))
            tokenized_txt_batch = tokenized_txt[start:end].to(device)
            logits_per_image_batch, logits_per_text_batch = clip_model(image_tensor, tokenized_txt_batch.to(device))
            similarities[start:end, j] = logits_per_image_batch.detach().cpu().numpy()

    # Find top 10 texts with the highest similarity scores for each image
    top_texts_for_img = {}
    for j, imgname in enumerate(all_images):
        top_text_indices = similarities[:, j].argsort()[-10:][::-1]
        top_texts = [all_captions[i] for i in top_text_indices]
        top_texts_for_img[imgname] = top_texts

    # Find top 10 images with the highest similarity scores for each text
    top_imgs_for_txt = {}
    for i, txt in enumerate(all_captions):
        top_img_indices = similarities[i, :].argsort()[-10:][::-1]
        top_imgs = [all_images[j] for j in top_img_indices]
        top_imgs_for_txt[txt] = top_imgs

    # Compute mean R@1, R@5, and R@10 for text
    mean_r_at_1_txt = compute_mean_r_at_k([top_imgs_for_txt[txt] for txt in all_captions], all_images, 1)
    mean_r_at_5_txt = compute_mean_r_at_k([top_imgs_for_txt[txt] for txt in all_captions], all_images, 5)
    mean_r_at_10_txt = compute_mean_r_at_k([top_imgs_for_txt[txt] for txt in all_captions], all_images, 10)

    # Compute mean R@1, R@5, and R@10 for image
    mean_r_at_1_img = compute_mean_r_at_k([top_texts_for_img[img] for img in all_images], all_captions, 1)
    mean_r_at_5_img = compute_mean_r_at_k([top_texts_for_img[img] for img in all_images], all_captions, 5)
    mean_r_at_10_img = compute_mean_r_at_k([top_texts_for_img[img] for img in all_images], all_captions, 10)
    print(
        f"Mean R@1 for text: {mean_r_at_1_txt:.1f}  "
        f"Mean R@5 for text: {mean_r_at_5_txt:.1f}  "
        f"Mean R@10 for text: {mean_r_at_10_txt:.1f}  "
        f"Mean R@1 for image: {mean_r_at_1_img:.1f}  "
        f"Mean R@5 for image: {mean_r_at_5_img:.1f}  "
        f"Mean R@10 for image: {mean_r_at_10_img:.1f}")

    print("Done")


def get_ir_dataset():
    path = _FLICKR_ANNOTATIONS if args.dataset == _FLICKR30 else _MSCOCO_ANNOTATIONS
    dataset = json.load(open(path))
    all_images = []
    all_captions = []
    for data in dataset['images']:
        if data['split'] == 'test':
            caption = data['sentences'][0]['raw']
            all_images.append(data['filename'])
            all_captions.append(caption)
    return all_captions, all_images

def compute_mean_r_at_k(rankings, labels, k):
    r_at_k = []
    for ranking, label in zip(rankings, labels):
        if label in ranking[:k]:
            r_at_k.append(1)
        else:
            r_at_k.append(0)
    return np.mean(r_at_k) * 100


if __name__ == '__main__':
    main()
