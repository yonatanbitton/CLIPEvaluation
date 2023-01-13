import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Device: {device} ***")
import clip
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# data_dir = '/Users/yonatanbitton/Documents/CLIPEvaluationData'
data_dir = '/cs/snapless/roys/yonatanbitton/CLIPEvaluationData'
_FLICKR_ANNOTATIONS = f'{data_dir}/caption_datasets/dataset_flickr30k.json'
_FLICKER_IMAGES = f'{data_dir}/relevant_images/Flickr'
_FLICKR30 = 'flickr30'

_MSCOCO_ANNOTATIONS = f'{data_dir}/caption_datasets/dataset_coco.json'
_MSCOCO_IMAGES = f"{data_dir}/relevant_images/COCO/val2014"
_MSCOCO = 'mscoco'
_PREFIX = 'a photo of'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--clip_backend', default='ViT-B/32', choices=['RN50', 'ViT-B/32', 'ViT-L/14', 'RN50x64'],
                    help='The CLIP backend version')
parser.add_argument('--batch_size', type=int, default=200, help='Text batch size for each image. Batch size of 200 should fit with a single RTX2080.')
parser.add_argument('--dataset', default=_MSCOCO, choices=[_FLICKR30, _MSCOCO],
                    help='The name of the file to process')
parser.add_argument('--add_prefix', action='store_const', default=True, help='Adds a prefix of "an image of a" to the textual prompt, following the original paper.', const=True)

args = parser.parse_args()


def main():
    print(f"Dataset: {args.dataset}, backend: {args.clip_backend}, add_prefix: {args.add_prefix}")
    # Get Image Retrieval Dataset
    all_captions, all_images = prepare_ir_dataset()
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

    print(f'Text-Retrieval R@1: {text_recall_at_k(similarities, k=1)}')
    print(f'Text-Retrieval R@5: {text_recall_at_k(similarities, k=5)}')
    print(f'Text-Retrieval R@10: {text_recall_at_k(similarities, k=10)}')

    print(f'Image-Retrieval R@1: {image_recall_at_k(similarities, k=1)}')
    print(f'Image-Retrieval R@5: {image_recall_at_k(similarities, k=5)}')
    print(f'Image-Retrieval R@10: {image_recall_at_k(similarities, k=10)}')

    print("Done")


def prepare_ir_dataset():
    path = _FLICKR_ANNOTATIONS if args.dataset == _FLICKR30 else _MSCOCO_ANNOTATIONS
    dataset = json.load(open(path))
    all_images = []
    all_captions = []
    for data in dataset['images']:
        if data['split'] == 'test':
            all_images.append(data['filename'])
            for caption in data['raw']:
                if args.add_prefix:
                    caption = f"{_PREFIX} {caption}"
                all_captions.append(caption)
    return all_captions, all_images

def text_recall_at_k(similarities, k):
    ''' the images are used to retrieve the corresponding sentences '''
    n_images = similarities.shape[1]
    recall = 0
    for i in range(n_images):
        # Find the top K captions for each image
        top_k_captions = similarities[:, i].argsort()[-k:][::-1][:k]
        # Check if at least one of the relevant captions is in the top K
        relevant_captions = np.arange(i*5, (i+1)*5)
        if any(np.isin(relevant_captions, top_k_captions)):
            recall += 1
    # Calculate the overall recall
    recall = recall / n_images
    return recall

def image_recall_at_k(similarities, k):
    ''' the sentences are used to retrieve the corresponding images '''
    n_captions = similarities.shape[0]
    recall = 0
    for i in range(n_captions):
        # Find the top K images for each caption
        top_k_images = similarities[i,:].argsort()[::-1][:k]
        # Check if at least one of the relevant images is in the top K
        relevant_images = [int(i/5)]
        if any(np.isin(relevant_images, top_k_images)):
            recall += 1
    # Calculate the overall recall
    recall = recall / n_captions
    return recall


if __name__ == '__main__':
    main()
