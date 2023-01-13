import pandas as pd
import torch
import numpy as np

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
parser.add_argument('--clip_backend', default='RN50', choices=['ViT-B/32', 'RN50'],
                    help='The CLIP backend version')
parser.add_argument('--batch_size', type=int, default=200, help='Text batch size for each image. Batch size of 200 should fit with a single RTX2080.')
parser.add_argument('--dataset', default=_MSCOCO, choices=[_FLICKR30, _MSCOCO],
                    help='The name of the file to process')
parser.add_argument('--add_prefix', action='store_const', default=True, help='Adds a prefix of "an image of a" to the textual prompt, following the original paper.', const=True)

args = parser.parse_args()


def main():
    print(f"Dataset: {args.dataset}, backend: {args.clip_backend}, add_prefix: {args.add_prefix}")
    # Get Image Retrieval Dataset
    all_captions, all_images, df = prepare_ir_dataset()
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

    print(f'text_recall_at_1: {text_recall_at_k(similarities, k=1)}')
    print(f'text_recall_at_5: {text_recall_at_k(similarities, k=5)}')
    print(f'text_recall_at_10: {text_recall_at_k(similarities, k=10)}')

    print(f'image_recall_at_1: {image_recall_at_k(similarities, k=1)}')
    print(f'image_recall_at_5: {image_recall_at_k(similarities, k=5)}')
    print(f'image_recall_at_10: {image_recall_at_k(similarities, k=10)}')
    print("That's it lets finish please!")

    # print(f'image_retrieval2: {image_retrieval2(similarities)}')
    # print(f'text_retrieval2: {text_retrieval2(similarities)}')
    print("Done")
    # print(f'i2t: {i2t(similarities)}')
    # print(f't2i: {t2i(similarities)}')

    num_captions = len(all_captions)
    ground_truth_indices = np.empty((num_captions, 5), dtype=int)
    # Populate the ground_truth_indices array
    for j in range(num_captions):
        ground_truth_indices[j] = np.arange(j * 5, (j + 1) * 5)

    print(f'Text-Retrieval R@1: {text_retrieval(similarities, ground_truth_indices, k=1)}')
    print(f'Text-Retrieval R@5: {text_retrieval(similarities, ground_truth_indices, k=5)}')
    print(f'Text-Retrieval R@10: {text_retrieval(similarities, ground_truth_indices, k=10)}')

    print(f'Image-Retrieval R@1: {image_retrieval(similarities, ground_truth_indices, k=1)}')
    print(f'Image-Retrieval R@5: {image_retrieval(similarities, ground_truth_indices, k=5)}')
    print(f'Image-Retrieval R@10: {image_retrieval(similarities, ground_truth_indices, k=10)}')

    print("***")



    print("DONE")
    # Compute R@1
    recall_at_1 = compute_recall(similarities, 1)
    mean_recall_at_1 = np.mean(recall_at_1)
    print("Mean recall at 1:", mean_recall_at_1)

    # Compute R@5
    recall_at_5 = compute_recall(similarities, 5)
    mean_recall_at_5 = np.mean(recall_at_5)
    print("Mean recall at 5:", mean_recall_at_5)

    # Compute R@10
    recall_at_10 = compute_recall(similarities, 10)
    mean_recall_at_10 = np.mean(recall_at_10)
    print("Mean recall at 10:", mean_recall_at_10)

    print(mean_recall_at_k(similarities, 1))
    print(mean_recall_at_k(similarities, 5))
    print(mean_recall_at_k(similarities, 10))

    # Find top 10 texts with the highest similarity scores for each image
    top_texts_for_img = {}
    for j, imgname in enumerate(all_images):
        top_text_indices = similarities[:, j].argsort()[-10:][::-1]
        top_texts = [all_captions[i] for i in top_text_indices]
        top_texts_for_img[imgname] = top_texts

    # Find top 10 images with the highest similarity scores for each text
    top_imgs_for_txt = {} # prob is here, the txt is low
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


def prepare_ir_dataset():
    sent_keys = ['tokens', 'raw', 'imgid', 'sentid']
    path = _FLICKR_ANNOTATIONS if args.dataset == _FLICKR30 else _MSCOCO_ANNOTATIONS
    dataset = json.load(open(path))
    all_images = []
    all_captions = []
    relevant_rows = []
    for data in dataset['images']:
        # if len(all_images) >= 5:
        #     break
        if data['split'] == 'test':
            all_images.append(data['filename'])
            sentences = data.pop('sentences')[:5]
            for k in sent_keys:
                k_lst = [sent[k] for sent in sentences]
                data[k] = k_lst
            relevant_rows.append(data)
            for caption in data['raw']:
                if args.add_prefix:
                    caption = f"{_PREFIX} {caption}"
                all_captions.append(caption)
    df = pd.DataFrame(relevant_rows)
    for c in ['sentids'] + sent_keys:
        df[c] = df[c].apply(json.dumps)
    return all_captions, all_images, df


def compute_mean_r_at_k(rankings, labels, k):
    r_at_k = []
    for ranking, label in zip(rankings, labels):
        if label in ranking[:k]:
            r_at_k.append(1)
        else:
            r_at_k.append(0)
    return np.mean(r_at_k) * 100


def mean_recall_at_k(similarities, k):
    """
    Calculates mean recall at k for text and image.

    Parameters:
        similarities (np.array): 2D array of similarities, shape (50, 10)
        k (int): the recall at k value (1, 5, or 10)

    Returns:
        tuple: mean recall at k for text and image
    """
    # Initialize variables
    text_recall = 0
    image_recall = 0

    # Iterate over rows (captions)
    for i in range(similarities.shape[0]):
        # Get top k indices for caption i
        top_k_indices = np.argsort(similarities[i])[-k:]

        # Calculate text recall
        text_recall += len(set(range(5*(i//5), 5*(i//5+1))).intersection(set(top_k_indices))) / 5

        # Iterate over columns (images)
        for j in range(similarities.shape[1]):
            # Get top k indices for image j
            top_k_indices = np.argsort(similarities[:, j])[-k:]

            # Calculate image recall
            image_recall += (i in top_k_indices) / k

    # Return mean recall for text and image
    return text_recall / similarities.shape[0], image_recall / similarities.shape[1]

def compute_recall(similarities, k):
    num_captions = similarities.shape[0]
    num_images = similarities.shape[1]

    # Get the top k images for each caption
    top_k_images = np.argsort(-similarities, axis=1)[:, :k]

    # Create a binary matrix where 1 denotes a relevant image for a caption
    relevance_matrix = np.zeros((num_captions, num_images))
    for j in range(num_images):
        relevance_matrix[j*5:j*5+5, j] = 1

    # Compute recall for each caption
    recall = np.sum(relevance_matrix[np.arange(num_captions)[:, np.newaxis], top_k_images], axis=1) / 5

    return recall


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

def image_recall_at_k_prev(similarities, k):
    n_captions = similarities.shape[0]
    recall = 0
    for i in range(n_captions):
        # Find the top K images for each caption
        top_k_images = np.argpartition(similarities[i, :], -k)[-k:]
        # Check if at least one of the relevant images is in the top K
        relevant_images = np.arange(int(i/5), int(i/5)+5)
        if any(np.isin(relevant_images, top_k_images)):
            recall += 1
    # Calculate the overall recall
    recall = recall / n_captions
    return recall




def text_retrieval(similarities, ground_truth_indices, k):
    ''' the images are used to retrieve the corresponding sentences '''
    # Initialize variables to store the number of correct matches and total queries
    correct_matches = 0
    total_queries = similarities.shape[0]

    for i in range(similarities.shape[0]):
        # Get the indices of the top k highest similarities for the current query
        k_top_images_for_caption = np.argpartition(similarities[i], -k)[-k:]
        gt_images = ground_truth_indices[i]

        # Check if any of the ground truth indices are present in the top k indices
        if any(np.isin(gt_images, k_top_images_for_caption)):
            correct_matches += 1

    # Calculate and return recall@k
    recall_at_k = correct_matches / total_queries
    return recall_at_k


def image_retrieval(similarities, ground_truth_indices, k):
    ''' the sentences are used to retrieve the corresponding images '''
    # Initialize variables to store the number of correct matches and total queries
    correct_matches = 0
    total_queries = similarities.shape[1]

    for i in range(similarities.shape[1]):
        # Get the indices of the top k highest similarities for the current query
        k_top_captions_for_image = np.argpartition(similarities[:, i], -k)[-k:]
        gt_captions = ground_truth_indices[i]
        # Check if any of the ground truth indices are present in the top k indices
        if any(np.isin(gt_captions, k_top_captions_for_image)):
            correct_matches += 1

    # Calculate and return recall@k
    recall_at_k = correct_matches / total_queries
    return recall_at_k


def i2t(similarities, npts=1000):
    """
    Images->Text (Image Annotation)
    similarities: (5N, N) matrix of similarities between captions and images
    """
    ranks = np.zeros(npts)
    for index in range(npts):
        # Get query image similarities
        im = similarities[index, :].reshape(1, -1)

        # Get indices of the top 5 most similar captions
        inds = np.argsort(im.flatten())[-5:][::-1]
        ranks[index] = np.where(np.isin(inds, range(5 * index, 5 * (index + 1))))[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks == 0)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return (r1, r5, r10)


def t2i(similarities, npts=1000):
    """
    Text->Images (Image Search)
    similarities: (5N, N) matrix of similarities between captions and images
    """
    ranks = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions similarities
        queries = similarities[5 * index: 5 * index + 5, :]

        # Get indices of the top 5 most similar images for each caption
        inds = np.argsort(queries, axis=1)[:, -5:][:, ::-1]
        for i in range(5):
            ranks[5 * index + i] = np.where(inds[i] == 5 * index + i)[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks == 0)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return (r1, r5, r10)

def text_retrieval2(similarities, num_captions_per_image=5):
    num_images = len(similarities[:,0])
    num_captions = num_images * num_captions_per_image
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    for i in range(num_captions):
        caption_similarities = similarities[i]
        caption_similarities_sorted = np.argsort(caption_similarities)[::-1]
        relevant_images = [i//num_captions_per_image for i in range(i, i+num_captions_per_image)]
        if caption_similarities_sorted[0] in relevant_images:
            recall_at_1 += 1
        if any(x in relevant_images for x in caption_similarities_sorted[:5]):
            recall_at_5 += 1
        if any(x in relevant_images for x in caption_similarities_sorted[:10]):
            recall_at_10 += 1
    return recall_at_1/num_captions, recall_at_5/num_captions, recall_at_10/num_captions

def image_retrieval2(similarities, num_captions_per_image=5):
    num_images = len(similarities[:, 0])
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    for i in range(num_images):
        image_similarities = similarities[i*num_captions_per_image:(i+1)*num_captions_per_image].mean(axis=0)
        image_similarities_sorted = np.argsort(image_similarities)[::-1]
        relevant_captions = [i*num_captions_per_image+j for j in range(num_captions_per_image)]
        if image_similarities_sorted[0] in relevant_captions:
            recall_at_1 += 1
        if any(x in relevant_captions for x in image_similarities_sorted[:5]):
            recall_at_5 += 1
        if any(x in relevant_captions for x in image_similarities_sorted[:10]):
            recall_at_10 += 1
    return recall_at_1/num_images, recall_at_5/num_images, recall_at_10/num_images


if __name__ == '__main__':
    main()
