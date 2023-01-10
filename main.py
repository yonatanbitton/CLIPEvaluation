import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

annotations_file = '/Users/yonatanbitton/Documents/Flickr/flickr30k/filtered_test_by_flickr_entities.token'
images_root = "/Users/yonatanbitton/Documents/Flickr/flickr30k-images"


def main():
    # Initialize CLIP model and tokenizer
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = open(annotations_file).readlines()
    print(f"Read {len(dataset)} lines")
    all_images = []
    all_captions = []

    for data in dataset:
        # if len(all_images) > 100:
        #     continue
        imgname, caption = data.split("\t")
        caption = caption.strip(".\n").strip()
        imgname, imgnum = imgname.split("#")
        if int(imgnum) > 0:
            continue
        all_images.append(imgname)
        all_captions.append(caption)
    print(f"Aggregated {len(all_images)} images and {len(all_captions)} captions")

    # Pre-compute image-text similarities
    similarities = np.empty((len(all_captions), len(all_images)))
    for i, txt in tqdm(enumerate(all_captions), desc='calculating similarities', total=len(all_captions)):
        for j, imgname in enumerate(all_images):
            similarities[i, j] = get_img_txt_similarity(clip_model, clip_processor, imgname, txt)

    # Find top 10 texts with highest similarity scores for each image
    top_texts_for_img = {}
    for j, imgname in enumerate(all_images):
        top_text_indices = similarities[:, j].argsort()[-10:][::-1]
        top_texts = [all_captions[i] for i in top_text_indices]
        top_texts_for_img[imgname] = top_texts

    # Find top 10 images with highest similarity scores for each text
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


def get_img_txt_similarity(clip_model, clip_processor, imgname, txt):
    image = Image.open(os.path.join(images_root, imgname))
    inputs = clip_processor(text=[txt], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image.item()
    return logits_per_image


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
