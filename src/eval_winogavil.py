import argparse
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--clip_backend', default='clip-vit-base-patch32', choices=['clip-vit-base-patch32'],
                    help='clip backend version')
args = parser.parse_args()


def main():
    # Get WinoGAViL Dataset
    winogavil = load_dataset("nlphuji/winogavil")["test"]

    # Initialize CLIP model and processor
    clip_model = CLIPModel.from_pretrained(f"openai/{args.clip_backend}")
    clip_processor = CLIPProcessor.from_pretrained(f"openai/{args.clip_backend}")

    all_jaccards_for_ids = {}

    # Iterate WinoGAViL Instances
    for idx, example in tqdm(enumerate(winogavil)):
        # Solve instance
        clip_predictions = solve_winogavil_instance(clip_model, clip_processor, example["cue"],
                                                    example["num_associations"], example['candidates'],
                                                    example['candidate_images'])
        assert len(clip_predictions) == len(example["associations"])

        # Evaluate with Jaccard
        model_jaccard_score = get_jaccard(clip_predictions, example["associations"])
        all_jaccards_for_ids[example['ID']] = model_jaccard_score

        if idx > 0 and idx % 10 == 0:
            print(f"idx: {idx}, current Jaccard index average: {np.mean(list(all_jaccards_for_ids.values()))}")

    print(f"Result: {np.mean(list(all_jaccards_for_ids.values()))}")
    print("Done")


def solve_winogavil_instance(clip_model, clip_processor, cue, num_associations, candidates, candidates_images):
    clip_text = get_clip_txt(cue)

    sim_for_image = {}
    for img_name, img in zip(candidates, candidates_images):
        processed_cue_img = clip_processor(text=[clip_text], images=img, return_tensors="pt")
        output_cue_img = clip_model(**processed_cue_img).logits_per_image.item()
        sim_for_image[img_name] = output_cue_img

    sorted_sim_for_image = Counter(sim_for_image).most_common()[:num_associations]
    clip_predictions = [x[0] for x in sorted_sim_for_image]
    return clip_predictions


def get_clip_txt(item):
    item = item.lower()
    vowels = ["a", "e", "i", "o", "u"]
    if any(item.startswith(x) for x in vowels):
        clip_txt = f"An {item}"
    else:
        clip_txt = f"A {item}"
    return clip_txt


def get_vectors_similarity(v1, v2):
    similarity = v1.detach().numpy() @ v2.detach().numpy().T
    similarity_item = similarity.item()
    return similarity_item


def get_jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    jaccard = int(len(s1.intersection(s2)) / len(s1.union(s2)) * 100)
    return jaccard


if __name__ == '__main__':
    main()
