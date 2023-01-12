import argparse
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--clip_backend', default='clip-vit-base-patch32', choices=['clip-vit-base-patch32'],
                    help='clip backend version')
parser.add_argument('--access_token', default='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',
                    help='The name of the file to process')
args = parser.parse_args()


def main():
    # Get WinoGAViL Dataset
    winoground = load_dataset("facebook/winoground", use_auth_token=args.access_token)["test"]

    # Initialize CLIP model and processor
    clip_model = CLIPModel.from_pretrained(f"openai/{args.clip_backend}")
    clip_processor = CLIPProcessor.from_pretrained(f"openai/{args.clip_backend}")

    # Get CLIP image-caption scores from the whole dataset
    winoground_clip_scores = []
    for example in tqdm(winoground):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        # Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
        input_c0_i0 = clip_processor(text=[example["caption_0"]], images=[example["image_0"].convert("RGB")],
                                     return_tensors="pt")
        input_c1_i0 = clip_processor(text=[example["caption_1"]], images=[example["image_0"].convert("RGB")],
                                     return_tensors="pt")
        input_c0_i1 = clip_processor(text=[example["caption_0"]], images=[example["image_1"].convert("RGB")],
                                     return_tensors="pt")
        input_c1_i1 = clip_processor(text=[example["caption_1"]], images=[example["image_1"].convert("RGB")],
                                     return_tensors="pt")
        output_c0_i0 = clip_model(**input_c0_i0)
        output_c1_i0 = clip_model(**input_c1_i0)
        output_c0_i1 = clip_model(**input_c0_i1)
        output_c1_i1 = clip_model(**input_c1_i1)
        clip_score_c0_i0 = output_c0_i0.logits_per_image.item()
        clip_score_c1_i0 = output_c1_i0.logits_per_image.item()
        clip_score_c0_i1 = output_c0_i1.logits_per_image.item()
        clip_score_c1_i1 = output_c1_i1.logits_per_image.item()
        winoground_clip_scores.append(
            {"id": example["id"], "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0,
             "c1_i1": clip_score_c1_i1})

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in winoground_clip_scores:
        text_correct_count += 1 if text_correct(result) else 0
        image_correct_count += 1 if image_correct(result) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(winoground_clip_scores)
    print("text score:", text_correct_count / denominator)
    print("image score:", image_correct_count / denominator)
    print("group score:", group_correct_count / denominator)

    print("Done")


# Define the text, image, and group metrics, and compute the overall performance of CLIP
def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]


def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]


def group_correct(result):
    return image_correct(result) and text_correct(result)


if __name__ == '__main__':
    main()
