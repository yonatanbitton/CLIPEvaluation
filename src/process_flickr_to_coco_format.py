# Flickr images URL: https://uofi.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl
# Flickr karpaty splits URL: http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
import json
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

flickr_annotations_path = '/Users/yonatanbitton/Documents/CLIPEvaluationData/Flickr/caption_datasets/dataset_flickr30k.json'
coco_annotations_path = '/Users/yonatanbitton/Documents/OpenFlamingo/datasets/coco/annotations/captions_train2017.json'
flickr_images_path = '/Users/yonatanbitton/Documents/Visual_Q2_Data/Datasets/SNLI-VE/SNLI-VE_GITHUB/data/Flickr30K/flickr30k_images'
coco_images_path = '/Users/yonatanbitton/Documents/OpenFlamingo/datasets/coco/train2017'
flickr_coco_style_annotations_path = '/Users/yonatanbitton/Documents/CLIPEvaluationData/Flickr/caption_datasets/dataset_flickr30k_coco_style.json'

def main():
    flickr_annotations = json.load(open(flickr_annotations_path))
    coco_annotations = json.load(open(coco_annotations_path))

    full_dataset_coco = COCODataset(
        image_dir_path=coco_images_path, annotations_path=coco_annotations_path
    )

    flickr_new_annotations = []
    running_id = 0
    for item in flickr_annotations['images']:
        for sent in item['sentences']:
            r_dict = {'caption': sent['raw'], 'id': running_id, 'image_id': item['filename'].split('.jpg')[0], 'original_image_id': item['imgid'], 'sentid': sent['sentid']}
            flickr_new_annotations.append(r_dict)
            running_id += 1
    flickr_coco_style = {'annotations': flickr_new_annotations}
    print(f'Dumping Flickr COCO style to {flickr_coco_style_annotations_path}')
    json.dump(flickr_coco_style, open(flickr_coco_style_annotations_path, 'w'))

    full_dataset_flickr = COCODataset(
        image_dir_path=flickr_images_path, annotations_path=flickr_coco_style_annotations_path, source='flickr'
    )

    for x_coco in tqdm(full_dataset_coco, total=len(full_dataset_coco)):
        pass

    for x_flickr in tqdm(full_dataset_flickr, total=len(full_dataset_flickr)):
        pass

    print("Done")


class COCODataset(Dataset):
    def __init__(
        self,
        image_dir_path,
        annotations_path,
        source='coco'
    ):
        self.image_dir_path = image_dir_path
        self.annotations = json.load(open(annotations_path))["annotations"]
        self.source = source

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, idx):
        if self.source == 'coco':
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']:012d}.jpg"
        else:
            return f"{self.image_dir_path}/{self.annotations[idx]['image_id']}.jpg"

    def __getitem__(self, idx):
        image = Image.open(self.get_img_path(idx))
        caption = self.annotations[idx]["caption"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["image_id"],
        }


if __name__ == '__main__':
    main()