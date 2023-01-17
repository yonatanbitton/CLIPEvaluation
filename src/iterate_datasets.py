import pandas as pd
import datasets
from datasets import load_dataset


def main():
    dollar_street()
    open_images()
    fairface()

    print("Done")


def dollar_street():
    dollarstreet_dataset = load_dataset("nlphuji/dollar_street_test")
    # Iterate DollarStreet
    for ds_example in dollarstreet_dataset['test']:
        break
    print("DollarStreet")
    print(ds_example)

def open_images():
    openimages_dataset = load_dataset("nlphuji/open_images_dataset_v7")
    for oi_example in openimages_dataset['test']:
        break
    print("OpenImages")
    print(oi_example)


def fairface():
    fairface_dataset = load_dataset("nlphuji/fairface_val")
    # Iterate FairFace
    for ff_example in fairface_dataset['test']:
        break
    print("FairFace")
    print(ff_example)


if __name__ == '__main__':
    main()