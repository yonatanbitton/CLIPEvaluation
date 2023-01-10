# Read in test image names
with open("/Users/yonatanbitton/Documents/Flickr/test_flickr_entities.txt", "r") as f:
    test_image_names = [line.strip() for line in f]

written_lines = 0
# Open full dataset file and filtered dataset file
with open("/Users/yonatanbitton/Documents/Flickr/flickr30k/results_20130124.token", "r") as f_full, \
     open("/Users/yonatanbitton/Documents/Flickr/flickr30k/filtered_test_by_flickr_entities.token", "w") as f_filtered:
    # Iterate over lines in full dataset file
    for line in f_full:
        # Extract image name from line
        image_name = line.split("#")[0].split(".")[0]
        # If image name is in the test set, write line to filtered dataset file
        if image_name in test_image_names:
            f_filtered.write(line)
            written_lines += 1

print(f"Wrote {written_lines} items")