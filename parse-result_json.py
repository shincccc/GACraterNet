import json

def process_json_to_dict(file_path, save_path):
    """Process JSON file to group bounding boxes by image ID and save results."""
    # Load JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create dictionary to store bounding boxes per image
    image_dict = {}
    for item in data:
        image_id = item['image_id']
        bbox = item['bbox']
        # Append bbox to existing list or create new list
        image_dict.setdefault(image_id, []).append(bbox)

    # Calculate total number of bounding boxes
    total_bboxes = sum(len(bboxes) for bboxes in image_dict.values())
    print("Total number of bounding boxes:", total_bboxes)

    # Save dictionary to JSON file
    with open(save_path, 'w') as file:
        json.dump(image_dict, file, indent=4)
    print("Dictionary saved to:", save_path)

if __name__ == "__main__":
    """Main execution: process and save JSON data."""
    file_path = '/home/xgq/Desktop/HF/yunshi/results/no_dcn.json'
    save_path = '/home/xgq/Desktop/HF/yunshi/results/no_dcn.json'
    process_json_to_dict(file_path, save_path)
