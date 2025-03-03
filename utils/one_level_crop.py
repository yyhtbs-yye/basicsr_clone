import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count

def process_image(img_file_output):
    """Crop an image to the nearest multiple of 8 and save it."""
    img_file, output_subfolder = img_file_output
    img = cv2.imread(str(img_file))

    if img is not None:
        height, width = img.shape[:2]
        new_height = (height // 8) * 8
        new_width = (width // 8) * 8
        cropped_img = img[:new_height, :new_width]

        output_img_path = output_subfolder / img_file.name
        cv2.imwrite(str(output_img_path), cropped_img)
        print(f"Cropped and saved: {output_img_path}")

def crop_images_to_multiple_of_8_parallel(root_folder, output_folder):
    root_path = Path(root_folder)
    output_path = Path(output_folder)

    tasks = []

    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in root_path.glob("*.JPEG"):
        tasks.append((img_file, output_path))

    if not tasks:
        print("No tasks found. Exiting.")
        return

    # Ensure at least one worker is used
    num_workers = max(1, min(cpu_count(), len(tasks)))
    with Pool(num_workers) as pool:
        pool.map(process_image, tasks)

# Example usage
if __name__ == "__main__":
    crop_images_to_multiple_of_8_parallel("datasets/ImageNet/GT", "datasets/ImageNet/GT_M8")
