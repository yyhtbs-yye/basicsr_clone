import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count

def validate_image(img_file):
    """
    Reads an image and checks if its height and width are multiples of 8.
    Returns a tuple: (filename, valid_flag, height, width).
    """
    img = cv2.imread(str(img_file))
    if img is None:
        # Unable to read the image file
        return (img_file.name, False, None, None)

    height, width = img.shape[:2]
    valid = (height % 8 == 0) and (width % 8 == 0)
    return (img_file.name, valid, height, width)

def validate_images_parallel(root_folder):
    """
    Validates all JPEG images in the specified folder using multiprocessing.
    """
    root_path = Path(root_folder)
    image_files = list(root_path.glob("*.JPEG"))

    if not image_files:
        print("No images found.")
        return

    num_workers = max(1, min(cpu_count(), len(image_files)))
    with Pool(num_workers) as pool:
        results = pool.map(validate_image, image_files)

    valid_count = 0
    for filename, valid, height, width in results:
        if height is None or width is None:
            print(f"{filename}: Unable to read the image.")
        elif valid:
            print(f"{filename}: Valid image ({height}x{width})")
            valid_count += 1
        else:
            print(f"{filename}: Invalid image ({height}x{width}) - dimensions are not multiples of 8")

    print(f"\nTotal valid images: {valid_count} out of {len(image_files)}")

if __name__ == "__main__":
    # Change the folder path as needed
    folder_to_validate = "datasets/ImageNet/GT_M8"
    validate_images_parallel(folder_to_validate)
