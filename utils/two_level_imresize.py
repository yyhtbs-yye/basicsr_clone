import cv2
from pathlib import Path

from basicsr.utils.matlab_functions import imresize

def resize_images_in_folder(root_folder, scale, output_folder):
    root_path = Path(root_folder)
    output_path = Path(output_folder)

    for subfolder in root_path.iterdir():
        if subfolder.is_dir():
            output_subfolder = output_path / subfolder.name
            output_subfolder.mkdir(parents=True, exist_ok=True)

            for img_file in subfolder.glob("*.JPEG"):  # Process only PNG files
                img = cv2.imread(str(img_file))
                if img is not None:
                    resized_img = imresize(img, scale)
                    output_img_path = output_subfolder / img_file.name
                    cv2.imwrite(str(output_img_path), resized_img)
                    print(f"Resized and saved: {output_img_path}")

# Example usage
resize_images_in_folder("datasets/ImageNet/GT", scale=0.5, output_folder="datasets/ImageNet/bicubic/X2")