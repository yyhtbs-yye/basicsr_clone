import cv2
from pathlib import Path
from basicsr.utils.matlab_functions import imresize
from multiprocessing import Process

def process_image(img_file, scale, output_path):
    img = cv2.imread(str(img_file))
    output_img_name = str(img_file.name)[:-5] + '.png'
    output_img_path = output_path / output_img_name

    # Check if the output file already exists
    if output_img_path.exists():
        print(f"Skipping {output_img_name}, already exists.")
        return

    if img is not None:
        if scale == 1:
            resized_img = img
        else:
            # Resize the image using imresize
            resized_img = imresize(img, scale)
        # Convert the resized image to 8-bit to avoid the warning
        resized_img = cv2.convertScaleAbs(resized_img)
        cv2.imwrite(str(output_img_path), resized_img)
        print(f"Resized and saved: {output_img_path}")

def process_images_chunk(img_files_chunk, scale, output_path):
    """Process a chunk (sublist) of image files."""
    for img_file in img_files_chunk:
        process_image(img_file, scale, output_path)

def split_list(lst, n):
    """
    Splits lst into n nearly equal sublists.

    Uses divmod to account for lists where len(lst) is not perfectly divisible by n.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def resize_images_in_folder(root_folder, scale, output_folder):
    root_path = Path(root_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    img_files = list(root_path.glob("*.JPEG"))
    print("Finished listing the folder")

    # Determine the number of processes (one per sublist)
    num_processes = 256
    # Use fewer processes if there are less images
    num_processes = min(num_processes, len(img_files))

    # Split the list of images into num_processes chunks
    chunks = split_list(img_files, num_processes)

    processes = []
    for chunk in chunks:
        p = Process(target=process_images_chunk, args=(chunk, scale, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# Example usage
if __name__ == '__main__':
    downscale = 1
    resize_images_in_folder("datasets/ImageNet/GT_M8", scale=1 / downscale, output_folder=f"/mnt/ramdisk/ImageNet/bicubic/X{downscale}")
