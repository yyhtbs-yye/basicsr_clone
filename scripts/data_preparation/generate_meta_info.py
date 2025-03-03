from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info_div2k(gt_folder='datasets/DIV2K/DIV2K_train_HR_sub/',
                             meta_info_txt='basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'):
    """Generate meta info for DIV2K dataset.
    """

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')


if __name__ == '__main__':
    import argparse

    # Create the argument parser with a description
    parser = argparse.ArgumentParser(
        description="Generate meta info for the DIV2K dataset."
    )

    # Add the required arguments for gt_folder and meta_info_txt
    parser.add_argument(
        "--gt_folder",
        type=str,
        required=True,
        help="Path to the ground truth folder."
    )
    parser.add_argument(
        "--meta_info_txt",
        type=str,
        required=True,
        help="Path to save the meta info text file."
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    generate_meta_info_div2k(args.gt_folder, args.meta_info_txt)

