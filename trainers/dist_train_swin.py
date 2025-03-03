import os
import os.path as osp

import sys
from basicsr.train import train_pipeline

def main():
    sys.argv = [
        'basicsr/train.py',  # Simulate script name
        '-opt', '/home/admyyh/python_workspaces/basicsr_sopy/options/yye/train_SwinIR_SRx2_from_ImageNet_250k_bigger.yml',
        '--launcher', 'none',                # Launcher type
        '--auto_resume',                     # Include this flag if you want auto-resume
        # '--debug',                           # Include this flag to enable debug mode
        '--local_rank=0',                    # Local rank for distributed training
        # Add any additional flags or options as needed
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Main entry logic
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)

if __name__ == "__main__":
    main()
