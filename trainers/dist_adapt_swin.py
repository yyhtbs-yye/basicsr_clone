import os
import os.path as osp

import sys
from basicsr.adapt import adapt_pipeline

def main():
    sys.argv = [
        'basicsr/adapt.py',  # Simulate script name
        '-opt', '/home/admyyh/python_workspaces/basicsr_clone/options/yye/adapt_SwinIR_SRx2_from_DF2K_250k_smaller.yml',
        '--launcher', 'none',                # Launcher type
        # '--auto_resume',                     # Include this flag if you want auto-resume
        # '--debug',                           # Include this flag to enable debug mode
        '--local_rank=0',                    # Local rank for distributed training
        # Add any additional flags or options as needed
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Main entry logic
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    adapt_pipeline(root_path)

if __name__ == "__main__":
    main()
