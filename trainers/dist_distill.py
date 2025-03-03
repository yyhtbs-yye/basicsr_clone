import os
import os.path as osp

import sys
from basicsr.distill import distill_pipeline

def main():
    sys.argv = [
        'basicsr/distill.py',  # Simulate script name
        '-opt', 'options/distill/HMA/HMA_SRx2_finetune_from_pretrain_250k_custom.yml',
        '--launcher', 'none',             # Launcher type
        '--auto_resume',                     # Include this flag if you want auto-resume
        # '--debug',                           # Include this flag to enable debug mode
        '--local_rank=0',                    # Local rank for distributed training
        # Add any additional flags or options as needed
        "--nproc_per_node=8"
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,8,9,10,11"

    # Main entry logic
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    distill_pipeline(root_path)

if __name__ == "__main__":
    main()
