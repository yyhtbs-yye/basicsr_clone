import os
import os.path as osp

import sys
from basicsr.plugin import plugin_pipeline

def main():
    sys.argv = [
        'basicsr/distill.py',  # Simulate script name
        '-opt', 'options/yye/plugin_Ysrt_SRx2_from_DF2K_250k_smaller.yml',
        '--launcher', 'none',             # Launcher type
        '--auto_resume',                     # Include this flag if you want auto-resume
        # '--debug',                           # Include this flag to enable debug mode
        '--local_rank=0',                    # Local rank for distributed training
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Main entry logic
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    plugin_pipeline(root_path)

if __name__ == "__main__":
    main()
