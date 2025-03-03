import torch

from copy import deepcopy

if __name__=="__main__":

    load_path = "experiments/train_SwinIR_SRx2_from_DF2K_250k_smaller/models/net_g_298000.pth"

    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

    feature_extractor_dict = {
        key.replace("feature_extractor.", ""): value
        for key, value in load_net['params'].items()
        if key.startswith("feature_extractor.")
    }
    a = 1

    save_path = "experiments/pretrained_models/sisr/SwinIR_SRx2_from_DF2K_250k_smaller_net_g_298000.pth"
    torch.save(feature_extractor_dict, save_path)
