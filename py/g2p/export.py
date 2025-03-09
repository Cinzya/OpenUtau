import torch
import os
import sys
import hydra
from omegaconf import OmegaConf

sys.path.append(os.path.abspath('.'))
from g2p.model import GreedyG2p
from g2p.trainer import G2pTrainer
from g2p.dataset import SphinxDataset


def load_model(cfg_path, model_path):
    """Loads the model using the provided configuration and checkpoint file."""
    print("Loading model...")
    cfg = OmegaConf.load(cfg_path)
    model = hydra.utils.instantiate(cfg)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    return model


def export_onnx(model, onnx_path):
    """Exports the model to ONNX format."""
    print(f"Exporting model to {onnx_path}...")
    
    # Assuming max_len is required from the model
    greedy = GreedyG2p(model.max_len, model.encoder, model.decoder)
    greedy.export(onnx_path)
    
    print("ONNX export completed!")


if __name__ == "__main__":
    # Define paths
    cfg_path = 'g2p/de_DE/cfg.yaml'
    model_path = 'g2p-459.ptsd'
    onnx_path = 'g2p.onnx'

    # Load model
    model = load_model(cfg_path, model_path)

    # Export to ONNX
    export_onnx(model, onnx_path)
