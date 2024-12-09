# hubconf.py
dependencies = ["torch", "torchaudio", "transformers"]

from transformers import HubertForCTC

def hubert_ee(model_path: str):
    """
    Load a huggingface HubertForCTC model from a local directory.
    """
    model = HubertForCTC.from_pretrained(model_path)
    model.eval()
    return model
