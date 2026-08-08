"""PatchEX — interpretable enzyme optimal-temperature and optimal-pH prediction.

    from patchex import PatchEX
    mdl = PatchEX.load("topt")
    preds = mdl.predict_fasta("proteins.fasta")

See README.md for the CLI (`patchex-predict`) and for training.
"""
__version__ = "1.0.0"

from .infer import PatchEX, Prediction, predictions_to_frame, write_weights
from .embed import embed_sequences, read_fasta
from .model import Model

__all__ = ["PatchEX", "Prediction", "Model", "embed_sequences", "read_fasta",
           "predictions_to_frame", "write_weights", "__version__"]
