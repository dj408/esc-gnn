import torch
from torch import nn
from torch_geometric.data import Data

from models.base_module import BaseModule


class ComparisonModel(BaseModule):
    """
    Thin wrapper that adapts a PyG comparison model to the BaseModule API used by
    the training loop. It delegates forward passes to the wrapped PyG model and
    exposes predictions under the expected 'preds' key.

    The wrapper adds small compatibility shims:
    - Ensures a node index attribute named 'atoms' exists (falls back to 'z' or
      creates a zero-index tensor) for models that embed atomic numbers.
    """

    def __init__(
        self,
        pyg_model: nn.Module,
        *,
        base_module_kwargs: dict,
        atomic_number_key: str = 'z',
    ) -> None:
        super().__init__(**base_module_kwargs)
        self.pyg_model = pyg_model
        self.atomic_number_key = atomic_number_key

    def _ensure_atoms_attribute(self, batch: Data) -> None:
        """
        Ensure `batch.atoms` exists for compatibility with some baseline models.

        If absent, try to alias from `self.atomic_number_key` (default 'z'). If
        still absent, create a zero-index LongTensor of length num_nodes.
        """
        # If the wrapped model explicitly supports operating without `atoms`,
        # and does not have an embedding table, avoid synthesizing one so the
        # model can use its bias-initialized constant node features.
        if hasattr(self.pyg_model, 'use_bias_if_no_atoms') \
        and getattr(self.pyg_model, 'use_bias_if_no_atoms'):
            emb_in = getattr(self.pyg_model, 'emb_in', None)
            if emb_in is None:
                return
        if hasattr(batch, 'atoms') and batch.atoms is not None:
            return

        if hasattr(batch, self.atomic_number_key):
            atoms = getattr(batch, self.atomic_number_key)
            if not torch.is_tensor(atoms):
                atoms = torch.as_tensor(atoms)
            batch.atoms = atoms.to(dtype=torch.long, device=batch.pos.device if hasattr(batch, 'pos') else self.get_device())
            return

        # Fallback: synthesize a single-type atom index vector
        num_nodes = None
        if hasattr(batch, 'pos') and isinstance(batch.pos, torch.Tensor):
            num_nodes = batch.pos.size(0)
        elif hasattr(batch, 'x') and isinstance(batch.x, torch.Tensor):
            num_nodes = batch.x.size(0)
        if num_nodes is None:
            raise ValueError("Could not infer number of nodes to create dummy 'atoms' attribute.")
        device = batch.pos.device if hasattr(batch, 'pos') else self.get_device()
        batch.atoms = torch.zeros(num_nodes, dtype=torch.long, device=device)

    def forward(self, batch: Data) -> dict:
        # Small input normalization for comparison models that expect an 'atoms' field
        self._ensure_atoms_attribute(batch)

        preds = self.pyg_model(batch)
        # If wrapped model already returns a dict, pass through; else wrap tensor
        if isinstance(preds, dict):
            return preds
        return {'preds': preds}


