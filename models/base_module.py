"""
This file contains useful classes
and functions for extending the
torch.nn.Module class for model-building,
and works with the accelerate library: 

(1) Class definition for 'BaseModule',
an extension of torch.nn.Module with
built-in loss and metrics methods for
regressor or binary classifier models.

(2) Function 'test_nn', which computes
basic metrics for regression and binary
classification models built from 
BaseModule.
"""

import models.nn_utilities as nnu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    Dataset, 
    DataLoader
)
from torchmetrics.regression import (
    MeanSquaredError,
    # R2Score,
    MeanAbsoluteError
)
from torchmetrics.classification import (
    Accuracy,
    BinaryAccuracy,
    BinaryRecall,
    BinaryF1Score,
    BinaryPrecision,
    BinarySpecificity,
    BinaryAUROC,
    BinaryConfusionMatrix
)
from torchmetrics import Metric, MetricCollection
from torch_geometric.data import Data
from typing import Tuple, Dict, Optional, Callable, Any, List
import torch.distributed as dist


class BaseModule(nn.Module):
    """
    Subclass of torch.nn.Module, designed to work
    flexibly with the 'train_model' function (in
    train_fn.py). Has built-in loss functions and 
    metrics calculation methods specific to regression 
    and binary classification models (if not overridden).
    
    __init__ args:
        task: string key description of the model task, e.g.,
            'regression' or 'binary classification'.
        loss_fn: (optional) functional loss to use; if
            'None', will attempt to assign a default loss
            function based on the 'task' argument in 
            __init__().
        loss_fn_kwargs: for a torch.nn.functional loss.
        target_name: string key for the prediction target.
        metrics_kwargs: kwargs for setting up metric calcu-
            lator objects, e.g., num_classes for multiclass
            accuracy.
        key_prefix: string prefix for each metric column
             name in the training records. Should end in '_'.
        on_best_model_kwargs: kwargs for the 'on_best_model'
            method (overridden in subclasses, if implemented).
        target_preproc_stats: Optional dictionary containing 
            target preprocessing statistics
        device: manual device onto which to move the model
            weights.
        has_lazy_parameter_initialization: Whether the model 
            instantiates parameters lazily (after __init__)
        has_normalized_train_targets: Whether the model has 
            train set target normalization
    """
    def __init__(
        self,
        task: str,
        loss_fn: Optional[Callable] = None,
        loss_fn_kwargs: Dict[str, Any] = {},
        target_name: str = None,
        metrics_kwargs: Dict[str, Any] = {},
        key_prefix: str = '',
        on_best_model_kwargs: Dict[str, Any] = {},
        target_preproc_stats: Optional[Dict[str, Any]] = None,
        device = None,
        has_lazy_parameter_initialization: bool = False,
        has_normalized_train_targets: bool = False,
        verbosity: int = 0
    ):
        super(BaseModule, self).__init__()
        # ------------------------------------------------------------------
        # Whether the model instantiates parameters lazily (after __init__)
        # If True, training utilities should run a dummy forward pass before
        # wrapping the model with DistributedDataParallel so that all
        # parameters are registered. Sub-classes can override this flag.
        # ------------------------------------------------------------------
        self.has_lazy_parameter_initialization = has_lazy_parameter_initialization

        # Whether the model has train set target normalization
        # (Subclasses can override this flag)
        self.has_normalized_train_targets = has_normalized_train_targets

        self.task = task.lower()
        self.device = device
        self.target_name = target_name
        self.metrics_kwargs = metrics_kwargs
        self.target_dim = metrics_kwargs.get('num_outputs', 1)
        self.key_prefix = key_prefix
        self.on_best_model_kwargs = on_best_model_kwargs
        self.verbosity = verbosity

        # If needed, store train target preprocessing statistics 
        # for de-normalization during metrics calculation
        self.target_preproc_stats = None
        if has_normalized_train_targets \
        and target_preproc_stats is not None \
        and ('center' in target_preproc_stats) \
        and ('scale' in target_preproc_stats):
            
            # Flag to indicate the model has normalized train set targets
            self.has_normalized_train_targets = True
            center = target_preproc_stats['center']
            scale = target_preproc_stats['scale']
            self.register_buffer('_target_center', center)
            self.register_buffer('_target_scale', scale)
            self.target_preproc_stats = {'center': center, 'scale': scale}

        self._set_up_metrics()
        self.set_device()
        
        if loss_fn is None:
            if 'reg' in self.task:
                self.loss_fn = F.mse_loss
            elif 'class' in self.task and 'bin' in self.task:
                # F.binary_cross_entropy_with_logits removes need
                # for sigmoid activation after last layer, but targets
                # need to be floats between 0 and 1
                # https://pytorch.org/docs/stable/generated/
                # torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
                self.loss_fn = F.binary_cross_entropy_with_logits
            elif 'class' in self.task and 'multi' in self.task:
                self.loss_fn = F.cross_entropy
            else:
                raise NotImplementedError(
                    f"No loss function implemented for task = '{self.task}'!"
                )
        else:
            self.loss_fn = loss_fn

        self.loss_fn_kwargs = {'reduction': 'mean'} \
            if loss_fn_kwargs is None else loss_fn_kwargs
        self.loss_keys = None # set in first call to update_metrics()

        self.epoch_loss_dict = {
            'train': {'size': 0.}, # init as floats since used in divison
            'valid': {'size': 0.}
        }

        self._wandb_watched = False  # Track if wandb.watch has been called
        
    def set_device(self):
        # optional: manually enforce device
        if self.device is not None:
            # self.device = torch.device(device)
            self.to(self.device)

    
    def get_device(self):
        # find device that the model weights are on
        return next(self.parameters()).device


    def _loss(
        self, 
        input_dict, 
        output_dict,
        phase
    ):
        """
        This fn wraps loss_fn so it takes dicts storing 
        preds, targets as inputs, and outputs a loss_dict.
        Separated out in case this module is just one head
        of a multi-head model, and its loss is just one
        term of a composite loss function.
        """
        preds = output_dict['preds']
        # may need to de-normalize preds for valid set
        # (if the model is trained to predict normalized targets)
        if phase == 'valid' and self.has_normalized_train_targets:
            preds = self.get_denormalized(preds)
            if self.verbosity > 0:
                print(f"[DEBUG] BaseModule._loss: de-normalizing valid set preds")
                preds_print = [f'{v:.2f}' for v in preds[:5].squeeze().detach().cpu().tolist()]
                print(f"\tpreds: {preds_print}")
                
        # print('preds.shape =', preds.shape)
        targets = input_dict['target']
        # print('targets.shape =', targets.shape)

        if phase == 'valid' and self.verbosity > 0:
            targets_print = [f'{v:.2f}' for v in targets[:5].squeeze().detach().cpu().tolist()]
            print(f"\ttargets = {targets_print}")
        
        # 'targets' may itself be a dict holding
        # multiple targets
        if (self.target_name is not None) \
        and (isinstance(targets, dict)):
            targets = targets[self.target_name]
        # print('targets.shape =', targets.shape)
        # print('targets =', targets)
        
        # [BYPASSED] Align loss with metric definition for vector node regression:
        # when the task is vector-node regression and using MSE, compute the
        # mean of squared vector norms per node (sum over coordinates, mean over nodes),
        # which matches MultiTargetMSE(mode='vector').
        # if (
        #     ('reg' in self.task) and ('vector' in self.task) and ('node' in self.task)
        #     and (self.loss_fn is F.mse_loss)
        # ):
        #     diff = (preds.squeeze() - targets.squeeze())
        #     # sum across last dim (coordinates), mean across nodes/samples
        #     loss = (diff * diff).sum(dim=-1).mean()
        # else:
        loss = self.loss_fn(
            input=preds.squeeze(),
            target=targets.squeeze(),
            **self.loss_fn_kwargs 
        )
        loss_dict = {
            'loss': loss,
            'size': targets.shape[0]
        }
        return loss_dict

    
    def loss(self, input_dict, output_dict, phase):
        """
        Simply grabs preds and targets from dictionary
        containers and calls 'self._fc_loss', unless 
        overridden by subclass.
        """
        loss_dict = self._loss(input_dict, output_dict, phase)
        return loss_dict

    
    def _set_up_metrics(self):
        """
        Convenience method to set output layer activation and 
        metrics based on model task type.
        """
        if 'reg' in self.task:
            if 'node' in self.task:
                # Node-level: use custom MultiTargetMSE so we can average per-graph
                # (no MAE equivalent yet)
                self.mse = MultiTargetMSE(
                    num_targets=self.target_dim,
                    mode=('vector' if 'vector' in self.task else 'per_target')
                )
            else:
                # Graph-level: standard torchmetrics
                if self.target_dim == 1:
                    self.mse = MeanSquaredError(sync_on_compute=True)
                    self.mae = MeanAbsoluteError(sync_on_compute=True)
                else:
                    self.mse = MeanSquaredError(sync_on_compute=True)
                    self.mae = MeanAbsoluteError(
                        num_outputs=self.target_dim,
                        sync_on_compute=True
                    )
                
            # in multi-target regression, R^2 is computed for each target
            # separately
            # self.R2_score = R2Score(
            #     # num_outputs=self.num_targets, # this may be a bug in v1.7.1
            #     multioutput='raw_values'
            # )
            
        elif 'class' in self.task:

            # binary classification
            if 'bin' in self.task:
                self.accuracy = BinaryAccuracy(sync_on_compute=True)
                self.balanced_accuracy = Accuracy(
                    task='multiclass', 
                    num_classes=2, 
                    average='macro',
                    sync_on_compute=True
                )
                self.specificity = BinarySpecificity(sync_on_compute=True)
                self.f1 = BinaryF1Score(sync_on_compute=True)
                self.f1_neg = BinaryF1Score(sync_on_compute=True)
                self.auroc = BinaryAUROC(sync_on_compute=True)
                self.class_1_pred_ct = 0
                # self.running_class_1_probs = []

            # multi-class classification
            elif 'multi' in self.task:
                self.accuracy = Accuracy(
                    task='multiclass',
                    num_classes=self.metrics_kwargs['num_classes'],
                    sync_on_compute=True
                )
                self.balanced_accuracy = Accuracy(
                    task='multiclass', 
                    num_classes=self.metrics_kwargs['num_classes'], 
                    average='macro',
                    sync_on_compute=True
                )
    
        else:
            raise NotImplementedError(
                f"Metrics for task='{self.task}' not yet implemented"
                f" in BaseModule!"
            )


    def on_best_model(self) -> None:
        """
        Overridable method to perform special
        methods whenever a new best model is
        achieved during training.
        """
        return None

    
    def get_printable_metrics(self) -> List[str]:
        """
        Get the list of metrics that should be printed for this model's task.
        
        Returns:
            List of metric names that should be printed during training/evaluation
        """
        return MetricDefinitions.get_printable_metrics_for_task(self.task)

    
    def update_metrics(
        self, 
        phase,
        loss_dict,
        input_dict = None, 
        output_dict = None
    ) -> None:
        if self.verbosity > 0 and phase == 'valid':
            print(f"[DEBUG] BaseModule.update_metrics: phase = {phase}")
        
        device = self.get_device()
        
        # on first call only: initialize loss counters
        if self.loss_keys is None:
            self.loss_keys = [
                k for k, v in loss_dict.items() \
                if 'loss' in k.lower()
            ]
            for k in self.epoch_loss_dict.keys():
                for loss_key in self.loss_keys:
                    self.epoch_loss_dict[k][loss_key] = 0.0
                    
        # Accumulate *sum* of loss, not mean, so final average is correct
        for loss_key in self.loss_keys:
            # loss_dict[loss_key] is a per-batch MEAN; multiply by batch size to get the sum
            self.epoch_loss_dict[phase][loss_key] += loss_dict[loss_key] * loss_dict['size']
        # Keep running total of number of samples seen
        self.epoch_loss_dict[phase]['size'] += loss_dict['size'] 

        # validation metrics
        if phase == 'valid':
            preds = output_dict['preds']
            # print('preds.shape:', preds.shape)
            target = input_dict['target']
            
            # 'target' may itself be a dict containing
            # multiple targets
            if (self.target_name is not None) \
            and (isinstance(target, dict)):
                target = target[self.target_name]
                # print('target:', target)
            
            if 'reg' in self.task:
                # If we normalized train set targets, de-normalize 
                # before computing metrics
                if self.has_normalized_train_targets:
                    # Model is trained to predict normalized targets...
                    if phase == 'train':
                        # For train set metrics (if calculated), both preds and 
                        # targets need de-normalization, since targets were 
                        # normalized during data loading
                        preds = self.get_denormalized(preds)
                        target = self.get_denormalized(target)
                    elif phase == 'valid':
                        # De-normalize predictions to match never-normalized targets
                        if self.verbosity > 0:
                            print(f"\tde-normalizing valid set preds")
                        preds = self.get_denormalized(preds)
                        # (targets are not normalized in valid set, so no 
                        # de-normalization needed)
                    else:
                        raise ValueError(f"Invalid phase: {phase}")
                
                # Squeeze after de-normalization (same order as loss calculation)
                if 'multi' not in self.task:
                    preds = preds.squeeze()
                    target = target.squeeze()
                    
                if phase == 'valid' and self.verbosity > 0:
                    preds_print = [f'{v:.2f}' for v in preds[:5].detach().cpu().tolist()]
                    target_print = [f'{v:.2f}' for v in target[:5].detach().cpu().tolist()]
                    print(f"\tpreds: {preds_print}")
                    print(f"\ttarget: {target_print}")

                # print("update_metrics:")
                # print(f"\tpreds.shape: {preds.shape}")
                # print(f"\ttarget.shape: {target.shape}")
                # Optional node-level normalization by per-graph node counts
                node_counts = None
                batch_index = None
                if isinstance(input_dict, dict):
                    node_counts = input_dict.get('node_counts', None)
                    batch_index = input_dict.get('batch_index', None)
                if isinstance(self.mse, MultiTargetMSE):
                    self.mse.update(preds, target, batch_index=batch_index, node_counts=node_counts)
                else:
                    self.mse.update(preds, target)
                if hasattr(self, 'mae'):
                    self.mae.update(preds, target)
                
                # R^2
                # R2_score = self.R2_score.compute().detach().cpu().numpy()
                # if self.num_targets == 1:
                #     R2_score = R2_score.item()
                # metrics_dict = metrics_dict \
                #     | {(self.key_prefix + 'R2_valid'): R2_score}
                
            elif 'class' in self.task and 'bin' in self.task:
                # accuracy and f1
                # when using BCE with logits, need to convert
                # logit preds to 0 or 1 -> 
                # logit = log(p/(1-p)) -> logit>0 -> p>0.5 -> predicted '1'
                class_preds = torch.tensor(
                    [(logit > 0.0) for logit in preds],
                    dtype=torch.long,
                    device=device
                )
                class_targets = torch.tensor(
                    [int(t) for t in target],
                    dtype=torch.long,
                    device=device
                )
                self.accuracy.update(class_preds, class_targets)
                self.balanced_accuracy.update(class_preds, class_targets)
                self.f1.update(class_preds, class_targets)
                self.f1_neg.update(
                    torch.logical_not(class_preds).to(torch.long), 
                    torch.logical_not(class_targets).to(torch.long)
                )
                self.specificity.update(class_preds, class_targets)
                # auroc detects logits if preds are outside of [0, 1]
                self.auroc.update(preds, class_targets)
                self.class_1_pred_ct += torch.sum(preds > 0.).item()

            elif 'class' in self.task and 'multi' in self.task:
                class_preds = torch.argmax(preds, dim=1)
                print(f"class_preds.shape: {class_preds.shape}")
                print(f"target.shape: {target.shape}")
                class_targets = torch.tensor(
                    [int(t) for t in target],
                    dtype=torch.long,
                    device=device
                )
                self.accuracy.update(class_preds, class_targets)
                self.balanced_accuracy.update(class_preds, class_targets)
                

    
    def calc_metrics(
        self,
        epoch: int,
        is_validation_epoch: bool = True,
        # input_dict: Optional[Dict[str, Any]] = None, 
        # output_dict: Optional[Dict[str, Any]] = None, 
        # loss_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float | int]:

        phases = ('train', 'valid') if is_validation_epoch else ('train', )
        metrics_dict = {'epoch': epoch}
        
        # include train (and maybe validation) mean losses in metrics_dict
        for phase in phases:
            for loss_key in self.loss_keys:
                sample_count = self.epoch_loss_dict[phase]['size']
                # Avoid division-by-zero on ranks that saw zero batches
                # (e.g., small validation shard under DDP)
                if sample_count > 0:
                    avg_loss = self.epoch_loss_dict[phase][loss_key] / sample_count
                else:
                    # Use a zero tensor on the correct device to preserve dtype/device
                    avg_loss = torch.tensor(0.0, device=self.get_device())
                
                metrics_dict = metrics_dict \
                    | {(loss_key + '_' + phase): avg_loss.item()}

                # reset epoch loss to 0.
                self.epoch_loss_dict[phase][loss_key] = 0.
                
            self.epoch_loss_dict[phase]['size'] = 0.

        # in validation epochs, calc validation set metrics        
        if is_validation_epoch:
            
            if 'reg' in self.task:
                # For node-level regression we use a custom metric that maintains
                # running numerator/denominator so we can safely aggregate across DDP ranks
                if isinstance(self.mse, MultiTargetMSE):
                    # Capture local accumulators before reset
                    local_numer = self.mse.sum_mse_across_graphs.detach().clone()
                    local_denom = self.mse.graph_count.detach().clone().to(torch.float32)

                    # Compute local mean only if we actually updated on this rank
                    if local_denom.item() > 0:
                        local_mean = (local_numer / local_denom).detach().cpu().numpy()
                        if self.target_dim == 1:
                            local_mean = local_mean.item()
                    else:
                        # No updates on this rank; avoid calling compute() to prevent warnings
                        # Report NaN locally; a proper global mean will be computed upstream using numer/denom
                        local_mean = float('nan') if self.target_dim == 1 else (local_numer.detach().cpu().numpy() * float('nan'))

                    # Stash accumulators for weighted cross-rank reduction in the trainer
                    metrics_dict[self.key_prefix + 'mse_valid_numer'] = local_numer.detach().cpu().numpy()
                    metrics_dict[self.key_prefix + 'mse_valid_denom'] = float(local_denom.item())

                    # Coerce size-1 arrays to scalar for logging/printing compatibility
                    try:
                        if hasattr(local_mean, 'size') and getattr(local_mean, 'size', 0) == 1:
                            local_mean = local_mean.item()
                    except Exception:
                        pass
                    metrics_dict = metrics_dict | {(self.key_prefix + 'mse_valid'): local_mean}
                    self.mse.reset()
                else:
                    # Graph-level regression: standard torchmetrics compute (already DDP-aware)
                    mse_score = self.mse.compute().detach().cpu().numpy()
                    if self.target_dim == 1:
                        mse_score = mse_score.item()
                    metrics_dict = metrics_dict | {(self.key_prefix + 'mse_valid'): mse_score}
                    self.mse.reset()
    
                # MAE
                if hasattr(self, 'mae'):
                    mae_score = self.mae.compute().detach().cpu().numpy()
                    if self.target_dim == 1:
                        mae_score = mae_score.item()
                    metrics_dict = metrics_dict \
                        | {(self.key_prefix + 'mae_valid'): mae_score}
                    self.mae.reset()
                
            elif 'class' in self.task and 'bin' in self.task:
                accuracy_score = self.accuracy.compute().detach().cpu().numpy().item()
                bal_accuracy_score = self.balanced_accuracy.compute().detach().cpu().numpy().item()
                f1_score = self.f1.compute().detach().cpu().numpy().item()
                f1_neg_score = self.f1_neg.compute().detach().cpu().numpy().item()
                specificity_score = self.specificity.compute().detach().cpu().numpy().item()
                auroc_score = self.auroc.compute().detach().cpu().numpy().item()
                metrics_dict = metrics_dict \
                    | {(self.key_prefix + 'accuracy_valid'): accuracy_score} \
                    | {(self.key_prefix + 'bal_accuracy_valid'): bal_accuracy_score} \
                    | {(self.key_prefix + 'f1_valid'): f1_score} \
                    | {(self.key_prefix + 'f1_neg_valid'): f1_neg_score} \
                    | {(self.key_prefix + 'specificity_valid'): specificity_score} \
                    | {(self.key_prefix + 'auroc_valid'): auroc_score} \
                    | {(self.key_prefix + 'class_1_pred_ct_valid'): self.class_1_pred_ct} 
                self.accuracy.reset()
                self.balanced_accuracy.reset()
                self.f1.reset()
                self.f1_neg.reset()
                self.specificity.reset()
                self.auroc.reset()
                self.class_1_pred_ct = 0
    
            elif 'class' in self.task and 'multi' in self.task:
                accuracy_score = self.accuracy.compute().detach().cpu().numpy().item()
                bal_accuracy_score = self.balanced_accuracy.compute().detach().cpu().numpy().item()
                metrics_dict = metrics_dict \
                    | {(self.key_prefix + 'accuracy_valid'): accuracy_score} \
                    | {(self.key_prefix + 'bal_accuracy_valid'): bal_accuracy_score}
                self.accuracy.reset()
                self.balanced_accuracy.reset()
                
        return metrics_dict

    def on_fully_initialized_for_wandb(self, config=None):
        """
        Call this method after the model (and its MLP head, if any) is fully initialized and after wandb.init() (or acc.init_trackers(...)) has been called. This will call wandb.watch(self) only once, if wandb is available, logging is enabled, and wandb.run is not None.
        """
        if getattr(self, '_wandb_watched', False):
            print("[BaseModule] wandb.watch already called, skipping")
            return
        try:
            import wandb
            if config is not None \
            and getattr(config, 'use_wandb_logging', False) \
            and (getattr(wandb, 'run', None) is not None):
                wandb_log_freq = getattr(config, 'wandb_log_freq', 2048)
                print(f"[BaseModule] Calling wandb.watch with log_freq={wandb_log_freq}")
                wandb.watch(self, log='all', log_freq=wandb_log_freq)
                self._wandb_watched = True
                print("[BaseModule] wandb.watch called successfully")
            else:
                print(f"[BaseModule] wandb.watch conditions not met: config={config is not None}, use_wandb_logging={getattr(config, 'use_wandb_logging', False) if config else False}, wandb.run={getattr(wandb, 'run', None) is not None}")
        except Exception as e:
            print(f"[BaseModule] error using wandb; will not log to wandb: {e}")


    def on_fully_initialized(self, config=None):
        """
        Call this method after the model (and its MLP head, if any) is fully initialized and after wandb.init() (or acc.init_trackers(...)) has been called. This will call any post-initialization hooks, such as wandb.watch.
        """
        self.on_fully_initialized_for_wandb(config)


    def run_epoch_zero_methods(self, Any):
        """
        Run any methods that need to be run at the start of the first epoch.
        """
        return None


    def get_denormalized(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Undo normalization of targets using the stored statistics.

        The center / scale buffers might have shape (num_targets,) or
        (1, num_targets) (older checkpoints). To ensure broadcasting works
        regardless of dimensionality we flatten away any singleton
        dimensions.
        """
        if not self.has_normalized_train_targets:
            return tensor

        center = self._target_center.to(tensor.device)
        scale = self._target_scale.to(tensor.device)

        # Remove possible (leading) singleton dimensions, e.g. (1, D) -> (D,)
        if center.dim() > 1:
            center = center.squeeze()
        if scale.dim() > 1:
            scale = scale.squeeze()

        return tensor * scale + center



def test_nn(
    trained_model: BaseModule,
    data_container: Dict[str, DataLoader] | Data,
    task: str,
    device: str = 'cpu',
    target_name: str = 'target',
    set_key: str = 'test',
    metrics_kwargs: Dict[str, Any] = {},
    using_pytorch_geo: bool = False,
    accelerator: Optional[Any] = None,  # Add accelerator parameter
    verbosity: int = 0
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Computes standard regression or binary
    classification metrics for the 'test' set 
    in a data_container, given a trained 
    BaseModule (or extension).

    Args:
        trained_model: trained BaseModule model.
        task: string key coding model task, e.g.,
            'regression' or 'binary classification.'
        device: device key on which the tensors live, 
            e.g. 'cpu' or 'cuda.'
        target_name: string key for the model target.
        data_container: dictionary of Data-
            Loaders, with a keyed 'test' set, or
            a pytorch geometric Data object (e.g.
            of one graph, with train/valid/test masks).
        set_key: which set ('train'/'valid'/'test') to 
            compute metrics for (default: 'test').
        metrics_kwargs: kwargs for setting up metric calcu-
            lator objects, e.g., num_classes for multiclass
            accuracy.
        using_pytorch_geo: whether the DataLoaders
            hold PyTorch Geometric datasets (i.e.
            where data are loaded as sparse block-
            diagonal matrices, requiring batch indices).
        accelerator: Optional accelerator object for DDP synchronization.
        verbosity: verbosity level for debug print statements
    Returns:
        2-tuple: (1) dictionary of metric scores,
        and (2) dictionary of other task-specific
        metric objects (e.g. a confusion matrix
        calculator object for classification).
    """
    if verbosity > 2:
        if accelerator is not None:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Starting test_nn evaluation')
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Calculating metrics for evaluation ({set_key}) set:')
        else:
            print(f'[DEBUG] Calculating metrics for evaluation ({set_key}) set:')

    task = task.lower()
    num_targets = metrics_kwargs.get('num_outputs', 1)
    
    # record raw model preds and targets, for option
    # to calculate other metrics no calculated here
    auxiliary_metrics_dict = {
        'preds': [],
        'targets': []
    }
    
    # set up metrics collections based on task
    metric_collection = None
    if 'reg' in task:
        if ('node' in task):
            # Node-level: custom MSE (graph-aware)
            # No MAE equivalent yet
            mse_calculator = MultiTargetMSE(
                num_targets=num_targets,
                mode='vector' if 'vector' in task else 'per_target'
            )
            # no MAE equivalent yet
            # metric_collection = MetricCollection({'mae': mae_calculator}).to(device)
        else:
            # Graph-level: standard torchmetrics for both MSE and MAE
            if num_targets > 1:
                mae_calculator = MeanAbsoluteError(
                    num_outputs=num_targets,
                    sync_on_compute=True
                )
            else:
                mae_calculator = MeanAbsoluteError(sync_on_compute=True)
            mse_standard = MeanSquaredError(sync_on_compute=True)
            metric_collection = MetricCollection({
                'mse': mse_standard,
                'mae': mae_calculator,
            }).to(device)
        
    elif 'class' in task and 'bin' in task:
        metric_collection = MetricCollection({
            'acc': BinaryAccuracy(),
            'f1': BinaryF1Score(),
            'ppv': BinaryPrecision(),
            'sensitivity': BinaryRecall(),
            'specificity': BinarySpecificity(),
            'auroc': BinaryAUROC()
        }).to(device)

        # auxiliary metrics (not part of the MetricCollection)
        bal_acc_calculator = Accuracy(
            task='multiclass', 
            num_classes=2, 
            average='macro'
        ).to(device)
        f1_neg_calculator = BinaryF1Score().to(device)
        confusion_mat_calculator = BinaryConfusionMatrix().to(device)
        auxiliary_metrics_dict |= {
            'bal_acc': bal_acc_calculator,
            'f1_neg': f1_neg_calculator,
            'confusion_matrix': confusion_mat_calculator,
        }

    elif 'class' in task and 'multi' in task:
        metric_collection = MetricCollection({
            'acc': Accuracy(
                task='multiclass', 
                num_classes=metrics_kwargs['num_classes']
            ),
            'bal_acc': Accuracy(
                task='multiclass', 
                num_classes=metrics_kwargs['num_classes'], 
                average='macro'
            )
        }).to(device)

    def update(preds, targets, *, batch_obj=None) -> None:
        """
        Inner function to update metrics (in containers);
        called more than once if calculating in batches.
        """
        # --------------------------------------------------
        # Ensure targets match prediction dimensionality
        # --------------------------------------------------
        # Subset target tensor if extra dimensions are present (e.g., QM9 19 targets)
        if ('reg' in task) and (preds.dim() <= targets.dim()):
            tii = metrics_kwargs.get('target_include_indices', None)
            if (tii is not None) and (targets.dim() == 2):
                # Slice columns before any further processing
                targets = targets[:, tii]

        # Record predictions and (possibly sliced) targets for auxiliary output
        auxiliary_metrics_dict['preds'].append(preds)
        auxiliary_metrics_dict['targets'].append(targets)
        
        if 'reg' in task:
            if ('node' in task):
                # Optional node-level grouping for per-graph normalization
                batch_index = None
                node_counts = None
                if using_pytorch_geo and (batch_obj is not None) and hasattr(batch_obj, 'batch'):
                    batch_index = batch_obj.batch
                    node_counts = torch.bincount(batch_index)
                elif (batch_obj is not None) and hasattr(batch_obj, 'valid_mask'):
                    # Single-graph masked case (valid/test mask applied upstream when present)
                    mask_attr = 'test_mask' if 'test' in set_key else ('val_mask' if 'val' in set_key else 'train_mask')
                    if hasattr(batch_obj, mask_attr):
                        mask = getattr(batch_obj, mask_attr)
                        node_counts = mask.sum()
                # Update custom MSE with grouping info when available
                mse_calculator.update(preds, targets, batch_index=batch_index, node_counts=node_counts)
            
        elif 'class' in task and 'bin' in task:
            targets = targets.long()
            preds_binary = (preds > 0.).long()

            bal_acc_calculator.update(preds_binary, targets)
            f1_neg_calculator.update(
                torch.logical_not(preds_binary).to(torch.long),
                torch.logical_not(targets).to(torch.long)
            )
            confusion_mat_calculator.update(preds_binary, targets)

        elif 'class' in task and 'multi' in task:
            targets = targets.long()
            preds = torch.argmax(preds, dim=1)

        if metric_collection is not None:
            metric_collection.update(preds, targets.squeeze())

    # Ensure all processes are synchronized before evaluation
    if accelerator is not None:
        if verbosity > 2:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Waiting for all processes before evaluation')
        accelerator.wait_for_everyone()
        if verbosity > 2:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: All processes synchronized, starting evaluation')
    
    # get model predictions on test set
    trained_model.eval()
    with torch.set_grad_enabled(False):
        if isinstance(data_container, dict):
            if accelerator is not None and verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Starting batch processing')
            
            for batch_i, batch in enumerate(data_container[set_key]):
                if accelerator is not None and batch_i % 10 == 0 and verbosity > 2:
                    accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Processing batch {batch_i}')
                
                if using_pytorch_geo:
                    data = batch.to(device)
                    batch_output_dict = trained_model(data)
                    preds = batch_output_dict['preds'].squeeze()
                    targets = batch[target_name]

                    if (batch.num_graphs == 1) \
                    and hasattr(batch, 'test_mask'):
                        preds = preds[batch.test_mask]
                        targets = targets[batch.test_mask]
                else:
                    if isinstance(batch, (tuple, list)):
                        features, targets = batch
                        features = features.to(device)
                        targets = targets.to(device).squeeze()
                        batch_output_dict = trained_model(features)
                        preds = batch_output_dict['preds'].squeeze() 
                    else:
                        batch_output_dict = trained_model(batch)
                        targets = batch[target_name]
                        if isinstance(targets, dict):
                            targets = targets[target_name]
                        targets = targets.squeeze()
                        preds = batch_output_dict['preds'].squeeze()
                
                # Unwrap model if it's a DDP model (wrapped by accelerate)
                if (accelerator is not None) and hasattr(trained_model, 'module'):
                    trained_model = trained_model.module

                if trained_model.has_normalized_train_targets:
                    preds = trained_model.get_denormalized(preds)
                    # note: targets are not normalized in test set
                update(preds, targets, batch_obj=batch if using_pytorch_geo else None)
                
                # Print progress only on main process
                if batch_i % 10 == 0 and verbosity > 2:
                    if (accelerator is not None) and accelerator.is_main_process:
                        accelerator.print(f"Processed {batch_i} batches...")
                    elif accelerator is None:
                        print(f"Processed {batch_i} batches...")

        elif using_pytorch_geo:
            if accelerator is not None and verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Processing single PyG graph')
            data = data_container.to(device)
            output_dict = trained_model(data)
            targets = data[target_name]
            preds = output_dict['preds'].squeeze()
            if ('val' in set_key):
                mask = data.val_mask
            elif ('test' in set_key):
                mask = data.test_mask
            elif ('train' in set_key):
                mask = data.train_mask
            if trained_model.has_normalized_train_targets:
                preds = trained_model.get_denormalized(preds)
                # targets are not normalized in test set
            update(preds[mask], targets[mask], batch_obj=data)
            
        # Ensure all processes have finished processing their batches
        if accelerator is not None:
            if verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Waiting for all processes to finish batch processing')
            accelerator.wait_for_everyone()
            if verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: All processes finished batch processing')
            
        # Compute metrics
        if accelerator is not None and verbosity > 2:
            accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Computing metrics')
        if metric_collection is not None:
            metric_scores_dict = metric_collection.compute()
        else:
            metric_scores_dict = {}

        # Add custom MSE result for node-level regression; for graph-level it is already in metric_collection
        if 'reg' in task and ('node' in task):
            metric_scores_dict['mse'] = mse_calculator.compute()
        
        # Process auxiliary metrics
        if auxiliary_metrics_dict is not None:
            if accelerator is not None and verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Processing auxiliary metrics')
            for k, v in auxiliary_metrics_dict.items():
                if k in ('preds', 'targets'):
                    # Patch: handle empty list case for DDP
                    if len(v) == 0:
                        # Try to infer shape from metric_collection or metrics_kwargs
                        if 'reg' in task:
                            # Regression: shape (0, num_targets)
                            shape = (0, num_targets)
                            dtype = torch.float32
                        else:
                            # Classification: shape (0,)
                            shape = (0,)
                            dtype = torch.float32
                        device_ = device if isinstance(device, torch.device) else torch.device(device)
                        metric_scores_dict[k] = torch.empty(shape, dtype=dtype, device=device_)
                    else:
                        metric_scores_dict[k] = torch.cat(v)
                else:
                    metric_scores_dict[k] = v.compute()

        # For auxiliary metrics that aren't DDP-aware, we need to gather them
        if accelerator is not None and accelerator.num_processes > 1:
            if accelerator.is_main_process and verbosity > 2:
                accelerator.print(f'[DEBUG] Main process: Starting auxiliary metrics gathering')
            # Patch: ensure all processes have a tensor to gather, even if empty
            for k in ['preds', 'targets']:
                if k in metric_scores_dict:
                    tensor = metric_scores_dict[k]
                    if tensor is None or tensor.numel() == 0:
                        # Try to infer shape from num_targets
                        if 'reg' in task:
                            shape = (0, num_targets)
                            dtype = torch.float32
                        else:
                            shape = (0,)
                            dtype = torch.float32
                        device_ = device if isinstance(device, torch.device) else torch.device(device)
                        metric_scores_dict[k] = torch.empty(shape, dtype=dtype, device=device_)
            # Gather predictions and targets for auxiliary metrics
            gathered_preds = accelerator.gather(metric_scores_dict['preds'])
            gathered_targets = accelerator.gather(metric_scores_dict['targets'])
            
            # Only main process computes auxiliary metrics
            if accelerator.is_main_process and verbosity > 2:
                accelerator.print(f'[DEBUG] Main process: Computing auxiliary metrics on gathered data')
                
                # Update auxiliary metrics with full dataset
                if 'class' in task and 'bin' in task:
                    bal_acc_calculator.reset()
                    f1_neg_calculator.reset()
                    confusion_mat_calculator.reset()
                    
                    preds_binary = (gathered_preds > 0.).long()
                    bal_acc_calculator.update(preds_binary, gathered_targets)
                    f1_neg_calculator.update(
                        torch.logical_not(preds_binary).to(torch.long),
                        torch.logical_not(gathered_targets).to(torch.long)
                    )
                    confusion_mat_calculator.update(preds_binary, gathered_targets)
                    
                    # Update auxiliary metrics in results
                    metric_scores_dict['bal_acc'] = bal_acc_calculator.compute()
                    metric_scores_dict['f1_neg'] = f1_neg_calculator.compute()
                    metric_scores_dict['confusion_matrix'] = confusion_mat_calculator.compute()
                    metric_scores_dict['preds'] = gathered_preds
                    metric_scores_dict['targets'] = gathered_targets
            
            # Broadcast auxiliary metrics to all processes
            if accelerator.is_main_process and verbosity > 2:
                accelerator.print(f'[DEBUG] Main process: Broadcasting auxiliary metrics')
            for k in ['bal_acc', 'f1_neg', 'confusion_matrix', 'preds', 'targets']:
                if k in metric_scores_dict and isinstance(metric_scores_dict[k], torch.Tensor):
                    if dist.is_initialized():
                        dist.broadcast(metric_scores_dict[k], src=0)

        if accelerator is not None:
            if verbosity > 2:
                accelerator.print(f'[DEBUG] Process {accelerator.process_index}: Evaluation complete')
            accelerator.print('\nDone calculating metrics.')
        else:
            print('\nDone calculating metrics.')
        
        return metric_scores_dict



class MetricDefinitions:
    """
    Centralized definitions for metrics used across different tasks.
    Ensures consistency between training validation metrics and test metrics.
    """
    
    # Core metrics that should be printed for each task type
    PRINTABLE_METRICS = {
        'regression': [
            'mse', 'mae'  # removed rmse
        ],
        'binary_classification': [
            'acc', 'f1', 'sensitivity', 'specificity', 'auroc'
        ],
        'multiclass_classification': [
            'acc', 'bal_acc'
        ]
    }
    
    # Mapping between different naming conventions used in calc_metrics vs test_nn
    METRIC_NAME_MAPPING = {
        # Standard name -> validation name (with _valid suffix)
        'acc': 'accuracy_valid',
        'bal_acc': 'bal_accuracy_valid', 
        'f1': 'f1_valid',
        'f1_neg': 'f1_neg_valid',
        'sensitivity': 'sensitivity_valid',  # Note: sensitivity = recall
        'specificity': 'specificity_valid',
        'auroc': 'auroc_valid',
        'mse': 'mse_valid',
        'mae': 'mae_valid'
    }
    
    @classmethod
    def get_printable_metrics_for_task(cls, task: str) -> List[str]:
        """
        Get the list of metrics that should be printed for a given task.
        
        Args:
            task: Task string (e.g., 'regression', 'binary_classification')
            
        Returns:
            List of metric names to print
        """
        task_lower = task.lower()
        
        if 'reg' in task_lower:
            return cls.PRINTABLE_METRICS['regression']
        elif 'class' in task_lower and 'bin' in task_lower:
            return cls.PRINTABLE_METRICS['binary_classification'] 
        elif 'class' in task_lower and 'multi' in task_lower:
            return cls.PRINTABLE_METRICS['multiclass_classification']
        else:
            # Return empty list for unknown tasks
            return []
    
    @classmethod
    def get_validation_metric_name(cls, metric_name: str) -> str:
        """
        Get the validation metric name (with _valid suffix) for a given metric.
        
        Args:
            metric_name: Standard metric name (e.g., 'acc', 'mse')
            
        Returns:
            Validation metric name (e.g., 'accuracy_valid', 'mse_valid')
        """
        return cls.METRIC_NAME_MAPPING.get(metric_name, metric_name + '_valid')


class MultiTargetMSE(Metric):
    """
    A custom metric class for computing Mean Squared Error (MSE) separately for each target 
    (if mode == 'per_target') in a multi-target regression task, or for a vector target 
    (if mode == 'vector').

    This metric is particularly useful when you need to track the MSE for each output
    independently, rather than computing a single aggregated MSE across all targets.

    Note that the incoming batch_size could be number of nodes or number of graphs,
    depending on the task.

    Attributes:
        num_targets (int): The number of target variables in the regression task.
        mode (str): 'per_target' computes per-target MSE, 'vector' computes vector-norm MSE.
        sum_mse_across_graphs (torch.Tensor): Running sum of per-graph mean squared errors.
            Shape is (num_targets,) in 'per_target' mode, or (1,) in 'vector' mode.
        graph_count (torch.Tensor): Running total of graphs aggregated across updates.

    Example:
        >>> metric = PerTargetMSE(num_targets=3)
        >>> preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> target = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
        >>> metric.update(preds, target)
        >>> mse_per_target = metric.compute()
        >>> metric.reset()
    """
    def __init__(
            self,
            num_targets: int,
            mode: str = 'per_target',
            dist_sync_on_step: bool = False,  # deprecated (more expensive)
            sync_on_compute: bool = True
        ) -> None:
        # Enable automatic cross-process reduction when .compute() is called
        super().__init__(
            dist_sync_on_step=dist_sync_on_step, 
            sync_on_compute=sync_on_compute
        )
        self.num_targets = num_targets
        self.mode = mode
        if mode not in ['per_target', 'vector']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'per_target' or 'vector'.")
        # Running sum over graphs of per-graph mean squared error
        # Shape: (num_targets,) for per_target else (1,) for vector
        out_dim = num_targets if mode == 'per_target' else 1
        self.add_state("sum_mse_across_graphs", default=torch.zeros(out_dim), dist_reduce_fx="sum")
        # Total number of graphs aggregated
        self.add_state("graph_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        *,
        batch_index: torch.Tensor | None = None,
        node_counts: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with a new batch.

        For node-level tasks (when ``batch_index`` is provided and preds are node-aligned),
        computes per-graph MSEs (mean over nodes in each graph) and accumulates their sum
        along with the number of graphs. For single-graph masked updates, pass
        ``node_counts=torch.tensor([num_masked_nodes])`` and leave ``batch_index`` as None.

        For graph-level tasks (no ``batch_index`` and no ``node_counts``), falls back to
        treating each row as a separate "graph" (i.e., standard mean across rows).
        """
        if preds.shape != target.shape:
            raise AssertionError(
                f"Predictions and targets must have the same shape. "
                f"Got {preds.shape} and {target.shape}."
            )

        device = preds.device
        # Ensure internal state tensors live on the same device as inputs
        if self.sum_mse_across_graphs.device != device:
            self.sum_mse_across_graphs = self.sum_mse_across_graphs.to(device)
        if self.graph_count.device != device:
            self.graph_count = self.graph_count.to(device)

        # Determine per-node squared error tensor
        if self.mode == 'per_target':
            # Shape: (N, T)
            per_node_sq_err = (preds - target) ** 2
        else:  # 'vector'
            # Sum across coordinates -> (N,)
            per_node_sq_err = ((preds - target) ** 2).sum(dim=1)

        # Case 1: Multi-graph node-level batch with graph assignments
        if batch_index is not None:
            # Normalize per graph by its node count (mean over nodes), then sum across graphs
            unique_graphs = torch.unique(batch_index)
            num_graphs_batch = unique_graphs.numel()

            if self.mode == 'per_target':
                # Accumulate sum of per-graph mean (shape (T,)) across graphs in this batch
                batch_sum = torch.zeros(self.num_targets, device=device)
                for g in unique_graphs:
                    mask = (batch_index == g)
                    graph_mean = per_node_sq_err[mask].mean(dim=0)  # (T,)
                    batch_sum += graph_mean
            else:  # 'vector'
                batch_sum = torch.tensor(0.0, device=device)
                for g in unique_graphs:
                    mask = (batch_index == g)
                    graph_mean = per_node_sq_err[mask].mean()  # scalar
                    batch_sum = batch_sum + graph_mean

            # Update running totals
            if self.mode == 'per_target':
                self.sum_mse_across_graphs += batch_sum
            else:
                self.sum_mse_across_graphs += batch_sum.unsqueeze(0)
            self.graph_count += torch.tensor(num_graphs_batch, device=device, dtype=torch.long)
            return

        # Case 2: Single-graph masked node-level batch (no batch_index but node_counts provided)
        if node_counts is not None and node_counts.numel() == 1:
            if self.mode == 'per_target':
                graph_mean = per_node_sq_err.mean(dim=0)  # (T,)
                self.sum_mse_across_graphs += graph_mean
            else:
                graph_mean = per_node_sq_err.mean()  # scalar
                self.sum_mse_across_graphs += graph_mean.unsqueeze(0)
            self.graph_count += torch.tensor(1, device=device, dtype=torch.long)
            return

        # Case 3: Graph-level batch (or fallback) — treat each row as a graph
        if self.mode == 'per_target':
            row_means = per_node_sq_err  # each row is one graph already
            # Sum across rows (graphs) -> (T,)
            batch_sum = row_means.mean(dim=0)  # mean across rows
            # To be consistent with per-graph averaging across batches, count number of rows as graphs
            num_graphs_batch = preds.shape[0]
            # Convert mean back to sum over graphs for accumulator:
            batch_sum = batch_sum * num_graphs_batch
            self.sum_mse_across_graphs += batch_sum
            self.graph_count += torch.tensor(num_graphs_batch, device=device, dtype=torch.long)
        else:
            # vector mode: per_node_sq_err is (N,), each row treated as its own graph
            batch_mean = per_node_sq_err.mean()
            num_graphs_batch = preds.shape[0]
            batch_sum = batch_mean * num_graphs_batch
            self.sum_mse_across_graphs += batch_sum.unsqueeze(0)
            self.graph_count += torch.tensor(num_graphs_batch, device=device, dtype=torch.long)

    def compute(self):
        # Avoid division by zero
        denom = torch.clamp(self.graph_count.to(torch.float32), min=1.0)
        return self.sum_mse_across_graphs / denom

    def reset(self):
        self.sum_mse_across_graphs.zero_()
        self.graph_count.zero_()