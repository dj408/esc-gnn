"""
Function to train/fine-tune a PyTorch model, 
using an Accelerator wrapper for CPU-GPU(s)
device support (for multi-GPU training, implements 
the distributed data parallel approach).

To use accelerate in a distributed environment, use command 
`accelerate launch` instead of `python3` when calling a script.

Example:
accelerate launch --num_processes=2 training_script.py

Ref:
https://huggingface.co/docs/accelerate/en/package_reference/accelerator
"""

from os_utilities import (
    ensure_dir_exists, 
    smart_pickle, 
    get_time_hr_min_sec
)
import models.nn_utilities as nnu
import os
import copy
import time
from datetime import datetime
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from accelerate.utils import DistributedDataParallelKwargs
# Conditionally import AcceleratedScheduler (may be unavailable in older Accelerate versions)
try:
    from accelerate.scheduler import AcceleratedScheduler
except (ImportError, ModuleNotFoundError):
    AcceleratedScheduler = None  # Fallback for older versions
# from dataclasses import dataclass
from config.train_config import TrainingConfig
from typing import Type, Optional, Tuple, Literal, List, Dict, Any
from models.base_module import MetricDefinitions
from training.metrics_utils import metric_to_str
from training.prep_model import load_pretrained_weights_if_specified

# Import guard for wandb
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None


def train_model(
    config: TrainingConfig,
    dataloader: Dict[str, DataLoader] | Data,
    model: nn.Module,
    optimizer_class: Type[optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    scheduler_class: Optional[Type[_LRScheduler]] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    accelerator: Optional[Accelerator] = None
) -> Tuple[
        Optional[nn.Module], # final model
        Optional[List[Dict[str, Any]]], # train history records
        Optional[nnu.EpochCounter] # EpochCounter object
]:
    """
    A general training function for base_module.BaseModule models, and
    potentially other pytorch models.

    Features:
    - Pass an optional `config.snapshot_path` to load an accelerate model 
      snapshot and resume training; else leave `None`.
    - Supports gradient clipping, learning rate scheduling, and mixed 
      precision training.
    - Optimizer and scheduler are initialized after the first forward pass
      to ensure all model parameters (including dynamically initialized ones)
      are properly set up.
    
    Notes:
    - The loss function must be an attrib. of the model, to get placed on 
      the correct device with Accelerate.
    - It's possible to keep training to the max number of epochs if
      validation loss kept improving by an arbitrarily small amount, but 
      return a final model from a much earlier epoch, if 
      'config.main_metric_rel_improv_thresh' is 1.0 or not None.
    
    Args:
        config: TrainingConfig object containing experiment configuration.
        dataloader: dictionary of Dataloaders by set, or pytorch_geometric 
            Data object containing the training data and set masks.
        model: the torch.nn.Module model object to train. Must have 
            'forward', 'loss', and 'update_metrics' methods that
            take dictionaries, as done in base_module.BaseModule.
        optimizer_class: The optimizer class to use (e.g., optim.Adam, optim.AdamW).
        optimizer_kwargs: Dictionary of keyword arguments to pass to the optimizer
            constructor. Note that 'lr' and 'weight_decay' will be overridden by
            values from config if specified.
        scheduler_class: Optional learning rate scheduler class. If provided, must be a 
            PyTorch scheduler class (e.g., ReduceLROnPlateau, CosineAnnealingLR).
        scheduler_kwargs: Optional dictionary of keyword arguments to pass to the
            scheduler constructor.
        accelerator: Optional pre-initialized Accelerator object. If None, a new one
            will be created using config parameters.
    Returns:
        3-tuple of (model, records, epoch_ctr) if 'config.return_best' is True, 
        or else (None, None, None) if error or 'config.return_best' is False. 
    """
    optimizer = None
    scheduler = None
    scheduler_step_kwargs = {}

    # Placeholder for best model parameters; set when return_best=True
    best_model_wts = None

    """
    Parameter validation
    """
    # Catch missing train_logs_save_dir before attempting to save logs
    if config.train_logs_save_dir is None and accelerator.is_main_process:
        raise ValueError("config.train_logs_save_dir cannot be None on main process")
    
    # Check if this is a single-graph dataset (Batch with 1 graph):
    # if so, it must have a 'set_node_mask' attribute
    if isinstance(dataloader, dict) and \
    (len(dataloader['train'].dataset) == 1):
        single_graph_task = True
        batch = next(iter(dataloader['train']))
        if ((not hasattr(batch, 'train_mask')) or (not hasattr(batch, 'valid_mask'))):
            raise Exception(
                "A single-graph training task using a DataLoader"
                " requires a 'train_mask' and 'valid_mask' attributes"
                " in the Batch/Data object."
            )
        if len(dataloader) > 1:
            raise Exception(
                "A single-graph training task using a DataLoader"
                " requires only one 'train' dataloader, with one Batch/Data"
                " object, with 'train_mask' and 'valid_mask' attributes."
            )
    else:
        single_graph_task = False

    """
    Inner functions
    """
    def _save_snapshot(name, acc):
        # We want the following layout:
        #   models/
        #     ├── best/        <- directory containing snapshot files
        #     ├── final/
        #     └── checkpoint/
        if name in {"best", "final", "checkpoint"}:
            save_dir = os.path.join(config.model_save_dir, name)
        else:
            save_dir = config.model_save_dir

        ensure_dir_exists(save_dir, raise_exception=True)
        acc.save_state(save_dir)


    def _run_dummy_forward_pass(model, dataloader, acc):
        """
        Run a dummy forward pass to initialize all submodules (for snapshot loading).
        - This will make sure all parameters exist, so their snapshot values can be loaded.
        """
        model.eval()
        with torch.no_grad():
            try:
                # Get a dummy batch from the dataloader
                if isinstance(dataloader, dict):
                    # Try in order: train -> valid -> test, skip empty shards
                    dummy_batch = None
                    for _key in ('train', 'valid', 'test'):
                        if _key in dataloader:
                            try:
                                dummy_batch = next(iter(dataloader[_key]))
                                break
                            except StopIteration:
                                dummy_batch = None
                    if dummy_batch is None:
                        raise ValueError("No non-empty DataLoader found for dummy forward pass on this rank.")
                elif isinstance(dataloader, Data):
                    dummy_batch = dataloader
                else:
                    dummy_batch = dataloader

                # Move dummy_batch to the accelerator device to ensure all
                # contained tensors (including sparse ones) live on the same
                # device before the forward pass.
                if hasattr(dummy_batch, 'to'):
                    dummy_batch = dummy_batch.to(acc.device)
                elif isinstance(dummy_batch, (tuple, list)):
                    dummy_batch = [x.to(acc.device) for x in dummy_batch]
                
                # Prefer model-provided lightweight initializer when available
                if hasattr(model, 'run_epoch_zero_methods'):
                    model.run_epoch_zero_methods(dummy_batch)
                else:
                    # Fallback: run a forward pass to initialize all submodules
                    model(dummy_batch)
                
            except Exception as e:
                import traceback, sys
                acc.print(f"\n[DEBUG] Dummy forward pass error: {e}")
                traceback.print_exc(file=sys.stdout)
                print()

    # --------------------------------------------------------------
    # Helper – list & print uninitialized Lazy modules (with labeling)
    # --------------------------------------------------------------
    def _print_uninitialized_lazy_modules(model, acc, label: str | None = None):
        """Print names of Lazy modules that still have uninitialized params."""
        try:
            from torch.nn.modules.lazy import LazyModuleMixin
        except ImportError:
            return
        uninit = [
            n for n, m in model.named_modules() \
            if isinstance(m, LazyModuleMixin) \
            and m.has_uninitialized_params()
        ]
        prefix = f"[DEBUG] Lazy module status ({label}):" if label else "[DEBUG] Lazy module status:"
        if uninit:
            acc.print(f"{prefix} {len(uninit)} uninitialized")
            for n in uninit:
                acc.print(f"  - {n}")
        else:
            acc.print(f"{prefix} all initialized")


    def _register_custom_checkpoints(acc, epoch_ctr):
        """
        Register custom objects for checkpointing.

        We only need to register objects that Accelerate does *not* already
        track automatically. Optimizer and scheduler are captured when they
        are wrapped by ``accelerate.prepare``.  Registering the scheduler a
        second time leads to a mismatch between the number of checkpoint
        files on disk and the number of registered objects when loading an
        older snapshot. Therefore we register *only* the EpochCounter.
        """
        acc.register_for_checkpointing(epoch_ctr)


    def _prepare_optimizer_and_scheduler(model, optimizer_class, optimizer_kwargs, scheduler_class, scheduler_kwargs, acc):
        """
        Create and prepare optimizer and scheduler as done after first forward pass.
        """
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        if scheduler_class is not None:
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            scheduler = acc.prepare(scheduler)
        else:
            scheduler = None
        optimizer = acc.prepare(optimizer)
        return optimizer, scheduler

    def _log_output(out, acc):
        # Always use acc.print for consistent output across processes
        acc.print(out)
        
        # Only main process writes to log file
        if (config.train_logs_save_dir is not None) \
        and acc.is_main_process:
            log_filepath = os.path.join(
                config.train_logs_save_dir,
                config.train_print_filename
            )
            with open(log_filepath, 'a') as f:
                f.write(out + '\n')

            
    def _train_batch(
        optimizer, scheduler, epoch, training, phase, batch_i, batch, acc
    ) -> Tuple[optim.Optimizer, _LRScheduler, Dict[str, Any], float]:
        """
        Helper function to process a single batch during training
        (train or validation phase).

        Note that the forward pass may initialize an MLP or other 
        layers, so the optimizer and scheduler must be initialized after
        the first forward pass.
        """
        # DEBUG: print batch number
        # print(f"\n[DEBUG] batch_i: {batch_i}")

        # Skip empty/None batches that can appear on some ranks when dataset
        # size is not perfectly divisible across DDP shards.
        if batch is None:
            return optimizer, scheduler, {}

        # Handle different types of batch data
        if config.using_pytorch_geo:
            # PyTorch-Geometric data
            # Move batch to device with non_blocking=True
            batch = batch.to(acc.device, non_blocking=config.non_blocking)

            # DEBUG: Only print if any batch index has zero nodes
            # if hasattr(batch, 'batch'):
            #     unique, counts = batch.batch.unique(return_counts=True)
            #     zero_indices = unique[counts == 0]
            #     if len(zero_indices) > 0:
            #         print(f"[DEBUG] WARNING: batch indices with zero nodes: {zero_indices.tolist()}")

            if epoch == 0:
                if hasattr(model, 'run_epoch_zero_methods'):
                    model.run_epoch_zero_methods(batch)
                if batch_i == 0:
                    out = f"[DEBUG] train._train_batch: run_epoch_zero_methods: Batch attribute shapes:\n{batch}"
                    _log_output(out, acc)

            # Time forward pass only for inference timing (validation)
            _forward_t0 = time.time()
            output_dict = model(batch)
            forward_elapsed_sec = float(time.time() - _forward_t0)
            preds = output_dict['preds']

            # Extract targets
            _target = batch[config.dataset_config.target_key]
            input_dict = {'target': _target}

            # Provide node-level grouping info for metrics when applicable
            # If the model outputs per-node predictions for a PyG batch with multiple graphs,
            # attach the per-node graph index vector and per-graph node counts so metrics can
            # normalize by graph size before averaging across graphs.
            if 'node' in config.dataset_config.task:
                try:
                    if hasattr(batch, 'batch'):
                        # Heuristic: node-level if preds align with nodes
                        num_nodes = getattr(batch, 'num_nodes', None)
                        x_nodes = getattr(batch, 'x', None)
                        n_nodes_from_x = x_nodes.shape[0] if isinstance(x_nodes, torch.Tensor) else None
                        is_node_level = False
                        if isinstance(preds, torch.Tensor):
                            if num_nodes is not None and preds.shape[0] == num_nodes:
                                is_node_level = True
                            elif n_nodes_from_x is not None and preds.shape[0] == n_nodes_from_x:
                                is_node_level = True
                        if is_node_level:
                            bi = batch.batch
                            input_dict['batch_index'] = bi
                            input_dict['node_counts'] = torch.bincount(bi)
                except Exception:
                    # Best-effort only; metrics will fall back to standard averaging
                    pass

            if batch_i == 0 and config.verbosity > 0:
                print(f"[DEBUG] train._train_batch (phase = {phase}, batch_i = {batch_i})")
                preds_print = [f'{v:.2f}' for v in preds[:5].squeeze().detach().cpu().tolist()]
                target_print = [f'{v:.2f}' for v in _target[:5].squeeze().detach().cpu().tolist()]
                print(f"\tpreds: {preds_print}")
                print(f"\ttarget: {target_print}")
        else:
            # Regular tensor data (from TensorDataset)
            # batch is a tuple of (features, targets)
            features, targets = batch
            # Move tensors to device with non_blocking=True
            features = features.to(acc.device, non_blocking=config.non_blocking)
            targets = targets.to(acc.device, non_blocking=config.non_blocking)
            input_dict = {'x': features, 'target': targets}
            _forward_t0 = time.time()
            output_dict = model(input_dict)
            forward_elapsed_sec = float(time.time() - _forward_t0)
            preds = output_dict['preds']
        
        # Initialize optimizer and scheduler after first forward pass 
        # if not done yet
        if optimizer is None:
            optimizer, scheduler = _prepare_optimizer_and_scheduler(
                model, optimizer_class, optimizer_kwargs, scheduler_class, scheduler_kwargs, acc
            )
        
        # For single-graph tasks, we need to mask the predictions and targets
        if config.using_pytorch_geo:
            node_mask_name = 'train_mask' if training else 'valid_mask'
            if hasattr(batch, node_mask_name):
                node_mask = batch[node_mask_name]
                output_dict['preds'] = preds[node_mask]
                input_dict['target'] = batch.y[node_mask]
                # Provide node-count for masked single-graph updates so per-graph
                # normalization can be applied in metrics
                if 'node' in config.dataset_config.task: 
                    try:
                        input_dict['node_counts'] = node_mask.sum()
                    except Exception:
                        pass

        # Calculate loss and check for NaN
        # Use model.module to access the current state of the model's attributes if in DDP
        real_model = model.module if hasattr(model, 'module') else model
        loss_dict = real_model.loss(input_dict, output_dict, phase)
        if torch.isnan(loss_dict['loss']):
            if torch.isnan(output_dict['preds']).any():
                acc.print(f"\ntrain._train_batch: output_dict['preds'] has nan value(s)")
            if torch.isnan(input_dict['target']).any():
                acc.print(f"\ntrain._train_batch: input_dict['target'] has nan value(s)")
            raise Exception("Loss function returned NaN!")
        
        # train phase only: backward pass and optimizer step
        if training:
            acc.backward(loss_dict['loss'])
            
            # Apply gradient clipping if specified
            if config.max_grad_norm is not None:
                acc.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            else:
                # Default gradient clipping if not specified
                acc.clip_grad_norm_(model.parameters(), 1.0)
            
            # log any tracked gradients and/or weights (before step) - main process only
            if config.grad_track_param_names is not None and acc.is_main_process:
                nnu.log_parameter_grads_weights(
                    path=config.train_logs_save_dir,
                    model=model,
                    grad_track_param_names=config.grad_track_param_names,
                    epoch_i=(epoch - 1), 
                    batch_i=batch_i
                )
            
            # After backward, identify any parameters without gradients (potentially unused)
            if acc.is_main_process and batch_i == 0 and epoch == 1:
                unused = [
                    name for name, p in model.named_parameters() \
                    if p.grad is None
                ]
                if unused:
                    acc.print(f"[DEBUG] Parameters with no gradients in first backward pass ({len(unused)}):")
                    for i, name in enumerate(unused[:20]):
                        acc.print(f"  ({i+1}) {name}")
                    if len(unused) > 20:
                        acc.print("  ... (truncated)")
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Ensure all processes are synchronized after optimizer step
            # (NOT needed: synchronization should already be done by optimizer.step() in DDP)
            # acc.wait_for_everyone()
        
        # Update train or validation batch loss and metrics
        real_model.update_metrics(
            phase, 
            loss_dict, 
            input_dict, 
            output_dict
        )
        
        # [Optional] classification task: update class 1 preds counts
        if (config.verbosity > 0) and ('class' in config.dataset_config.task):
            class1_preds_ctr.update(output_dict, phase)
            
        # NOTE: If DataLoader drop_last=False, we can get potential out-of-order collective calls that can trigger mismatched sequence errors (e.g., BROADCAST vs REDUCE) when some ranks finish a batch much earlier than others. Adding a sync here will prevent this, but could slow training with many GPUs (costing one all-reduce per batch). Alternatively, we could set drop_last=True, so each GPU gets the same number of batches.
        if not config.drop_last: 
            acc.wait_for_everyone()
        
        return optimizer, scheduler, input_dict, forward_elapsed_sec
            

    """
    INITIALIZE DIRS
    """
    # Only main process creates directories (wait for accelerator initialization first)
        
    # store metrics by epoch in list of dicts 
    records = []
    
    # initialize EpochCounter
    epoch_ctr = nnu.EpochCounter(0, config.main_metric)
    best_epoch = 1


    """
    ACCELERATOR WRAPPER
    """
    # Use provided accelerator or create new one
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config.ddp_find_unused_parameters)

    acc = accelerator if accelerator is not None else Accelerator(
        device_placement=False if config.using_pytorch_geo else config.device,  # Disable automatic device placement to avoid issues with PyG Data objects
        # cpu=(not torch.cuda.is_available()),
        mixed_precision='no' if config.mixed_precision == 'none' else config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        split_batches=config.dataloader_split_batches,
        kwargs_handlers=[ddp_kwargs] if config.ddp_find_unused_parameters else None
    )

    # Manually move model to device before acc.prepare
    model = model.to(acc.device)

    # ------------------------------------------------------------------
    # Run a dummy forward pass BEFORE wrapping the model with DDP
    # ------------------------------------------------------------------
    # Some models lazily create sub-modules during the first forward pass
    # (either via PyTorch LazyModules or by constructing layers on-demand).
    # Ensure all parameters are created before DDP hooks are attached.
    def _has_uninitialized_lazy_params(mod: nn.Module) -> bool:
        try:
            from torch.nn.modules.lazy import LazyModuleMixin  # type: ignore
        except Exception:
            return False
        for _name, _m in mod.named_modules():
            if isinstance(_m, LazyModuleMixin) and _m.has_uninitialized_params():
                return True
        return False

    # Print status before attempting initialization
    _print_uninitialized_lazy_modules(model, acc, label='before initialization')

    need_dummy_init = bool(getattr(model, 'has_lazy_parameter_initialization', False)) or _has_uninitialized_lazy_params(model)
    if need_dummy_init:
        try:
            _run_dummy_forward_pass(model, dataloader, acc)
        except Exception as e:
            acc.print(f"[WARNING] Dummy forward pass before DDP wrapping failed: {e}")

    # Print status after attempting initialization
    _print_uninitialized_lazy_modules(model, acc, label='after initialization')

    # Optional: load pretrained weights (after lazy initialization)
    load_pretrained_weights_if_specified(model, config)

    # Prepare model for distributed training
    model = acc.prepare(model)

    # Print/log total and trainable params
    if acc.is_main_process:
        # only need to count on one process
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except ValueError:
            # Fallback: skip uninitialized parameters when counting
            try:
                from torch.nn.parameter import UninitializedParameter  # type: ignore
                total_params = sum(p.numel() for p in model.parameters() if not isinstance(p, UninitializedParameter))
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad and not isinstance(p, UninitializedParameter))
                _log_output("[INFO] Skipping uninitialized parameters during counting.", acc)
            except Exception:
                raise
        out = f"Model parameters:\n - Total: {total_params}\n - Trainable: {trainable_params}"
        _log_output(out, acc)
    
    # Initialize wandb.watch after model is fully initialized (main process only)
    if getattr(config, 'use_wandb_logging', False) \
    and _WANDB_AVAILABLE \
    and acc.is_main_process:
        real_model = model.module if hasattr(model, 'module') else model
        if hasattr(real_model, 'on_fully_initialized_for_wandb'):
            real_model.on_fully_initialized_for_wandb(config)
        else:
            # Fallback: call wandb.watch directly if model doesn't have the method
            wandb_log_freq = getattr(config, 'wandb_log_freq', 2048)
            wandb.watch(model, log='all', log_freq=wandb_log_freq)
    
    if isinstance(dataloader, dict):
        dataloader['train'], dataloader['valid'] = acc.prepare(
            dataloader['train'], 
            dataloader['valid']
        )
    elif isinstance(dataloader, Data):
        dataloader = acc.prepare(dataloader)
    
    # Initialize optimizer and scheduler as None (set in dummy forward pass)
    optimizer = None
    scheduler = None

    # If loading a snapshot, run a dummy forward pass, create optimizer/scheduler, and register checkpoints in correct order
    if config.snapshot_name is not None:
        optimizer, scheduler = _prepare_optimizer_and_scheduler(
            model, optimizer_class, optimizer_kwargs, scheduler_class, scheduler_kwargs, acc
        )
        _register_custom_checkpoints(acc, epoch_ctr)
    else:
        # Only register epoch_ctr if not resuming
        acc.register_for_checkpointing(epoch_ctr)
    
    # Create directories (main process only)
    if config.model_save_dir is not None and acc.is_main_process:
        ensure_dir_exists(config.model_save_dir, raise_exception=True)
    
    # Wait for directory creation before proceeding
    acc.wait_for_everyone()
    
    """
    Log key settings
    """
    # log training hardware info
    num_devices, device_type, distr_type = (
        acc.num_processes, 
        acc.device.type.upper(), 
        acc.distributed_type
    )
    distributed_str = distr_type.split('.')[0]
    out = f"Training on {num_devices} x {device_type}" \
        + f" device (distributed: {distributed_str})" \
        + f" with mixed precision ('{config.mixed_precision}')"
    _log_output(out, acc)

    # Print device assignment for each process to confirm multi-GPU usage
    acc.print(
        f"[DEBUG] Process {acc.process_index + 1}/{acc.num_processes} "
        f"on device: {acc.device} "
        f"(distributed type: {acc.distributed_type})"
    )
    if torch.cuda.is_available():
        acc.print(
            f"[DEBUG] CUDA device name: {torch.cuda.get_device_name(acc.device)}"
        )

    # Ensure all processes are synchronized after printing
    acc.wait_for_everyone()
    
    # ------------------------------------------------------------------
    # Log effective batch size (per rank and global)
    # ------------------------------------------------------------------
    # if isinstance(dataloader, dict):
    #     _train_dl = dataloader.get('train', None)
    # else:
    #     _train_dl = dataloader

    # if _train_dl is not None:
    #     local_bs = getattr(_train_dl, 'batch_size', None)
    #     # For DataLoader shards created by Accelerate, `.total_batch_size` is
    #     # the global batch size before splitting.
    #     global_bs = getattr(_train_dl, 'total_batch_size', None)
    #     if global_bs is None and local_bs is not None:
    #         global_bs = local_bs * acc.num_processes if config.dataloader_split_batches else local_bs

    #     acc.print(
    #         f"[BATCH SIZE] local={local_bs}, global={global_bs} "
    #         f"(split_batches={config.dataloader_split_batches})"
    #     )

    """
    [Optional] Load model snapshot
    - to resume training from saved model state
    """
    if config.verbosity > 0:
        acc.print(f"[DEBUG] Entering snapshot loading block. config.snapshot_name = {config.snapshot_name}")
        acc.print(f"[DEBUG] model_save_dir = {config.model_save_dir}")
    if config.snapshot_name is not None:
        # --------------------------------------------------------------
        # NEW – simplified snapshot loading (flat directory layout)
        # --------------------------------------------------------------
        snapshot_dir = os.path.join(config.model_save_dir, config.snapshot_name)

        if config.verbosity > 0:
            acc.print(f"[DEBUG] Looking for snapshot directory: {snapshot_dir}")

        if os.path.exists(snapshot_dir):
            try:
                acc.load_state(snapshot_dir)
                out = (
                    f"Resumed training from '{config.snapshot_name}' snapshot "
                    f"at epoch {epoch_ctr.n} ({snapshot_dir})"
                )
                # ------------------------------------------------------
                # Optional LR override — if the user supplied a new
                # learning-rate (via CLI `--learn_rate`) it is stored in
                # `config.optimizer_config.learn_rate`.  After restoring the
                # optimizer state we force-set each param-group to the
                # requested value so training continues with the new LR.
                # ------------------------------------------------------
                new_lr = getattr(config.optimizer_config, 'learn_rate', None)
                if new_lr is not None and optimizer is not None:
                    lr_changed = False
                    for pg in optimizer.param_groups:
                        if pg.get('lr', None) != new_lr:
                            pg['lr'] = new_lr
                            lr_changed = True
                    if lr_changed:
                        # Update scheduler base_lrs if present
                        if scheduler is not None and hasattr(scheduler, 'base_lrs'):
                            scheduler.base_lrs = [new_lr for _ in scheduler.base_lrs]
                        acc.print(f"[INFO] Learning rate overridden to {new_lr} after snapshot load.")
                _log_output(out, acc)
            except Exception as e:
                out = (
                    f"Error loading '{config.snapshot_name}' snapshot at "
                    f"'{snapshot_dir}': {e}"
                )
                _log_output(out, acc)
                raise
        else:
            out = f"Snapshot directory '{snapshot_dir}' not found."
            _log_output(out, acc)
            raise FileNotFoundError(out)

    
    """
    Training loop
    """
    time_0 = time.time()
    ul_str = '-' * 16
    num_epochs_no_metric_improve = 0
    best_main_metric_score = -1
    last_epoch_flag = False    

    # ------------------------------------------------------------------
    # Helper – freeze BatchNorm stats (set to eval) when requested
    # ------------------------------------------------------------------
    def _freeze_batch_norm(mod):
        """
        Recursively set all BatchNorm layers to eval mode and disable stats updates."""
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()
            # Ensure the module does not update running stats even if .train() is called later
            mod.track_running_stats = False
    bn_frozen = False

    # classification task: print class 1 preds proportion for
    # each epoch and phase; here, init empty counters container
    if (config.verbosity > 0) and ('class' in config.dataset_config.task):
        class1_preds_ctr = nnu.Class1PredsCounter()

    # --- LR scheduler tracking ---
    last_lr = None
    last_lr_change_epoch = epoch_ctr.n
    if scheduler is not None and hasattr(optimizer, 'param_groups'):
        last_lr = optimizer.param_groups[0]['lr']
    # ----------------------------
    # Plateau restart tracking (reload-best + LR reduce)
    plateau_restart_count = 0

    # loop through (marginal) epochs
    last_ctr_epoch = epoch_ctr.n + config.n_epochs
    for epoch in range(epoch_ctr.n + 1, last_ctr_epoch + 1):
        time_epoch_0 = time.time()
        epoch_ctr += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out = f'\nEpoch {epoch}/{config.n_epochs}\n{ul_str}\n[{current_time}]'
        _log_output(out, acc)

        # Optional: freeze BatchNorm layers at the specified epoch (once)
        if (not bn_frozen) \
        and (config.batch_norm_freeze_epoch is not None) \
        and (config.batch_norm_freeze_epoch > 0) \
        and (epoch >= config.batch_norm_freeze_epoch):
            real_model = model.module if hasattr(model, 'module') else model
            real_model.apply(_freeze_batch_norm)
            bn_frozen = True
            _log_output(f"[INFO] BatchNorm layers frozen at epoch {epoch}", acc)

        # each epoch has a training and maybe a validation phase
        is_validation_epoch = (epoch % config.validate_every == 0)
        phases = ('train', 'valid') if is_validation_epoch else ('train', )
        if single_graph_task:
            # There is only one dataloader, with one Batch, with set masks
            # so we still access the 'train' set even in a validation epoch
            dataloader_phase_keys = ('train', )
        else:
            dataloader_phase_keys = phases

        # Track per-phase timing within the epoch
        train_time_this_epoch = 0.0
        valid_inference_time_this_epoch = 0.0

        for phase, dataloader_phase_key in zip(phases, dataloader_phase_keys):
            training = (phase == 'train')
            model.train() if training else model.eval()

            with torch.set_grad_enabled(training):
                phase_t0 = time.time()
                forward_time_this_phase = 0.0
                for batch_i, batch in enumerate(dataloader[dataloader_phase_key]):
                    optimizer, scheduler, input_dict, fwd_sec = _train_batch(
                        optimizer, scheduler, epoch, training, phase, batch_i, batch, acc
                    )
                    # Accumulate forward pass time for this phase
                    forward_time_this_phase += float(fwd_sec)
                    # On first epoch
                    if epoch == 1:
                        # save initial model weights if specified
                        if config.return_best:
                            # Always save from the unwrapped model to avoid 'module.' prefixes
                            _real = model.module if hasattr(model, 'module') else model
                            best_model_wts = copy.deepcopy(_real.state_dict())

                        # save initial model weights/gradients (main process only)
                        if (config.grad_track_param_names is not None) and acc.is_main_process:
                            nnu.log_parameter_grads_weights(
                                path=config.train_logs_save_dir,
                                model=model,
                                grad_track_param_names=config.grad_track_param_names,
                                epoch_i=-1, 
                                batch_i=-1,
                                save_grads=config.save_grads
                            )
                        

                # End of phase timing
                phase_elapsed = time.time() - phase_t0
                if training:
                    # For training, count the entire phase (forward + loss + backward)
                    train_time_this_epoch += float(phase_elapsed)
                else:
                    # For validation, count only the forward pass time across batches
                    valid_inference_time_this_epoch += float(forward_time_this_phase)

        # Synchronize all ranks at the end of the train/valid phase loop
        # This guarantees that every process completed the same number of batches before moving on to epoch-level metric reductions and scheduler steps.
        acc.wait_for_everyone()

        # after epoch's train and maybe valid sets are complete
        # calc epoch losses/metrics (updated with every batch)
        real_model = model.module if hasattr(model, 'module') else model
        epoch_hist_d = real_model.calc_metrics(
            epoch, 
            is_validation_epoch
        )

        # ------------------------------------------------------------------
        # Aggregate metrics across ranks (DDP) so every process sees the same
        # values before any control-flow or scheduler logic that depends on
        # them.  For loss averages we take a mean across ranks.
        # ------------------------------------------------------------------
        if acc.num_processes > 1:
            # Keys that always exist
            keys_to_reduce = ['loss_train']
            # Keys that exist only on validation epochs
            if is_validation_epoch:
                keys_to_reduce += ['loss_valid', config.main_metric]
            for _k in keys_to_reduce:
                if _k in epoch_hist_d:
                    _t = torch.tensor(
                        epoch_hist_d[_k], 
                        device=acc.device, 
                        dtype=torch.float32
                    )
                    _t = acc.reduce(_t, reduction='mean')  # All-reduce mean
                    epoch_hist_d[_k] = _t.item()

            # Special-case weighted reduction for node-level regression MSE
            # For node-level regression metrics (custom MultiTargetMSE), we reduce via a weighted
            # average using the stored numerator/denominator to avoid empty-shard
            # issues where some ranks may have processed zero validation batches.
            # if is_validation_epoch and config.main_metric.startswith('mse'):
            #     # Expect optional accumulators set by BaseModule.calc_metrics when node-level
            #     numer_key = 'mse_valid_numer'
            #     denom_key = 'mse_valid_denom'
            #     if numer_key in epoch_hist_d and denom_key in epoch_hist_d:
            #         local_numer = torch.as_tensor(epoch_hist_d[numer_key], device=acc.device, dtype=torch.float32)
            #         local_denom = torch.tensor(epoch_hist_d[denom_key], device=acc.device, dtype=torch.float32)
            #         # Reduce (sum) numerator and denominator across ranks
            #         global_numer = acc.reduce(local_numer, reduction='sum')
            #         global_denom = acc.reduce(local_denom, reduction='sum')
            #         # Compute safe mean (avoid divide-by-zero)
            #         if torch.is_tensor(global_denom) and global_denom.numel() == 1:
            #             denom_val = max(1.0, float(global_denom.item()))
            #         else:
            #             denom_val = float(global_denom.max(torch.tensor(1.0, device=acc.device)))
            #         global_mean = (global_numer / denom_val).detach().cpu().numpy()
            #         # Overwrite validation MSE with globally correct value
            #         epoch_hist_d['mse_valid'] = global_mean.item() if isinstance(global_mean, float) or (hasattr(global_mean, 'shape') and getattr(global_mean, 'shape', ()) == ()) else global_mean
        # ------------------------------------------------------------------

        # Log to wandb via accelerator.log if enabled (main process only)
        if getattr(config, 'use_wandb_logging', False):
            if _WANDB_AVAILABLE:
                pass  # wandb.watch(model) should be called only after model is fully initialized (e.g., after first forward pass for ESCGNN)
                # The model is responsible for calling wandb.watch(self) after full initialization.
            else:
                acc.print('[WARNING] wandb not installed, but use_wandb_logging=True')
            log_dict = {k: float(v) if hasattr(v, 'item') else v for k, v in epoch_hist_d.items() if isinstance(v, (int, float)) or (hasattr(v, 'item') and v.size == 1)}
            log_dict['epoch'] = epoch
            acc.log(log_dict, step=epoch)

        # Defer LR scheduler step until after plateau/restart logic
        # classification task: print class 1 preds counts
        if is_validation_epoch and (config.verbosity > 0) \
        and ('class' in config.dataset_config.task):
            class1_preds_ctr.print_preds_counts()
            class1_preds_ctr.reset()

        # Attach timing details and epoch number; we persist only on validation epochs
        epoch_hist_d['epoch'] = epoch
        epoch_hist_d['train_time_sec'] = float(train_time_this_epoch)
        epoch_hist_d['valid_inference_time_sec'] = float(valid_inference_time_this_epoch)

        # log/print losses
        train_loss = epoch_hist_d['loss_train']
        valid_loss = epoch_hist_d['loss_valid'] if is_validation_epoch else None
        epoch_time_elapsed = time.time() - time_epoch_0
        epoch_min, epoch_sec = int(epoch_time_elapsed // 60), epoch_time_elapsed % 60
        out = f"\nlosses ({config.loss_fn}):\n\ttrain: {train_loss:.6e}"
        if is_validation_epoch:
            out += f"\n\tvalid: {valid_loss:.6e}"
            out += "\nvalid set metrics:"

            # For regression tasks, append MAE and MSE metrics for validation set
            if 'reg' in config.dataset_config.task.lower():
                for metric_key in ('mae', 'mse'):
                    val_key = MetricDefinitions.get_validation_metric_name(metric_key)
                    if val_key in epoch_hist_d:
                        out += "\n\t" + metric_to_str(metric_key, epoch_hist_d[val_key], proportion_check=False)
            elif 'class' in config.dataset_config.task.lower():
                # Print common classification metrics defined for the task
                cls_metrics = MetricDefinitions.get_printable_metrics_for_task(
                    config.dataset_config.task
                )
                for metric_key in cls_metrics:
                    val_key = MetricDefinitions.get_validation_metric_name(metric_key)
                    if val_key in epoch_hist_d:
                        out += "\n\t" + metric_to_str(metric_key, epoch_hist_d[val_key], proportion_check=True)

            if scheduler is not None:
                out += f"\nlearning rate: {optimizer.param_groups[0]['lr']:.2e}"

        # Print time elapsed and phase breakdown
        out += f"\nepoch time: {epoch_min}m, {epoch_sec:.4f}s"
        out += f"\nphase times: train={train_time_this_epoch:.4f}s"
        if is_validation_epoch:
            out += f", inference={valid_inference_time_this_epoch:.4f}s"
        _log_output(out, acc)

        # Periodic checkpointing (main process only)
        if (
            config.checkpoint_every and config.checkpoint_every > 0 and
            (epoch % config.checkpoint_every == 0) and acc.is_main_process
        ):
            _save_snapshot('checkpoint', acc)
            _log_output(f'Checkpoint saved for epoch {epoch}.', acc)

        # validation phases: early stopping and train history / best model saving steps
        if is_validation_epoch:
            # Track whether we trigger a plateau restart this epoch
            did_plateau_restart = False

            # reset 'new best metric score reached' flags
            new_best_score_reached, score_thresh_reached = False, False

            # grab current epoch's score for main metric
            epoch_main_metric_score = epoch_hist_d[config.main_metric]

            # first validation phase only: set first main metric score as the 
            # score to beat in subsequent validation epochs
            if epoch_ctr.n == config.validate_every:
                epoch_ctr.set_best(config.main_metric, epoch, epoch_main_metric_score)
            
            # Reset/improve counter based on *main metric* improvement
            best_main_metric_score = epoch_ctr.best[config.main_metric]['score']
            if (
                (config.main_metric_is_better == 'lower' and epoch_main_metric_score < best_main_metric_score) or
                (config.main_metric_is_better == 'higher' and epoch_main_metric_score > best_main_metric_score)
            ):
                num_epochs_no_metric_improve = 0
            else:
                num_epochs_no_metric_improve += config.validate_every

            # if in final desired epoch, or (burn-in period passed AND 'patience' num 
            # epochs w/o valid loss improvement reached): set 'last_epoch_flag=True', 
            # which will break the epochs' for-loop at end of the current epoch
            if (epoch == last_ctr_epoch):
                last_epoch_flag = True
                out = f'Reached final desired epoch {last_ctr_epoch}.'
                _log_output(out, acc)
                break
            elif (
                (epoch > config.burnin_n_epochs)
            ):
                # Determine patience for plateau detection (share with scheduler when enabled)
                plateau_patience = config.scheduler_config.patience
                use_reload_on_plateau = bool(config.scheduler_config.reload_best_on_plateau)
                plateau_max_restarts = int(config.scheduler_config.plateau_max_restarts or 0)
                scheduler_factor = float(config.scheduler_config.factor)
                min_lr_cfg = float(config.scheduler_config.min_lr or 0.0)

                # Fallback patience to early-stopping patience if scheduler patience is missing
                if plateau_patience is None:
                    plateau_patience = config.no_valid_metric_improve_patience

                if num_epochs_no_metric_improve >= plateau_patience:
                    if use_reload_on_plateau and (plateau_restart_count < plateau_max_restarts):
                        # Reload best weights kept in-memory to avoid I/O
                        real_model = model.module if hasattr(model, 'module') else model
                        if best_model_wts is not None:
                            real_model.load_state_dict(best_model_wts)
                        else:
                            _log_output('[PLATEAU] Warning: best_model_wts is None; skipping weight reload.', acc)

                        # Reduce LR using scheduler factor and clamp by min_lr
                        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                            current_lr = optimizer.param_groups[0].get('lr', None)
                        else:
                            current_lr = None

                        new_lr = None
                        if current_lr is not None:
                            new_lr = max(current_lr * scheduler_factor, min_lr_cfg)

                        # Recreate optimizer/scheduler to reset internal state with new LR
                        if new_lr is not None:
                            opt_kwargs_restart = dict(optimizer_kwargs)
                            opt_kwargs_restart['lr'] = new_lr
                        else:
                            opt_kwargs_restart = optimizer_kwargs
                        optimizer, scheduler = _prepare_optimizer_and_scheduler(
                            model, optimizer_class, opt_kwargs_restart, scheduler_class, scheduler_kwargs, acc
                        )

                        plateau_restart_count += 1
                        num_epochs_no_metric_improve = 0
                        did_plateau_restart = True
                        # Sync LR-tracking with manual reduction to avoid false "scheduler reduced" logs later
                        try:
                            if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
                                last_lr = optimizer.param_groups[0]['lr']
                                last_lr_change_epoch = epoch
                                # Also align scheduler base_lrs if present
                                if scheduler is not None and hasattr(scheduler, 'base_lrs'):
                                    try:
                                        scheduler.base_lrs = [last_lr for _ in scheduler.base_lrs]
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        if new_lr is not None:
                            _log_output(
                                f"[PLATEAU] Reloaded best weights and reduced LR to {new_lr:.2e} (restart {plateau_restart_count}/{plateau_max_restarts}).",
                                acc
                            )
                        else:
                            _log_output(
                                f"[PLATEAU] Reloaded best weights (no LR change detected). Restart {plateau_restart_count}/{plateau_max_restarts}.",
                                acc
                            )
                    else:
                        if (config.stop_rule is not None) and ('no' in config.stop_rule) and ('improv' in config.stop_rule):
                            last_epoch_flag = True
                            out = f'Validation loss did not improve for' \
                                  + f' {num_epochs_no_metric_improve} epochs: stopping.'
                            _log_output(out, acc)
                            break

            # check for new best key validation score (by a relative margin)
            best_main_metric_score = epoch_ctr.best[config.main_metric]['score']
            if config.main_metric_is_better == 'lower':
                score_thresh = best_main_metric_score \
                    if config.main_metric_rel_improv_thresh is None \
                    else (best_main_metric_score * (1. - config.main_metric_rel_improv_thresh))
                score_thresh_reached = (epoch_main_metric_score < score_thresh)
            elif config.main_metric_is_better == 'higher':
                score_thresh = best_main_metric_score \
                    if config.main_metric_rel_improv_thresh is None \
                    else (best_main_metric_score * (1. + config.main_metric_rel_improv_thresh))
                score_thresh_reached = (epoch_main_metric_score > score_thresh)
    
            # if new best validation score threshold reached, record it
            if score_thresh_reached:
                new_best_score_reached = True
                best_main_metric_score = epoch_hist_d[config.main_metric]
                # Ensure scalar for storage/logging if metric returns size-1 arrays
                try:
                    if hasattr(best_main_metric_score, 'size') and getattr(best_main_metric_score, 'size', 0) == 1:
                        best_main_metric_score = best_main_metric_score.item()
                except Exception:
                    pass
                epoch_ctr.set_best(config.main_metric, epoch, best_main_metric_score)
                # epoch_key = f"epoch_{epoch}"
                real_model = model.module if hasattr(model, 'module') else model
                real_model.on_best_model()

            # append this epoch's losses, metrics, and time elapsed to records ONLY on validation epochs
            # include epoch training time and reigning epoch with best validation score
            epoch_hist_d['sec_elapsed'] = epoch_time_elapsed
            epoch_hist_d['best_epoch'] = epoch_ctr.best[config.main_metric]['epoch']
            records.append(epoch_hist_d)

            # Incrementally persist training logs after each VALIDATION epoch (main process only)
            if acc.is_main_process and (config.train_logs_save_dir is not None):
                ensure_dir_exists(config.train_logs_save_dir, raise_exception=True)
                train_logs_filepath = os.path.join(
                    config.train_logs_save_dir,
                    config.train_logs_filename
                )
                # Overwrite to maintain a single up-to-date file
                smart_pickle(train_logs_filepath, records, overwrite=True)
    
            # if new best epoch (by main metric validation set score):
            if new_best_score_reached:
                out = f"-> New best model!"
                _log_output(out, acc)
                    
                if config.save_best_model_state:
                    _log_output(f'Saving best model...', acc)
                    _save_snapshot('best', acc)  # now saves to 'best' subfolder
                    # also save records up to best model in same dir as snapshot (main process only)
                    if acc.is_main_process:
                        records_filepath = os.path.join(config.model_save_dir, 'best', 'training_records.pkl')
                        smart_pickle(records_filepath, records, overwrite=True)
                    _log_output(f'Best model saved.', acc)
                        
                if config.return_best or config.scheduler_config.reload_best_on_plateau:
                    real_model = model.module if hasattr(model, 'module') else model
                    best_model_wts = copy.deepcopy(real_model.state_dict())
            else:
                # new main metric score wasn't the best seen
                prev_best_epoch = epoch_ctr.best[config.main_metric]['epoch']
                # Ensure printf-safe scalar formatting even if numpy array slipped through
                try:
                    bms_print = best_main_metric_score.item() if hasattr(best_main_metric_score, 'item') else float(best_main_metric_score)
                except Exception:
                    bms_print = best_main_metric_score
                out = f"[Current best epoch: {prev_best_epoch} ({config.main_metric}={bms_print:.4e})]"
                _log_output(out, acc)

            # last_epoch_flag has been set, break out of epochs' for-loop and
            # jump to POST-TRAINING section
            # After plateau logic, perform scheduler step unless we reloaded this epoch
            if (epoch >= config.burnin_n_epochs) and is_validation_epoch and (scheduler is not None) and (not did_plateau_restart):
                metric_val = epoch_hist_d[config.main_metric]
                sched_key = config.scheduler_config.scheduler_key
                if sched_key == 'ReduceLROnPlateau':
                    scheduler.step(metric_val)
                    if hasattr(optimizer, 'param_groups'):
                        current_lr = optimizer.param_groups[0]['lr']
                        if last_lr is not None and current_lr < last_lr:
                            epochs_since_last_change = epoch - last_lr_change_epoch
                            msg = (
                                f"[LR SCHEDULER] Learning rate reduced at epoch {epoch}: "
                                f"new lr = {current_lr:.2e} (was {last_lr:.2e}), "
                                f"{epochs_since_last_change} epoch(s) since last change."
                            )
                            _log_output(msg, acc)
                            last_lr_change_epoch = epoch
                        last_lr = current_lr
                else:
                    scheduler.step()

            if last_epoch_flag:
                break

    """
    Post-training
    """
    # get total time elapsed
    time_str = get_time_hr_min_sec(time.time(), time_0, return_str=True)
    out = f'{epoch_ctr.n} epochs complete in {time_str}'
    _log_output(out, acc)
    acc.print(out)

    # log final best validation score and epoch
    if epoch_ctr.n > config.burnin_n_epochs:
        best_epoch = epoch_ctr.best[config.main_metric]['epoch']
        out = f'Best {config.main_metric}: {best_main_metric_score:.4f}' \
            + f' at epoch {best_epoch}'
        _log_output(out, acc)

    # save final training log (main process only)
    if acc.is_main_process:
        ensure_dir_exists(config.train_logs_save_dir, raise_exception=True)
        train_logs_filepath = os.path.join(
            config.train_logs_save_dir, 
            config.train_logs_filename
        )
        # Overwrite final to avoid creating suffixed duplicates
        smart_pickle(train_logs_filepath, records, overwrite=True)
    _log_output('Final training log saved.', acc)

    # optional: save final epoch's model state
    if config.save_final_model_state:
        _save_snapshot('final', acc)  # saves to 'final' subfolder
        _log_output('Last model state saved.', acc)
        
    # optional: load best model weights and return tuple with history log
    if config.return_best:
        if best_model_wts is not None:
            out = f'Returning model with best weights (from epoch {best_epoch}).\n'
            _log_output(out, acc)
            real_model = model.module if hasattr(model, 'module') else model
            real_model.load_state_dict(best_model_wts)
            train_fn_output = (model, records, epoch_ctr)
        else:
            # Fallback: return the final model so that downstream code has
            # non-None objects and DDP ranks stay in sync.
            out = 'No best model weights were recorded — returning final model.\n'
            _log_output(out, acc)
            train_fn_output = (model, records, epoch_ctr)
    else:
        train_fn_output = (None, None, None)
    
    return train_fn_output

