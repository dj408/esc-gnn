#!/usr/bin/env python3

"""
TO DO
[ ] wandb logging? (cloud logging needs web proxies)
[ ] Add more datasets (in data_processing/process_pyg_data.py and dataset_config.post_init)
[ ] Add support for stratified cross-validation
"""
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
import yaml
import time
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import (
    broadcast_object_list, 
    DistributedDataParallelKwargs
)
import os
import sys
sys.path.insert(0, '../')
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from config.arg_parsing import get_clargs
from config.config_manager import ConfigManager
from config.train_config import TrainingConfig
from config.dataset_config import DatasetConfig
import training.train as train
from models.base_module import test_nn, MetricDefinitions
from training.prep_dataset import (
    load_dataset, 
    create_dataloaders, 
    get_torchmd_qm9_splits
)
from training.metrics_utils import metric_to_str
from training.prep_optimizer import prepare_optimizer, prepare_scheduler
from training.prep_model import prepare_escgnn_model, prepare_comparison_model
from os_utilities import create_experiment_dir, smart_pickle, ensure_dir_exists
from data_processing.data_utilities import (
    get_random_splits,
    get_kfold_splits,
    process_kfold_splits,
    multi_strat_multi_fold_idx_sample
)

# Import guard for wandb
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    wandb = None


def main_print(*args, timestamp=False, **kwargs):
    # Helper function for main process printing
    # Expects 'acc' or 'accelerator' to be in the caller's scope
    # (or pass as a kwarg if needed)
    acc = kwargs.pop('acc', None) or kwargs.pop('accelerator', None)
    config = kwargs.pop('config', None)
    if acc is not None:
        if acc.is_main_process:
            if timestamp:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}]", *args, **kwargs)
            else:
                print(*args, **kwargs)
            # Only main process writes to log file
            if config is not None and hasattr(config, 'train_logs_save_dir') and config.train_logs_save_dir is not None:
                log_filepath = os.path.join(
                    config.train_logs_save_dir,
                    getattr(config, 'train_logs_filename', 'logs.txt')
                )
                with open(log_filepath, 'a') as f:
                    if timestamp:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{current_time}] {' '.join(map(str, args))}\n")
                    else:
                        f.write(f"{' '.join(map(str, args))}\n")
    else:
        # Fallback: just print
        if timestamp:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}]", *args, **kwargs)
        else:
            print(*args, **kwargs)


def log_to_train_print(
    out: str,
    acc: Accelerator,
    config: TrainingConfig,
) -> None:
    """
    Append a line to the training print log file (train_print.txt) on the main process.
    """
    try:
        if acc.is_main_process and (config.train_logs_save_dir is not None):
            log_filepath = os.path.join(
                config.train_logs_save_dir,
                config.train_print_filename
            )
            with open(log_filepath, 'a') as f:
                f.write(out + '\n')
    except Exception as e:
        # Fallback warning; avoid raising to not disrupt training flow
        main_print(f"[WARNING] Could not write to train_print log: {e}", acc=acc, config=config)


def run_one_fold(
    config: TrainingConfig,
    dataloader_dict: Dict[str, Any],
    fold_idx: int = 0,
    save_results: bool = True,
    accelerator: Optional[Accelerator] = None
) -> Dict[str, Any]:
    """
    Run a single fold of training, validation, and testing for a model.
    
    This function handles the complete training pipeline for one fold, including:
    1. Preparing the optimizer and learning rate scheduler classes and kwargs
    2. Initializing the model
    3. Training the model
    4. Testing the model on the appropriate evaluation set
    5. Saving results if specified ('smart_pickle' prevents overwriting existing 
       files)
    
    Args:
        config: TrainingConfig object containing all training parameters and configurations
        dataloader_dict: Dictionary mapping set names ('train', 'valid', 'test') to their
            respective DataLoader objects
        fold_idx: Index of the current fold (default: 0, used for k-fold cross-validation)
        save_results: Whether to save the results to disk (default: True)
        accelerator: Optional Accelerator object for distributed training
        
    Returns:
        Dictionary containing:
        - 'model': Name of the model used
        - 'dataset': Name of the dataset used
        - 'fold_i': Index of the current fold
        - 'epoch_times': List of times taken for each epoch up to the best epoch
        - Metric scores from testing (keys depend on the task and metrics used)
        
    Raises:
        ValueError: If the model key in config is not supported
    """
    
    # Use the shared main_print function
    def _main_print(*args, **kwargs):
        main_print(*args, acc=accelerator, config=config, **kwargs)
    
    # Prepare optimizer and (optional) scheduler 
    # prep_start_time = time.time()
    # _main_print(f"\n{'='*60}")
    # _main_print(f"STARTING OPTIMIZER/SCHEDULER PREPARATION - FOLD {fold_idx}", timestamp=True)
    optimizer_class, optimizer_kwargs = prepare_optimizer(config.optimizer_config)
    if config.scheduler_config.scheduler_key:
        scheduler_class, scheduler_kwargs = prepare_scheduler(
            config.scheduler_config,
            config.validate_every
        )
    else:
        scheduler_class, scheduler_kwargs = None, None

    # prep_elapsed = time.time() - prep_start_time
    # prep_min, prep_sec = prep_elapsed // 60, prep_elapsed % 60
    # _main_print(f"Complete.")
    # _main_print(f"Total optimizer/scheduler preparation time: {prep_min}m {prep_sec:.2f}s")
    # _main_print(f"{'='*60}")

    # Prepare model
    model_start_time = time.time()
    _main_print(f"\n{'='*60}")
    _main_print(f"FOLD {fold_idx}: STARTING MODEL PREPARATION ['{config.model_config.model_key}']", timestamp=True)

    key_lower = config.model_config.model_key.lower()
    if 'escgnn' in key_lower:
        # ESCGNN-family models (escgnn, escgnn_radial, escgnn_modular)
        config, model, dataloader_dict = prepare_escgnn_model(config, dataloader_dict, acc=accelerator)
    elif key_lower == 'visnet':
        from training.prep_model import prepare_visnet_model  # local import to avoid circular deps when not needed
        config, model, dataloader_dict = prepare_visnet_model(config, dataloader_dict, acc=accelerator)
    elif key_lower in ('dimenet', 'schnet', 'egnn', 'tfn', 'legs', 'gin', 'gat', 'gcn'):
        # Comparison models prepared via dedicated wrapper
        config, model, dataloader_dict = prepare_comparison_model(config, dataloader_dict, acc=accelerator)
    else:
        raise ValueError(
            f"Model key '{config.model_config.model_key}' not supported"
        )
    if config.verbosity > 0:
        if config.dataset_config.target_preprocessing_type is not None:
            _main_print(f"Target rescaling stats set in model: center: {model._target_center}, scale: {model._target_scale}")
        else:
            _main_print(f"No target rescaling stats set in model")
    model_elapsed = time.time() - model_start_time
    model_min, model_sec = model_elapsed // 60, model_elapsed % 60
    _main_print(f"Complete.")
    _main_print(f"Total model preparation time: {int(model_min)}m {model_sec:.2f}s")

    # TRAINING START
    train_start_time = time.time()
    _main_print(f"\n{'='*60}")
    _main_print(f"FOLD {fold_idx}: STARTING TRAINING", timestamp=True)
        
    # Train model
    config.return_best = True
    trained_model, records, epoch_ctr = train.train_model(
        config,
        dataloader_dict,
        model,
        optimizer_class,
        optimizer_kwargs,
        scheduler_class,
        scheduler_kwargs,
        accelerator=accelerator
    )
    
    # TRAINING COMPLETE
    train_elapsed = time.time() - train_start_time
    _main_print(f"Complete.")
    train_min, train_sec = train_elapsed // 60, train_elapsed % 60
    _main_print(f"Training time: {int(train_min)}m {train_sec:.2f}s")
    # _main_print(f"{'='*60}")


    # Ensure we have an accelerator for DDP-aware testing
    if accelerator is None:
        raise ValueError("accelerator parameter is required for DDP-aware testing and results saving")

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    eval_start_time = time.time()
    _main_print(f"\n{'='*60}")
    _main_print(f"FOLD {fold_idx}: STARTING EVALUATION", timestamp=True)

    # Test model (DDP-aware: all processes evaluate their subset)
    eval_set_key = 'test'
    # The evaluation set is 'test' unless we're using k-fold cross-validation
    # and we are not using a test set.
    if (('kfold' in config.experiment_type) \
         and (not config.use_cv_test_set)):
        eval_set_key = 'valid'
    metrics_kwargs = {}
    metrics_kwargs['num_outputs'] = config.dataset_config.target_dim

    # Set model to eval mode for all processes
    trained_model.eval()
    
    # Ensure all processes are synchronized before evaluation
    accelerator.wait_for_everyone()

    # Call the new evaluation function
    fold_results_dict = run_evaluation(
        config=config,
        dataloader_dict=dataloader_dict,
        trained_model=trained_model,
        accelerator=accelerator,
        fold_idx=fold_idx,
        eval_set_key=eval_set_key,
        metrics_kwargs=metrics_kwargs,
        eval_start_time=eval_start_time
    )

    # Ensure all processes wait for testing and saving to complete
    accelerator.wait_for_everyone()

    # Append test metric printout to logs.txt (main process only)
    # NOW PRINTED UPSTREAM
    # if accelerator.is_main_process \
    # and (config.train_logs_save_dir is not None):
    #     log_filepath = os.path.join(
    #         config.train_logs_save_dir,
    #         getattr(config, 'train_logs_filename', 'logs.txt')
    #     )
    #     try:
    #         with open(log_filepath, 'a') as f:
    #             f.write(f"\n[TEST METRICS] Fold {fold_idx} results:\n")
    #             for k, v in fold_results_dict.items():
    #                 f.write(f"  {k}: {v}\n")
    #     except Exception as e:
    #         print(f"[WARNING] Could not write test metrics to log file {log_filepath}: {e}")

    # If using wandb, you can call wandb.watch(model) here after tracker init:
    #   import wandb; wandb.watch(model)

    return fold_results_dict


def run_evaluation(
    config: TrainingConfig,
    dataloader_dict: Dict[str, Any],
    trained_model: torch.nn.Module,
    accelerator: Accelerator,
    fold_idx: int = 0,
    eval_set_key: str = 'test',
    metrics_kwargs: Optional[Dict[str, Any]] = None,
    eval_start_time: Optional[float] = None
):
    """
    Run evaluation of a model and dataloader_dict using accelerator; print and save results.
    """
    if metrics_kwargs is None:
        metrics_kwargs = {}
    if eval_start_time is None:
        eval_start_time = time.time()
    
    # All processes evaluate their subset of data
    main_print(f"Running evaluation on {eval_set_key.upper()} set...", acc=accelerator, config=config)
    log_to_train_print(f"Running evaluation on {eval_set_key.upper()} set...", acc=accelerator, config=config)
    metric_scores_dict = test_nn(
        trained_model=trained_model,
        data_container=dataloader_dict,
        task=config.dataset_config.task,
        device=accelerator.device,  # Use accelerator device instead of config.device
        target_name=config.dataset_config.target_key,
        set_key=eval_set_key,
        metrics_kwargs=metrics_kwargs,
        using_pytorch_geo=config.using_pytorch_geo,
        accelerator=accelerator  # Pass accelerator for DDP synchronization
    )

    # Ensure all processes are synchronized after evaluation
    accelerator.wait_for_everyone()

    # Only main process handles results and printing
    if accelerator.is_main_process:
        # Parameter counts
        try:
            _model_for_count = trained_model.module if hasattr(trained_model, 'module') else trained_model
            num_params_total = sum(p.numel() for p in _model_for_count.parameters())
            num_params_trainable = sum(p.numel() for p in _model_for_count.parameters() if p.requires_grad)
        except Exception:
            num_params_total, num_params_trainable = None, None

        fold_results_dict = {
            'model': config.model_config.model_key,
            'dataset': config.dataset_config.dataset,
            'fold_i': fold_idx,
            'num_params_total': num_params_total,
            'num_params_trainable': num_params_trainable,
        }
        
        # Print evaluation set metric scores
        if config.dataset_config.target_dim > 1:
            main_print(f"[Target dim: {config.dataset_config.target_dim}]", acc=accelerator, config=config)
            log_to_train_print(f"[Target dim: {config.dataset_config.target_dim}]", acc=accelerator, config=config)
        printable_metrics = MetricDefinitions.get_printable_metrics_for_task(
            config.dataset_config.task
        )
        # Heading for test metrics per fold
        log_to_train_print(f"[TEST METRICS] Fold {fold_idx} results:", acc=accelerator, config=config)
        for metric, score in metric_scores_dict.items():
            score = score.detach().cpu().numpy()
            # Preserve array vs scalar in results dict as-is
            fold_results_dict[metric] = score

            if (config.verbosity > -1) and (metric in printable_metrics):
                _line = '\t' + metric_to_str(metric, score)
                main_print(_line, acc=accelerator, config=config)
                log_to_train_print(_line, acc=accelerator, config=config)

        # Save results (only main process)
        if hasattr(config, 'results_save_dir') and hasattr(config, 'results_filename'):
            import os
            from os_utilities import smart_pickle
            results_filepath = os.path.join(
                config.results_save_dir,
                config.results_filename
            )
            smart_pickle(results_filepath, fold_results_dict, overwrite=False)
        
        # EVALUATION COMPLETE
        import time
        eval_elapsed = time.time() - eval_start_time
        eval_min, eval_sec = eval_elapsed // 60, eval_elapsed % 60
        main_print(f"Complete.", acc=accelerator, config=config)
        log_to_train_print("Complete.", acc=accelerator, config=config)
        main_print(f"Evaluation time: {int(eval_min)}m {eval_sec:.2f}s", acc=accelerator, config=config)
        log_to_train_print(f"Evaluation time: {int(eval_min)}m {eval_sec:.2f}s", acc=accelerator, config=config)
    else:
        # Non-main processes: don't need to save results
        fold_results_dict = None
    return fold_results_dict


# --------------------------------------------------------------
# Helper – warn if global batch size exceeds split sizes
# (this could lead to cryptic errors like "UnboundLocalError: cannot access 
# local variable 'current_batch' where it is not associated with a value")
# --------------------------------------------------------------
# def _batch_size_sanity_check(split_sizes: dict, train_dl, acc, fold_idx=None):
#     """Print an all-caps warning if global batch size > any split size."""
#     try:
#         local_bs = getattr(train_dl, 'batch_size', None)
#         global_bs = getattr(train_dl, 'total_batch_size', None)
#         if global_bs is None and local_bs is not None:
#             global_bs = local_bs * acc.num_processes \
#                 if not acc.split_batches else local_bs

#         if global_bs is None:
#             return

#         for split_name, split_size in split_sizes.items():
#             if split_size > 0 and split_size < global_bs:
#                 fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
#                 acc.print(
#                     f"*** WARNING: GLOBAL BATCH SIZE ({global_bs}) IS LARGER THAN {split_name.upper()} SPLIT SIZE ({split_size}){fold_str}. "
#                     f"THIS CAN CREATE EMPTY SHARDS AND RUNTIME ERRORS. REDUCE batch_size OR GPU COUNT.***"
#                 )
#                 break
#     except Exception as _e:
#         acc.print(f"[DEBUG] Batch-size validation skipped: {_e}")


# def _batch_remainder_sanity_check(
#     split_sizes: dict,
#     dataloader_dict: Dict[str, Any],
#     acc,
#     fold_idx: Optional[int] = None,
#     min_multiplier: int = 4,
#     min_remainder_ratio: float = 0.5
# ) -> None:
#     """
#     Warn in ALL CAPS when, for small splits, drop_last would drop a large remainder.

#     For each available split dataloader:
#       - compute effective global batch size
#       - if split_size < min_multiplier * global_bs and drop_last is True, 
#         ensure remainder < (min_remainder_ratio * global_bs)
#       - otherwise, warn to adjust batch size
#     """
#     try:
#         for split_name, split_size in split_sizes.items():
#             dl = dataloader_dict.get(split_name)
#             if dl is None:
#                 continue

#             local_bs = getattr(dl, 'batch_size', None)
#             global_bs = getattr(dl, 'total_batch_size', None)
#             if global_bs is None and local_bs is not None:
#                 global_bs = local_bs * acc.num_processes if not acc.split_batches else local_bs

#             # Respect user's preference: avoid getattr with non-None defaults
#             if hasattr(dl, 'drop_last'):
#                 drop_last = dl.drop_last
#             else:
#                 drop_last = None

#             try:
#                 global_bs = int(global_bs) if global_bs is not None else None
#             except Exception:
#                 global_bs = None

#             if (global_bs is None) or (global_bs <= 0) or (split_size is None) or (split_size <= 0):
#                 continue

#             if (split_size < min_multiplier * global_bs) and bool(drop_last):
#                 remainder = split_size % global_bs
#                 if remainder >= (min_remainder_ratio * global_bs):
#                     fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
#                     acc.print(
#                         f"*** WARNING: {split_name.upper()} SPLIT SIZE ({split_size}) IS < 4x GLOBAL BATCH ({global_bs}){fold_str}. "
#                         f"WITH drop_last=True THIS WILL DROP {remainder} SAMPLES (>= HALF A BATCH). CONSIDER ADJUSTING batch_size. ***"
#                     )
#     except Exception as _e:
#         acc.print(f"[DEBUG] Batch remainder validation skipped: {_e}")


def _batching_sanity_check(
    split_sizes: dict,
    dataloader_dict: Dict[str, Any],
    acc,
    fold_idx: Optional[int] = None,
    min_multiplier: int = 4,
    min_remainder_ratio: float = 0.5
) -> None:
    """
    Validate batching across ALL splits:
      1) Warn if global batch size > split size (empty shards risk)
      2) When drop_last=True and split is small, warn if dropped remainder >= threshold
    """
    try:
        for split_name, split_size in split_sizes.items():
            dl = dataloader_dict.get(split_name)
            if dl is None:
                continue

            local_bs = getattr(dl, 'batch_size', None)
            global_bs = getattr(dl, 'total_batch_size', None)
            if global_bs is None and local_bs is not None:
                global_bs = local_bs * acc.num_processes if not acc.split_batches else local_bs

            # Respect preference: avoid getattr with non-None default
            drop_last = dl.drop_last if hasattr(dl, 'drop_last') else None

            try:
                global_bs = int(global_bs) if global_bs is not None else None
            except Exception:
                global_bs = None

            if (global_bs is None) or (global_bs <= 0) or (split_size is None) or (split_size <= 0):
                continue

            # Check 1: batch size vs split size
            if split_size > 0 and split_size < global_bs:
                fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
                acc.print(
                    f"*** WARNING: GLOBAL BATCH SIZE ({global_bs}) IS LARGER THAN {split_name.upper()} SPLIT SIZE ({split_size}){fold_str}. "
                    f"THIS CAN CREATE EMPTY SHARDS AND RUNTIME ERRORS. REDUCE batch_size OR GPU COUNT.***"
                )
                # Do not break; check remainder too for this split

            # Check 2: drop_last remainder magnitude
            if (split_size < (min_multiplier * global_bs)) and bool(drop_last):
                remainder = split_size % global_bs
                if remainder >= (min_remainder_ratio * global_bs):
                    fold_str = f" IN FOLD {fold_idx}" if fold_idx is not None else ""
                    # Use variables in message to match thresholds
                    ratio_pct = int(min_remainder_ratio * 100)
                    acc.print(
                        f"*** WARNING: {split_name.upper()} SPLIT SIZE ({split_size}) IS < {min_multiplier}x GLOBAL BATCH ({global_bs}){fold_str}. "
                        f"WITH drop_last=True THIS WILL DROP {remainder} SAMPLES (>= {ratio_pct}% OF A BATCH). CONSIDER ADJUSTING batch_size. ***"
                    )
    except Exception as _e:
        acc.print(f"[DEBUG] Batching validation skipped: {_e}")


def print_first_batch_summary(
    dataloader_dict: Dict[str, Any],
    set_key: str,
    acc: Accelerator,
    config: TrainingConfig,
) -> None:
    """
    Print a brief summary of the first batch from a specified dataloader set.

    Runs only on the main process. Handles PyG `Batch`/`Data`, tuple/list of
    tensors, and generic objects by type.
    """
    try:
        if not acc.is_main_process:
            return
        if (set_key not in dataloader_dict) or (dataloader_dict[set_key] is None):
            return

        first_batch = next(iter(dataloader_dict[set_key]))
        heading = f"[INFO] First {set_key.upper()} batch:"

        # PyG Batch/Data prints shapes via __repr__
        cls_name = first_batch.__class__.__name__
        if hasattr(first_batch, 'to_data_list') or cls_name in ('Batch', 'Data'):
            main_print(f"{heading} object:", acc=acc, config=config)
            main_print(f"{first_batch}", acc=acc, config=config)
            return

        # Tuple/list batches
        if isinstance(first_batch, (tuple, list)):
            shape_strs: List[str] = []
            for i, item in enumerate(first_batch):
                if isinstance(item, torch.Tensor):
                    shape_strs.append(f"arg{i} shape: {tuple(item.shape)}")
                else:
                    shape_strs.append(f"arg{i} type: {type(item).__name__}")
            main_print(f"{heading} (non-PyG) shapes:", acc=acc, config=config)
            main_print("; ".join(shape_strs), acc=acc, config=config)
            return

        # Fallback
        main_print(f"{heading} type: {type(first_batch)}", acc=acc, config=config)
    except Exception as e:
        main_print(f"[WARNING] Could not sample first {set_key.upper()} batch: {e}", acc=acc, config=config)


def main(clargs):
    # EXPERIMENT START
    experiment_start_time = time.time()
    
    # Load configuration
    config_manager = ConfigManager(clargs)
    config = config_manager.config
    
    # Set environment variable to suppress device warnings
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Set up wandb offline mode if requested or on SLURM
    if getattr(config, 'use_wandb_logging', False):
        if getattr(config, 'wandb_offline', False) \
        or os.environ.get('SLURM_JOB_ID'):
            os.environ['WANDB_MODE'] = 'offline'

    # Initialize accelerator object
    # Note: we disable automatic device placement to avoid issues with batching PyG Data objects
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config.ddp_find_unused_parameters)

    acc = Accelerator(
        device_placement=False if config.using_pytorch_geo else config.device,  
        # cpu=(not torch.cuda.is_available()),
        mixed_precision='no' if config.mixed_precision == 'none' else config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        split_batches=config.dataloader_split_batches,
        log_with='wandb' \
            if (getattr(config, 'use_wandb_logging', False) and _WANDB_AVAILABLE) \
            else None,
        kwargs_handlers=[ddp_kwargs] if config.ddp_find_unused_parameters else None
    )

    # If using wandb, initialize tracker (only once, after Accelerator is created)
    if getattr(config, 'use_wandb_logging', False):
        if _WANDB_AVAILABLE:
            project_name = getattr(config, 'experiment_id', 'escgnn')
            wandb_init_kwargs = {}

            # Set wandb mode to offline if requested or on SLURM    
            if getattr(config, 'wandb_offline', False) or os.environ.get('SLURM_JOB_ID'):
                wandb_init_kwargs['mode'] = 'offline'

            # Initialize wandb tracker
            acc.init_trackers(
                project_name=project_name,
                config=config.__dict__,
                init_kwargs={'wandb': wandb_init_kwargs}
            )
        else:
            acc.print('[WARNING] wandb not installed, but use_wandb_logging=True')

    # Print PyTorch version
    main_print(f"PyTorch version: {torch.__version__}", acc=acc)
    
    # Print CUDA info
    if torch.cuda.is_available():
        try:
            main_print(f"CUDA info:", acc=acc)
            main_print(f"   - version: {torch.version.cuda}", acc=acc)
            main_print(f"   - device count: {torch.cuda.device_count()}", acc=acc)
        except Exception as e:
            main_print(f"Error getting CUDA info: {e}", acc=acc)
    
    # Print accelerator info
    main_print(f"Accelerator info:", acc=acc)
    main_print(f"   - num. processes: {acc.num_processes}", acc=acc)
    main_print(f"   - distributed type: {acc.distributed_type}", acc=acc)

    # Print warning if running in single-process mode with multiple GPUs
    if torch.cuda.device_count() > 1 and acc.num_processes == 1:
        main_print("WARNING: Multiple GPUs detected but running in single-process mode!", acc=acc)
        main_print("To use multiple GPUs, launch with: accelerate launch --num_processes=N --multi_gpu script.py", acc=acc)
    # elif acc.num_processes > 1:
    #     main_print(f"Multi-GPU training enabled with {acc.num_processes} processes", acc=acc)
    
    # Ensure experiment_id is deterministic for DDP (all processes use same ID)
    if config.experiment_id is None:
        # Generate a deterministic experiment ID based on current time
        # All processes will generate the same timestamp since they start simultaneously
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_id = f"exp_{timestamp}"
    
    # Only main process creates directories, then broadcast exact paths to all ranks
    shared_dirs_list = [None]
    if acc.is_main_process and config.snapshot_name is None:
        # Compose a model+target slug to separate runs with different targets
        model_slug = config.model_config.model_key
        try:
            tgt_key = getattr(config.dataset_config, 'target_key', None)
            if isinstance(tgt_key, str) and tgt_key:
                # Sanitize target key for filesystem
                safe_tgt = ''.join(c if c.isalnum() or c in ('-', '_') else '-' for c in tgt_key)
                model_slug = f"{model_slug}_{safe_tgt}"
        except Exception:
            pass

        # For CV runs, create only the parent experiment directory now; per-fold subdirs will be created later.
        dirs = create_experiment_dir(
            root_dir=config.save_dir,
            model_name=model_slug,
            dataset_name=config.dataset_config.dataset,
            experiment_id=config.experiment_id,
            config=config_manager.config.__dict__,
            config_format='yaml',
            verbosity=config.verbosity,
            create_subdirs=(config.experiment_type == 'tvt')
        )
        if dirs is None:
            main_print("Error: Could not create experiment directories. Exiting.", acc=acc)
            return
        shared_dirs_list[0] = dirs

    # Broadcast the directory map so every rank uses the same (possibly suffixed) model directory
    if config.snapshot_name is None:
        broadcast_object_list(shared_dirs_list)
        dirs = shared_dirs_list[0]
        if dirs is None:
            main_print("Error: Directory broadcast failed.", acc=acc)
            return

        # Update config paths on all ranks consistently
        if config.experiment_type == 'tvt':
            config.results_save_dir = dirs['metrics']
            config.model_save_dir = dirs['models'] if config.model_save_dir is None else config.model_save_dir
            config.train_logs_save_dir = dirs['logs']
            config.config_save_path = dirs['config_save_path']
        else:
            # For CV, leave top-level subdirs unset; will be set per fold
            config.results_save_dir = None
            config.model_save_dir = None
            config.train_logs_save_dir = None
            config.config_save_path = None

    # ------------------------------------------------------------------
    # Resume-from-snapshot adjustments
    # ------------------------------------------------------------------
    if config.snapshot_name is not None:
        # Experiment directory (parent of "models/")
        exp_dir = os.path.dirname(config.model_save_dir)

        # Path to the *original* config used in the first run
        legacy_yaml_path = os.path.join(exp_dir, 'config', 'config.yaml')

        # Decide whether to load the legacy YAML.  We do so **only** when the
        # user did *not* pass a different --config_path on the command line.
        cl_yaml_path = getattr(config_manager.clargs, 'config_path', None)
        # Load the legacy YAML only when the caller **did not pass** a
        # --config_path (CLI parameter omitted).  If a path was provided we
        # assume the user wants to use that file exactly as-is.
        load_legacy = (cl_yaml_path is None)

        if load_legacy and os.path.exists(legacy_yaml_path):
            with open(legacy_yaml_path, 'r') as f:
                loaded_yaml = yaml.safe_load(f)

            # Merge – CLI overrides have already been applied by ConfigManager
            for k, v in loaded_yaml.get('training', {}).items():
                if hasattr(config, k):
                    setattr(config, k, v)
            for k, v in loaded_yaml.get('model', {}).items():
                if hasattr(config.model_config, k):
                    setattr(config.model_config, k, v)
            for k, v in loaded_yaml.get('optimizer', {}).items():
                if hasattr(config.optimizer_config, k):
                    setattr(config.optimizer_config, k, v)
            for k, v in loaded_yaml.get('scheduler', {}).items():
                if hasattr(config.scheduler_config, k):
                    setattr(config.scheduler_config, k, v)

            # Keep the snapshot we are resuming from
            # (it could have been overwritten above)
            config.snapshot_name = Path(config_manager.clargs.snapshot_path).name

        # --- Paths that always point inside *exp_dir* ---
        config.train_logs_save_dir = os.path.join(exp_dir, 'logs')
        config.model_save_dir      = os.path.join(exp_dir, 'models')
        config.results_save_dir    = os.path.join(exp_dir, 'metrics')

        # ------------------------------------------------------------------
        # Prepare a restart-config filename but delay the actual save until
        # later (guarded by `is_main_process`) so we write it only once.
        # ------------------------------------------------------------------
        config_dir = os.path.join(exp_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)

        n = 1
        while True:
            restart_config_path = os.path.join(config_dir, f'config_restart_{n}.yaml')
            if not os.path.exists(restart_config_path):
                break
            n += 1

        # Remember where we will save the (possibly edited) config later
        config.config_save_path = restart_config_path

    # All processes load dataset
    # for now, this includes independently computing target rescaling stats,
    # applying them to targets, and subsetting targets if needed
    # TODO: compute stats on one process, and move these steps into a 
    # transform method of the dataset class (this requires awareness of which
    # split a data object is in though...unless all splits are transformed and
    # un-transformed)
    dataset = load_dataset(
        config, 
        model_key=config.model_config.model_key
    )
    # print(f"len(dataset) after load_dataset: {len(dataset)}")
    if dataset is None:
        raise RuntimeError("Dataset failed to load! Check your config file and data filepaths.")
    
    # Run experiment (all processes must participate)
    if config.experiment_type == 'tvt':  # 'train/valid/[test]' splits
        if acc.is_main_process:
            # ----------------------------------------------------------
            # Choose split strategy – TorchMD QM9 splits vs random splits
            # ----------------------------------------------------------
            if (
                config.dataset_config.use_torchmd_qm9_splits
                and config.dataset_config.dataset.lower() == 'qm9'
            ):
                # Use TorchMD splits (110k/10k/rest) as used by Equiformer
                splits_dict = get_torchmd_qm9_splits(len_dataset=len(dataset))
            else:
                # Default: random proportional splits
                splits_dict = get_random_splits(
                    n=len(dataset),
                    seed=config.dataset_config.split_seed,
                    train_prop=config.dataset_config.train_prop,
                    valid_prop=config.dataset_config.valid_prop,
                )

            splits_list = [splits_dict]
        else:
            splits_list = [None]
        broadcast_object_list(splits_list)
        splits_dict = splits_list[0]
        dataloader_dict, config = create_dataloaders(dataset, splits_dict, config)
        main_print(f"After create_dataloaders:\n\tconfig.dataset_config.target_include_indices: {config.dataset_config.target_include_indices}", acc=acc)
        if config.dataset_config.target_preprocessing_type is not None:
            main_print(f"\tconfig.dataset_config.target_preproc_stats: {config.dataset_config.target_preproc_stats}", acc=acc)
        
        # Sanity check: print first train batch object (main process only)
        print_first_batch_summary(dataloader_dict=dataloader_dict, set_key='train', acc=acc, config=config)
        
        # Print split and batch sizes (main process only)
        if acc.is_main_process:
            split_sizes = {k: len(v.dataset) for k, v in dataloader_dict.items()}
            acc.print(
                f"Dataset split sizes:" 
                f" train: {split_sizes.get('train', 0)},"
                f" valid: {split_sizes.get('valid', 0)},"
                f" test: {split_sizes.get('test', 0)}"
            )
            acc.print(
                f"Batch sizes:"
                f" train: {dataloader_dict.get('train', {}).batch_size},"
                f" valid: {dataloader_dict.get('valid', {}).batch_size},"
                f" test: {dataloader_dict.get('test', {}).batch_size}"
            )

            # Batching sanity-checks across all splits
            _batching_sanity_check(split_sizes, dataloader_dict, acc)
        
        # Save config after all processing is complete (main process only)
        if config.config_save_path and acc.is_main_process:
            config_manager.save_config(config.config_save_path)
        
        # Run one fold (whole experiment)
        _ = run_one_fold(
            config, 
            dataloader_dict, 
            save_results=True,
            accelerator=acc
        )
        
    elif 'kfold' in config.experiment_type:  # k-fold cross-validation
        if 'stratified' in config.experiment_type:
            raise NotImplementedError(
                "Stratified k-fold cross-validation not yet implemented"
            )
        else:
            # Create k-fold split indexes on main process, broadcast to all
            if acc.is_main_process:
                kfold_splits = get_kfold_splits(
                    seed=config.dataset_config.split_seed,
                    k=config.dataset_config.k_folds,
                    n=len(dataset)
                )
                # Also choose a deterministic validation fold per test fold using the split_seed
                # Ensure valid fold != test fold
                rng = torch.Generator().manual_seed(int(config.dataset_config.split_seed))
                val_folds = []
                for test_i in range(config.dataset_config.k_folds):
                    candidates = [j for j in range(config.dataset_config.k_folds) if j != test_i]
                    # Deterministic pick using seeded generator
                    pick_idx = int(torch.randint(low=0, high=len(candidates), size=(1,), generator=rng).item())
                    val_folds.append(candidates[pick_idx])
                kfold_splits_list = [kfold_splits, val_folds]
            else:
                kfold_splits_list = [None, None]
            broadcast_object_list(kfold_splits_list)
            kfold_splits = kfold_splits_list[0]
            val_folds = kfold_splits_list[1]
        
        # Loop through folds
        for fold_idx in range(config.dataset_config.k_folds):
            # Process k-fold splits into train/valid/test sets (all processes, deterministic)
            split_dict = process_kfold_splits(
                kfold_splits,
                k=config.dataset_config.k_folds,
                fold_idx=fold_idx,
                include_test_set=True,
                valid_fold_idx=val_folds[fold_idx]
            )
            
            # Create per-fold output directories under a common parent
            if acc.is_main_process and config.snapshot_name is None:
                parent_dir = dirs['exp_dir']
                fold_root = os.path.join(parent_dir, f"fold_{fold_idx}")
                fold_dirs = {
                    'metrics': os.path.join(fold_root, 'metrics'),
                    'models': os.path.join(fold_root, 'models'),
                    'logs': os.path.join(fold_root, 'logs'),
                    'config': os.path.join(fold_root, 'config')
                }
                for _p in fold_dirs.values():
                    ensure_dir_exists(_p, raise_exception=True)
                # Prepare a unique config save path per fold
                fold_config_path = os.path.join(fold_dirs['config'], 'config.yaml')
                fold_dirs_map = {
                    'results_save_dir': fold_dirs['metrics'],
                    'model_save_dir': fold_dirs['models'],
                    'train_logs_save_dir': fold_dirs['logs'],
                    'config_save_path': fold_config_path,
                }
            else:
                fold_dirs_map = None
            # Broadcast fold directories so all ranks agree
            shared_fold_dirs = [fold_dirs_map]
            broadcast_object_list(shared_fold_dirs)
            fold_dirs_map = shared_fold_dirs[0]

            # Update config paths for this fold
            config.results_save_dir = fold_dirs_map['results_save_dir']
            config.model_save_dir = fold_dirs_map['model_save_dir']
            config.train_logs_save_dir = fold_dirs_map['train_logs_save_dir']
            config.config_save_path = fold_dirs_map['config_save_path']

            # Create dataloaders for fold
            dataloader_dict, config = create_dataloaders(dataset, split_dict, config)
            
            # Sanity check: print first TRAIN batch object (main process only)
            print_first_batch_summary(dataloader_dict=dataloader_dict, set_key='train', acc=acc, config=config)
            
            # Print split sizes (main process only)
            if acc.is_main_process:
                split_sizes = {k: len(v.dataset) for k, v in dataloader_dict.items()}
                acc.print(
                    f"[Dataset split sizes]" \
                    f" train: {split_sizes.get('train', 0)},"\
                    f" valid: {split_sizes.get('valid', 0)},"\
                    f" test: {split_sizes.get('test', 0)}"\
                    f" (fold {fold_idx})"
                )

                # Batching sanity-checks across all splits (fold-aware)
                _batching_sanity_check(split_sizes, dataloader_dict, acc, fold_idx)
            
            # Save config for this fold (main process only)
            if config.config_save_path and acc.is_main_process:
                config_manager.save_config(config.config_save_path)
            
            # Train model on fold and save results
            _ = run_one_fold(
                config, 
                dataloader_dict, 
                fold_idx,
                save_results=True,
                accelerator=acc
            )
    else:
        raise ValueError(
            f"Experiment type '{config.experiment_type}' not supported"
        )
    
    # EXPERIMENT COMPLETE (only main process prints summary)
    if acc.is_main_process:
        total_experiment_elapsed = time.time() - experiment_start_time
        hours = int(total_experiment_elapsed // 3600)
        minutes = int((total_experiment_elapsed % 3600) // 60)
        seconds = total_experiment_elapsed % 60
        
        main_print(f"\n{'='*60}")
        main_print(f"EXPERIMENT COMPLETE")
        main_print(f"Total experiment time: {hours:02d}h {minutes:02d}m {seconds:05.2f}s")
        # main_print(f"{'='*60}\n")

    # ------------------------------------------------------
    # End Accelerator tracking (e.g., wandb) after all work
    # ------------------------------------------------------
    if getattr(config, 'use_wandb_logging', False):
        acc.end_training()

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    clargs = get_clargs()
    main(clargs)

