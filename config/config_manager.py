# config/config_manager.py
import os
import yaml
import importlib
import torch
from pathlib import Path
from typing import Dict, Any, Type
from utilities import merge_dicts, read_yaml
from .train_config import TrainingConfig
from .dataset_config import DatasetConfig
from .model_config import ModelConfig
from .optimizer_config import OptimizerConfig
from .scheduler_config import SchedulerConfig
from .arg_parsing import get_clargs
from dataclasses import fields


def check_bf16_support() -> bool:
    """
    Check if the current device supports BFloat16 operations.
    Tests both dense and sparse tensor operations.
    Returns True if BFloat16 is supported, False otherwise.
    """
    if not torch.cuda.is_available():
        return False
        
    device = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(device)
    
    try:
        # Test 1: Dense tensor operation
        x = torch.randn(2, 2, dtype=torch.bfloat16, device='cuda')
        y = torch.randn(2, 2, dtype=torch.bfloat16, device='cuda')
        z = torch.matmul(x, y)
        
        # Test 2: Sparse tensor operation
        # Create a simple sparse tensor
        indices = torch.tensor([[0, 1], [1, 0]], device='cuda')
        values = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, device='cuda')
        sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 2), device='cuda')
        
        # Try sparse-dense multiplication
        dense_tensor = torch.randn(2, 2, dtype=torch.bfloat16, device='cuda')
        _ = torch.sparse.mm(sparse_tensor, dense_tensor)
        
        return True
    except Exception as e:
        print(f"BFloat16 not supported: {str(e)}")
        return False


class ConfigManager:
    """
    Manages configuration for the training script.
    The parameter hierarchy is:
    - command line args
    - yaml config file
    - default values in TrainingConfig

    Note that parameter names must match between any yaml files, 
    the command line args, and the TrainingConfig/ModelConfig/
    OptimizerConfig/SchedulerConfig classes.
    """
    def __init__(self, clargs):
        self.clargs = clargs
        self.config = self._load_config()
        

    def _load_class_from_string(self, class_path: str) -> Type:
        """Convert a string class path to the actual class."""
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
        
        
    def _override_with_clargs(self, training_config: TrainingConfig) -> TrainingConfig:
        """
        Override configuration parameters with command line arguments if provided.
        
        Args:
            training_config: The current training configuration
            
        Returns:
            Updated training configuration with command line argument overrides
        """
        # Dataset config overrides
        if self.clargs.dataset is not None:
            training_config.dataset_config.dataset = self.clargs.dataset
            
        # Model config overrides
        if self.clargs.model_key is not None:
            training_config.model_config.model_key = self.clargs.model_key
            
        if self.clargs.model_mode is not None:
            training_config.model_config.model_mode = self.clargs.model_mode
            
        if self.clargs.wavelet_scales_type is not None:
            training_config.model_config.wavelet_scales_type = self.clargs.wavelet_scales_type
            
        if self.clargs.J is not None:
            training_config.model_config.J = self.clargs.J

        if self.clargs.use_dirac_nodes is not None:
            training_config.model_config.use_dirac_nodes = self.clargs.use_dirac_nodes

        if (self.clargs.invariant_pred is not None) and self.clargs.invariant_pred:
            training_config.model_config.equivar_pred = False

        # Training config overrides
        if self.clargs.scalar_feat_key is not None:
            training_config.dataset_config.scalar_feat_key = self.clargs.scalar_feat_key
        
        if self.clargs.vector_feat_key is not None:
            training_config.dataset_config.vector_feat_key = self.clargs.vector_feat_key
            
        if self.clargs.validate_every_n_epochs is not None:
            training_config.validate_every = self.clargs.validate_every_n_epochs
            
        if self.clargs.burn_in is not None:
            training_config.burnin_n_epochs = self.clargs.burn_in
            
        if self.clargs.patience is not None:
            training_config.no_valid_metric_improve_patience = self.clargs.patience
            
        if self.clargs.experiment_id is not None:
            training_config.experiment_id = self.clargs.experiment_id
            
        # Training config overrides (continued)
        if self.clargs.batch_size is not None:
            training_config.batch_size = self.clargs.batch_size
            
        if self.clargs.verbosity is not None:
            training_config.verbosity = self.clargs.verbosity
            
        if self.clargs.slurm:
            training_config.slurm = True
            
        # Dataset config override for subsample_n
        if hasattr(self.clargs, 'subsample_n') and self.clargs.subsample_n is not None:
            training_config.dataset_config.subsample_n = self.clargs.subsample_n
            
        # Ablation
        if self.clargs.ablate_vector_track:
            training_config.ablate_vector_track = True
        if self.clargs.ablate_scalar_track:
            training_config.ablate_scalar_track = True

        if self.clargs.task_key is not None:
            training_config.dataset_config.task = self.clargs.task_key
        if self.clargs.target_key is not None:
            training_config.dataset_config.target_key = self.clargs.target_key
        if self.clargs.target_dim is not None:
            training_config.dataset_config.target_dim = self.clargs.target_dim
        if hasattr(self.clargs, 'k_folds') and self.clargs.k_folds is not None:
            training_config.dataset_config.k_folds = int(self.clargs.k_folds)
        
        if self.clargs.n_epochs is not None:
            training_config.n_epochs = self.clargs.n_epochs
        
        # Optimizer config overrides
        if self.clargs.learn_rate is not None:
            training_config.optimizer_config.learn_rate = self.clargs.learn_rate
        
        if hasattr(self.clargs, 'save_best_model_state') and self.clargs.save_best_model_state is not None:
            training_config.save_best_model_state = self.clargs.save_best_model_state

        if hasattr(self.clargs, 'save_final_model_state') and self.clargs.save_final_model_state is not None:
            training_config.save_final_model_state = self.clargs.save_final_model_state
        
        if hasattr(self.clargs, 'use_wandb_logging') and self.clargs.use_wandb_logging:
            training_config.use_wandb_logging = True
        if hasattr(self.clargs, 'wandb_offline') and self.clargs.wandb_offline:
            training_config.wandb_offline = True
        
        if hasattr(self.clargs, 'wandb_log_freq') and self.clargs.wandb_log_freq is not None:
            training_config.wandb_log_freq = self.clargs.wandb_log_freq
        
        if hasattr(self.clargs, 'subsample_seed') and self.clargs.subsample_seed is not None:
            training_config.dataset_config.subsample_seed = self.clargs.subsample_seed

        # Dataset preparation: compute edge distances flag
        if hasattr(self.clargs, 'compute_edge_distances') and self.clargs.compute_edge_distances:
            training_config.dataset_config.compute_edge_distances = True

        # Experiment type override (e.g., 'kfold')
        if hasattr(self.clargs, 'experiment_type') and self.clargs.experiment_type is not None:
            training_config.experiment_type = self.clargs.experiment_type

        
        # -----------------------------------------------------
        # Snapshot path override – simplifies resuming training
        # -----------------------------------------------------
        if getattr(self.clargs, 'snapshot_path', None) is not None:
            snap_path = Path(self.clargs.snapshot_path).expanduser().resolve()

            # Determine snapshot_name and model_save_dir. The new structure
            # always saves checkpoints into a single-level sub-directory under
            # "models/" (e.g. "models/checkpoint", "models/best").  Therefore
            # we only need to strip one parent directory to recover the
            # corresponding "models" folder – no legacy nested layouts.

            training_config.snapshot_name = snap_path.name  # e.g. 'checkpoint'

            # Whether the user passed the directory itself or a specific file
            # inside that directory, its parent is always the desired
            # ``models`` directory in the new layout.
            training_config.model_save_dir = str(snap_path.parent)

            # Derive experiment_dir so other paths can be constructed later
            exp_dir = Path(training_config.model_save_dir).parent
            training_config.experiment_dir = str(exp_dir)
            # Set other save dirs if not already set
            if training_config.train_logs_save_dir is None:
                training_config.train_logs_save_dir = str(exp_dir / 'logs')
            if training_config.results_save_dir is None:
                training_config.results_save_dir = str(exp_dir / 'metrics')
        
        if getattr(self.clargs, 'pretrained_weights_dir', None) is not None:
            training_config.pretrained_weights_dir = self.clargs.pretrained_weights_dir
        
        # Checkpoint frequency override
        if getattr(self.clargs, 'checkpoint_every', None) is not None:
            training_config.checkpoint_every = self.clargs.checkpoint_every
        
        return training_config
        
        
    def _load_config(self) -> TrainingConfig:
        # Start with default configurations
        dataset_config = DatasetConfig()
        model_config = ModelConfig()
        optimizer_config = OptimizerConfig()
        scheduler_config = SchedulerConfig()
        training_config = TrainingConfig()
        
        # Load YAML if provided (supports experiment+model layered config)
        if self.clargs.config_path:
            # -----------------------------------------------------
            # Load YAML files and merge
            # -----------------------------------------------------
            # Determine model YAML and (optional) experiment YAML in same directory
            model_yaml_path = Path(self.clargs.config_path).expanduser().resolve()
            model_parent = model_yaml_path.parent
            exp_candidate_1 = model_parent / 'experiment.yaml'
            exp_candidate_2 = model_parent / f"{model_parent.name}.yaml"

            # If user passed the experiment file itself, treat as experiment-only
            is_experiment_file = model_yaml_path == exp_candidate_1 or model_yaml_path == exp_candidate_2

            model_yaml_dict = read_yaml(model_yaml_path) if not is_experiment_file else {}
            exp_yaml_dict = {}
            exp_used_path = None
            if exp_candidate_1.exists():
                exp_yaml_dict = read_yaml(exp_candidate_1)
                exp_used_path = exp_candidate_1
            elif exp_candidate_2.exists():
                exp_yaml_dict = read_yaml(exp_candidate_2)
                exp_used_path = exp_candidate_2

            # User visibility of which layered YAMLs are being applied
            if is_experiment_file:
                print(f"[CONFIG] CONFIG PATH POINTS TO EXPERIMENT YAML: {model_yaml_path}. MODEL YAML LAYER WILL NOT BE AUTO-LOADED.")
            else:
                if exp_used_path is not None:
                    print(f"[CONFIG] FOUND EXPERIMENT YAML: {exp_used_path}. WILL APPLY ITS SETTINGS, THEN OVERLAY MODEL YAML: {model_yaml_path}.")
                else:
                    print(f"[CONFIG] EXPERIMENT YAML NOT FOUND IN DIR: {model_parent}. USING MODEL YAML ONLY; DEFAULTS WILL FILL UNSPECIFIED SECTIONS.")
            
            # Precedence for YAML files: model YAML > experiment YAML > defaults in Python
            if exp_yaml_dict:
                yaml_config = exp_yaml_dict.copy()
                yaml_config = merge_dicts(base=yaml_config, override=model_yaml_dict)
            else:
                yaml_config = model_yaml_dict

            # Fallback: if no exp+model logic applies (e.g., user passed a single-file config)
            if not yaml_config:
                yaml_config = read_yaml(model_yaml_path)

            # -----------------------------------------------------
            # Populate dataclasses from merged yaml_config
            # -----------------------------------------------------
            def _filter_valid_fields(config_dict, config_type):
                valid_fields = set(f.name for f in fields(config_type))
                valid_entries = {k: v for k, v in config_dict.items() if k in valid_fields}
                invalid_entries = {k: v for k, v in config_dict.items() if k not in valid_fields}
                if invalid_entries:
                    print(f"[CONFIG] WARNING: Invalid entries found in {config_type.__name__} config: {invalid_entries}")
                return valid_entries

            if 'dataset' in yaml_config:
                dataset_yaml = yaml_config['dataset'].copy()
                if 'dataset_class' in dataset_yaml:
                    dataset_yaml['dataset_class'] = self._load_class_from_string(dataset_yaml['dataset_class'])
                dataset_yaml = _filter_valid_fields(dataset_yaml, DatasetConfig)
                dataset_config = DatasetConfig(**dataset_yaml)

            if 'model' in yaml_config:
                model_yaml = _filter_valid_fields(yaml_config['model'], ModelConfig)
                model_config = ModelConfig(**model_yaml)

            if 'optimizer' in yaml_config:
                optimizer_yaml = _filter_valid_fields(yaml_config['optimizer'], OptimizerConfig)
                optimizer_config = OptimizerConfig(**optimizer_yaml)

            if 'scheduler' in yaml_config:
                scheduler_yaml = _filter_valid_fields(yaml_config['scheduler'], SchedulerConfig)
                scheduler_config = SchedulerConfig(**scheduler_yaml)

            if 'training' in yaml_config:
                training_yaml = _filter_valid_fields(yaml_config['training'], TrainingConfig)
                training_config = TrainingConfig(**training_yaml)
        
        # -----------------------------------------------------
        # Create final config with nested configs
        # -----------------------------------------------------
        training_config.dataset_config = dataset_config
        training_config.model_config = model_config
        training_config.optimizer_config = optimizer_config
        training_config.scheduler_config = scheduler_config
        
        # -----------------------------------------------------
        # Override with command line arguments
        # -----------------------------------------------------
        training_config = self._override_with_clargs(training_config)

        # -----------------------------------------------------
        # Validate config (to catch some errors early)
        # -----------------------------------------------------
       
        # Validate target_include_indices
        if dataset_config.target_include_indices is not None:
            if len(dataset_config.target_include_indices) != dataset_config.target_dim:
                raise ValueError(
                    f"[CONFIG] The length of 'target_include_indices' ({len(dataset_config.target_include_indices)}) "
                    f"must be equal to 'target_dim' ({dataset_config.target_dim})"
                )
        
        # Validate dataset split proportions for tvt experiment type
        if training_config.experiment_type == 'tvt':
            if (dataset_config.train_prop + dataset_config.valid_prop >= 1.0) \
            and not dataset_config.use_torchmd_qm9_splits:
                raise ValueError(
                    f"[CONFIG] For experiment_type 'tvt', the sum of train_prop ({dataset_config.train_prop}) "
                    f"and valid_prop ({dataset_config.valid_prop}) must be less than 1.0 to allow "
                    f"for a test set. Current sum: {dataset_config.train_prop + dataset_config.valid_prop}"
                )
            
        # Validate target preprocessing stats
        if dataset_config.target_preprocessing_type is not None:
            if dataset_config.target_preproc_stats is not None:
                if 'scale' not in dataset_config.target_preproc_stats:
                    raise ValueError(
                        f"[CONFIG] target_preproc_stats must contain 'scale' when target_preprocessing_type is specified"
                    )
                if 'center' not in dataset_config.target_preproc_stats:
                    raise ValueError(
                    f"[CONFIG] target_preproc_stats must contain 'center' when target_preprocessing_type is specified"
                )
        
        # Check BFloat16 support and update mixed precision if needed
        if training_config.mixed_precision == 'bf16' and not check_bf16_support():
            print("[CONFIG] Warning: BFloat16 not supported on this device. Falling back to standard precision.")
            training_config.mixed_precision = 'no'
        
        # Validation: if reload-on-plateau is enabled, require saving best model state
        if getattr(training_config.scheduler_config, 'reload_best_on_plateau', False):
            if not bool(getattr(training_config, 'save_best_model_state', False)):
                raise ValueError(
                    "[CONFIG] 'scheduler.reload_best_on_plateau' requires 'training.save_best_model_state: true' to "
                    "persist best checkpoints. Please enable it in the training config."
                )

        # Print a warning if early-stopping patience (epochs) is not greater than
        # scheduler patience (epochs). Note: LR scheduler internally operates on
        # validation steps, so we will scale patience by validate_every when constructing
        # it. Here we compare in epoch units and also report the derived validation-steps.
        if hasattr(training_config.scheduler_config, 'patience') and training_config.scheduler_config.patience is not None:
            sched_pat_epochs = training_config.scheduler_config.patience
            es_pat_epochs = training_config.no_valid_metric_improve_patience
            validate_every = max(1, int(getattr(training_config, 'validate_every', 1)))
            # Derived number of validation steps the scheduler will wait
            try:
                import math
                sched_pat_val_steps = max(1, math.ceil(sched_pat_epochs / validate_every))
            except Exception:
                sched_pat_val_steps = sched_pat_epochs
            if es_pat_epochs < sched_pat_epochs:
                print(
                    f"[CONFIG] Warning: Early stopping patience ({es_pat_epochs} epochs) should be >= "
                    f"scheduler patience ({sched_pat_epochs} epochs). With validate_every={validate_every}, the scheduler "
                    f"patience corresponds to {sched_pat_val_steps} validation checks."
                )
        
        # Add HPC environment variables if they exist
        training_config = self._add_hpc_env(training_config)
        
        return training_config
    

    def _add_hpc_env(self, config: TrainingConfig) -> TrainingConfig:
        """Add HPC environment variables to config if they exist."""
        # Check for SLURM environment variables
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        
        if slurm_job_id is not None:
            config.slurm_job_id = slurm_job_id
        if slurm_array_task_id is not None:
            config.slurm_array_task_id = slurm_array_task_id
            
        return config
    

    def save_config(self, save_path: str, overwrite: bool = False):
        """Save the current configuration to a YAML file.

        If *overwrite* is False (default) and *save_path* already exists, a
        numeric suffix ( "_1", "_2", … ) is appended *before* the file
        extension so existing files are preserved.  For example, saving to
        "config.yaml" when it already exists will create "config_1.yaml",
        "config_2.yaml", etc.
        """

        # ------------------------------------------------------------------
        # Determine a non-clobbering filename unless overwriting is allowed
        # ------------------------------------------------------------------
        if not overwrite:
            base, ext = os.path.splitext(save_path)
            candidate_path = save_path
            counter = 1
            while os.path.exists(candidate_path):
                candidate_path = f"{base}_{counter}{ext}"
                counter += 1
            save_path = candidate_path

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Custom dumper to avoid YAML anchors/references
        class NoAliasDumper(yaml.SafeDumper):
            def ignore_aliases(self, data):
                return True

        # Assemble the nested config dict
        config_dict = {
            'dataset': {
                k: v for k, v in self.config.dataset_config.__dict__.items()
                if k != 'dataset_class'  # Skip the class object when dumping
            },
            'model': self.config.model_config.__dict__,
            'optimizer': self.config.optimizer_config.__dict__,
            'scheduler': self.config.scheduler_config.__dict__,
            'training': {
                k: v for k, v in self.config.__dict__.items()
                if k not in ['dataset_config', 'model_config', 'optimizer_config', 'scheduler_config']
            },
        }

        # ------------------------------------------------------------------
        # Sanitize: YAML cannot serialize callables (e.g., transform functions)
        # ------------------------------------------------------------------
        def _sanitize(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if callable(v):
                        # Store the callable's qualified name as string
                        out[k] = f"<callable:{v.__name__}>"
                    else:
                        out[k] = _sanitize(v)
                return out
            elif isinstance(obj, (list, tuple)):
                return [_sanitize(i) for i in obj]
            else:
                return obj

        config_dict = _sanitize(config_dict)

        # Add the dataset class path as a dotted string, if present
        if hasattr(self.config.dataset_config, 'dataset_class'):
            cls = self.config.dataset_config.dataset_class
            config_dict['dataset']['dataset_class'] = f"{cls.__module__}.{cls.__name__}"

        # Finally, write the YAML
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, Dumper=NoAliasDumper)
    
