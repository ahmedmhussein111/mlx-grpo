import logging
from openai import OpenAI
import shutil
import tempfile
import psutil
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union,  Callable
import sys
import os
import gc
from mlx.utils import tree_flatten, tree_unflatten, tree_map, tree_map_with_path
import mlx.core as mx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import OpenAI
import openai  # Required for retry exception types

import re
from mlx.utils import tree_map
from pathlib import Path
from mlx_lm.models.llama import ModelArgs


# ---------------- LOGGING SETUP ----------------
def setup_logging(verbose: bool = True):
    log_level = logging.INFO if verbose else logging.ERROR
    logger = logging.getLogger()
    logger.setLevel(log_level)
    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    # Clear existing handlers to avoid duplicates if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # File Handler
    try:
        fh = logging.FileHandler("aggregated_grpo_training.log", mode="a")
        fh.setLevel(logging.DEBUG)  # Log debug messages to file
        fh.setFormatter(console_formatter)
        logger.addHandler(fh)
    except Exception as e:
        logging.error(f"Could not set up file logging: {e}")

    logging.debug("Logging set up.")
    return logging


# ======================================================================
# Utility Functions
# ======================================================================
# --- Disk Space Check Configuration ---
MIN_FREE_SPACE_GB = 2.0  # Minimum required free space in Gigabytes
MIN_REQUIRED_BYTES = int(MIN_FREE_SPACE_GB * 1024**3)


# --- Globals for Script Modification Check ---
try:
    # Get the absolute path of the script being run
    SCRIPT_PATH = os.path.abspath(sys.argv[0])
    # Record the initial modification time
    INITIAL_MOD_TIME = os.path.getmtime(SCRIPT_PATH)
    logging.debug(f"Script modification check initialized. Watching: {SCRIPT_PATH}")
except Exception as e:
    logging.warning(f"Could not initialize script modification check: {e}")
    SCRIPT_PATH = None
    INITIAL_MOD_TIME = None
# --- End Globals ---


def _check_disk_space(path_to_check: str, required_bytes: int) -> bool:
    """
    Checks if sufficient free disk space is available on the partition
    containing the given path.

    Args:
        path_to_check: The directory path to check the disk space for.
        required_bytes: The minimum number of free bytes required.

    Returns:
        True if sufficient space is available, False otherwise.
    """
    try:
        # Ensure the path exists or check its parent if it's a file path
        check_dir = (
            path_to_check
            if os.path.isdir(path_to_check)
            else os.path.dirname(path_to_check)
        )
        if not os.path.exists(check_dir):
            # If the directory doesn't exist yet, check the first existing parent
            check_dir = os.path.abspath(check_dir)
            while not os.path.exists(check_dir) and check_dir != os.path.dirname(
                check_dir
            ):
                check_dir = os.path.dirname(check_dir)
            if not os.path.exists(check_dir):
                logging.error(
                    f"Cannot determine filesystem for path '{path_to_check}'. Cannot check disk space."
                )
                return False  # Cannot verify space

        usage = shutil.disk_usage(check_dir)
        free_gb = usage.free / (1024**3)
        required_gb = required_bytes / (1024**3)
        logging.debug(f"Disk usage for '{check_dir}': Free={free_gb:.2f} GB")
        if usage.free >= required_bytes:
            return True
        else:
            logging.warning(
                f"Insufficient disk space at '{check_dir}'. "
                f"Required: {required_gb:.2f} GB, "
                f"Available: {free_gb:.2f} GB"
            )
            return False
    except FileNotFoundError:
        logging.error(
            f"Cannot check disk space: Path component not found for '{path_to_check}'"
        )
        return False  # Assume insufficient if path doesn't resolve
    except Exception as e:
        logging.error(
            f"Error checking disk space for '{path_to_check}': {e}", exc_info=True
        )
        return False  # Assume insufficient on error


def _save_and_commit(temp_prefix, target_path, save_fn):
    """
    Safely writes to a temporary file with the correct extension, checking disk
    space first, then atomically moves it to target_path.
    """
    target_dir = os.path.dirname(target_path)
    # Ensure target directory exists before checking space/creating tmp
    os.makedirs(target_dir, exist_ok=True)
    tmp_dir = os.path.join(target_dir, "tmp")  # Place tmp dir within target dir

    # --- Disk Space Check ---
    # Check space on the filesystem containing the target directory
    if not _check_disk_space(target_dir, MIN_REQUIRED_BYTES):
        logging.warning(
            f"Attempting to clean temporary directory to free space: {tmp_dir}"
        )
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
                logging.debug(f"Removed temporary directory: {tmp_dir}")
            except Exception as e_rm:
                logging.error(f"Failed to remove temporary directory {tmp_dir}: {e_rm}")
                # Continue anyway, maybe space issue isn't related to tmp dir

        # Recreate tmp_dir after attempting removal
        os.makedirs(tmp_dir, exist_ok=True)

        # Re-check space after cleaning attempt
        if not _check_disk_space(target_dir, MIN_REQUIRED_BYTES):
            logging.critical(
                f"Insufficient disk space on the partition for '{target_dir}' "
                f"(needs ~{MIN_FREE_SPACE_GB:.1f} GB free) even after attempting to clean {tmp_dir}. "
                "Cannot proceed with saving. Exiting."
            )
            sys.exit(1)  # Exit script with error code
        else:
            logging.debug(
                f"Disk space sufficient after cleaning attempt for {tmp_dir}."
            )
    else:
        # Ensure tmp_dir exists even if space check passed first time
        os.makedirs(tmp_dir, exist_ok=True)

    # --- Proceed with Saving ---
    base_ext = os.path.splitext(target_path)[1]
    if base_ext not in [".safetensors", ".npz", ".json"]:  # Allow .json for configs
        raise ValueError(
            f"Unsupported file extension for {target_path}. Must be '.safetensors', '.npz', or '.json'."
        )

    # Create temporary file within the tmp_dir
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=temp_prefix, suffix=base_ext, dir=tmp_dir
    )
    os.close(tmp_fd)  # Close file descriptor immediately

    try:
        logging.debug(f"Attempting to save data to temporary file: {tmp_path}")
        save_fn(tmp_path)  # Execute the actual saving function

        # Verify temporary file was created and is not empty
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            logging.debug(
                f"Temporary file {tmp_path} created (Size: {os.path.getsize(tmp_path)} bytes). Moving to {target_path}"
            )
            # Atomically move the temporary file to the final target path
            shutil.move(tmp_path, target_path)
            # try:
            #     # change_ownership_and_permissions(target_path, mode=644, uid=501, gid=20)
            # except Exception as e:
            #     logging.wardning(f"Failed to change permiossions forcommit file {target_path}: {e}", exc_info=True)

            logging.debug(f"Committed => {target_path}")
        else:
            # Clean up empty or non-existent temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise IOError(f"Invalid or empty temporary file generated: {tmp_path}")

    except Exception as e:
        logging.error(
            f"Failed to save or commit file {target_path}: {e}", exc_info=True
        )
        # Clean up the temporary file if anything failed after its creation
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logging.debug(f"Removed failed temporary file: {tmp_path}")
            except OSError as rm_err:
                logging.error(
                    f"Could not remove temporary file {tmp_path} after error: {rm_err}"
                )
        raise  # Re-raise the original exception to signal failure


def _save_directory_and_commit(temp_prefix: str, target_dir_path: Path, save_fn: Callable[[str], None]):
    """
    Safely saves directory content using a temporary directory and atomic move/replace.
    Assumes save_fn populates the provided temporary directory path.
    """
    target_dir_path = Path(target_dir_path)
    if not target_dir_path.is_dir():
        logging.error(f"Target path '{target_dir_path}' for directory commit must be an existing directory.")
        raise ValueError("Target path for directory commit must be an existing directory.")

    # Use a tmp directory within the parent of the target directory for atomicity potential
    parent_dir = target_dir_path.parent
    tmp_base_dir = parent_dir / "tmp_dirs" # Specific temp area for directory commits
    os.makedirs(tmp_base_dir, exist_ok=True)

    # --- Disk Space Check (Check space in the parent directory's partition) ---
    if not _check_disk_space(str(parent_dir), MIN_REQUIRED_BYTES):
         # Simplified: Critical exit if low space. Could add cleanup attempt like in _save_and_commit.
        logging.critical(f"Insufficient disk space for saving to '{target_dir_path}'. Exiting.")
        sys.exit(1)

    # Create a unique temporary directory *name* first
    tmp_dir_name = tempfile.mkdtemp(prefix=temp_prefix, dir=str(tmp_base_dir))
    tmp_dir_path = Path(tmp_dir_name) # Convert to Path object
    logging.debug(f"Attempting to save directory content to temporary directory: {tmp_dir_path}")

    try:
        # Execute the saving function, which should populate tmp_dir_path
        save_fn(str(tmp_dir_path)) # Pass path as string

        # Verify temporary directory exists and is not empty (basic check)
        if tmp_dir_path.exists() and any(tmp_dir_path.iterdir()):
            logging.debug(f"Temporary directory {tmp_dir_path.name} populated. Replacing {target_dir_path}")

            # --- Atomic Replace Logic ---
            # 1. Rename the final target directory to a temporary backup name (if it exists)
            backup_target_path = None
            if target_dir_path.exists():
                 # Create a unique backup name *within the same parent directory*
                 backup_target_path_str = tempfile.mktemp(prefix=f"{target_dir_path.name}_backup_", dir=str(parent_dir))
                 backup_target_path = Path(backup_target_path_str)
                 logging.debug(f"Backing up existing target '{target_dir_path.name}' to '{backup_target_path.name}'")
                 os.rename(target_dir_path, backup_target_path)

            # 2. Move the newly created temporary directory to the final target name
            try:
                shutil.move(str(tmp_dir_path), str(target_dir_path))
                logging.debug(f"Successfully committed directory => {target_dir_path}")

                # 3. If move successful, remove the backup (if one was created)
                if backup_target_path and backup_target_path.exists():
                    logging.debug(f"Removing backup directory: {backup_target_path.name}")
                    shutil.rmtree(backup_target_path, ignore_errors=True) # Use ignore_errors for robustness

            except Exception as move_err:
                 logging.error(f"Failed to move temporary directory {tmp_dir_path.name} to {target_dir_path.name}: {move_err}")
                 # Attempt to restore backup if move failed
                 if backup_target_path and backup_target_path.exists() and not target_dir_path.exists():
                     logging.warning(f"Attempting to restore backup '{backup_target_path.name}' to '{target_dir_path.name}'")
                     try:
                         os.rename(backup_target_path, target_dir_path)
                     except Exception as restore_err:
                          logging.error(f"Failed to restore backup: {restore_err}. Target directory might be inconsistent.")
                 raise move_err # Re-raise the error that caused the failure

        else:
            # Clean up empty or non-existent temporary directory
            if tmp_dir_path.exists():
                shutil.rmtree(tmp_dir_path, ignore_errors=True)
            raise IOError(f"Temporary directory not populated or is empty: {tmp_dir_path}")

    except Exception as e:
        logging.error(f"Failed to save or commit directory {target_dir_path.name}: {e}", exc_info=True)
        # Clean up the temporary directory if anything failed
        if tmp_dir_path.exists():
            logging.debug(f"Removing failed temporary directory: {tmp_dir_path.name}")
            shutil.rmtree(tmp_dir_path, ignore_errors=True)
        raise # Re-raise the original exception

    finally:
        # Clean up the base temporary directory if it's empty (optional)
        try:
            if tmp_base_dir.exists() and not any(tmp_base_dir.iterdir()):
                 tmp_base_dir.rmdir()
        except OSError:
            pass # Ignore if not empty or other issues

def clear_mlx_cache_safe():
    """Safely clear MLX cache."""
    # Task 21: Call mx.clear_cache()
    try:
        mx.clear_cache()
        logging.debug("Called mx.clear_cache()")
    except AttributeError:
        logging.warning("mx.clear_cache() not available in this MLX version.")
    except Exception as e:
        logging.warning(f"Error calling mx.clear_cache(): {e}")


def perform_memory_cleanup(tensors_to_delete: Optional[List[Any]] = None):
    """Explicitly delete tensors, clear cache, and run GC."""
    return
    # Task 21: Add explicit deletion, clear_cache, and GC calls
    if tensors_to_delete:
        logging.debug(
            f"Explicitly deleting {len(tensors_to_delete)} tensor(s)/object(s)..."
        )
        for tensor in tensors_to_delete:
            try:
                del tensor
            except NameError:
                pass  # Already deleted or out of scope
    clear_mlx_cache_safe()
    gc.collect()
    logging.debug("Performed memory cleanup (del, clear_cache, gc.collect).")


def default_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks


# ---------------- MEMORY MANAGEMENT & PRINT OVERRIDE ----------------
def limit_memory(max_memory_gb, relaxed=False):
    cache_mem = mx.get_cache_memory()

    mx.clear_cache()
    max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
    mx.reset_peak_memory()
    mx.set_wired_limit(max_memory_bytes)
    # mx.set_cache_limit(0)
    previous_limit = mx.set_memory_limit(max_memory_bytes)

    return previous_limit + cache_mem


def validate_text_format(generated_text: str) -> Tuple[bool, List[str]]:
    """
    Validates whether the given generated_text adheres to the proper format.

    Required format:
      - The trimmed response must start with "<think>" on its own line.
      - The final line must be "</answer>" on its own line.
      - Somewhere in the text, a line with "</think>" must be immediately followed by a line with "<answer>".
      - Each required tag must appear on its own line (i.e., the line contains only that tag).

    Required tags:
      - "<think>"
      - "</think>"
      - "<answer>"
      - "</answer>"

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean status (True if valid, False otherwise)
                                  and a list of warning/error messages for missing or misformatted tags.
    """
    warnings = []

    # Check for the presence of each required tag
    required_tags = {
        "<think>": "Missing '<think>' tag with all the internal chain of thoughts (Not visible to user) for internal use only ending with </think>",
        "</think>": "Missing '</think>' tag.",
        "<answer>": "Missing '<answer>' tag with the comprehensive answer visible for the user, ending your response with </answer>",
        "</answer>": "Missing '</answer>' tag.",
    }

    for tag, error_msg in required_tags.items():
        if tag not in generated_text:
            warnings.append(error_msg)

    # Enforce proper formatting: trim and split into non-empty lines
    formatted_text = generated_text.strip()
    lines = [line.strip() for line in formatted_text.splitlines() if line.strip() != ""]

    if not lines:
        warnings.append("The generated text is empty after trimming.")
        return (False, warnings)

    # Check that the very first line is exactly "<think>"
    if lines[0] != "<think>":
        warnings.append(
            "The first line must be exactly '<think>' on its own line with all the internal chain of thoughts (Not visible to user)."
        )

    # Check that the last line is exactly "</answer>"
    if lines[-1] != "</answer>":
        warnings.append("The last line must be exactly '</answer>' on its own line.")

    # Check that there is a line with "</think>" immediately followed by a line with "<answer>"
    found_thinking_answer = False
    for i in range(len(lines) - 1):
        if lines[i] == "</think>" and lines[i + 1] == "<answer>":
            found_thinking_answer = True
            break
    if not found_thinking_answer:
        warnings.append(
            "There must be a '</think>' line immediately followed by a '<answer>' line and then the comprehensive answer."
        )

    # Ensure each required tag appears on its own line.
    for tag in required_tags.keys():
        for line in lines:
            if tag in line and line != tag:
                warnings.append(
                    f"The tag '{tag}' must appear on its own line on itself (When Answering, Write the comprehensive answer for the user)."
                )
                break

    is_valid = True # len(warnings) == 0
    return (is_valid, warnings)


# ---------------- SCRIPT MODIFICATION CHECK ----------------
def check_script_updated_and_exit(agent=None, perform_save: bool = True):
    """
    Checks if the running script file has been modified. If so, attempts
    a final save (if agent provided and saving enabled) and performs controlled shutdown.
    """
    global SCRIPT_PATH, INITIAL_MOD_TIME

    if SCRIPT_PATH is None or INITIAL_MOD_TIME is None:
        return  # Feature disabled or failed to initialize

    try:
        current_mod_time = os.path.getmtime(SCRIPT_PATH)
        if current_mod_time > INITIAL_MOD_TIME:
            logging.warning(
                f"Script file '{SCRIPT_PATH}' modified since execution start."
            )
            logging.warning("Initiating controlled shutdown...")

            # Task 25: Replace sys.exit(1) with controlled shutdown
            # Task 25: Add cleanup callback before exit
            if agent and perform_save:
                try:
                    logging.debug(
                        f"Attempting final save for agent (Update Count: {agent.update_count})..."
                    )
                    # Ensure save_llm is True only if it makes sense (e.g., LLM was trained)
                    should_save_llm = (
                        getattr(agent.config, "train_llm", True)
                        and agent.update_count > 0
                    )
                    agent.save(save_llm=should_save_llm)
                    logging.debug("Final save completed.")
                except Exception as e_save:
                    logging.error(f"Error during final save: {e_save}", exc_info=True)

            # Perform any other necessary cleanup here
            logging.debug("Exiting program due to script modification.")
            sys.exit(2)  # Use a specific exit code for script modification exit

    except FileNotFoundError:
        logging.error(
            f"Script file '{SCRIPT_PATH}' not found during modification check. Disabling check."
        )
        SCRIPT_PATH = None  # Disable further checks
    except Exception as e:
        logging.error(f"Error during script modification check: {e}", exc_info=True)


# ---------------- UTILITY FUNCTIONS ----------------
def ensure_float_values(metrics_dict: Dict[str, Any]) -> Dict[str, float]:
    """Ensure all values in the metrics dictionary are float."""
    if not isinstance(metrics_dict, dict):
        return {}
    safe_metrics = {}
    for k, v in metrics_dict.items():
        try:
            if isinstance(v, mx.array):
                item_val = v.item()  # Extract scalar value first
                if np.isnan(item_val) or np.isinf(item_val):
                    logging.warning(f"Metric '{k}' is NaN or Inf, using 0.0")
                    safe_metrics[k] = 0.0
                else:
                    safe_metrics[k] = float(item_val)
            elif isinstance(v, (int, float, np.number)):
                if np.isnan(v) or np.isinf(v):
                    logging.warning(f"Metric '{k}' is NaN or Inf, using 0.0")
                    safe_metrics[k] = 0.0
                else:
                    safe_metrics[k] = float(v)
            else:
                # Attempt conversion for other types, log warning if fails
                safe_metrics[k] = float(v)
        except (ValueError, TypeError, AttributeError) as e:
            logging.warning(
                f"Could not convert metric '{k}' (value: {v}, type: {type(v)}) to float: {e}. Using 0.0"
            )
            safe_metrics[k] = 0.0
    return safe_metrics

def convert_dict_to_model_args(config_dict: Dict[str, Any]) -> Optional[Any]:
    """
    Converts a dictionary to ModelArgs (or a similar config object) if the class is available.
    Handles missing keys gracefully based on the target class definition.
    Returns an instance of the class or None on failure.
    """
    if not isinstance(config_dict, dict):
        logging.error("Cannot convert non-dict config to ModelArgs.")
        return None

    # Check if the target ModelArgs class (from mlx_lm) is available
    if "ModelArgs" not in globals() or not inspect.isclass(ModelArgs):
        logging.warning(
            "ModelArgs class not available or not imported correctly. Cannot convert config dict."
        )
        # Return the original dict as a fallback? Or None? Let's return None for clarity.
        return None

    import inspect  # Needed to inspect constructor

    target_class = ModelArgs

    # Get expected arguments from the class constructor (__init__)
    try:
        sig = inspect.signature(target_class.__init__)
        expected_args = set(sig.parameters.keys()) - {"self"}
    except Exception as e:
        logging.error(
            f"Could not inspect {target_class.__name__} constructor: {e}. Cannot safely convert."
        )
        return None

    # Prepare args for constructor, only including keys expected by __init__
    constructor_args = {}
    unknown_args = []
    for k, v in config_dict.items():
        if k in expected_args:
            constructor_args[k] = v
        else:
            # Track args present in the dict but not expected by the constructor
            unknown_args.append(k)

    if unknown_args:
        logging.warning(
            f"Unknown parameters found in config dict ignored by {target_class.__name__}: {unknown_args}"
        )

    # Attempt to instantiate the class
    try:
        instance = target_class(**constructor_args)
        logging.debug(
            f"Successfully created {target_class.__name__} instance from config dict."
        )
        return instance
    except TypeError as e:
        # This likely means a required argument in __init__ was missing
        logging.error(
            f"TypeError creating {target_class.__name__} (likely missing required field or type mismatch): {e}"
        )
        return None
    except Exception as e:
        logging.error(
            f"Unexpected error creating {target_class.__name__} from dict: {e}",
            exc_info=True,
        )
        return None


def convert_model_to_bfloat16(model):
    """Converts all parameters in an MLX model to bfloat16."""
    # Get the current parameters
    params = model.parameters()

    # Convert floating point parameters to bfloat16
    bfloat16_params = tree_map(
        lambda p: p.astype(mx.bfloat16)
        if (isinstance(p, mx.array) and mx.issubdtype(p.dtype, mx.floating))
        else p,
        params,
    )

    # Update the model with converted parameters
    model.update(bfloat16_params)

    # Verify conversion
    logging.debug("Converted model parameters to bfloat16")

    return model


# Add update_shared method to the LLM model
def patch_llm_model_with_update_shared(model_instance):
    """Adds update_shared method to LLM model instance"""
    if not hasattr(model_instance, "update_shared"):

        def update_shared_method(self, params):
            self.update(params)

        import types

        model_instance.update_shared = types.MethodType(
            update_shared_method, model_instance
        )

    return model_instance


def quantize_model_mlx(
    model, config, quantize: bool = True, q_group_size: int = 64, q_bits: int = 4
):
    return model
    if quantize:
        weights = dict(tree_flatten(model.parameters()))
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(model, config, q_group_size, q_bits, None)

        params = model.parameters()
        bfloat16_params = tree_map(
            lambda p: p.astype(mx.bfloat16)
            if isinstance(p, mx.array) and mx.issubdtype(p.dtype, mx.floating)
            else p,
            params,
        )
        model.update(bfloat16_params)
        return model
