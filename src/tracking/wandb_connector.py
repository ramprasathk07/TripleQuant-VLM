import os
import time
import logging
from typing import Dict, Any, Optional

import wandb
import pynvml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WandBLogger:
    """
    A WandB logger that supports logging multiple runs under the same project,
    with run name provided dynamically at logging time.
    
    Reads WandB credentials from .env file:
        WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
    """
    
    def __init__(self, 
                 project_name: Optional[str] = None,
                 entity: Optional[str] = None,
                 log_gpu_interval: int = 15,
                 track_gpu: bool = True):
        """
        Initialize the logger, but does not start a WandB run yet.
        
        Args:
            project_name: WandB project name. If None, reads from env WANDB_PROJECT.
            entity: WandB entity (user/team). If None, reads from env WANDB_ENTITY.
            log_gpu_interval: Seconds between GPU metric samples.
            track_gpu: Whether to monitor GPU usage.
        """
        # Load from env if not provided
        self.project_name = project_name or os.getenv("WANDB_PROJECT")
        if not self.project_name:
            raise ValueError("Project name must be provided or set in .env as WANDB_PROJECT")
        
        self.entity = entity or os.getenv("WANDB_ENTITY")
        # Ignore an unset/placeholder entity (e.g. the .env scaffold value) so wandb
        # falls back to the API key's default account instead of erroring with
        # "entity ... not found during upsertBucket". wandb.init also reads
        # WANDB_ENTITY from the environment, so the placeholder must be cleared there
        # too — otherwise passing entity=None still picks up the bad value.
        if self.entity and ("your_" in self.entity or "username_or_team" in self.entity
                            or not self.entity.strip()):
            self.entity = None
            os.environ.pop("WANDB_ENTITY", None)
        
        self.log_gpu_interval = log_gpu_interval
        self.track_gpu = track_gpu
        
        # Internal state
        self._current_run = None
        self._step = 0
        self._gpu_logger_thread = None
        self._stop_gpu_thread = False
        self._gpu_metrics_enabled = False
        
        # Initialize GPU monitoring capability (but not start logging until a run exists)
        if self.track_gpu:
            self._init_gpu_monitoring()
    

    # Run management

    
    def start_run(self, run_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Start a new WandB run with the given name. Finishes any previously active run.
        
        Args:
            run_name: Name for this run (appears in WandB UI).
            config: Optional hyperparameters/config to log for this run.
        """
        # Finish existing run if any
        if self._current_run is not None:
            self._stop_gpu_thread = True
            if self._gpu_logger_thread and self._gpu_logger_thread.is_alive():
                self._gpu_logger_thread.join(timeout=5)
            self._current_run.finish()
            logging.info(f"Finished previous run: {self._current_run.name}")
        
        # Reset step counter for new run
        self._step = 0
        
        # Initialize new run
        try:
            self._current_run = wandb.init(
                project=self.project_name,
                name=run_name,
                entity=self.entity,
                config=config or {},
                reinit=True
            )
            logging.info(f"Started new WandB run: {run_name}")
        except Exception as e:
            logging.error(f"Failed to start WandB run '{run_name}': {e}")
            self._current_run = None
            return
        
        # Restart GPU logging thread for this run
        if self.track_gpu and self._gpu_metrics_enabled:
            self._stop_gpu_thread = False
            import threading
            self._gpu_logger_thread = threading.Thread(target=self._log_gpu_metrics_loop, daemon=True)
            self._gpu_logger_thread.start()
            logging.info(f"GPU monitoring started for run '{run_name}'")
    
    def finish_current_run(self):
        """Finish the currently active WandB run and stop GPU logging."""
        if self._current_run is not None:
            self._stop_gpu_thread = True
            if self._gpu_logger_thread and self._gpu_logger_thread.is_alive():
                self._gpu_logger_thread.join(timeout=5)
            self._current_run.finish()
            logging.info(f"Finished run: {self._current_run.name}")
            self._current_run = None
    

    # Logging methods (with optional run_name override)
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, run_name: Optional[str] = None):
        """
        Log metrics to the current run (or to a new temporary run if run_name is provided).
        
        If run_name is given and differs from the current active run, a new run is started.
        """
        if run_name is not None:
            # Check if we need to switch runs
            if self._current_run is None or self._current_run.name != run_name:
                self.start_run(run_name)
        
        if self._current_run is None:
            logging.warning("No active WandB run. Call start_run() first or provide run_name.")
            return
        
        if step is None:
            step = self._step
        
        self._current_run.log(metrics, step=step)
        self._step = step + 1
    
    def log_llm_metrics(self,
                        llm_name: Optional[str] = None,
                        input_tokens: Optional[int] = None,
                        output_tokens: Optional[int] = None,
                        total_tokens: Optional[int] = None,
                        cost: Optional[float] = None,
                        latency: Optional[float] = None,
                        run_name: Optional[str] = None,
                        **kwargs):
        """
        Log standardized LLM metrics to the specified run.
        """
        metrics = {}
        if llm_name: metrics["llm/model_name"] = llm_name
        if input_tokens: metrics["llm/tokens_input"] = input_tokens
        if output_tokens: metrics["llm/tokens_output"] = output_tokens
        if total_tokens: metrics["llm/tokens_total"] = total_tokens
        if cost: metrics["llm/cost_usd"] = cost
        if latency: metrics["llm/latency_seconds"] = latency
        metrics.update(kwargs)
        
        self.log_metrics(metrics, run_name=run_name)
    
    def log_text_sample(self, name: str, text: str, step: Optional[int] = None, run_name: Optional[str] = None):
        """Log a text sample as a WandB Table."""
        if run_name is not None:
            if self._current_run is None or self._current_run.name != run_name:
                self.start_run(run_name)
        
        if self._current_run is None:
            logging.warning("No active run.")
            return
        
        if step is None:
            step = self._step
        
        text_table = wandb.Table(columns=["text_sample"], data=[[text]])
        self._current_run.log({name: text_table}, step=step)
        self._step = step + 1
    

    # GPU monitoring internals
    def _init_gpu_monitoring(self):
        """Initialize NVML for GPU monitoring (but don't start thread yet)."""
        try:
            pynvml.nvmlInit()
            self._gpu_metrics_enabled = True
            logging.info("NVML initialized – GPU monitoring available.")
        except Exception as e:
            logging.warning(f"Could not initialize NVML. GPU metrics disabled. Error: {e}")
            self._gpu_metrics_enabled = False
    
    def _log_gpu_metrics_loop(self):
        """Background thread: logs GPU metrics to the current run."""
        while not self._stop_gpu_thread:
            if self._current_run is not None:
                gpu_metrics = self._fetch_gpu_metrics()
                if gpu_metrics:
                    self._current_run.log(gpu_metrics)
            time.sleep(self.log_gpu_interval)
    
    def _fetch_gpu_metrics(self) -> Dict[str, Any]:
        """Fetch current GPU metrics using pynvml."""
        metrics = {}
        if not self._gpu_metrics_enabled:
            return metrics
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f"gpu/gpu_{i}/utilization_percent"] = util.gpu
                metrics[f"gpu/gpu_{i}/memory_utilization_percent"] = util.memory
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"gpu/gpu_{i}/memory_used_mb"] = mem_info.used / 1024 / 1024
                metrics[f"gpu/gpu_{i}/memory_total_mb"] = mem_info.total / 1024 / 1024
                metrics[f"gpu/gpu_{i}/memory_free_mb"] = mem_info.free / 1024 / 1024
                
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics[f"gpu/gpu_{i}/temperature_celsius"] = temp
                except:
                    pass
        except Exception as e:
            logging.warning(f"Error fetching GPU metrics: {e}")
        return metrics
    

    # Context manager (optional)
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish_current_run()
        if self._gpu_metrics_enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass