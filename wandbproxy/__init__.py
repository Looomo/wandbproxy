import sys
import types
import wandb as original_wandb
import mlflow
class WANDBProxyRun:

    def __init__(self, run_instance):
        if not isinstance(run_instance, original_wandb.wandb_sdk.wandb_run.Run):
            raise TypeError("WANDBProxyRun must be initialized with a wandb.Run instance.")
        self._original_run = run_instance
        print("WANDBProxyRun: Successfully wrapped wandb.Run instance.")

    def __getattr__(self, name):
        return getattr(self._original_run, name)

    def log(self, data, commit=True, step=None):
        self._original_run.log(data, commit=commit, step=step)
        mlflow.log_metrics( data, step=step )
    
    def finish(self, *args, **kwargs):
        """
        Intercepts wandb.run.finish() to also end the MLflow run.
        """
        print("WANDBProxyRun: Intercepting wandb.run.finish()...")
        self._original_run.finish(*args, **kwargs)
        mlflow.end_run() # End the active MLflow run
        print("WANDBProxyRun: wandb.run.finish() wrapped successfully, MLflow run ended.")

class WandbModuleProxy(types.ModuleType):
    def __init__(self, name, original_module):
        super().__init__(name)
        self._original_wandb = original_module 
        self._current_proxied_run = None 
        for attr_name in dir(self._original_wandb):
            if not attr_name.startswith("_") and hasattr(self._original_wandb, attr_name):
                if attr_name not in ["init", "log", "run"]:
                    setattr(self, attr_name, getattr(self._original_wandb, attr_name))

    def __getattr__(self, name):
    
        if name == "run":
            return self._current_proxied_run if self._current_proxied_run else self._original_wandb.run
        return getattr(self._original_wandb, name)

    def init(self, *args, mlflow_params = {}, **kwargs):
        print(f"WandbModuleProxy: Intercepting wandb.init()...")
        original_run = self._original_wandb.init(*args, **kwargs)

        acceptable_keys = {
            # 'experiment_id': 'project',
            'run_name': 'name',
        }
        
        wandb_exp_name = getattr(original_run, 'project')
        if wandb_exp_name is not None: 
            mlflow.set_experiment(wandb_exp_name)

        for k,v in acceptable_keys.items():
            if k not in mlflow_params.keys():
                target = getattr(original_run, v)
                if target is not None:
                    mlflow_params[k] = target
                    print(f"::debug: updating {k} -> {mlflow_params[k]}")
        # acceptable_keys.update(mlflow_params)
        mlflow.start_run(**mlflow_params)

        run_config = getattr(original_run, 'config')
        if run_config is not None: 
            mlflow.log_params(run_config)
        self._current_proxied_run = WANDBProxyRun(original_run)
        print(f"WandbModuleProxy: wandb.init() wrapped successfully, returning proxied run.")
        return self._current_proxied_run

    def log(self, data, step=None, commit = None):

        if self._current_proxied_run:
            self._current_proxied_run.log(data, step, commit)
        else:
            self._original_wandb.log(data, step, commit)
        mlflow.log_metrics( data, step=step )

_proxy_module_instance = WandbModuleProxy(__name__, original_wandb)
sys.modules['wandb'] = _proxy_module_instance
print("wandbproxy.py: Successfully installed as 'wandb' transparent proxy in sys.modules.")