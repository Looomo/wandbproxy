import sys
import types
import wandb as original_wandb # 导入原始的wandb模块，并将其重命名为 original_wandb
import mlflow
class WANDBProxyRun:
    """
    代理 wandb.Run 实例的类，重写其 log 函数。
    所有其他对 wandb.Run 实例的操作将透明地转发给原始实例。
    """
    def __init__(self, run_instance):
        if not isinstance(run_instance, original_wandb.wandb_sdk.wandb_run.Run):
            raise TypeError("WANDBProxyRun must be initialized with a wandb.Run instance.")
        self._original_run = run_instance # 保存原始的 wandb.Run 实例
        print("WANDBProxyRun: Successfully wrapped wandb.Run instance.")

    def __getattr__(self, name):
        """
        拦截对 WANDBProxyRun 实例属性的访问。
        将所有未在 WANDBProxyRun 中定义的操作转发给原始的 wandb.Run 实例。
        """
        # print(f"WANDBProxyRun: Accessing attribute '{name}' via __getattr__ (forwarding to original run).")
        return getattr(self._original_run, name)

    def log(self, data, commit=True, step=None):
        """
        重写 wandb.Run 实例的 log 函数。
        在这里执行你对 log 数据的自定义修改。
        """
        # print(f"WANDBProxyRun: --- Intercepting wandb.run.log() call ---")
        # print(f"WANDBProxyRun: Original data: {data}")
        
        # 示例：将所有数值类型的数据乘以 2
        # modified_data = {k: v * 2 if isinstance(v, (int, float)) else v for k, v in data.items()}
        # print(f"WANDBProxyRun: Modified data: {modified_data}")
        
        # 调用原始 wandb.Run 实例的 log 函数，传入修改后的数据
        self._original_run.log(data, commit=commit, step=step)
        mlflow.log_metrics( data, step=step )
        # print("WANDBProxyRun: --- wandb.run.log() operation complete ---")

# --- 2. 定义 WandbModuleProxy 类 (用于代理整个 wandb 模块) ---
# 这个类负责拦截所有对 wandb 模块的顶级操作
class WandbModuleProxy(types.ModuleType):
    """
    透明代理 wandb 模块的类。
    只重写了模块级的 log 函数，并确保 wandb.init() 返回 WANDBProxyRun 实例。
    所有其他对 wandb 模块的访问都将转发给原始的 wandb 模块。
    """
    def __init__(self, name, original_module):
        super().__init__(name)
        self._original_wandb = original_module # 保存原始 wandb 模块的引用
        self._current_proxied_run = None # 用于存储当前活动的 WANDBProxyRun 实例

        # 将原始 wandb 模块的所有公共属性和方法（除了我们自己要重写的）
        # 复制到这个代理模块中。这实现了透明转发。
        for attr_name in dir(self._original_wandb):
            if not attr_name.startswith("_") and hasattr(self._original_wandb, attr_name):
                # 排除我们即将自定义的方法，避免覆盖或循环引用
                if attr_name not in ["init", "log", "run"]:
                    setattr(self, attr_name, getattr(self._original_wandb, attr_name))

    def __getattr__(self, name):
        """
        拦截对 WandbModuleProxy 实例属性的访问。
        这主要用于处理那些原始 wandb 模块中存在的，但我们没有显式复制或重写的方法/属性。
        """
        # 特殊处理 'run' 属性，它应该返回当前活动的 WANDBProxyRun 实例
        if name == "run":
            # 如果 init 已经创建了代理 run，则返回它；否则返回原始 wandb.run 对象
            return self._current_proxied_run if self._current_proxied_run else self._original_wandb.run
        
        # 对于其他所有未捕获的属性，直接从原始 wandb 模块中获取并返回
        # 这确保了例如 wandb.config, wandb.watch, wandb.agent 等都能正常工作
        # print(f"WandbModuleProxy: Accessing attribute '{name}' via __getattr__ (forwarding to original wandb).")
        return getattr(self._original_wandb, name)

    def init(self, *args, **kwargs):
        """
        重写 wandb.init() 函数。
        它会调用原始的 wandb.init()，然后用 WANDBProxyRun 包装返回的 run 实例。
        """
        print(f"WandbModuleProxy: Intercepting wandb.init()...")
        original_run = self._original_wandb.init(*args, **kwargs)
        
        # 将原始 run 实例包装到我们的代理类中
        self._current_proxied_run = WANDBProxyRun(original_run)
        print(f"WandbModuleProxy: wandb.init() wrapped successfully, returning proxied run.")
        
        # 返回包装后的 run 实例，这样用户就可以直接对它进行操作
        return self._current_proxied_run

    def log(self, data, step=None, commit = None):
        """
        重写模块级的 wandb.log() 函数。
        如果存在活动的 Run 实例，则调用其代理的 log 方法；
        否则，调用原始 wandb 模块的 log 方法，并应用自定义修改。
        """
        
        # 示例：将所有数值类型的数据乘以 10 (与 WANDBProxyRun 的修改逻辑可以不同或一致)
        # 这里的修改是针对直接调用 wandb.log() 而没有 active run 的情况
        # modified_data = {k: v * 10 if isinstance(v, (int, float)) else v for k, v in data.items()}
        # print(f"WandbModuleProxy: Modified data: {modified_data}")

        if self._current_proxied_run:
            # 如果有活动的 run 实例，通常用户会通过 wandb.run.log() 或 wandb.log() 来记录数据
            # 如果是 wandb.log()，且有 active run，我们会调用代理 run 的 log 方法
            # 这会触发 WANDBProxyRun.log() 的拦截逻辑
            self._current_proxied_run.log(data, step, commit)
        else:
            # 如果没有 active run，直接调用原始 wandb 模块的 log
            # 这里应当报错.
            self._original_wandb.log(data, step, commit)
        mlflow.log_metrics( data, step=step )
        # if type(data) is dict:
        #     for k,v in data.items():
        #         mlflow.log_metric(k,v, step = step)
        
        # print("WandbModuleProxy: --- Module-level wandb.log() operation complete ---")

   
# --- 3. 替换 sys.modules 中的原始 wandb 模块 ---
# 当这个文件被导入时，执行替换逻辑
# 这使得任何后续的 `import wandb` 语句都将加载我们的代理模块
_proxy_module_instance = WandbModuleProxy(__name__, original_wandb)
sys.modules['wandb'] = _proxy_module_instance
print("wandbproxy.py: Successfully installed as 'wandb' transparent proxy in sys.modules.")