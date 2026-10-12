"""Use the host's own model manager while the isolated converter owns the GPU."""
from __future__ import annotations

import gc
import sys
from contextlib import contextmanager

from .bridge import BridgeError


def native_manager(sd_models):
    # These modules are already initialized by Forge/reForge at host startup.
    # Importing a second backend here can initialize conflicting GPU settings.
    for attribute, label in (("memory_management", "Forge"), ("model_management", "reForge")):
        manager = getattr(sd_models, attribute, None)
        if manager is not None:
            return manager, label
    for name, label in (("backend.memory_management", "Forge"),
                        ("ldm_patched.modules.model_management", "reForge")):
        if name in sys.modules:
            return sys.modules[name], label
    return None, "AUTOMATIC1111"


@contextmanager
def gpu_session(sd_models, devices):
    """Restore after success, cancellation, process failure or generator close.

    Caller must hold the host queue lock, and close the converter process stream
    before leaving this context. Keep strong patcher references: reForge's loaded
    wrappers use weak references which may disappear during unload/collection.
    """
    manager, label = native_manager(sd_models)
    if manager is not None:
        required = ("current_loaded_models", "unload_all_models", "load_models_gpu", "soft_empty_cache")
        if not all(hasattr(manager, name) for name in required):
            raise BridgeError(f"{label} 的显存接口不兼容，请提供 WebUI 版本和运行记录。")
        models = []
        for loaded in list(manager.current_loaded_models):
            model = loaded.model
            if model is not None and all(model is not previous for previous in models):
                models.append(model)
        try:
            manager.unload_all_models()
            gc.collect()
            manager.soft_empty_cache(force=True)
            yield label
        finally:
            try:
                if models:
                    # The native manager preserves low VRAM / CPU swap policies.
                    manager.load_models_gpu(models)
                manager.soft_empty_cache(force=True)
            except Exception as error:
                raise BridgeError(f"恢复 {label} 模型失败：{error}。请重新加载 SD 模型；已导出文件仍可下载。") from error
    else:
        data = getattr(sd_models, "model_data", None)
        if data is None or not all(callable(getattr(sd_models, name, None))
                                   for name in ("send_model_to_cpu", "send_model_to_device")):
            raise BridgeError("当前 WebUI 的显存接口不兼容，请提供版本和运行记录。")
        active = data.sd_model
        models = list(getattr(data, "loaded_sd_models", []))
        if active is not None and all(model is not active for model in models):
            models.append(active)
        try:
            for model in models:
                sd_models.send_model_to_cpu(model)
            gc.collect()
            devices.torch_gc()
            yield label
        finally:
            if active is not None:
                sd_models.send_model_to_device(active)
            devices.torch_gc()
