import os
import sys

import platform

import torch
import yaml
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from panopticon.config import SETTINGS_FILE


def _platform_cpu_label():
    """Lightweight CPU label using stdlib only (safe under PyInstaller)."""
    for candidate in (platform.processor(), platform.uname().processor, platform.machine()):
        if candidate:
            return f"CPU ({candidate})"
    return "CPU"


def _safe_cpu_label():
    """Return a CPU label while avoiding cpuinfo multiprocessing issues when frozen."""
    if getattr(sys, "frozen", False):
        return _platform_cpu_label()

    os.environ.setdefault("CPUINFO_NOMULTI_PROCESSING", "1")
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw") if isinstance(info, dict) else None
        if brand:
            return f"CPU ({brand})"
    except Exception as exc:
        print(f"CPU info unavailable: {exc}")
    return _platform_cpu_label()


class DeviceSelector(QMenu):
    device_changed = Signal(str)

    def __init__(self, main_window, tr_func, status_bar, auto_load_save=True, settings_key="device"):
        super().__init__(tr_func("select_inference_device_menu"), main_window)
        self.main_window = main_window
        self.tr = tr_func
        self.status_bar = status_bar
        self.auto_load_save = auto_load_save
        self.settings_key = settings_key
        self.current_device = "cpu"
        self.device_actions = {}
        self._create_device_actions()
        if self.auto_load_save:
            self.load_device_setting()
        self._set_checked_device(self.current_device)
        self.retranslate_ui()

    def _create_device_actions(self):
        self.device_group = QActionGroup(self)
        self.device_group.setExclusive(True)

        # CPU
        cpu_name = _safe_cpu_label()
        cpu_action = QAction(cpu_name, self)
        cpu_action.setCheckable(True)
        cpu_action.setData("cpu")
        cpu_action.triggered.connect(self._on_device_selected)
        self.addAction(cpu_action)
        self.device_group.addAction(cpu_action)
        self.device_actions["cpu"] = cpu_action

        # CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                cuda_name = f"CUDA:{i} ({torch.cuda.get_device_name(i)})"
                cuda_action = QAction(cuda_name, self)
                cuda_action.setCheckable(True)
                cuda_action.setData(f"cuda:{i}")
                cuda_action.triggered.connect(self._on_device_selected)
                self.addAction(cuda_action)
                self.device_group.addAction(cuda_action)
                self.device_actions[f"cuda:{i}"] = cuda_action
        else:
            cuda_action = QAction(self.tr("cuda_unavailable"), self)
            cuda_action.setEnabled(False)
            self.addAction(cuda_action)
            self.device_actions["cuda_unavailable"] = cuda_action

        # OpenVINO (split into CPU / GPU / NPU)
        try:
            from openvino import Core

            core = Core()
            openvino_devices = core.available_devices

            # OpenVINO CPU (show specific device name(s) if provided by Core)
            cpu_devs = [d for d in openvino_devices if d.upper().startswith("CPU")]
            if cpu_devs:
                cpu_suffix = cpu_name[5:-1] if cpu_name.startswith("CPU (") and cpu_name.endswith(")") else cpu_name
                openvino_cpu_action = QAction(f"openVINO CPU ({cpu_suffix})", self)
                openvino_cpu_action.setProperty("device_suffix", cpu_suffix)
                openvino_cpu_action.setCheckable(True)
                openvino_cpu_action.setData("openvino_cpu")
                openvino_cpu_action.triggered.connect(self._on_device_selected)
                self.addAction(openvino_cpu_action)
                self.device_group.addAction(openvino_cpu_action)
                self.device_actions["openvino_cpu"] = openvino_cpu_action
            else:
                openvino_cpu_action = QAction(self.tr("openvino_cpu_unavailable"), self)
                openvino_cpu_action.setEnabled(False)
                self.addAction(openvino_cpu_action)
                self.device_actions["openvino_cpu_unavailable"] = openvino_cpu_action

            # OpenVINO GPU (show specific device name(s) if provided by Core)
            gpu_devs = [d for d in openvino_devices if d.upper().startswith("GPU")]
            if gpu_devs:
                gpu_names = []
                for dev in gpu_devs:
                    try:
                        full_name = core.get_property(dev, "FULL_DEVICE_NAME")
                        gpu_names.append(full_name)
                    except Exception:
                        gpu_names.append(dev)  # fallback to device ID
                gpu_names_str = ", ".join(gpu_names)
                openvino_gpu_action = QAction(f"openVINO GPU ({gpu_names_str})", self)
                openvino_gpu_action.setProperty("device_suffix", gpu_names_str)
                openvino_gpu_action.setCheckable(True)
                openvino_gpu_action.setData("openvino_gpu")
                openvino_gpu_action.triggered.connect(self._on_device_selected)
                self.addAction(openvino_gpu_action)
                self.device_group.addAction(openvino_gpu_action)
                self.device_actions["openvino_gpu"] = openvino_gpu_action
            else:
                openvino_gpu_action = QAction(self.tr("openvino_gpu_unavailable"), self)
                openvino_gpu_action.setEnabled(False)
                self.addAction(openvino_gpu_action)
                self.device_actions["openvino_gpu_unavailable"] = openvino_gpu_action

            # OpenVINO NPU (e.g., MYRIAD/VPU/HDDL) - show specific device name(s)
            npu_devs = [d for d in openvino_devices if any(k in d.upper() for k in ("MYRIAD", "VPU", "HDDL"))]
            if npu_devs:
                npu_names = []
                for dev in npu_devs:
                    try:
                        full_name = core.get_property(dev, "FULL_DEVICE_NAME")
                        npu_names.append(full_name)
                    except Exception:
                        npu_names.append(dev)  # fallback to device ID
                npu_names_str = ", ".join(npu_names)
                openvino_npu_action = QAction(f"openVINO NPU({npu_names_str})", self)
                openvino_npu_action.setProperty("device_suffix", npu_names_str)
                openvino_npu_action.setCheckable(True)
                openvino_npu_action.setData("openvino_npu")
                openvino_npu_action.triggered.connect(self._on_device_selected)
                self.addAction(openvino_npu_action)
                self.device_group.addAction(openvino_npu_action)
                self.device_actions["openvino_npu"] = openvino_npu_action
            else:
                openvino_npu_action = QAction(self.tr("openvino_npu_unavailable"), self)
                openvino_npu_action.setEnabled(False)
                self.addAction(openvino_npu_action)
                self.device_actions["openvino_npu_unavailable"] = openvino_npu_action

        except ImportError:
            openvino_not_installed_action = QAction(self.tr("openvino_not_installed"), self)
            openvino_not_installed_action.setEnabled(False)
            self.addAction(openvino_not_installed_action)
            self.device_actions["openvino_not_installed"] = openvino_not_installed_action

        # ONNX, TensorRT, CoreML (via ONNX Runtime)
        try:
            import onnxruntime as ort

            onnx_providers = ort.get_available_providers()

            # ONNX CPU
            onnx_cpu_action = QAction(f"ONNX CPU ({cpu_name})", self)
            onnx_cpu_action.setCheckable(True)
            onnx_cpu_action.setData("onnx_cpu")
            onnx_cpu_action.triggered.connect(self._on_device_selected)
            self.addAction(onnx_cpu_action)
            self.device_group.addAction(onnx_cpu_action)
            self.device_actions["onnx_cpu"] = onnx_cpu_action

            # ONNX CUDA
            if "CUDAExecutionProvider" in onnx_providers:
                gpu_names_str = ", ".join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
                onnx_cuda_action = QAction(f"ONNX CUDA ({gpu_names_str})", self)
                onnx_cuda_action.setCheckable(True)
                onnx_cuda_action.setData("onnx_cuda")
                onnx_cuda_action.triggered.connect(self._on_device_selected)
                self.addAction(onnx_cuda_action)
                self.device_group.addAction(onnx_cuda_action)
                self.device_actions["onnx_cuda"] = onnx_cuda_action
            else:
                onnx_cuda_action = QAction(self.tr("onnx_cuda_unavailable"), self)
                onnx_cuda_action.setEnabled(False)
                self.addAction(onnx_cuda_action)
                self.device_actions["onnx_cuda_unavailable"] = onnx_cuda_action

            # TensorRT
            if "TensorrtExecutionProvider" in onnx_providers:
                gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                gpu_str = ", ".join(gpu_names)
                tensorrt_action = QAction(f"TensorRT ({gpu_str})", self)
                tensorrt_action.setCheckable(True)
                tensorrt_action.setData("tensorrt")
                tensorrt_action.triggered.connect(self._on_device_selected)
                self.addAction(tensorrt_action)
                self.device_group.addAction(tensorrt_action)
                self.device_actions["tensorrt"] = tensorrt_action
            else:
                tensorrt_action = QAction(self.tr("tensorrt_unavailable"), self)
                tensorrt_action.setEnabled(False)
                self.addAction(tensorrt_action)
                self.device_actions["tensorrt_unavailable"] = tensorrt_action

            # CoreML (via ONNX Runtime)
            if "CoreMLExecutionProvider" in onnx_providers:
                coreml_onnx_action = QAction("CoreML (TODO show GPU name)", self)
                coreml_onnx_action.setCheckable(True)
                coreml_onnx_action.setData("coreml_onnx")
                coreml_onnx_action.triggered.connect(self._on_device_selected)
                self.addAction(coreml_onnx_action)
                self.device_group.addAction(coreml_onnx_action)
                self.device_actions["coreml_onnx"] = coreml_onnx_action
            else:
                coreml_onnx_action = QAction(self.tr("coreml_onnx_unavailable"), self)
                coreml_onnx_action.setEnabled(False)
                self.addAction(coreml_onnx_action)
                self.device_actions["coreml_onnx_unavailable"] = coreml_onnx_action

        except ImportError:
            onnx_unavailable_action = QAction(self.tr("onnxruntime_not_installed"), self)
            onnx_unavailable_action.setEnabled(False)
            self.addAction(onnx_unavailable_action)
            self.device_actions["onnxruntime_not_installed"] = onnx_unavailable_action

            tensorrt_unavailable_action = QAction(self.tr("tensorrt_unavailable"), self)
            tensorrt_unavailable_action.setEnabled(False)
            self.addAction(tensorrt_unavailable_action)
            self.device_actions["tensorrt_unavailable"] = tensorrt_unavailable_action

            coreml_onnx_unavailable_action = QAction(self.tr("coreml_onnx_unavailable"), self)
            coreml_onnx_unavailable_action.setEnabled(False)
            self.addAction(coreml_onnx_unavailable_action)
            self.device_actions["coreml_onnx_unavailable"] = coreml_onnx_unavailable_action

        # CoreML (via PyTorch MPS)
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            coreml_mps_action = QAction(self.tr("coreml_mps_available"), self)
            coreml_mps_action.setCheckable(True)
            coreml_mps_action.setData("coreml_mps")
            coreml_mps_action.triggered.connect(self._on_device_selected)
            self.addAction(coreml_mps_action)
            self.device_group.addAction(coreml_mps_action)
            self.device_actions["coreml_mps"] = coreml_mps_action
        else:
            coreml_mps_action = QAction(self.tr("coreml_mps_unavailable"), self)
            coreml_mps_action.setEnabled(False)
            self.addAction(coreml_mps_action)
            self.device_actions["coreml_mps_unavailable"] = coreml_mps_action

        # Set initial device
        self._set_checked_device(self.current_device)

    def save_device_setting(self, device_code):
        all_settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    loaded_content = yaml.safe_load(f)
                    if isinstance(loaded_content, dict):
                        all_settings = loaded_content
                    else:
                        all_settings = {}
            except Exception as e:
                print(f"Error reading settings file for saving device: {e}")
                if self.status_bar:
                    self.status_bar.showMessage(self.tr("status_error_reading_settings", e=str(e)), 5000)
                all_settings = {}

        all_settings[self.settings_key] = device_code
        self.current_device = device_code

        try:
            with open(SETTINGS_FILE, "w") as f:
                yaml.dump(all_settings, f, default_flow_style=False, sort_keys=False)
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_device_changed", device_name=device_code), 2000)
        except Exception as e:
            print(f"Error writing device setting: {e}")
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_error_saving_device", err=str(e)), 5000)

    def load_device_setting(self):
        default_device = "cpu"
        loaded_device = default_device

        if not os.path.exists(SETTINGS_FILE):
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_device_not_found"), 3000)
            self.current_device = default_device
            return

        try:
            with open(SETTINGS_FILE) as f:
                all_settings = yaml.safe_load(f)
                if all_settings and isinstance(all_settings, dict) and self.settings_key in all_settings:
                    device_code = all_settings.get(self.settings_key)
                    # TODO: Validate if the loaded device is actually available
                    loaded_device = device_code
                    if self.status_bar:
                        self.status_bar.showMessage(self.tr("status_device_loaded", device_name=loaded_device), 2000)
                else:
                    if self.status_bar:
                        self.status_bar.showMessage(self.tr("status_device_not_found"), 3000)
        except Exception as e:
            print(f"Error loading device setting: {e}")
            if self.status_bar:
                self.status_bar.showMessage(self.tr("status_error_loading_device", err=str(e)), 5000)

        self.current_device = loaded_device

    def _on_device_selected(self):
        selected_action = self.sender()
        if isinstance(selected_action, QAction):
            device = selected_action.data()
            if device != self.current_device:
                if self.auto_load_save:
                    self.save_device_setting(device)
                self.device_changed.emit(self.current_device)

    def _set_checked_device(self, device_key):
        action = self.device_actions.get(device_key)
        if action and action.isEnabled():
            action.setChecked(True)
            self.current_device = device_key
        elif "cpu" in self.device_actions and self.device_actions["cpu"].isEnabled():
            self.device_actions["cpu"].setChecked(True)
            self.current_device = "cpu"
        else:
            # Fallback if no valid device can be set, should not happen with CPU always available
            pass

    def get_selected_device(self):
        return self.current_device

    def retranslate_ui(self):
        self.setTitle(self.tr("select_inference_device_menu"))
        # Retranslate actions that are not dynamically named (e.g., unavailable messages)
        if "cuda_unavailable" in self.device_actions:
            self.device_actions["cuda_unavailable"].setText(self.tr("cuda_unavailable"))
        if "openvino_cpu_unavailable" in self.device_actions:
            self.device_actions["openvino_cpu_unavailable"].setText(self.tr("openvino_cpu_unavailable"))
        if "openvino_gpu_unavailable" in self.device_actions:
            self.device_actions["openvino_gpu_unavailable"].setText(self.tr("openvino_gpu_unavailable"))
        if "openvino_npu_unavailable" in self.device_actions:
            self.device_actions["openvino_npu_unavailable"].setText(self.tr("openvino_npu_unavailable"))
        if "openvino_not_installed" in self.device_actions:
            self.device_actions["openvino_not_installed"].setText(self.tr("openvino_not_installed"))
        if "onnx_cuda_unavailable" in self.device_actions:
            self.device_actions["onnx_cuda_unavailable"].setText(self.tr("onnx_cuda_unavailable"))
        if "tensorrt_available" in self.device_actions:
            self.device_actions["tensorrt_available"].setText(self.tr("tensorrt_available"))
        if "tensorrt_unavailable" in self.device_actions:
            self.device_actions["tensorrt_unavailable"].setText(self.tr("tensorrt_unavailable"))
        if "coreml_onnx_available" in self.device_actions:
            self.device_actions["coreml_onnx_available"].setText(self.tr("coreml_onnx_available"))
        if "coreml_onnx_unavailable" in self.device_actions:
            self.device_actions["coreml_onnx_unavailable"].setText(self.tr("coreml_onnx_unavailable"))
        if "coreml_mps_available" in self.device_actions:
            self.device_actions["coreml_mps_available"].setText(self.tr("coreml_mps_available"))
        if "coreml_mps_unavailable" in self.device_actions:
            self.device_actions["coreml_mps_unavailable"].setText(self.tr("coreml_mps_unavailable"))
        if "onnxruntime_not_installed" in self.device_actions:
            self.device_actions["onnxruntime_not_installed"].setText(self.tr("onnxruntime_not_installed"))