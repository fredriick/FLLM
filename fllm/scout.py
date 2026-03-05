"""
scout.py — Cross-platform hardware detection.
Produces a HardwareProfile that drives all downstream decisions.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import psutil


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    vendor: str                    # "nvidia" | "amd" | "intel" | "apple"
    architecture: str              # e.g. "Ampere", "RDNA3", "Apple M2"
    compute_capability: Optional[str] = None   # CUDA only, e.g. "8.6"
    memory_bandwidth_gbps: Optional[float] = None
    index: int = 0


@dataclass
class HardwareProfile:
    # ── CPU ─────────────────────────────────────────────────────────────────
    cpu_model: str = "Unknown"
    cpu_physical_cores: int = 1
    cpu_logical_cores: int = 1
    cpu_instructions: List[str] = field(default_factory=list)  # AVX2, AVX512, AMX, NEON

    # ── RAM ─────────────────────────────────────────────────────────────────
    total_ram_gb: float = 0.0
    ram_type: str = "Unknown"          # DDR4, DDR5, LPDDR5, …
    ram_speed_mhz: Optional[float] = None
    dual_channel: bool = False

    # ── Storage ─────────────────────────────────────────────────────────────
    storage_type: str = "Unknown"      # NVMe, SATA_SSD, HDD
    storage_speed_mbps: float = 0.0
    pcie_version: Optional[float] = None
    pcie_lanes: Optional[int] = None

    # ── GPU(s) ──────────────────────────────────────────────────────────────
    gpus: List[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0

    # ── System flags ───────────────────────────────────────────────────────
    unified_memory: bool = False       # Apple Silicon / AMD APU
    is_wsl2: bool = False            # Running under WSL2
    is_docker: bool = False           # Running in Docker
    has_igpu: bool = False           # Has integrated GPU alongside discrete
    os: str = "Unknown"
    arch: str = "Unknown"

    # ── Derived recommendations ─────────────────────────────────────────────
    tier: str = "C"                    # "A" | "B" | "C"
    optimal_quant_bits: int = 4
    max_context_tokens: int = 2048
    recommended_backend: str = "llama.cpp"

    # ── Human-readable warnings ─────────────────────────────────────────────
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scout
# ---------------------------------------------------------------------------

class HardwareScout:
    """
    Detect all hardware capabilities on the current machine, across
    Windows, macOS (Intel + Apple Silicon), and Linux.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._os = platform.system()          # "Linux" | "Darwin" | "Windows"
        self._machine = platform.machine()    # "x86_64" | "arm64" | "AMD64" …
        self._detect_environment()
        self.profile = HardwareProfile(
            os=self._os,
            arch=self._machine,
            is_wsl2=self._is_wsl2,
            is_docker=self._is_docker,
        )

    def _detect_environment(self):
        """Detect WSL2, Docker, and Rosetta 2 environments."""
        self._is_wsl2 = False
        self._is_docker = False

        # Rosetta 2 detection (macOS only)
        if self._os == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "sysctl.proc_translated"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip() == "1":
                    self.profile.warnings.append(
                        "Running under Rosetta 2 — reinstall Python as native ARM for Tier B performance."
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
                pass

        # WSL2 detection
        if self._os == "Linux":
            # Check /proc/version for WSL
            try:
                with open("/proc/version", "r") as f:
                    version = f.read().lower()
                    if "microsoft" in version and "wsl2" in version:
                        self._is_wsl2 = True
            except (FileNotFoundError, PermissionError):
                pass

            # Alternative: check /proc/sys/fs/binfmt_misc/WSLInterop
            if not self._is_wsl2:
                if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
                    self._is_wsl2 = True

            # Docker detection
            # Check for /.dockerenv file
            if Path("/.dockerenv").exists():
                self._is_docker = True

            # Alternative: check cgroup
            try:
                with open("/proc/1/cgroup", "r") as f:
                    cgroup = f.read()
                    if "docker" in cgroup:
                        self._is_docker = True
            except (FileNotFoundError, PermissionError):
                pass

    # ── Public entry point ───────────────────────────────────────────────────

    def detect_all(self) -> HardwareProfile:
        self._log("Detecting CPU …")
        self._detect_cpu()
        self._log("Detecting RAM …")
        self._detect_ram()
        self._log("Detecting storage …")
        self._detect_storage()
        self._log("Detecting GPUs …")
        self._detect_gpus()
        self._log("Classifying hardware tier …")
        self._classify_tier()
        return self.profile

    # ── CPU ─────────────────────────────────────────────────────────────────

    def _detect_cpu(self):
        p = self.profile

        # Model name — cross-platform
        try:
            import cpuinfo          # py-cpuinfo
            info = cpuinfo.get_cpu_info()
            p.cpu_model = info.get("brand_raw", "Unknown")
            flags = info.get("flags", [])

            if "avx512f" in flags:
                p.cpu_instructions.append("AVX512")
            if "avx2" in flags:
                p.cpu_instructions.append("AVX2")
            if "amx_bf16" in flags:
                p.cpu_instructions.append("AMX")
            if "neon" in flags or (self._os == "Darwin" and "arm" in self._machine.lower()):
                p.cpu_instructions.append("NEON")
        except ImportError:
            p.cpu_model = platform.processor() or "Unknown"
            p.warnings.append("py-cpuinfo not installed; CPU instruction detection skipped.")

        p.cpu_physical_cores = psutil.cpu_count(logical=False) or 1
        p.cpu_logical_cores = psutil.cpu_count(logical=True) or 1

    # ── RAM ─────────────────────────────────────────────────────────────────

    def _detect_ram(self):
        p = self.profile
        vm = psutil.virtual_memory()
        p.total_ram_gb = vm.total / (1024 ** 3)

        if self._os == "Linux":
            self._detect_ram_linux(p)
        elif self._os == "Darwin":
            self._detect_ram_macos(p)
        elif self._os == "Windows":
            self._detect_ram_windows(p)

    def _detect_ram_linux(self, p: HardwareProfile):
        out = self._run(["dmidecode", "-t", "memory"], sudo=True)
        if not out:
            return
        # Speed
        m = re.search(r"Speed:\s*(\d+)\s*MT/s", out)
        if m:
            p.ram_speed_mhz = float(m.group(1))
        # Type
        for t in ("DDR5", "DDR4", "DDR3", "LPDDR5", "LPDDR4"):
            if t in out:
                p.ram_type = t
                break
        # Dual channel (heuristic: two populated channels)
        p.dual_channel = ("ChannelA-DIMM0" in out and "ChannelB-DIMM0" in out)

    def _detect_ram_macos(self, p: HardwareProfile):
        out = self._run(["system_profiler", "SPMemoryDataType"])
        if not out:
            return
        m = re.search(r"Type:\s*(.+)", out)
        if m:
            p.ram_type = m.group(1).strip()
        m = re.search(r"Speed:\s*(\d+)\s*MHz", out)
        if m:
            p.ram_speed_mhz = float(m.group(1))

    def _detect_ram_windows(self, p: HardwareProfile):
        out = self._run([
            "wmic", "memorychip", "get",
            "Speed,MemoryType,BankLabel", "/format:csv"
        ])
        if not out:
            return
        # MemoryType 26 = DDR4, 34 = DDR5
        if "34" in out:
            p.ram_type = "DDR5"
        elif "26" in out:
            p.ram_type = "DDR4"
        m = re.search(r",(\d{3,5}),", out)
        if m:
            p.ram_speed_mhz = float(m.group(1))

    # ── Storage ─────────────────────────────────────────────────────────────

    def _detect_storage(self):
        p = self.profile
        if self._os == "Linux":
            self._detect_storage_linux(p)
        elif self._os == "Darwin":
            self._detect_storage_macos(p)
        elif self._os == "Windows":
            self._detect_storage_windows(p)

    def _detect_storage_linux(self, p: HardwareProfile):
        out = self._run(["lsblk", "-d", "-o", "NAME,ROTA,TRAN"])
        if out:
            if "nvme" in out.lower():
                p.storage_type = "NVMe"
            elif "sata" in out.lower() and "0" in out:
                p.storage_type = "SATA_SSD"
            else:
                p.storage_type = "HDD"

        # PCIe version from lspci
        lspci = self._run(["lspci", "-vvv"])
        if lspci:
            # Match "LnkSta: Speed 16GT/s (ok), Width x4"
            m = re.search(r"LnkSta:\s*Speed\s*(\d+)GT/s.*?Width x(\d+)", lspci)
            if m:
                gt_s = int(m.group(1))
                lanes = int(m.group(2))
                p.pcie_lanes = lanes
                # GT/s → PCIe generation
                p.pcie_version = {8: 3.0, 16: 4.0, 32: 5.0}.get(gt_s, gt_s / 8)

    def _detect_storage_macos(self, p: HardwareProfile):
        out = self._run(["system_profiler", "SPNVMeDataType"])
        if out and "nvme" in out.lower():
            p.storage_type = "NVMe"
        else:
            out2 = self._run(["system_profiler", "SPSerialATADataType"])
            p.storage_type = "SATA_SSD" if out2 else "Unknown"

    def _detect_storage_windows(self, p: HardwareProfile):
        out = self._run([
            "powershell", "-Command",
            "Get-PhysicalDisk | Select-Object MediaType,BusType | ConvertTo-Json"
        ])
        if out:
            if "NVMe" in out:
                p.storage_type = "NVMe"
            elif "SSD" in out:
                p.storage_type = "SATA_SSD"
            else:
                p.storage_type = "HDD"

    # ── GPUs ─────────────────────────────────────────────────────────────────

    def _detect_gpus(self):
        p = self.profile

        # WSL2 warnings
        if self._is_wsl2:
            p.warnings.append("Running under WSL2. GPU passthrough depends on Windows driver version.")

            # Check for AMD in WSL2 (ROCm support in WSL2 is limited)
            if self._os == "Linux":
                amd_check = self._run(["lspci"])
                if amd_check and "AMD" in amd_check:
                    p.warnings.append(
                        "WSL2 + AMD GPU detected. ROCm support in WSL2 is experimental. "
                        "Consider using native Linux or Windows for AMD GPUs."
                    )

        # Docker warnings
        if self._is_docker:
            if not self._has_docker_gpu():
                p.warnings.append(
                    "Running in Docker without GPU passthrough. "
                    "Use --gpus flag or NVIDIA Container Toolkit for GPU access."
                )
            else:
                p.warnings.append("Running in Docker with GPU passthrough detected.")

        # Apple Silicon — unified memory, no discrete GPU
        if self._os == "Darwin" and ("arm" in self._machine.lower() or "M1" in platform.processor() or "M2" in platform.processor()):
            self._detect_apple_silicon(p)
            return

        # Try CUDA via torch
        cuda_found = self._detect_cuda_gpus(p)

        # Fallback: nvidia-smi (CUDA without torch installed)
        if not cuda_found:
            self._detect_nvidia_smi(p)

        # AMD ROCm
        if not p.gpus:
            self._detect_amd_gpus(p)

        # Intel integrated/discrete GPUs (check for both iGPU and dGPU)
        self._detect_intel_gpus(p)

        p.total_vram_gb = sum(g.vram_gb for g in p.gpus)

        # Sort largest VRAM first, prefer discrete over integrated
        p.gpus.sort(key=lambda g: (g.vram_gb, g.vendor not in ("intel",)), reverse=True)

        # Detect iGPU + dGPU combo
        discrete_gpus = [g for g in p.gpus if g.vendor in ("nvidia", "amd")]
        integrated_gpus = [g for g in p.gpus if g.vendor in ("intel",)]

        if discrete_gpus and integrated_gpus:
            p.has_igpu = True
            # Use discrete GPU only for recommendations
            p.total_vram_gb = sum(g.vram_gb for g in discrete_gpus)
            p.warnings.append(
                f"Detected {len(integrated_gpus)} integrated GPU(s). Using discrete GPU for inference."
            )

    def _has_docker_gpu(self) -> bool:
        """Check if Docker has GPU passthrough."""
        # Check nvidia-smi in container
        out = self._run(["nvidia-smi"])
        if out:
            return True
        # Check rocm-smi
        out = self._run(["rocm-smi"])
        if out:
            return True
        return False

    def _detect_apple_silicon(self, p: HardwareProfile):
        p.unified_memory = True
        # sysctl gives us the chip name
        chip = self._run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
        # On Apple Silicon this is often empty; try system_profiler
        if not chip.strip():
            sp = self._run(["system_profiler", "SPHardwareDataType"]) or ""
            m = re.search(r"Chip:\s*(.+)", sp)
            chip = m.group(1).strip() if m else "Apple Silicon"

        # GPU core count from system_profiler
        sp_disp = self._run(["system_profiler", "SPDisplaysDataType"]) or ""
        cores_m = re.search(r"Total Number of Cores:\s*(\d+)", sp_disp)
        gpu_cores = int(cores_m.group(1)) if cores_m else 0

        # Bandwidth estimates (GB/s) per chip
        bw_map = {
            "M1": 68.25, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
            "M2": 100,   "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
            "M3": 120,   "M3 Pro": 150, "M3 Max": 300,
            "M4": 120,   "M4 Pro": 273, "M4 Max": 546,
        }
        bw = 100.0
        for key, val in bw_map.items():
            if key in chip:
                bw = val
                break

        gpu = GPUInfo(
            name=chip,
            vram_gb=p.total_ram_gb,   # unified: all RAM is GPU RAM
            vendor="apple",
            architecture=chip,
            memory_bandwidth_gbps=bw,
            index=0,
        )
        p.gpus.append(gpu)
        p.total_vram_gb = p.total_ram_gb

    def _detect_cuda_gpus(self, p: HardwareProfile) -> bool:
        try:
            import torch
            if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
                return False
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                arch = self._cuda_arch_name(props.major, props.minor)
                bw = self._cuda_bandwidth_gbps(props)
                gpu = GPUInfo(
                    name=props.name,
                    vram_gb=props.total_memory / (1024 ** 3),
                    vendor="nvidia",
                    architecture=arch,
                    compute_capability=f"{props.major}.{props.minor}",
                    memory_bandwidth_gbps=bw,
                    index=i,
                )
                p.gpus.append(gpu)
            return bool(p.gpus)
        except ImportError:
            return False

    def _detect_nvidia_smi(self, p: HardwareProfile):
        out = self._run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits"
        ])
        if not out:
            return
        for i, line in enumerate(out.strip().splitlines()):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            name = parts[0]
            vram = float(parts[1]) / 1024   # MiB → GiB
            gpu = GPUInfo(
                name=name,
                vram_gb=vram,
                vendor="nvidia",
                architecture=self._guess_nvidia_arch(name),
                index=i,
            )
            p.gpus.append(gpu)

    def _detect_amd_gpus(self, p: HardwareProfile):
        # rocm-smi or wmic/powershell fallback
        out = self._run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        if out:
            for i, line in enumerate(out.strip().splitlines()[1:]):
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        vram_bytes = int(parts[-1].strip())
                        gpu = GPUInfo(
                            name=f"AMD GPU {i}",
                            vram_gb=vram_bytes / (1024 ** 3),
                            vendor="amd",
                            architecture="RDNA",
                            index=i,
                        )
                        p.gpus.append(gpu)
                    except ValueError:
                        pass

    def _detect_intel_gpus(self, p: HardwareProfile):
        # Basic detection via lspci / wmic / system_profiler
        detected_intel = []

        if self._os == "Linux":
            out = self._run(["lspci"])
            if out:
                # Intel integrated GPUs
                if "Intel" in out:
                    if "Iris" in out or "UHD" in out:
                        # Integrated: estimate shared VRAM from system RAM
                        vram_estimate = min(p.total_ram_gb / 4, 2.0)  # Typically 1-2GB shared
                        gpu = GPUInfo(
                            name="Intel Integrated GPU (UHD/Iris)",
                            vram_gb=vram_estimate,
                            vendor="intel",
                            architecture="Gen12+",
                            index=len(p.gpus),
                        )
                        detected_intel.append(gpu)
                    elif "Arc" in out:
                        # Discrete Arc GPU - try to get actual VRAM
                        vram = self._get_intel_arc_vram()
                        gpu = GPUInfo(
                            name="Intel Arc (detected via lspci)",
                            vram_gb=vram,
                            vendor="intel",
                            architecture="Xe-HPG",
                            index=len(p.gpus),
                        )
                        detected_intel.append(gpu)

        elif self._os == "Darwin" and self._machine == "x86_64":
            # Intel Mac with Iris
            out = self._run(["system_profiler", "SPDisplaysDataType"])
            if out and "Intel" in out:
                vram_estimate = min(p.total_ram_gb / 4, 1.5)
                gpu = GPUInfo(
                    name="Intel Iris",
                    vram_gb=vram_estimate,
                    vendor="intel",
                    architecture="Gen11",
                    index=len(p.gpus),
                )
                detected_intel.append(gpu)

        # Add detected Intel GPUs
        for gpu in detected_intel:
            p.gpus.append(gpu)

        # Warning if we detected but couldn't get accurate VRAM
        if detected_intel and any(g.vram_gb < 2 for g in detected_intel):
            p.warnings.append(
                "Intel integrated GPU detected with estimated shared VRAM. "
                "For better detection, install Intel GPU tools or use discrete GPU."
            )

    def _get_intel_arc_vram(self) -> float:
        """Try to get Intel Arc VRAM via various methods."""
        # Try via intel_gpu_top or sysfs
        for path in [
            "/sys/class/drm/card0/meminfo",
            "/sys/class/drm/card1/meminfo",
        ]:
            try:
                with open(path, "r") as f:
                    content = f.read()
                    # Parse VRAM from meminfo
                    m = re.search(r"vram.*?(\d+)\s*kB", content, re.IGNORECASE)
                    if m:
                        return float(m.group(1)) / (1024 * 1024)  # kB to GB
            except (FileNotFoundError, PermissionError):
                pass

        # Fallback: estimate based on model
        return 8.0  # Arc A770 has 8GB, A750 has 8GB, A380 has 6GB

    # ── Tier classification ──────────────────────────────────────────────────

    def _classify_tier(self):
        p = self.profile

        # ── Tier A: discrete GPU with meaningful VRAM ──
        has_discrete = any(g.vendor in ("nvidia", "amd") for g in p.gpus)
        if has_discrete and p.total_vram_gb >= 8:
            p.tier = "A"
            if p.total_vram_gb >= 48:
                p.optimal_quant_bits = 4
                p.max_context_tokens = 32768
                p.recommended_backend = "vllm"
            elif p.total_vram_gb >= 24:
                p.optimal_quant_bits = 4
                p.max_context_tokens = 16384
                p.recommended_backend = "vllm"
            else:
                p.optimal_quant_bits = 4
                p.max_context_tokens = 8192
                p.recommended_backend = "llama.cpp"   # vLLM needs >= ~10 GB free
            return

        # ── Tier B: unified memory (Apple Silicon / APU) ──
        if p.unified_memory or (p.total_ram_gb >= 32 and not has_discrete):
            p.tier = "B"
            p.optimal_quant_bits = 4
            p.max_context_tokens = 32768
            # Apple Silicon uses llama.cpp with Metal (mlx-lm is experimental)
            p.recommended_backend = "llama.cpp"
            return

        # ── Tier C: CPU-only ──
        p.tier = "C"
        p.max_context_tokens = 2048
        if "AVX512" in p.cpu_instructions or "AMX" in p.cpu_instructions:
            p.optimal_quant_bits = 3
            p.max_context_tokens = 4096
        elif "AVX2" in p.cpu_instructions:
            p.optimal_quant_bits = 4
        elif "NEON" in p.cpu_instructions:
            p.optimal_quant_bits = 4
        else:
            p.optimal_quant_bits = 8
            p.warnings.append(
                "No modern SIMD instructions found. Inference will be very slow."
            )
        p.recommended_backend = "llama.cpp"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _run(self, cmd: list, sudo: bool = False) -> Optional[str]:
        """Run a subprocess and return stdout, or None on failure."""
        if sudo and self._os != "Linux":
            sudo = False
        full_cmd = (["sudo"] if sudo else []) + cmd
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return None

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [scout] {msg}", file=sys.stderr)

    @staticmethod
    def _cuda_arch_name(major: int, minor: int) -> str:
        return {
            (9, 0): "Hopper",
            (8, 9): "Ada Lovelace",
            (8, 6): "Ampere",
            (8, 0): "Ampere",
            (7, 5): "Turing",
            (7, 0): "Volta",
            (6, 1): "Pascal",
            (5, 2): "Maxwell",
        }.get((major, minor), f"CUDA sm_{major}{minor}")

    @staticmethod
    def _cuda_bandwidth_gbps(props) -> float:
        """
        Estimate bandwidth from memory clock and bus width.
        memory_clock_rate is in kHz; bus width in bits.
        GDDR6X is DDR so ×2, standard GDDR6 is also DDR.
        """
        clock_ghz = props.memory_clock_rate / 1e6   # kHz → GHz
        bus_bytes = props.memory_bus_width / 8
        return round(clock_ghz * bus_bytes * 2, 1)  # DDR factor

    @staticmethod
    def _guess_nvidia_arch(name: str) -> str:
        name_lower = name.lower()
        if any(x in name_lower for x in ("40", "4080", "4090", "4070", "4060")):
            return "Ada Lovelace"
        if any(x in name_lower for x in ("30", "3090", "3080", "3070", "3060", "a100", "a6000")):
            return "Ampere"
        if any(x in name_lower for x in ("20", "2080", "2070", "2060", "t4")):
            return "Turing"
        if "h100" in name_lower or "h200" in name_lower:
            return "Hopper"
        return "Unknown"
