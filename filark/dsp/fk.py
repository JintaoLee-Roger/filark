import numpy as np
import scipy.fft
import warnings
from typing import Union, Tuple, Optional, Literal

# --- Soft Imports for Optional Backends ---
try:
    import pyfftw
    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Backend Literal Type for type hinting
BackendType = Literal['numpy', 'scipy', 'pyfftw', 'torch', 'jax', 'cupy']

# =========================================================================
# Part 1: Helper Functions (Physics & Utils)
# =========================================================================

def get_fk_axes(n_t: int, n_x: int, dt: float, dx: float, centered: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Frequency (f) and Wavenumber (k) axes.
    """
    # Frequency axis
    f = np.fft.fftfreq(n_t, d=dt)
    # Wavenumber axis
    k = np.fft.fftfreq(n_x, d=dx)
    
    if centered:
        f = np.fft.fftshift(f)
        k = np.fft.fftshift(k)
        
    return f, k

# =========================================================================
# Part 2: Functional API (Analysis / Exploration)
# =========================================================================

def fk_transform(
    data: Union[np.ndarray, 'torch.Tensor', 'cp.ndarray', 'jnp.ndarray'],
    dt: float = 1.0,
    dx: float = 1.0,
    backend: BackendType = 'scipy',
    return_power: bool = False,
    shift: bool = True
) -> Tuple[Union[np.ndarray, object], np.ndarray, np.ndarray]:
    """
    Perform f-k transform on 2D data (Time x Space).
    Suitable for exploration (`explore/`) and ad-hoc analysis.

    Args:
        data: Input 2D array (shape: [nt, nx]).
        dt: Sampling interval in time.
        dx: Sampling interval in space.
        backend: Calculation engine ('numpy', 'scipy', 'torch', etc.).
        return_power: If True, returns magnitude squared (Power Spectrum).
        shift: If True, shifts zero-frequency to center.

    Returns:
        (spectrum, f_axis, k_axis)
        Note: f_axis and k_axis are always numpy arrays for plotting convenience.
    """
    nt, nx = data.shape
    
    # --- Backend Execution ---
    if backend == 'numpy':
        spec = np.fft.fft2(data)
        if shift: spec = np.fft.fftshift(spec)
        
    elif backend == 'scipy':
        spec = scipy.fft.fft2(data)
        if shift: spec = scipy.fft.fftshift(spec)
        
    elif backend == 'pyfftw':
        if not HAS_PYFFTW: raise ImportError("pyfftw not installed.")
        # Functional pyfftw is just a wrapper, less efficient than Class-based builder
        spec = pyfftw.interfaces.numpy_fft.fft2(data)
        if shift: spec = pyfftw.interfaces.numpy_fft.fftshift(spec)

    elif backend == 'torch':
        if not HAS_TORCH: raise ImportError("torch not installed.")
        # Ensure tensor
        if not isinstance(data, torch.Tensor):
            t_data = torch.from_numpy(data)
        else:
            t_data = data
        spec = torch.fft.fft2(t_data)
        if shift: spec = torch.fft.fftshift(spec)
        if return_power:
            spec = spec.abs()**2
            return spec, *get_fk_axes(nt, nx, dt, dx, shift) # Early return for Torch to keep gradients if needed

    elif backend == 'cupy':
        if not HAS_CUPY: raise ImportError("cupy not installed.")
        if not isinstance(data, cp.ndarray):
            c_data = cp.asarray(data)
        else:
            c_data = data
        spec = cp.fft.fft2(c_data)
        if shift: spec = cp.fft.fftshift(spec)

    elif backend == 'jax':
        if not HAS_JAX: raise ImportError("jax not installed.")
        spec = jnp.fft.fft2(data)
        if shift: spec = jnp.fft.fftshift(spec)
        
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # --- Post Processing ---
    if return_power and backend not in ['torch']: 
        # Torch handled above to preserve graph, others handled here
        spec = np.abs(spec)**2 if backend in ['numpy', 'scipy', 'pyfftw'] else spec.real**2 + spec.imag**2
        if backend == 'cupy': spec = spec.real # cupy returns complex even for abs

    # Axes generation (Always strictly Numpy for plotting libs like MPL/Vispy)
    f, k = get_fk_axes(nt, nx, dt, dx, centered=shift)

    return spec, f, k


# =========================================================================
# Part 3: Class API (Pipeline / Streaming / Training)
# =========================================================================


class FKProcessor:
    """
    高性能 F-K 处理器 (Stateful)
    专为 Pipeline 设计：
    1. 内存预分配 (Pre-allocation)
    2. 显式规划 (Explicit Planning for FFTW)
    3. 零拷贝/少拷贝设计
    """

    def __init__(
        self, 
        shape: Tuple[int, int], 
        backend: BackendType = 'numpy',
        threads: int = -1, # -1 means all cpus
        use_float32: bool = False,
        plan_flags: tuple = ('FFTW_MEASURE',) # Pipeline场景建议用 MEASURE，初始化慢但运行快
    ):
        self.nt, self.nx = shape
        self.backend = backend
        self.dtype_real = np.float32 if use_float32 else np.float64
        self.dtype_complex = np.complex64 if use_float32 else np.complex128
        
        # --- Backend Initialization ---
        self._fft_plan = None
        self._buf_in = None
        self._buf_out = None
        self._shift_fn = None
        
        # 1. PyFFTW: 极限性能模式
        if backend == 'pyfftw':
            if not HAS_PYFFTW: raise ImportError("pyfftw not installed.")
            
            # 开启 pyfftw 的缓存（可选，但在显式 Plan 下不是必须的）
            pyfftw.interfaces.cache.enable()
            
            # A. 内存对齐分配 (Aligned Memory Allocation)
            # F-K 分析通常需要看负波数，所以这里做全复数变换 (fft2) 而非 rfft2
            # 这样 shift 后才能看到完整的四个象限
            self._buf_in = pyfftw.empty_aligned(shape, dtype=self.dtype_complex)
            self._buf_out = pyfftw.empty_aligned(shape, dtype=self.dtype_complex)
            
            # B. 显式构建 Plan (Expensive step)
            # 使用 FFTW_MEASURE 会在初始化时实际跑几次来寻找最优解，适合 Pipeline 长期运行
            print(f"[FKProcessor] Building FFTW plan for shape {shape} (this may take a moment)...")
            self._fft_plan = pyfftw.FFTW(
                self._buf_in, 
                self._buf_out, 
                axes=(0, 1),
                direction='FFTW_FORWARD',
                flags=plan_flags,
                threads=multiprocessing.cpu_count() if threads == -1 else threads
            )
            print("[FKProcessor] Plan built.")
            
            # C. 预热 (Warmup) - 避免第一次推理时的抖动
            self._buf_in[:] = 0
            self._fft_plan()
            
            # Shift 函数
            self._shift_fn = np.fft.fftshift

        # 2. Torch: GPU 管线
        elif backend == 'torch':
            if not HAS_TORCH: raise ImportError("torch missing")
            self._fft_fn = torch.fft.fft2
            self._shift_fn = torch.fft.fftshift
            # Torch 可以在此预分配 Tensor，但在 Python 层面并不总是能显著加速
            # 除非使用 CUDA Graphs (过于复杂，暂时跳过)

        # 3. Cupy: GPU 数值计算
        elif backend == 'cupy':
            if not HAS_CUPY: raise ImportError("cupy missing")
            self._fft_fn = cp.fft.fft2
            self._shift_fn = cp.fft.fftshift
            # Cupy Plan 缓存通常是自动管理的

        # 4. Scipy/Numpy
        else:
            if backend == 'scipy':
                self._fft_fn = lambda x: scipy.fft.fft2(x, workers=threads)
                self._shift_fn = scipy.fft.fftshift
            else:
                self._fft_fn = np.fft.fft2
                self._shift_fn = np.fft.fftshift

    def forward(self, x: Union[np.ndarray, object], shift: bool = True):
        """
        Args:
            x: Input data.
               - If pyfftw, expects numpy array (will be copied to aligned buffer).
               - If torch/cupy, expects tensor/array on device.
        """
        
        # --- PyFFTW (High Performance CPU) ---
        if self.backend == 'pyfftw':
            # 1. 数据搬运: 这是唯一的 Overhead，必须做
            # 如果 x 已经是 complex 且对齐的最好，但通常是 float
            # 利用 [:] 赋值比 np.copyto 稍微通用一点，处理类型转换
            self._buf_in[:] = x 
            
            # 2. 执行 Plan (C-level speed)
            self._fft_plan()
            
            # 3. Shift (Optional)
            # 如果为了极致速度，viz 层面可以接受 unshifted 数据自行处理
            # 但为了接口统一，这里默认做 shift
            res = self._buf_out
            if shift:
                res = self._shift_fn(res)
            return res

        # --- GPU (Torch/Cupy) ---
        elif self.backend in ['torch', 'cupy']:
            res = self._fft_fn(x)
            if shift:
                res = self._shift_fn(res)
            return res

        # --- Standard CPU ---
        else:
            res = self._fft_fn(x)
            if shift:
                res = self._shift_fn(res)
            return res

    @property
    def output_buffer(self):
        """
        直接获取 PyFFTW 的输出 buffer，实现 Zero-Copy 给后续流程
        """
        if self.backend == 'pyfftw':
            return self._buf_out
        raise NotImplementedError("Only available for pyfftw backend")


# =========================================================================
# Part 4: F-K Utilities (Filtering / Masking)
# =========================================================================

def apply_velocity_filter(
    spectrum, 
    f_axis, 
    k_axis, 
    min_vel=None, 
    max_vel=None, 
    taper_width=0.1
):
    """
    Apply a fan filter (velocity mask) in the f-k domain.
    Velocity v = f / k.
    
    This is usually easier to do with Numpy arrays even in pipelines 
    unless strictly fully GPU resident.
    """
    # Create grid
    F, K = np.meshgrid(f_axis, k_axis, indexing='ij')
    
    # Avoid division by zero
    K[K == 0] = 1e-10
    
    V_map = F / K
    mask = np.ones_like(spectrum, dtype=float)
    
    if min_vel is not None:
        # Mask velocities slower than min_vel (steep slopes)
        mask[np.abs(V_map) < min_vel] = 0
        
    if max_vel is not None:
        # Mask velocities faster than max_vel
        mask[np.abs(V_map) > max_vel] = 0
        
    # TODO: Implement tapering (gaussian blur on edges of mask) for smoother results
    
    return spectrum * mask