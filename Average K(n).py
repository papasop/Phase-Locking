# 安装 mpmath 用于获取 Riemann 零点
!pip install mpmath

import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero
from scipy.stats import pearsonr

# 获取前 N 个非平凡 Riemann 零点的虚部 γ_n
N = 1000
gamma_n = np.array([float(zetazero(n).imag) for n in range(1, N + 1)])

# φ(n) 结构函数
n_vals = np.arange(1, N + 1)
phi_n = (4 / (np.pi * n_vals)) * (np.cos(np.log(gamma_n)) + 1.1)

# 熵函数 H(n)
H_n = np.log(1 + phi_n**2)

# 局部导数比率 K(n) ≈ Δ log|φ| / Δ log H
log_phi = np.log(np.abs(phi_n))
log_H = np.log(H_n)
K_n = np.gradient(log_phi) / np.gradient(log_H)

# 零点间隔 Δγ_n = γ_{n+1} - γ_n
delta_gamma = np.diff(gamma_n)
K_trimmed = K_n[:-1]  # 对齐长度

# 可视化 K(n) vs Δγ_n
plt.figure(figsize=(12, 5))
plt.scatter(K_trimmed, delta_gamma, alpha=0.7, c='teal', label='Δγₙ vs K(n)')
plt.axvline(0.5, color='red', linestyle='--', label='K = 0.5')
plt.xlabel(r'$K(n) \approx \frac{d \log |\phi(n)|}{d \log H(n)}$')
plt.ylabel(r'Zero gap $\Delta \gamma_n = \gamma_{n+1} - \gamma_n$')
plt.title('Riemann Zero Gaps vs Structural Derivative Ratio K(n)')
plt.legend()
plt.grid(True)
plt.show()

# 相关性分析
corr_coef, pval = pearsonr(K_trimmed, delta_gamma)
print(f"Pearson correlation between K(n) and Δγₙ: {corr_coef:.6f} (p = {pval:.2e})")

# 平均值和稳定性观察
print(f"Average K(n): {np.mean(K_n):.6f} ± {np.std(K_n):.6f}")
print(f"Average Δγₙ: {np.mean(delta_gamma):.6f} ± {np.std(delta_gamma):.6f}")
