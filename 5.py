# 安装 mpmath（Colab 需要，本地已安装可跳过）
!pip install mpmath --quiet

# 导入库
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf, cos, log, pi, zetazero
from scipy.stats import pearsonr  # 用于相关分析（可选）

# 设置高精度
mp.dps = 50

# --- 获取前 N 个 Riemann 零点的虚部 γₙ ---
def get_gamma_n(N):
    return [mp.im(zetazero(n)) for n in range(1, N + 1)]

# --- 构造 φ(n) ---
def phi(n, gamma_n):
    return (4 / (pi * n)) * (cos(log(gamma_n[n - 1])) + 1.1)

# --- 构造结构熵 H(n) ---
def H(phi_n):
    return np.log(1 + phi_n**2)

# --- 构造结构比率函数 K(n) ---
def compute_K(phi_vals, H_vals):
    log_phi = np.log(np.abs(phi_vals))
    log_H = np.log(H_vals)
    
    # 一阶导数（有限差分）
    d_log_phi = np.gradient(log_phi)
    d_log_H = np.gradient(log_H)
    
    return d_log_phi / d_log_H

# --- 主流程 ---
N = 200  # 可调整为300或更大（注意计算时间）
gamma_list = get_gamma_n(N)
gamma_array = np.array([float(g) for g in gamma_list])  # 转换为float数组，以支持NumPy函数如arctan
phi_vals = np.array([float(phi(n + 1, gamma_list)) for n in range(N)])
H_vals = H(phi_vals)
K_vals = compute_K(phi_vals, H_vals)

# --- 计算 φ / √H 的分布 (R(n)) ---
ratio_vals = phi_vals / np.sqrt(H_vals)  # 假设phi>0，无需abs

# --- 计算几何相位 (论文Section 4) ---
theta_n = np.arctan(2 * gamma_array)
tau_n = np.pi / 2 - theta_n

# --- 输出统计信息：K(n) ---
print(f"✅ 平均 K(n) ≈ {np.mean(K_vals):.5f}")
print(f"σ(K) ≈ {np.std(K_vals):.5f}")
print(f"min(K) ≈ {np.min(K_vals):.5f}, max(K) ≈ {np.max(K_vals):.5f}")

# --- 输出统计信息：R(n) ---
print(f"✅ 平均 R(n) ≈ {np.mean(ratio_vals):.5f}")
print(f"σ(R) ≈ {np.std(ratio_vals):.5f}")
print(f"min(R) ≈ {np.min(ratio_vals):.5f}, max(R) ≈ {np.max(ratio_vals):.5f}")

# --- 输出几何相位统计（示例） ---
print(f"✅ 平均 τ_n ≈ {np.mean(tau_n):.5f}")
print(f"σ(τ_n) ≈ {np.std(tau_n):.5f}")
print(f"min(τ_n) ≈ {np.min(tau_n):.5f}, max(τ_n) ≈ {np.max(tau_n):.5f}")

# --- 可选：τ_n 与 φ(n) 的Pearson相关系数 ---
corr, _ = pearsonr(tau_n, phi_vals)
print(f"τ_n 与 φ(n) 的Pearson相关系数: {corr:.5f}")

# --- 可视化 K(n) ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, N + 1), K_vals, label=r'$K(n) = \frac{d \log|\phi|}{d \log H}$')
plt.axhline(0.5, color='green', linestyle='--', label='K = 0.5')
plt.axhline(1, color='red', linestyle='--', label='K = 1')
plt.title('Entropy / Structure Ratio K(n)')
plt.xlabel('n')
plt.ylabel('K(n)')
plt.legend()
plt.grid(True)
plt.show()

# --- 可视化 R(n) ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, N + 1), ratio_vals, label=r'$R(n) = \phi / \sqrt{H}$')
plt.axhline(1, color='red', linestyle='--', label='R = 1')
plt.title('Distribution of R(n)')
plt.xlabel('n')
plt.ylabel('R(n)')
plt.legend()
plt.grid(True)
plt.show()

# --- 可视化 τ_n ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, N + 1), tau_n, label=r'$\tau_n = \pi/2 - \arctan(2 \gamma_n)$')
plt.title('Structural Residue \tau_n')
plt.xlabel('n')
plt.ylabel('\tau_n')
plt.legend()
plt.grid(True)
plt.show()