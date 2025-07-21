# 修正：将 phi_vals 转为 float 类型以支持 numpy 运算
import numpy as np
import matplotlib.pyplot as plt
from mpmath import zetazero, mp, atan, pi
from scipy.stats import entropy
from scipy.signal import savgol_filter
from tqdm import tqdm
import pandas as pd
from IPython.display import display

# 设置高精度
mp.dps = 50

# 获取前 N 个 Riemann 零点的虚部 γₙ
def get_gamma_n(N):
    return np.array([float(mp.im(zetazero(n))) for n in range(1, N + 1)])

# 几何相位逼近 φ(n) = π/2 - arctan(c * γₙ)
def geometric_phase(gamma_n, c=2.0):
    return (np.pi / 2 - np.arctan(c * gamma_n)) / np.arange(1, len(gamma_n) + 1)

# 计算结构熵 H = log(1 + φ²)
def compute_entropy(phi_vals):
    return np.log(1 + phi_vals ** 2)

# 设置参数
N = 100  # 零点数量
sigma_scan = np.linspace(0.01, 0.99, 99)

# 预计算 φ(n)
gamma_n = get_gamma_n(N)
phi_vals = geometric_phase(gamma_n)

# 初始化熵列表
entropy_values = []

# 对每个 σ 计算 H(n)^σ 的 Shannon 熵
for sigma in tqdm(sigma_scan, desc="Scanning σ"):
    H_vals = compute_entropy(phi_vals)
    H_sigma = H_vals ** sigma
    H_sigma = savgol_filter(H_sigma, 11, 3)
    hist, _ = np.histogram(H_sigma, bins=50, density=True)
    hist = hist[hist > 0]
    shannon_H = entropy(hist, base=2)
    entropy_values.append(shannon_H)

# 构建 DataFrame 并可视化
df = pd.DataFrame({"σ": sigma_scan, "Shannon Entropy": entropy_values})
display(df)

plt.figure(figsize=(10, 5))
plt.plot(df["σ"], df["Shannon Entropy"], color='darkgreen', marker='o', markersize=3)
plt.title("Shannon Entropy of Geometric Phase-Based H(n)^σ")
plt.xlabel("σ")
plt.ylabel("Entropy")
plt.grid(True)
plt.show()
