import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene

def main():
    # CSV 文件所在的目录
    folder_path = (
        "/Users/baoxuan/Library/Containers/com.tencent.xinWeChat/"
        "Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/"
        "e9a7feea2f06cc446a731d8c3e5a4a59/Message/MessageTemp/9e20f478899dc29eb19741386f9343c8/File/data(1)"
    )
    
    # 用于存储每个被试（文件）的 pumps 方差
    var_session1_list = []
    var_session2_list = []
    
    # 用于存储所有被试中 session1 和 session2 的 10 个 pumps 数值（用于绘制直方图）
    all_pumps_session1 = []
    all_pumps_session2 = []
    
    # 遍历目录下所有文件
    for filename in os.listdir(folder_path):
        # 仅处理 CSV 文件
        if not filename.endswith(".csv"):
            continue
        
        lower_filename = filename.lower()
        # 仅处理文件名中同时包含 "bart" 且包含 "session1" 或 "session2" 的文件
        if "bart" not in lower_filename:
            continue
        
        if "session1" in lower_filename:
            session_type = "session1"
        elif "session2" in lower_filename:
            session_type = "session2"
        else:
            continue
        
        csv_path = os.path.join(folder_path, filename)
        
        # 读取文件所有行
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 检查是否至少有 11 行（因为我们需要第2到第11行）
        if len(lines) < 11:
            print(f"警告: {filename} 行数不足 11 行，跳过。")
            continue
        
        pumps_list = []
        # 提取第2到第11行（索引 1~10）的第2列数据
        for i in range(1, 11):
            row = lines[i].strip().split(",")
            if len(row) < 2:
                print(f"警告: {filename} 第{i+1}行列数不足2列，跳过该文件。")
                pumps_list = []
                break
            try:
                # 将第2列（索引1）转换为浮点数
                val = float(row[1])
                pumps_list.append(val)
            except ValueError:
                print(f"警告: {filename} 第{i+1}行第2列无法转换为数字，跳过该文件。")
                pumps_list = []
                break
        
        if len(pumps_list) == 10:
            # 计算该被试的 pumps 次数的样本方差（ddof=1）
            variance_pumps = np.var(pumps_list, ddof=1)
            if session_type == "session1":
                var_session1_list.append(variance_pumps)
                all_pumps_session1.extend(pumps_list)
            elif session_type == "session2":
                var_session2_list.append(variance_pumps)
                all_pumps_session2.extend(pumps_list)
    
    # 计算两组被试的 pumps 方差平均值
    avg_var_session1 = np.mean(var_session1_list) if var_session1_list else None
    avg_var_session2 = np.mean(var_session2_list) if var_session2_list else None
    
    print("=== 每个被试的 pumps 方差平均值 ===")
    print(f"Session1 平均方差: {avg_var_session1 if avg_var_session1 is not None else '无数据'}")
    print(f"Session2 平均方差: {avg_var_session2 if avg_var_session2 is not None else '无数据'}")
    
    # 对每个被试的 pumps 方差进行 Levene 检验（注意这里是对各文件方差进行检验）
    if len(var_session1_list) > 1 and len(var_session2_list) > 1:
        stat, p_value = levene(var_session1_list, var_session2_list)
        print("\n=== Levene 方差齐性检验 ===")
        print(f"统计量: {stat:.4f}, p 值: {p_value:.4f}")
        if p_value < 0.05:
            print("结果显示，两组数据的方差存在显著差异。")
        else:
            print("结果显示，两组数据的方差无显著差异。")
    else:
        print("数据不足，无法进行方差差异检验。")
    
    # 绘制直方图：统计所有被试中 session1 和 session2 的 pumps 分布
    # Session1 直方图
    plt.figure(figsize=(8, 6))
    plt.hist(all_pumps_session1, bins=10, color="skyblue", edgecolor="black")
    plt.title("Session1 Pumps Distribution")
    plt.xlabel("Pumps Count")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    
    # Session2 直方图
    plt.figure(figsize=(8, 6))
    plt.hist(all_pumps_session2, bins=10, color="lightgreen", edgecolor="black")
    plt.title("Session2 Pumps Distribution")
    plt.xlabel("Pumps Count")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()