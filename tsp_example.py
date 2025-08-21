import numpy as np
import matplotlib.pyplot as plt
from ant_colony import AntColonyOptimizer

def create_distance_matrix(cities):
    """
    根据城市坐标创建距离矩阵
    
    参数:
    cities (list): 城市坐标列表 [(x1, y1), (x2, y2), ...]
    
    返回:
    numpy.array: 距离矩阵
    """
    n = len(cities)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # 计算欧几里得距离
                distances[i][j] = np.sqrt((cities[i][0] - cities[j][0])**2 + 
                                          (cities[i][1] - cities[j][1])**2)
            else:
                distances[i][j] = np.inf  # 对角线设为无穷大
    
    return distances

def plot_cities_and_path(cities, path, title="TSP Solution"):
    """
    绘制城市和路径
    
    参数:
    cities (list): 城市坐标列表
    path (list): 最优路径
    title (str): 图标题
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制城市
    x_coords = [city[0] for city in cities]
    y_coords = [city[1] for city in cities]
    plt.scatter(x_coords, y_coords, c='red', s=100, zorder=2)
    
    # 标注城市编号
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    
    # 绘制路径
    for i in range(len(path[0])):
        start_city = path[0][i][0]
        end_city = path[0][i][1]
        plt.plot([cities[start_city][0], cities[end_city][0]], 
                 [cities[start_city][1], cities[end_city][1]], 
                 'b-', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：使用蚁群算法解决TSP问题
    """
    # 定义城市坐标 (这里使用一个经典的TSP问题实例)
    cities = [
        (60, 200), (180, 200), (80, 180), (140, 180),
        (20, 160), (100, 160), (200, 160), (140, 140),
        (40, 120), (100, 120), (180, 100), (60, 80),
        (120, 80), (180, 60), (20, 40), (100, 40),
        (200, 40), (20, 20), (60, 20), (160, 20)
    ]
    
    print(f"城市数量: {len(cities)}")
    print("城市坐标:")
    for i, city in enumerate(cities):
        print(f"  城市 {i}: ({city[0]}, {city[1]})")
    
    # 创建距离矩阵
    distances = create_distance_matrix(cities)
    
    # 设置蚁群算法参数
    n_ants = 20        # 蚂蚁数量
    n_best = 5         # 最优蚂蚁数量
    n_iterations = 100 # 迭代次数
    decay = 0.95       # 信息素挥发率
    alpha = 1          # 信息素重要程度
    beta = 2           # 启发式因子重要程度
    
    print("\n开始运行蚁群算法...")
    print(f"参数设置: 蚂蚁数量={n_ants}, 迭代次数={n_iterations}, 挥发率={decay}")
    
    # 创建蚁群优化器实例
    aco = AntColonyOptimizer(distances, n_ants, n_best, n_iterations, decay, alpha, beta)
    
    # 运行算法
    shortest_path = aco.run()
    
    # 输出结果
    print(f"\n算法运行完成!")
    print(f"最短路径: {shortest_path[0]}")
    print(f"最短距离: {shortest_path[1]:.2f}")
    
    # 绘制结果
    try:
        plot_cities_and_path(cities, shortest_path, "蚁群算法求解TSP问题结果")
    except Exception as e:
        print(f"绘图时出错: {e}")
        print("跳过可视化步骤")

if __name__ == "__main__":
    main()