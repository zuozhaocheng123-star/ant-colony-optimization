import numpy as np
import random
import matplotlib.pyplot as plt

class AntColonyOptimizer:
    """
    蚁群优化算法实现类
    """
    
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        初始化蚁群优化算法
        
        参数:
        distances (2D numpy.array): 距离矩阵
        n_ants (int): 蚂蚁数量
        n_best (int): 每次迭代中更新信息素的最优蚂蚁数量
        n_iterations (int): 迭代次数
        decay (float): 信息素挥发率
        alpha (int): 信息素重要程度因子
        beta (int): 启发式因子重要程度因子
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
    def run(self):
        """
        运行蚁群算法
        
        返回:
        tuple: (最优路径, 最短距离)
        """
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        
        for i in range(self.n_iterations):
            # 所有蚂蚁构建路径
            all_paths = self.gen_all_paths()
            # 更新信息素
            self.spread_pheronome(all_paths, self.n_best)
            # 获取当前迭代最优路径
            shortest_path = min(all_paths, key=lambda x: x[1])
            print(f"迭代 {i+1}: 当前最优距离 = {shortest_path[1]}")
            
            # 更新全局最优路径
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            
            # 信息素挥发
            self.pheromone = self.pheromone * self.decay
            
        return all_time_shortest_path
    
    def spread_pheronome(self, all_paths, n_best):
        """
        在路径上释放信息素
        
        参数:
        all_paths (list): 所有蚂蚁的路径
        n_best (int): 释放信息素的最优蚂蚁数量
        """
        # 按路径长度排序
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                # 在路径上释放信息素
                self.pheromone[move] += 1.0 / self.distances[move]
    
    def gen_path_dist(self, path):
        """
        计算路径总距离
        
        参数:
        path (list): 路径
        
        返回:
        float: 路径总距离
        """
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist
    
    def gen_all_paths(self):
        """
        生成所有蚂蚁的路径
        
        返回:
        list: 所有蚂蚁的路径和距离
        """
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths
    
    def gen_path(self, start):
        """
        生成单个蚂蚁的路径
        
        参数:
        start (int): 起始节点
        
        返回:
        list: 蚂蚁的路径
        """
        path = []
        visited = set()
        visited.add(start)
        prev = start
        
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        
        # 回到起点
        path.append((prev, start))
        return path
    
    def pick_move(self, pheromone, dist, visited):
        """
        选择下一个移动节点
        
        参数:
        pheromone (numpy.array): 当前节点信息素
        dist (numpy.array): 当前节点到其他节点的距离
        visited (set): 已访问节点集合
        
        返回:
        int: 下一个节点
        """
        # 避免除零错误
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        
        # 计算转移概率
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        
        # 归一化
        norm_row = row / row.sum()
        
        # 按概率选择下一个节点
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

# 使用示例
if __name__ == "__main__":
    # 创建一个简单的距离矩阵 (表示城市之间的距离)
    distances = np.array([
        [np.inf, 2, 2, 5, 7],
        [2, np.inf, 4, 8, 2],
        [2, 4, np.inf, 1, 3],
        [5, 8, 1, np.inf, 2],
        [7, 2, 3, 2, np.inf]
    ])
    
    # 创建蚁群优化算法实例
    aco = AntColonyOptimizer(distances, 10, 3, 100, 0.95, alpha=1, beta=1)
    
    # 运行算法
    shortest_path = aco.run()
    
    print(f"最短路径: {shortest_path[0]}")
    print(f"最短距离: {shortest_path[1]}")