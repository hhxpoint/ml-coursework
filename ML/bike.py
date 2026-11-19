# 习题 3-7: 有界深度优先搜索求解八数码难题
import copy

def print_board(state):
  
    for i in range(0, 9, 3):
        print(f"[{state[i]} {state[i+1]} {state[i+2]}]")
    print("-------")

def get_successors(state):
    """获取当前状态所有可能的下一步状态"""
    successors = []
    zero_idx = state.index(0)  # 找到空格(0)的位置
    row, col = divmod(zero_idx, 3)

    # 向上移动 (0向上，实际上是上面的数字向下)
    if row > 0:
        new_state = list(state)
        new_state[zero_idx], new_state[zero_idx - 3] = new_state[zero_idx - 3], new_state[zero_idx]
        successors.append(tuple(new_state))
    # 向下移动
    if row < 2:
        new_state = list(state)
        new_state[zero_idx], new_state[zero_idx + 3] = new_state[zero_idx + 3], new_state[zero_idx]
        successors.append(tuple(new_state))
    # 向左移动
    if col > 0:
        new_state = list(state)
        new_state[zero_idx], new_state[zero_idx - 1] = new_state[zero_idx - 1], new_state[zero_idx]
        successors.append(tuple(new_state))
    # 向右移动
    if col < 2:
        new_state = list(state)
        new_state[zero_idx], new_state[zero_idx + 1] = new_state[zero_idx + 1], new_state[zero_idx]
        successors.append(tuple(new_state))

    return successors

def depth_limited_search(path, goal_state, depth_limit):
    """有界深度优先搜索递归函数"""
    current_state = path[-1]
    if current_state == goal_state:
        return path
    if depth_limit == 0:
        return None

    for next_state in get_successors(current_state):
        if next_state not in path: # 简单的环检测，避免重复访问当前路径上的节点
            path.append(next_state)
            result = depth_limited_search(path, goal_state, depth_limit - 1)
            if result is not None:
                return result
            path.pop() # 回溯
    return None

def iddfs(start_state, goal_state, max_depth=50):
    """迭代加深控制循环"""
    for depth in range(max_depth):
        # print(f"正在搜索深度界限: {depth}")
        result = depth_limited_search([start_state], goal_state, depth)
        if result is not None:
            return result
    return None

if __name__ == "__main__":
    # 定义初始状态和目标状态 (用一维元组表示，0代表空格)
    S0 = (2, 8, 0, 1, 6, 3, 7, 5, 4)
    Sg = (1, 2, 3, 8, 0, 4, 7, 6, 5)

    print("初始状态 S0:")
    print_board(S0)
    print("目标状态 Sg:")
    print_board(Sg)

    print("开始进行有界深度优先搜索 (IDDFS)...")
    solution_path = iddfs(S0, Sg)

    if solution_path:
        print(f"\n找到解答！总共需要 {len(solution_path) - 1} 步。")
        print("最终结果（搜索路径）如下:")
        for step, state in enumerate(solution_path):
            print(f"Step {step}:")
            print_board(state)
    else:
        print("在设定的最大深度内未找到解。")