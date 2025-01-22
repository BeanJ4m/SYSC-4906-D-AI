from collections import deque
import heapq

def print_forest(forest):
    for row in forest:
        print(" ".join(row))
    print()
class SearchAgent:
    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
    def is_valid(self, position):
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] != '#'
    def get_cost(self, position):
        cell = self.grid[position[0]][position[1]]
        if cell == 'f':
            return 3
        elif cell == 'F':
            return 5
        return 1
    def print_path(self, path):
        temp_grid = [row[:] for row in self.grid] # Create a copy of the grid
        for row, col in path[1:-1]: # Avoid overriding start and goal
            temp_grid[row][col] = 'P'
        for row in temp_grid:
            print(" ".join(row))
    def bfs(self):
    # Implement BFS logic: return exploration steps, path cost, and path length, or None if no path is found.
        cost = 0
        path = deque()
        explored = set()
        fringe = deque()
        explored.add(self.start)
        fringe.append(self.start)
        return len(explored), cost, len(path)
    def dfs(self):
        
        exploration_steps = 0
        cost = 0
        path = []  
        explored = set()  
        fringe = deque() 

        # Start DFS
        fringe.append((self.start, [self.start]))  # Add the start node with the initial path

        while fringe:
            # Get the next node and its path
            node, path = fringe.pop()

            # Skip if already explored
            if node in explored:
                continue

            # Mark as explored
            explored.add(node)
            exploration_steps += 1

            # Update cost
            cost += self.get_cost(node)

            # Stop if the goal is reached
            if node == self.goal:
                self.print_path(path)
                return len(explored), cost, len(path)

            # Backtracking happens here as the `current_path` is updated for each new neighbor
            neighbors = [
                (node[0] - 1, node[1]),  # North
                (node[0], node[1] + 1),  # East
                (node[0] + 1, node[1]),  # South
                (node[0], node[1] - 1),  # West
            ]
            for neighbor in neighbors:
                # If the neighbor is valid and hasn't been explored, append it to the fringe
                if self.is_valid(neighbor) and neighbor not in explored:
                    # The new path here represents the backtracking step: `current_path + [neighbor]`
                    fringe.append((neighbor, path + [neighbor]))  # Add new path with the neighbor

        # If no path is found
        return None
            
  
    # Implement DFS logic: return exploration steps, path cost, and path length, or None if no path is found.
    def ucs(self):
    # Implement UCS logic: return exploration steps, path cost, and path length, or None if no path is found.
        length = 0
        cost = 0
        path = deque()
        explored = deque()
        fringe = deque()
        explored.append(self.start)
        fringe.append(self.start)
        
        return len(explored), cost, len(path)
    def astar(self, heuristic=None):
        length = 0
        cost = 0
        path = deque()
        explored = deque()
        fringe = deque()
        explored.append(self.start)
        fringe.append(self.start)
        
        if heuristic is None:
            def heuristic(pos):
                return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
    # Implement A* logic: return exploration steps, path cost, and path length, or None if no path is found.
        return len(explored), cost, len(path)
def test_search_agent(agent):
    results = {}
    print("\n--- BFS ---")
    path_length, exploration_steps, cost_length = agent.bfs()
    results['BFS'] = (path_length, exploration_steps, cost_length)
    print(f"Path Length: {path_length} steps")
    print(f"Exploration Steps: {exploration_steps}")
    print(f"Cost Length: {cost_length}")
    print("\n--- DFS ---")
    path_length, exploration_steps, cost_length = agent.dfs()
    results['DFS'] = (path_length, exploration_steps, cost_length)
    print(f"Path Length: {path_length} steps")
    print(f"Exploration Steps: {exploration_steps}")
    print(f"Cost Length: {cost_length}")
    print("\n--- UCS ---")
    path_length, exploration_steps, cost_length = agent.ucs()
    results['UCS'] = (path_length, exploration_steps, cost_length)
    print(f"Path Length: {path_length} steps")
    print(f"Exploration Steps: {exploration_steps}")
    print(f"Cost Length: {cost_length}")
    print("\n--- A* ---")
    path_length, exploration_steps, cost_length = agent.astar(lambda pos: abs(pos[0] - agent.goal[0]) +
    abs(pos[1] - agent.goal[1]))
    results['A*'] = (path_length, exploration_steps, cost_length)
    print(f"Path Length: {path_length} steps")
    print(f"Exploration Steps: {exploration_steps}")
    print(f"Cost Length: {cost_length}")
    return results
Agents = {}
forest0 = [ ['S', '.', '.', '.', '#', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['#', '#', '.', '#', '#', '.', '.', '#', '#', '#', '#', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '#', '.', '#', '.', '.'],
            ['.', '#', '#', '#', '.', '.', '#', '.', '#', '#', '#', '.', '#', '#', '#'],
            ['.', '#', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['.', '#', '.', '#', '#', '#', '.', '#', '#', '.', '#', '#', '#', '.', '#'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
            ['#', '#', '#', '.', '#', '#', '#', '#', '#', '#', '.', '#', '#', '.', '#'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['#', '.', '#', '#', '#', '#', '#', '#', '.', '#', '#', '.', '#', '#', '#'],
            ['#', '.', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['#', '.', '#', '#', '.', '#', '.', '#', '#', '#', '#', '.', '#', '#', '#'],
            ['#', '.', '#', '.', '.', '#', '.', '.', '.', '.', '#', '.', '.', '.', '#'],
            ['#', '.', '.', '.', '#', '.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
            ['#', '#', '#', '#', '#', '#', '#', '.', '.', '.', '#', '#', '#', '#', '*']]
start0 = (0, 0)
goal0 = (14, 14)
Agents[0] = SearchAgent(start0, goal0, forest0)
forest1 = [["S", ".", ".", ".", "#", "#", "#", ".", ".", ".", ".", ".", ".", ".", "."],
    ["#", "#", ".", "#", "#", ".", "f", "#", "#", "#", "#", ".", "F", ".", "."],
    [".", ".", ".", "#", ".", ".", ".", ".", ".", "f", "#", ".", ".", "F", "."],
    [".", "#", "#", "#", ".", ".", "#", ".", "#", "#", "#", ".", "#", "#", "#"],
    [".", "#", ".", ".", ".", "#", ".", ".", ".", ".", ".", ".", ".", ".", "#"],
    [".", "#", ".", "#", "#", "#", ".", "#", "#", ".", "#", "#", "#", ".", "#"],
    [".", "f", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "F", ".", "#"],
    ["#", "#", "#", ".", "#", "#", "#", "#", "#", "#", ".", "#", "#", ".", "#"],
    [".", ".", ".", ".", ".", ".", ".", ".", "f", ".", ".", ".", ".", ".", "."],
    ["#", ".", "#", "#", "#", "#", "#", "#", ".", "#", "#", ".", "F", "#", "#"],
    ["#", ".", ".", ".", ".", "#", ".", ".", ".", ".", ".", ".", "f", ".", "."],
    ["#", ".", "#", "#", ".", "#", ".", "#", "#", "#", "#", ".", "#", "#", "#"],
    ["#", ".", "#", ".", ".", "#", ".", ".", ".", ".", "#", ".", "F", ".", "#"],
    ["#", ".", ".", ".", "#", ".", ".", "#", ".", ".", ".", ".", ".", ".", "F"],
    ["#", "#", "#", "#", "#", "#", "#", ".", "F", ".", "#", "#", "#", "f", "*"]]
start1 = (0, 0)
goal1 = (14, 14)
Agents[1] = SearchAgent(start1, goal1, forest1)
forest2 = [["S", ".", ".", ".", ".", "#", ".", ".", ".", "f", ".", "#", ".", ".", "."],
    [".", "F", ".", "#", ".", "#", ".", "#", ".", ".", ".", ".", ".", "F", "."],
    [".", "#", ".", ".", "f", "#", ".", ".", "#", ".", ".", "F", ".", ".", "."],
    ["f", ".", ".", "#", ".", "#", "#", ".", ".", "#", "#", "#", "#", "#", "#"],
    [".", ".", ".", "#", ".", "f", ".", ".", ".", ".", ".", ".", "f", ".", "."],
    [".", ".", "#", ".", "#", ".", "#", "#", "#", "#", ".", ".", "F", ".", "#"],
    [".", ".", ".", ".", "#", ".", ".", ".", ".", "f", ".", ".", ".", ".", "."],
    [".", ".", "F", ".", "#", ".", "#", "#", "#", ".", "#", "#", "#", ".", "#"],
    ["#", ".", "f", ".", ".", ".", ".", ".", ".", "#", "f", ".", ".", ".", "."],
    ["#", ".", "#", ".", "#", "#", "#", "#", ".", "#", "#", "#", ".", "F", "#"],
    ["f", ".", "#", ".", ".", ".", ".", ".", "#", ".", ".", ".", ".", ".", "."],
    ["#", ".", "#", ".", "#", ".", "#", ".", "#", ".", "#", "#", "#", "#", "#"],
    [".", ".", ".", ".", "#", "F", "#", ".", ".", ".", "f", ".", "f", ".", "."],
    [".", "#", ".", ".", ".", "f", ".", "#", "#", ".", ".", ".", ".", "F", "F"],
    [".", "#", ".", ".", "#", "#", ".", "#", ".", ".", ".", ".", ".", "f", "*"]]
start2 = (0, 0)
goal2 = (14, 14)
Agents[2] = SearchAgent(start2, goal2, forest2)
forest3 = [["S", ".", ".", "#", ".", ".", ".", "f", ".", "#", ".", ".", ".", ".", "."],
    ["#", ".", "F", ".", ".", "F", "#", ".", ".", ".", ".", ".", "F", "#", "#"],
    [".", ".", ".", ".", ".", "#", ".", ".", ".", ".", "#", ".", ".", ".", "."],
    ["#", "#", "#", ".", "#", ".", ".", ".", "#", "f", ".", ".", ".", ".", "#"],
    [".", ".", ".", ".", ".", ".", ".", "#", "F", ".", ".", ".", "#", ".", "."],
    ["#", ".", "#", "#", "#", "#", "#", "#", ".", ".", ".", "#", "#", ".", "."],
    ["#", ".", ".", ".", ".", ".", ".", "#", "f", ".", ".", ".", "#", ".", "."],
    [".", ".", "f", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", ".", "#"],
    [".", ".", ".", ".", "#", ".", ".", ".", "#", "#", "#", ".", ".", ".", "."],
    [".", ".", "F", ".", "#", ".", "F", ".", "#", ".", "f", ".", ".", "#", "#"],
    ["f", ".", ".", ".", "#", "#", "#", ".", ".", ".", "#", "#", "#", "#", "#"],
    [".", ".", ".", ".", ".", ".", ".", "#", ".", "F", ".", ".", ".", ".", "#"],
    [".", ".", "#", ".", "#", "#", ".", "#", "f", ".", ".", ".", ".", "f", "#"],
    ["#", ".", "#", ".", ".", "F", ".", ".", ".", ".", ".", "#", ".", ".", "F"],
    ["#", "#", "#", "#", ".", ".", ".", "f", ".", "#", ".", ".", ".", "f", "*"]]
start3 = (0, 0)
goal3 = (14, 14)
Agents[3] = SearchAgent(start3, goal3, forest3)
for AGENT in Agents:
    print(f"Forest {AGENT} Solution:")
    print(test_search_agent(Agents[AGENT]))