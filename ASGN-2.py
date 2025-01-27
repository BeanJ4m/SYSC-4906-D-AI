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
        exploration_steps = 0
        cost = 0
        path = []  
        explored = set()  
        fringe = deque() 

        # Start BFS
        fringe.append((self.start, [self.start]))  # Add the start node with the initial path

        while fringe:
            # Get the next node and its path
            node, path = fringe.popleft()

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
                return len(path), len(explored), cost

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
    
    def dfs(self):
     # Implement DFS logic: return exploration steps, path cost, and path length, or None if no path is found.
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
                return len(path), len(explored), cost

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
              
    def ucs(self):
    # Implement UCS logic: return exploration steps, path cost, and path length, or None if no path is found.
        exploration_steps = 0
        explored = set()
        fringe = []
        
        # The fringe will hold tuples of (cumulative_cost, current_position, path_taken)
        heapq.heappush(fringe, (0, self.start, [self.start]))

        while fringe:
            # Pop the lowest-cost path from the priority queue
            cumulative_cost, current_node, path = heapq.heappop(fringe)

            # Skip if already explored
            if current_node in explored:
                continue

            # Mark the current node as explored
            explored.add(current_node)
            exploration_steps += 1

            # Check if we reached the goal
            if current_node == self.goal:
                self.print_path(path)
                return len(path), exploration_steps, cumulative_cost

            # Explore neighbors
            neighbors = [
                (current_node[0] - 1, current_node[1]),  # North
                (current_node[0], current_node[1] + 1),  # East
                (current_node[0] + 1, current_node[1]),  # South
                (current_node[0], current_node[1] - 1),  # West
            ]
            
            for neighbor in neighbors:
                if self.is_valid(neighbor) and neighbor not in explored:
                    # Calculate the new cumulative cost to this neighbor
                    new_cost = cumulative_cost + self.get_cost(neighbor)
                    # Push this neighbor onto the fringe
                    heapq.heappush(fringe, (new_cost, neighbor, path + [neighbor]))

        # If no path is found
        return None
    
    def astar(self, heuristic=None):
        # Implement A* logic: return exploration steps, path cost, and path length, or None if no path is found.
        if heuristic is None:
            def heuristic(pos):
                return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
        
        fringe = []
        explored = set()
        
        heapq.heappush(fringe, (0 + heuristic(self.start), 0, self.start, deque([self.start])))  # (f, g, node, path)
        
        while fringe:
            f, g, node, path = heapq.heappop(fringe)
            if node == self.goal:
                self.print_path(list(path))
                return len(path), len(explored), g
            if node in explored:
                continue
            neighbors = [
                (node[0] - 1, node[1]),  # North
                (node[0], node[1] + 1),  # East
                (node[0] + 1, node[1]),  # South
                (node[0], node[1] - 1),  # West
            ]
            explored.add(node)

            for neighbor in neighbors:
                if self.is_valid(neighbor) and neighbor not in explored:
                    g_new = g + self.get_cost(neighbor)  # Cost from start to neighbor
                    f_new = g_new + heuristic(neighbor)  # Total estimated cost (f = g + h)
                    new_path = deque(path)
                    new_path.append(neighbor)
                    heapq.heappush(fringe, (f_new, g_new, neighbor, new_path))
                    
        return None
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