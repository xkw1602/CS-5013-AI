# Problem: Implement the Breadth-First Search (BFS), Depth-First Search (DFS) 
# and Greedy Best-First Search (GBFS) algorithms on the graph from Figure 1 in hw1.pdf.


# Instructions:
# 1. Represent the graph from Figure 1 in any format (e.g. adjacency matrix, adjacency list).
# 2. Each function should take in the starting node as a string. Assume the search is being performed on
#    the graph from Figure 1.
#    It should return a list of all node labels (strings) that were expanded in the order they where expanded.
#    If there is a tie for which node is expanded next, expand the one that comes first in the alphabet.
# 3. You should only modify the graph representation and the function body below where indicated.
# 4. Do not modify the function signature or provided test cases. You may add helper functions. 
# 5. Upload the completed homework to Gradescope, it must be named 'hw1.py'.

# Examples:
#     The test cases below call each search function on node 'S' and node 'A'
# -----------------------------
from queue import Queue, LifoQueue

fig1 = {
    'A' : [('B', 4), ('E', 1)],
    'B' : [('A', 4), ('C', 2), ('F', 2)],
    'C' : [('B', 4), ('H', 4), ('S', 3)],
    'D' : [('S', 2), ('L', 8)],
    'E' : [('A', 1), ('F', 3), ('I', 6)],
    'F' : [('B', 2), ('E', 3), ('J', 6), ('K', 4)],
    'G' : [('M', 4), ('N', 4), ('Q', 10)],
    'H' : [('C', 4), ('K', 3), ('L', 7)],
    'I' : [('E', 6), ('J', 1), ('M', 5)],
    'J' : [('F', 6), ('I', 1), ('K', 3), ('N', 3)],
    'K' : [('F', 4), ('H', 3), ('J', 3), ('L', 9), ('P', 3)],
    'L' : [('D', 8), ('H', 7), ('K', 9), ('Q', 10)],
    'M' : [('G', 4), ('I', 5)],
    'N' : [('G', 4), ('J', 3), ('P', 2)],
    'P' : [('K', 3), ('N', 2)],
    'Q' : [('G', 10), ('L', 10)],
    'S' : [('C', 3), ('D', 2)]
}

h_list = {'A':10, 'B':9, 'C':16, 'D':21, 'E':13, 'F':9, 'G':0, 'H':12, 'I':9, 'J':5, 'K':8, 'L':18, 'M':3, 'N':4, 'P':6, 'Q':9, 'S':17}

def BFS(start: str) -> list:

    # Initialize queue, list, set. Set ensures each node is only accessed once
    q = Queue()
    expanded = []
    visited = set()

    # Add start node to queue and visited
    q.put(start)
    visited.add(start)


    while not q.empty():

        # Add current node to expansion list
        current_node = q.get()
        expanded.append(current_node)

        # Explore all edges of current node
        for node, weight in fig1.get(current_node):

            # If node has not been visited, add to queue
            if node not in visited:
                q.put(node)
                visited.add(node)

                # If current node has a path to G, exit loop
                if node == 'G':
                    expanded.append(node)
                    return expanded

    return expanded

def DFS(start: str) -> list:

    # Initialize stack & list. set is not necessary for DFS
    s = LifoQueue()
    expanded = []

    # Add start node to stack 
    s.put(start)

    while not s.empty():
        # Pop node off stack
        current_node = s.get()
        expanded.append(current_node)

        # Same implementation as BFS, adding new nodes to the stack in reverse order, to pop them in alphabetical order
        for node, weight in reversed(fig1.get(current_node)):
            if node not in expanded:
                s.put(node)
                if node == 'G':
                    expanded.append(node)
                    return expanded

    return expanded

def GBFS(start: str) -> list:

    # initialize fringe and structures
    expanded = []
    fringe = set()
    visited = set()

    # Add start node to fringe and set to current node
    fringe.add(start)
    visited.add(start)
    current_node = start

    while fringe:
        # Add current node to list, mark as visited and remove from fringe
        expanded.append(current_node)
        fringe.remove(current_node)

        for node, weight in fig1.get(current_node):
            if(node not in visited):
                fringe.add(node)
                visited.add(node)
            if node == 'G':
                expanded.append(node)
                return expanded
            
        # set current node to the lowest h-value in the fringe and loop
        current_node = min(fringe, key=lambda node: h_list.get(node))
        
    return expanded

# test cases - DO NOT MODIFY THESE
def run_tests():
    # Test case 1: BFS starting from node 'A'
    assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
    # Test case 2: BFS starting from node 'S'
    assert BFS('S') == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    # Test case 3: DFS starting from node 'A'
    assert DFS('A') == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"
    
    # Test case 4: DFS starting from node 'S'
    assert DFS('S') == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"

    
    
    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()
