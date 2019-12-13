import random
import copy
import sys
import math
import pandas as pd
import time

class Node:
    
    def __init__(self,statestring="b12 345 678", hval=0):
        #Initialize variables
        self.state = self.setState(statestring)
        self.hval = hval
        self.parent = None
        self.path_direction = None
    
    def setState(self, statestring):
        #Takes a string such as 'b12 345 678' and represents as a state
        validIndexCount = 0
        i=0
        state=[["", "", ""],["", "", ""],["", "", ""]]
        for i in range(len(statestring)):
            if statestring[i] != ' ':
                state[int(validIndexCount/3)][int(validIndexCount%3)] = statestring[i]
                validIndexCount+=1
        self.state = state
        return state
        
    def getBlankPosition(self):
        #Returns the indices of the blank tile for the puzzle
        for row in enumerate(self.state):
            for char in enumerate(row[1]):
                if char[1] == 'b': #if the current tile is the blank tile return its position
                    position = [row[0], char[0]]
                    return position
        return None
    
    def printState(self):
        print(self.state[0])
        print(self.state[1])
        print(self.state[2])
        print()
        
    def move(self, direction):
        #Moves the blank tile in the specified direction
        #Returns Boolean if move was successful
        blank_pos = self.getBlankPosition()
        if direction == "up":
            target_pos = [blank_pos[0] - 1, blank_pos[1]]
        elif direction == "down":
            target_pos = [blank_pos[0] + 1, blank_pos[1]]
        elif direction == "left":
            target_pos = [blank_pos[0], blank_pos[1] - 1]
        elif direction == "right":
            target_pos = [blank_pos[0], blank_pos[1] + 1]
        if (3 > target_pos[0] >= 0) and (3> target_pos[1] >= 0):
            target_val = self.state[target_pos[0]][target_pos[1]]
            self.state[target_pos[0]][target_pos[1]] = 'b'
            self.state[blank_pos[0]][blank_pos[1]] = target_val
            return True
        else:
            return False
        
    def randomizeState(self, numMoves):
        #Randomizes the state by performing n pseudo-random moves
        #Moves are pseudo-random beacuse generator is seeded, so the moves are essentially already predetermined, but they are still random
        numMoves = int(numMoves)
        moves = ["up", "left", "down", "right"]
        seed_val = 0
        for i in range(numMoves):
            validMove=False
            while not validMove:
                try:
                    random.seed(a=seed_val)
                    randInt = random.randrange(0, 4)
                    currentMove = moves[randInt]
                    self.move(currentMove)
                    validMove=True
                except:
                    validMove=False
                    seed_val += 1
            seed_val += 5
            #print(self.state)
        
    def randomizeStateNoSeed(self, numMoves):
        #This method is used to generate different random solvable states upon function call and does not use a seed to ensure local randomness
        #It is used for the expirements section of the writeup
        
        numMoves = int(numMoves)
        moves = ["up", "left", "down", "right"]
        for i in range(numMoves):
            validMove=False
            while not validMove:
                try:
                    randInt = random.randrange(0, 4)
                    currentMove = moves[randInt]
                    self.move(currentMove)
                    validMove=True
                except:
                    validMove=False

class PuzzleSolve:
    
    def __init__(self, puz_type="node"):
        #Establish a master node that the puzzle solver will work off of
        if puz_type == "node":
            self.master_node = Node()
        elif puz_type == "cube":
            self.master_node = Cube()
        self.max_nodes = math.inf
    
    def run_commands(self, filename):
        
        #Read commands from text file
        with open(filename) as file:
            line = file.readline()
            while line:
                #Do stuff here
                print(line)
                line_tokens = line.split()
                method = line_tokens[0]
                if (method == "setState"):
                    self.master_node.setState(line_tokens[1] + line_tokens[2] + line_tokens[3])
                if (method == "printState"):
                    #print("Printing State", self.master_node.state)
                    self.master_node.printState()
                if (method == "move"):
                    if type(self.master_node) == Cube:
                        self.master_node.move(line_tokens[1], line_tokens[2])
                    else:
                        self.master_node.move(line_tokens[1])
                if (method == "randomizeState"):
                    self.master_node.randomizeState(line_tokens[1])
                if (method == "solve"):
                    self.solve(line_tokens[1], line_tokens[2])
                if (method == "maxNodes"):
                    self.maxNodes(line_tokens[1])
                line = file.readline()
                
    def solve(self, search, arg):
        """
        Determines which search to perform and handles printing results of search
        """

        if search == "A-star":
            final = self.Astar(arg)
        elif search == "beam":
            final = self.local_beam(arg)
            
        if final is None:
            print("No solution given max nodes: ", self.max_nodes)
            return None
            
        path_directions = self.generate_path_directions(final)
        print("Solution Length: ", len(path_directions))
        print("Solution (Directions): ", path_directions)
        print("Solution (Puzzle Path)")
        self.print_path(self.generate_path(final))
        if type(self.master_node) == Cube:
            self.master_node = Cube()
        else:
            self.master_node = Node("b12 345 678")
        
    def Astar(self, heuristic):
        """
        A-star search algorithm using the specified heuristic function
        """
        
        #List of successor states to be considered
        searchable = []
        #List of states that have been explored already
        searched = []
        node_depth = 0
        
        #define the start and goal state
        start_node = self.master_node
        goal_state = [["b", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]
        if type(start_node) == Cube:
            goal_state = [["r", "r", "r", "r"], #Front
                          ["b", "b", "b", "b"], #Left
                          ["g", "g", "g", "g"], #Back
                          ["y", "y", "y", "y"], #Right
                          ["w", "w", "w", "w"], #Top
                          ["o", "o", "o", "o"]] #Bottom
        
        #add the start node to the searchable list
        searchable.append(start_node)
        
        #While there are nodes to be searched
        while len(searchable) != 0 and node_depth < self.max_nodes:
            
            #Set the node to the node in searchable with the lowest heuristic value
            node = searchable[0]
                        
            #print(node.state, node.hval)
            
            #If the node contains the goal state
            if (node.state == goal_state):
                return node
            
            #add node to searched list
            searched.append(node.state)
            #remove node from searchable list
            searchable.pop(0)
            
            #Iterate through all the node's children
            for successor in self.generate_children_helper(node):
                
                #if the successor has already been searched, skip it
                if successor.state in searched:
                    continue
                    
                #Another node has been explored so increment count
                node_depth+=1
                    
                #generate heuristic value for child and assign it
                if heuristic == "h1":
                    successor.hval = self.heuristic1(successor) + node.hval
                elif heuristic == "h2":
                    successor.hval = self.heuristic2(successor) + node.hval
                elif heuristic == "cube":
                    successor.hval = self.cube_heuristic(successor) + node.hval
                
                #if the child has not already been queued to search, add it to the queue
                if successor.state not in [temp.state for temp in searchable]:
                    searchable.append(successor)
            
            #Sort the searchable list of successor nodes by heuristic value, so the front contains the 'best' node
            searchable.sort(key = lambda x:x.hval,reverse=False)
            
        #A solution can not be found
        return None
    
    def local_beam(self, num_states):
        """
        Local beam search with num_states as maximum stored states
        """
        
        #List of successor states to be considered
        searchable = []
        #List of states that have been explored already
        searched = []
        node_depth = 0
        
        num_states = int(num_states)
        
        #define the start and goal state
        start_node = self.master_node
        goal_state = [["b", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]
        if type(start_node) == Cube:
            goal_state = [["r", "r", "r", "r"], #Front
                          ["b", "b", "b", "b"], #Left
                          ["g", "g", "g", "g"], #Back
                          ["y", "y", "y", "y"], #Right
                          ["w", "w", "w", "w"], #Top
                          ["o", "o", "o", "o"]] #Bottom
        
        searchable.append(start_node)
        
        while len(searchable) != 0 and node_depth < self.max_nodes:
            
            temp_nodes = []
            
            ### This section generates all the successors of all nodes in the open list
            #for each state in the open list
            for state in searchable:
                #iterate through all of its children
                for child in self.generate_children_helper(state):
                    if child.state == goal_state:
                        return child
                    if type(self.master_node) == Cube:
                        child.hval = self.cube_heuristic(child)
                    else:
                        child.hval = self.heuristic2(child)
                    temp_nodes.append(child)
                #Another node has been explored so increment count
                node_depth+=1
                    
            #Empty the nodes contained in memory
            searchable = []
            
            #Sort the successor states by heuristic value
            temp_nodes.sort(key = lambda x:x.hval,reverse=False)
            
            #For each successor state
            for node in temp_nodes:
                #If the node has not already been explored, add it to the open list for consideration
                if node.state not in searched:
                    searched.append(node.state)
                    searchable.append(node)
                #Limit the search space to k best states
                if len(searchable) >= num_states:
                    break
                    
        return None

    def generate_children_helper(self, node):
        if type(node) == Cube:
            return self.generate_children_cube(node)
        else:
            return self.generate_children(node)
            
    def generate_children(self, node):
        """
        Generate the successors of the specified node, and return them in a list
        """

        moves = ["up", "down", "left", "right"]
        children = []
        for move in moves:
            child = copy.deepcopy(node)
            if child.move(move):
                child.parent = node
                child.path_direction = move
                children.append(child)
        return children

    def generate_children_cube(self, node):
        faces = ["front", "left", "back", "right", "top", "bottom"]
        valid_rotations_dict = {
            "front": ["left", "right"], 
            "left": ["towards", "away"], 
            "back": ["left", "right"], 
            "right": ["towards", "away"], 
            "top": ["left", "right"], 
            "bottom": ["left", "right"]
        }
        children = []
        for face in faces:
            for rotation in valid_rotations_dict[face]:
                child = copy.deepcopy(node)
                move = [face, rotation]
                child.move(move[0], move[1])
                child.parent = node
                child.path_direction = move
                children.append(child)
        return children
            
    def heuristic1(self, node):
        """
        Number of misplaced tiles
        The higher the number returned the less optimal the node is
        """

        num_tiles = 0
        for row in enumerate(node.state):
            for value in enumerate(row[1]):
                if value[1] == 'b':
                    if (row[0], value[0]) != (0, 0):
                        num_tiles+=1
                else:
                    if int(value[1]) != ((row[0]*3) + value[0]):
                        num_tiles+=1
        return num_tiles
    
    def heuristic2(self, node):
        """
        Distance of misplaced tiles from their goal positions
        The higher the number returned the less optimal the node is
        """
        goal_state = [["b", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]
        if type(node) == Cube:
            goal_state = [["r", "r", "r", "r"], #Front
                          ["b", "b", "b", "b"], #Left
                          ["g", "g", "g", "g"], #Back
                          ["y", "y", "y", "y"], #Right
                          ["w", "w", "w", "w"], #Top
                          ["o", "o", "o", "o"]] #Bottom
        heuristic_val = 0
        
        for row in enumerate(node.state):
            for value in enumerate(row[1]):
                if value[1] != goal_state[row[0]][value[0]]:
                    cur_position = [row[0], value[0]]
                    correct_position = self.lookup_position(goal_state, value[1])
                    heuristic_val += abs(cur_position[0] - correct_position[0]) + abs(cur_position[1] - correct_position[1])
        return heuristic_val

    def cube_heuristic(self, cube):
        #For each side, how many unique colors are there
        #The more unique colors per side the lower the heuristic value
        hval = 0
        for face in enumerate(cube.state):
            temp = set(face[1])
            hval+=len(temp)-1
        return hval
                    
    def lookup_position(self, state, desired):
        """
        Looks up the index of a character in the desired state
        """
        
        for row in enumerate(state):
            for value in enumerate(row[1]):
                if value[1] == desired:
                    return [row[0], value[0]]
                
    def generate_path(self, node):
        """
        Generates the path of board states following a nodes parent
        """
        
        path = [node]
        while node.parent != None:
            path.append(node.parent)
            node = node.parent
        path.reverse()
        return path
    
    def generate_path_directions(self, node):
        """
        Generates the path of move directions from the start node to the end position
        """
        
        path = [node.path_direction]
        while node.parent != None:
            path.append(node.parent.path_direction)
            node = node.parent
        path.pop(-1)
        path.reverse()
        return path
    
    def print_path(self, node_path):
        print("Start Node:")
        for node in node_path:
            node.printState()
            
    def maxNodes(self, nodes):
        nodes = int(nodes)
        self.max_nodes = nodes

#####################################################
###### Cube class for the extra credit portion ######
#####################################################

class Cube:
    
    def __init__(self):
        #Front, Left, Back, Right, Top, Bottom
        self.state = [["r", "r", "r", "r"], #Front
                      ["b", "b", "b", "b"], #Left
                      ["g", "g", "g", "g"], #Back
                      ["y", "y", "y", "y"], #Right
                      ["w", "w", "w", "w"], #Top
                      ["o", "o", "o", "o"]] #Bottom
        self.hval = 0
        self.parent = None
        self.path_direction = []
        
    def move(self, side, direction):
        
        #define side rotation indices
        front= None
        left = None
        back= None
        top = None
        right = None
        bottom = None
        
        #Define the side inheritance list
        sides = None
        
        #Set variables to rotate the front face of the cube
        if side == "front":
            
            #Index of side that is being rotated
            side_index = 0
            
            front=None
            left = [2,3]
            back=None
            top = [3,4]
            right = [1,4]
            bottom = [1,2]
                
            if direction == "right":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [0, 5, 2, 4, 1, 3]
                        
            elif direction == "left":
                
                #Set rotation direction
                rotation = "counter-clockwise"
                
                sides = [0, 4, 2, 5, 3, 1]
                
        if side == "left":
            
            side_index = 1
            
            front=[1,4]
            left = None
            back=[2,3]
            right = None
            top = [1,4]
            bottom = [1,4]
            
            if direction == "towards":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [4, 1, 5, 3, 2, 0]
                
            elif direction == "away":
                
                #Set rotation direction
                rotation = "counter-clockwise"
                
                sides = [5, 1, 4, 3, 0, 2]
                
        if side == "back":
            
            side_index = 2
            
            front= None
            left = [1,4]
            back= None
            right = [2,3]
            top = [1,2]
            bottom = [3,4,]
            
            if direction == "right":
                
                #Set rotation direction
                rotation = "counter-clockwise"
                
                sides = [0, 5, 2, 4, 1, 3]
                
            elif direction == "left":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [0, 4, 2, 5, 3, 1]
            
        if side == "right":
            
            side_index = 3
            
            front=[2,3]
            left = None
            back=[1,4]
            right = None
            top = [2,3]
            bottom = [2,3]
            
            if direction == "towards":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [4, 1, 5, 3, 2, 0]
                
            elif direction == "away":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [5, 1, 4, 3, 0, 2]
            
        if side == "top":
            
            side_index = 4
            
            front=[1,2]
            left = [1,2]
            back=[1,2]
            right = [1,2]
            top =None
            bottom = None
            
            if direction == "right":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [1, 2, 3, 0, 4, 5]
                
            elif direction == "left":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [3, 0, 1, 2, 4, 5]
            
        if side == "bottom":
            
            side_index = 5
            
            front=[3,4]
            left = [3,4]
            back=[3,4]
            right = [3,4]
            top = None
            bottom = None
            
            if direction == "right":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [1, 2, 3, 0, 4, 5]
                
            elif direction == "left":
                
                #Set rotation direction
                rotation = "clockwise"
                
                sides = [3, 0, 1, 2, 4, 5]
            
        #Rotate the left face of cube in specified direction
        self.state[side_index] = self.rotate_full_side(self.state[side_index], rotation)

        #Define the rotation indices list
        side_indices = [front, left, back, right, top, bottom]

        #Rotate the side indices according to lists above
        self.rotate_leftovers(sides, side_indices)
            
        return True
    
    def printState(self):
        print(self.state[0])
        print(self.state[1])
        print(self.state[2])
        print(self.state[3])
        print(self.state[4])
        print(self.state[5])
        print()
    
    def randomizeState(self, num_moves):
        numMoves = int(num_moves)
        faces = ["front", "left", "back", "right", "top", "bottom"]
        valid_rotations_dict = {
            "front": ["left", "right"], 
            "left": ["towards", "away"], 
            "back": ["left", "right"], 
            "right": ["towards", "away"], 
            "top": ["left", "right"], 
            "bottom": ["left", "right"]
        }
        face_seed_val = 0
        rotation_seed_val = 0
        for i in range(numMoves):
            random.seed(a=face_seed_val)
            randFaceIndex = random.randrange(0, len(faces))
            randFace = faces[randFaceIndex]
            
            valid_rotations = valid_rotations_dict[randFace]
            random.seed(a=rotation_seed_val)
            randRotationIndex = random.randrange(0, len(valid_rotations))
            randRotation = valid_rotations[randRotationIndex]
            
            self.move(randFace, randRotation)
            face_seed_val += 6
            rotation_seed_val += 1
            
    def rotate_full_side(self, side, angle):
        if angle == "clockwise":
            new_side = side[1:]
            new_side.append(side[0])
        else:
            new_side = side[-1]
            new_side = [new_side] + side[:-1]

        return new_side
    
    def rotate_leftovers(self, sides, side_indices):
        state_copy = self.copy_state(self.state)
        for side in enumerate(side_indices):
            if side[1] is None:
                continue

            inherited_side = sides[side[0]]
            inherited_side_indices = side_indices[inherited_side]

            #print("Side: ", self.state[side[0]])
            #print("Inherited Side: ", inherited_side)

            self.state[side[0]][side[1][0]-1] = state_copy[inherited_side][inherited_side_indices[0]-1]
            self.state[side[0]][side[1][1]-1] = state_copy[inherited_side][inherited_side_indices[1]-1]
    
    def copy_state(self, state):
        copy = [["", "", "", ""], #Front
                ["", "", "", ""], #Left
                ["", "", "", ""], #Back
                ["", "", "", ""], #Right
                ["", "", "", ""], #Top
                ["", "", "", ""]] #Bottom
        
        for side in enumerate(state):
            for tile in enumerate(side[1]):
                copy[side[0]][tile[0]] = tile[1]
                
        return copy


#####################################################
######              Main Method                ######
#####################################################

if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except:
        print("Please enter the name of a file with program commands")
        
    PuzzleSolver = PuzzleSolve("node")
    PuzzleSolver.run_commands(filename)