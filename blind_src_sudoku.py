import tkinter as tk
from tkinter import messagebox
from tkinter import *
import copy
import time
import tracemalloc
import json
import numpy as np
import random
import threading


class SudokuSolverUI:
    def __init__(self, root, file):
        self.root = root
        self.root.title("Sudoku Solver")
        self.grid_values = None


        # Stop search variable
        self.stop_search = False


        # Create frames to separate grid and buttons
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.grid(row=0, column=0, padx=10, pady=10)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        
        # matrix to use
        self.easy, self.medium, self.hard, self.expert = [], [], [], []
        self.load_db(file)

        # For chose level
        self.lvVar = tk.StringVar()
        self.lvVar.set("Easy")

        # For show step
        self.showVar = tk.StringVar()
        self.showVar.set("NotShow")



        self.difficult_level = ["Easy", "Medium", "Hard", "Expert"]
        self.new_game()
    
        self.grid = [[tk.StringVar(value=str(self.grid_values[row][col]) if self.grid_values[row][col] != 0 else "") for col in range(9)] for row in range(9)]
        self.create_grid()
        self.create_buttons()

    def new_game(self):
        level = self.lvVar.get()
        if level == "Easy":
            self.given = self.easy[random.randint(0,len(self.easy)-1)]
        elif level == "Medium":
            self.given = self.medium[random.randint(0, len(self.medium)-1)]
        elif level == "Hard":
            self.given = self.hard[random.randint(0, len(self.hard)-1)]
        elif level == "Expert":
            self.given = self.expert[random.randint(0, len(self.expert)-1)]
        else:
            self.given = np.zeros((9, 9), dtype=int)
            
        self.grid_values = np.array(list(self.given)).reshape((9, 9)).astype(int)
        # self.grid_init = self.grid.copy()
        # self.solution = np.zeros((9, 9), dtype=int)




    def load_db(self, file):
        with open(file) as f:
            data = json.load(f)
        self.easy = data['Easy']
        self.medium = data['Medium']
        self.hard = data['Hard']
        self.expert = data['Expert']
   


    def create_grid(self):
        """Create a 9x9 Sudoku grid inside `grid_frame`."""
        for row in range(9):
            for col in range(9):
                entry = tk.Entry(self.grid_frame, width=3, font=('Arial', 18),
                                 textvariable=self.grid[row][col], justify='center')
                entry.grid(row=row, column=col, padx=5, pady=5)

    def create_buttons(self):
        """Create control buttons inside `button_frame`."""
        solve_button = tk.Button(self.button_frame, text="Solve Directly", command=self.solve, width=15)
        solve_button.pack(pady=5)

        solvestep_button = tk.Button(self.button_frame, text="Solve with Step", command=self.solveshowstep, width=15)
        solvestep_button.pack(pady=5)

        reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset, width=15)
        reset_button.pack(pady=5)


        # Create a label with text
        label = tk.Label(self.button_frame, text="Chose level", font=("Arial", 30))
        label.pack(pady=40)

        
        for l in self.difficult_level:
            radio_button = tk.Radiobutton(self.button_frame, text=l, variable=self.lvVar, value=l, command=self.Level)
            radio_button.pack(anchor="w")


    
    def Level(self):
        self.new_game()
        self.display_solution(self.grid_values)

    def reset(self):
        self.stop_search = True
        self.display_solution(self.grid_values)

    def solveshowstep(self):
        self.showVar.set("Show")
        self.solve()


    def display_solution(self, solution):
        for row in range(9):
            for col in range(9):
                self.grid[row][col].set(solution[row][col])

    def DFS(self,problem):
        start = Node(problem.initial)
        if problem.check_legal(start.state):
            return start.state
        stack = [start]
        x=1
        show = self.showVar.get()
        while stack:
            if self.stop_search:
                return None

            node = stack.pop()
            if problem.check_legal(node.state):
                return node.state
            
            print(f"Lần thử thứ {x}:")
            x+=1
            for row in node.state:
                print(row)


            if show == "Show":
                self.display_solution(node.state)  # Update the UI with the current node
                self.root.update()  # Force UI refresh after updating the grid
                time.sleep(0.2)  # Optional: Slow down the visualization


            stack.extend(node.expand(problem))
        return None  

        

    def solve(self):
        self.stop_search = False
        print("\nSolving with DFS...")

        def run_solver():
            start_time = time.time()

            problem = Problem(self.grid_values)
            solution = self.DFS(problem)

            self.showVar.set("NotShow")

            if self.stop_search:
                return

            if solution.any():
                self.display_solution(solution)
                self.root.update()  # Force UI refresh after updating the grid
                eplased_time = time.time() - start_time
                messagebox.showinfo("Time", "Elapsed Time: " + str(eplased_time) + " seconds" ) 

            else:
                messagebox.showerror("Error", "No solution found")
        # Run DFS in a new thread
        threading.Thread(target=run_solver, daemon=True).start()

        

        



class Problem(object):

    def __init__(self, initial):
        self.initial = initial
        self.size = len(initial) # Size of grid
        self.height = int(self.size/3) # Size of a quadrant

    def check_legal(self, state):
        # Expected sum of each row, column or quadrant.
        total = sum(range(1, self.size+1))

        # Check rows and columns and return false if total is invalid
        for row in range(self.size):
            if (len(state[row]) != self.size) or (sum(state[row]) != total):
                return False

            column_total = 0
            for column in range(self.size):
                column_total += state[column][row]

            if (column_total != total):
                return False

        # Check quadrants and return false if total is invalid
        for column in range(0,self.size,3):
            for row in range(0,self.size,self.height):

                block_total = 0
                for block_row in range(0,self.height):
                    for block_column in range(0,3):
                        block_total += state[row + block_row][column + block_column]

                if (block_total != total):
                    return False

        return True

    # Return set of valid numbers from values that do not appear in used
    def filter_values(self, values, used):
        return [number for number in values if number not in used]

    # Return first empty spot on grid (marked with 0)
    def get_spot(self, board, state):
        for row in range(board):
            for column in range(board):
                if state[row][column] == 0:
                    return row, column

    # Filter valid values based on row
    def filter_row(self, state, row):
        number_set = range(1, self.size+1) # Defines set of valid numbers that can be placed on board
        in_row = [number for number in state[row] if (number != 0)]
        options = self.filter_values(number_set, in_row)
        return options

    # Filter valid values based on column
    def filter_col(self, options, state, column):
        in_column = []
        for column_index in range(self.size):
            if state[column_index][column] != 0:
                in_column.append(state[column_index][column])
        options = self.filter_values(options, in_column)
        return options

    # Filter valid values based on quadrant
    def filter_quad(self, options, state, row, column):
        in_block = [] # List of valid values in spot's quadrant
        row_start = int(row/self.height)*self.height
        column_start = int(column/3)*3
        
        for block_row in range(0, self.height):
            for block_column in range(0,3):
                in_block.append(state[row_start + block_row][column_start + block_column])
        options = self.filter_values(options, in_block)
        return options  


    def actions(self, state):
        row,column = self.get_spot(self.size, state) # Get first empty spot on board

        # Remove a square's invalid values
        options = self.filter_row(state, row)
        options = self.filter_col(options, state, column)
        options = self.filter_quad(options, state, row, column)

        # Return a state for each valid option (yields multiple states)
        for number in options:
            new_state = copy.deepcopy(state) # Norvig used only shallow copy to copy states; deepcopy works like a perfect clone of the original
            new_state[row][column] = number
            yield new_state


class Node:
    def __init__(self, state):
        self.state = state

    def expand(self, problem):
        return [Node(state) for state in problem.actions(self.state)]



if __name__ == "__main__":
    tracemalloc.start()
    file = "sudoku_db.json"
    root = tk.Tk()
    app = SudokuSolverUI(root, file)
    root.mainloop()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Dung lượng hiện tại: {current / 1024:.2f} KB")
    print(f"Dung lượng cao nhất: {peak / 1024:.2f} KB")

    tracemalloc.stop()
