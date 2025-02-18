import sys
import os
import random
import time
import json
import copy
import numpy as np
from tkinter import *
from tkinter.ttk import *
from tkinter import simpledialog
from tkinter import messagebox
from SudokuGA import EvolutionarySolver

class SudokuGUI(Frame):
    def __init__(self, master, file):
        Frame.__init__(self, master)
        if master:
            master.title("Sudoku Solver")
            
        self.grid = np.zeros((9, 9), dtype=int)
        self.grid_init = np.zeros((9, 9), dtype=int)
        self.solution = np.zeros((9, 9), dtype=int)  # Để lưu kết quả từ solver
        self.locked = []
        self.easy, self.medium, self.hard, self.expert = [], [], [], []
        self.load_db(file)
        self.make_grid()
        self.bframe = Frame(self)

        # select game difficult level
        self.lvVar = StringVar()
        self.lvVar.set("")
        difficult_level = ["Easy", "Medium", "Hard", "Expert"]
        Label(self.bframe, text="Difficulty level:", font="Times 18 underline").pack(anchor=S)
        for l in difficult_level:
            Radiobutton(self.bframe, text=l, width=20, variable=self.lvVar, value=l)\
                .pack(anchor=S)
        
        # generate new game
        self.ng = Button(self.bframe, text='Generate New Game', width=20, command=self.new_game)\
            .pack(anchor=S)
        # solver
        self.sg = Button(self.bframe, text='Solve by EA', width=20, command=self.solver).pack(anchor=S)
        # auto fill
        self.auto = Button(self.bframe, text='Auto Fill', width=20,
                           command=self.auto_fill).pack(anchor=S)
        # check win
        self.cw = Button(self.bframe, text='Check Win', width=20,
                         command=self.check_win).pack(anchor=S)
        
        self.bframe.pack(side='bottom', fill='x', expand='1')
        self.pack()
        
    def auto_fill(self):
        if np.array_equal(self.solution, np.zeros((9, 9))):
            return
        self.grid = self.solution.copy()
        self.sync_board_and_canvas()
        
    def rgb(self, red, green, blue):
        return "#%02x%02x%02x" % (red, green, blue)

    def load_db(self, file):
        with open(file) as f:
            data = json.load(f)
        self.easy = data['Easy']
        self.medium = data['Medium']
        self.hard = data['Hard']
        self.expert = data['Expert']

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
            
        self.grid = np.array(list(self.given)).reshape((9, 9)).astype(int)
        self.grid_init = self.grid.copy()
        self.solution = np.zeros((9, 9), dtype=int)
        self.locked = [(y, x) for y in range(9)
                       for x in range(9) if self.grid[y][x] != 0]
        self.sync_board_and_canvas()
        
    def solver(self):
        if np.array_equal(self.grid_init, np.zeros((9, 9))):
            print("Invalid inputs")
            return
            
        start_time = time.time()
        
        solver = EvolutionarySolver(self.grid_init)
        solution = solver.solve()
        
        if solution is not None:
            self.solution = solution.values
            time_elapsed = '{0:6.2f}'.format(time.time()-start_time)
            str_print = "Solution found!\n" + "Time elapsed: " + str(time_elapsed) + "s"
            messagebox.showinfo("Success", str_print + "\nClick 'Auto Fill' to see the solution")
        else:
            str_print = "No solution found, please try again"
            messagebox.showerror("Error", str_print)

    def make_grid(self):
        w, h = 400, 400  # Increased size for better visibility
        c = Canvas(self, bg='white', width=w, height=h)
        c.pack(side='top', fill='both', expand='1')

        self.rects = [[None for x in range(9)] for y in range(9)]
        self.handles = [[None for x in range(9)] for y in range(9)]
        rsize = w / 9
        
        # Create base grid with uniform borders
        for y in range(9):
            for x in range(9):
                (xr, yr) = (x * rsize, y * rsize)
                r = c.create_rectangle(xr, yr, xr + rsize, yr + rsize,
                                    width=1, outline='black')
                t = c.create_text(xr + rsize/2, yr + rsize/2,
                                font=("Arial", int(rsize/2)))
                self.handles[y][x] = (r, t)
        
        # Draw thicker lines for 3x3 blocks
        for i in range(0, 10, 3):
            # Vertical lines
            x = i * rsize
            c.create_line(x, 0, x, h, width=2, fill='black')
            # Horizontal lines
            y = i * rsize
            c.create_line(0, y, w, y, width=2, fill='black')
        
        # Add outer border
        c.create_rectangle(0, 0, w, h, width=2, outline='black')
                
        self.canvas = c
        self.sync_board_and_canvas()
        self.canvas.bind("<Button-1>", self.handle_click)

    def handle_click(self, event):
        x, y = event.x, event.y
        w = self.canvas.winfo_width()
        rsize = w / 9
        grid_x, grid_y = int(x // rsize), int(y // rsize)
        if (grid_y, grid_x) not in self.locked:
            self.enter_value(grid_y, grid_x)
            
    def enter_value(self, row, col):
        value = simpledialog.askinteger("Enter Value", "Enter a number (1-9):", 
                                      minvalue=1, maxvalue=9)
        if value:
            self.grid[row][col] = value
            self.sync_board_and_canvas()
            # Optional: Auto-check if the move is correct
            if not np.array_equal(self.solution, np.zeros((9, 9))):
                if self.grid[row][col] != self.solution[row][col]:
                    self.canvas.itemconfig(self.handles[row][col][1], 
                                         fill="red")
                else:
                    self.canvas.itemconfig(self.handles[row][col][1], 
                                         fill="blue")

    def check_win(self):
        if np.array_equal(self.grid_init, np.zeros((9, 9))):
           messagebox.showinfo("Warning", "Please, create the game first.")
           return True
        if np.array_equal(self.solution, np.zeros((9, 9))):
           messagebox.showinfo("Suggest","Please, check the solver's result first.")
           return True
        if np.array_equal(self.grid, self.solution):
            messagebox.showinfo("Congratulations!",
                                "You've completed the puzzle correctly!")
            return True
        else:
            messagebox.showwarning(
                "Not Yet!", "The puzzle is not solved correctly yet.")
            return False

    def sync_board_and_canvas(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y][x] != 0:
                    # Set different colors for initial numbers vs. user input
                    if (y, x) in self.locked:
                        self.canvas.itemconfig(self.handles[y][x][1],
                                            text=str(self.grid[y][x]),
                                            fill="black")
                    else:
                        self.canvas.itemconfig(self.handles[y][x][1],
                                            text=str(self.grid[y][x]),
                                            fill="blue")
                else:
                    self.canvas.itemconfig(self.handles[y][x][1],
                                           text='')

if __name__ == "__main__":
    file = "sudoku_db.json"
    tk = Tk()
    gui = SudokuGUI(tk, file)
    gui.mainloop()