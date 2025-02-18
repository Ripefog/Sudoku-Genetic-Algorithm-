# **Genetic Algorithm for Solving Sudoku**

## 🌟 **Introduction**

This project implements a **Genetic Algorithm (GA)** to solve Sudoku puzzles. GA is an optimization technique inspired by natural selection, involving **selection, crossover, and mutation** operations to evolve a population of potential solutions toward the correct answer.

---

## 🚀 **Features**

✅ Uses **Cycle Crossover** to generate offspring while maintaining row constraints.  
✅ Implements **Tournament Selection** for choosing the fittest individuals.  
✅ Includes **Mutation** for diversity, swapping values within rows under constraints.  
✅ Employs an **adaptive mutation rate** using **Rechenberg's 1/5 Success Rule**.  
✅ Handles **9×9 Sudoku puzzles** efficiently with a structured population approach.  

---

## 📌 **Requirements**

To run this project, install the necessary dependencies:

```bash
pip install numpy
```

---

## ⚙️ **How It Works**

### 🧬 **1. Chromosome Representation**
- Each individual in the population represents a **9×9 Sudoku grid**.
- Fixed values from the given puzzle remain unchanged.
- Empty cells are filled with valid numbers, ensuring each row contains `{1,2,...,9}`.

### 🎯 **2. Fitness Function**
The fitness function evaluates a Sudoku grid by counting the number of conflicts in:
- **Rows** (already ensured to be valid)
- **Columns** (conflicts penalized)
- **3×3 Subgrids** (conflicts penalized)
- The ideal fitness score is **1.0** (fully valid solution).

### 🔄 **3. Genetic Algorithm Steps**
1️⃣ **Initialize Population**: Generate **700 Sudoku grids** with randomized but valid row assignments.  
2️⃣ **Evaluate Fitness**: Calculate conflicts in columns and subgrids.  
3️⃣ **Selection**: Use **tournament selection** to pick parents for reproduction.  
4️⃣ **Crossover**: Apply **Cycle Crossover** to swap row segments between parents.  
5️⃣ **Mutation**: Randomly swap values in a row to introduce variation.  
6️⃣ **Elitism**: Retain the **top 5%** of best solutions each generation.  
7️⃣ **Adaptive Mutation**: Adjust the mutation rate dynamically.  
8️⃣ **Repeat** Until Solution Found or **Max Generations Reached**.  

---

## 🏁 **Running the Solver**

To solve a Sudoku puzzle, use the following script:

```python
import numpy as np
from SudokuGA import EvolutionarySolver

# Define a Sudoku puzzle (0 represents empty cells)
puzzle = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

solver = EvolutionarySolver(puzzle)
solution = solver.solve()

if solution:
    print("Sudoku Solved:")
    print(solution.values)
else:
    print("Solution not found within generation limits.")
```

---

## ⚡ **Performance Considerations**

- ✅ **Works well** for moderate Sudoku puzzles with **30+ given numbers**.  
- ❌ **May struggle** with extremely sparse puzzles due to GA randomness.  
- 🔄 **Adaptive mutation** helps prevent stagnation, but tuning parameters is key.  

---

## 🔮 **Future Improvements**

📌 Implement **Hybrid GA + Constraint Propagation** for faster convergence.  
📌 Optimize **column and subgrid validation** for performance gains.  
📌 Extend support to **larger Sudoku variants (16×16, 25×25)**.  

---

## 🎓 **Conclusion**

This **GA-based Sudoku solver** demonstrates how **evolutionary algorithms** can tackle constraint satisfaction problems. While not always as efficient as **backtracking**, it provides an alternative approach that scales well and can be improved further with hybrid techniques.

---

📌 Developed using **Python & NumPy**.

