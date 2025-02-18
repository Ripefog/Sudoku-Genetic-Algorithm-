import numpy as np
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass, field

GRID_SIZE = 9
BLOCK_SIZE = 3

@dataclass
class SudokuGrid:
    values: np.ndarray = field(default_factory=lambda: np.zeros((GRID_SIZE, GRID_SIZE), dtype=int))
    fitness: float = 0.0

    def update_fitness(self, initial_grid: np.ndarray):
        column_count = np.zeros(GRID_SIZE)
        row_count = np.zeros(GRID_SIZE)
        block_count = np.zeros(GRID_SIZE)
        row_sum = 0
        column_sum = 0
        block_sum = 0

        # Row check
        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE):
                row_count[self.values[j][i] - 1] += 1

            for k in range(len(row_count)):
                if row_count[k] == 1:
                    row_sum += (1/GRID_SIZE)/GRID_SIZE
            row_count = np.zeros(GRID_SIZE)
            
        # Column check
        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE):
                column_count[self.values[i][j] - 1] += 1

            for k in range(len(column_count)):
                if column_count[k] == 1:
                    column_sum += (1/GRID_SIZE)/GRID_SIZE
            column_count = np.zeros(GRID_SIZE)

        # Block check
        for i in range(0, GRID_SIZE, 3):
            for j in range(0, GRID_SIZE, 3):
                block = self.values[i:i+3, j:j+3].flatten()
                block_count = np.zeros(GRID_SIZE)
                for val in block:
                    block_count[val - 1] += 1

                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1/GRID_SIZE)/GRID_SIZE

        # Compute overall fitness
        if int(column_sum) == 1 and int(block_sum) == 1 and int(row_sum) == 1:
            self.fitness = 1.0
        else:
            self.fitness = column_sum * block_sum * row_sum

        return self.fitness

class EvolutionarySolver:
    def __init__(self, initial_grid: np.ndarray):
        self.initial_grid = initial_grid.astype(int)
        self.population_size = 700
        self.elite_size = int(0.05 * self.population_size)
        self.mutation_rate = 0.06
        self.max_generations = 3000
        
        # Precompute legal values for each empty cell
        self.legal_values = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.initial_grid[row][col] == 0:
                    for value in range(1, 10):
                        if (not self._is_row_duplicate(row, value) and 
                            not self._is_column_duplicate(col, value) and 
                            not self._is_block_duplicate(row, col, value)):
                            self.legal_values[row][col].append(value)
                else:
                    self.legal_values[row][col].append(self.initial_grid[row][col])

    def _is_row_duplicate(self, row, value):
        return value in self.initial_grid[row]

    def _is_column_duplicate(self, col, value):
        return value in self.initial_grid[:, col]

    def _is_block_duplicate(self, row, col, value):
        block_row, block_col = (row // 3) * 3, (col // 3) * 3
        return value in self.initial_grid[block_row:block_row+3, block_col:block_col+3]

    def seed_population(self) -> List[SudokuGrid]:
        population = []
        for _ in range(self.population_size):
            candidate = SudokuGrid(self.initial_grid.copy())
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if self.initial_grid[row][col] == 0:
                        candidate.values[row][col] = self.legal_values[row][col][random.randint(0, len(self.legal_values[row][col]) - 1)]
                
                # Ensure no row duplicates
                while len(set(candidate.values[row])) != GRID_SIZE:
                    for col in range(GRID_SIZE):
                        if self.initial_grid[row][col] == 0:
                            candidate.values[row][col] = self.legal_values[row][col][random.randint(0, len(self.legal_values[row][col]) - 1)]
            
            population.append(candidate)
        return population

    def tournament_selection(self, population):
        tournament_size = 2
        candidates = random.sample(population, tournament_size)
        return max(candidates, key=lambda x: x.fitness)

    def cycle_crossover(self, parent1, parent2):
        child1 = SudokuGrid(parent1.values.copy())
        child2 = SudokuGrid(parent2.values.copy())
        
        if random.random() < 1.0:  # Always crossover
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            
            if crossover_point1 > crossover_point2:
                crossover_point1, crossover_point2 = crossover_point2, crossover_point1
            
            for row in range(crossover_point1, crossover_point2):
                child1.values[row], child2.values[row] = self._crossover_row(child1.values[row], child2.values[row])
        
        return child1, child2

    def _crossover_row(self, row1, row2):
        remaining = list(range(1, 10))
        child_row1 = np.zeros(GRID_SIZE, dtype=int)
        child_row2 = np.zeros(GRID_SIZE, dtype=int)
        cycle = 0

        while 0 in child_row1 and 0 in child_row2:
            if cycle % 2 == 0:
                index = self._find_unused_index(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next_val = row2[index]
            else:
                index = self._find_unused_index(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next_val = row2[index]
            
            while next_val != start:
                index = self._find_index(row1, next_val)
                if cycle % 2 == 0:
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                else:
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                next_val = row2[index]
            
            cycle += 1
        
        return child_row1, child_row2

    def _find_unused_index(self, parent_row, remaining):
        return next(i for i in range(len(parent_row)) if parent_row[i] in remaining)

    def _find_index(self, parent_row, value):
        return next(i for i in range(len(parent_row)) if parent_row[i] == value)

    def _mutate(self, candidate, initial_grid):
        if random.random() > self.mutation_rate:
            return False

        success = False
        while not success:
            row = random.randint(0, 8)
            from_column, to_column = random.sample(range(9), 2)

            if initial_grid[row][from_column] == 0 and initial_grid[row][to_column] == 0:
                if (not self._is_column_duplicate(to_column, candidate.values[row][from_column]) and 
                    not self._is_column_duplicate(from_column, candidate.values[row][to_column]) and 
                    not self._is_block_duplicate(row, to_column, candidate.values[row][from_column]) and 
                    not self._is_block_duplicate(row, from_column, candidate.values[row][to_column])):
                    
                    candidate.values[row][from_column], candidate.values[row][to_column] = \
                        candidate.values[row][to_column], candidate.values[row][from_column]
                    success = True

        return success


    def solve(self):
        print("Creating an initial population.")
        population = self.seed_population()
        stale_generations = 0
        best_fitness = 0
        sigma = 1
        phi = 0
        mutation_rate = 0.06

        for generation in range(self.max_generations):
            # Update fitness for all candidates
            for candidate in population:
                candidate.update_fitness(self.initial_grid)

            # Check for solution
            solution = next((candidate for candidate in population if candidate.fitness == 1.0), None)
            if solution:
                print(f"Solution found at generation {generation}")
                return solution

            # Sort population
            population.sort(key=lambda x: x.fitness, reverse=True)
            current_best_fitness = population[0].fitness
            current_worst_fitness = population[-1].fitness  # Get worst fitness

            # Track best fitness
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                stale_generations = 0
            else:
                stale_generations += 1

            # Re-seed if population becomes stale
            if stale_generations >= 70:
                print("Population stale. Re-seeding...")
                population = self.seed_population()
                stale_generations = 0
                sigma = 1
                phi = 0
                mutation_rate = 0.06

            # Create next generation
            next_population = population[-self.elite_size:]  # Keep elite solutions

            while len(next_population) < self.population_size:
                # Tournament selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                # Crossover
                child1, child2 = self.cycle_crossover(parent1, parent2)

                # Mutation with adaptive rate
                success1 = self._mutate(child1, self.initial_grid)
                success2 = self._mutate(child2, self.initial_grid)

                if success1:
                    phi += 1
                if success2:
                    phi += 1

                next_population.extend([child1, child2])

            population = next_population[:self.population_size]

            # Adaptive mutation rate (Rechenberg's 1/5 success rule)
            if phi > 0.2:
                sigma /= 0.998
            elif phi < 0.2:
                sigma *= 0.998

            mutation_rate = abs(np.random.normal(loc=0.0, scale=sigma))

            print(f"Generation {generation}: Best Fitness = {best_fitness:.6f}, Worst Fitness = {current_worst_fitness:.6f}")

        return None

    
# def solve_sudoku(puzzle: List[List[int]]) -> Optional[List[List[int]]]:
#     grid = np.array(puzzle, dtype=int)
#     solver = EvolutionarySolver(grid)
#     solution = solver.solve()
    
#     return solution.values.tolist() if solution else None