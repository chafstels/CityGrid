import random
from collections import deque
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CityGrid:
    def __init__(self, rows, cols, obstruction_probability=0.3):
        # Инициализация параметров сетки и вероятности препятствий
        self.rows = rows
        self.cols = cols
        self.obstruction_probability = obstruction_probability
        # Инициализация сетки с препятствиями и списком башен
        self.grid = self._generate_grid()
        self.towers = []

    def _generate_grid(self):
        # Генерация сетки с препятствиями на основе заданной вероятности
        return [[1 if random.random() < self.obstruction_probability else 0 for _ in range(self.cols)] for _ in range(self.rows)]

    def display_grid(self):
        # Вывод сетки в консоль
        for row in self.grid:
            print(" ".join(map(str, row)))

    def place_tower(self, row, col, tower_range):
        # Попытка установить башню в заданных координатах
        if self.grid[row][col] == 0:
            self.towers.append((row, col, tower_range))
            self._update_coverage()

    def _update_coverage(self):
        # Обновление информации о покрытии после установки новой башни
        coverage_grid = self._initialize_coverage_grid()

        for tower in self.towers:
            self._mark_coverage(coverage_grid, *tower)

        self.display_with_towers(coverage_grid)

    def _initialize_coverage_grid(self):
        # Инициализация пустой сетки покрытия
        return [[0] * self.cols for _ in range(self.rows)]

    def _mark_coverage(self, coverage_grid, row, col, tower_range):
        # Обозначение зоны покрытия для заданной башни
        for i in range(max(0, row - tower_range), min(self.rows, row + tower_range + 1)):
            for j in range(max(0, col - tower_range), min(self.cols, col + tower_range + 1)):
                coverage_grid[i][j] = 1

    def display_with_towers(self, coverage_grid):
        # Визуализация сетки с башнями и их зонами покрытия
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)

        self._draw_obstructions(ax)
        self._draw_towers_and_coverage(ax)

        plt.show()
        time.sleep(2)

    def _draw_obstructions(self, ax):
        # Отрисовка препятствий
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 1:
                    rect = patches.Rectangle(
                        (j, i),
                        1,
                        1,
                        linewidth=1,
                        edgecolor='k',
                        facecolor='k'
                    )
                    ax.add_patch(rect)

    def _draw_towers_and_coverage(self, ax):
        # Отрисовка башен и их зон покрытия
        for tower in self.towers:
            self._draw_tower(ax, *tower)
            self._draw_tower_center(ax, *tower)

    def _draw_tower_center(self, ax, row, col, tower_range):
        # Отрисовка центра башни
        center_rect = patches.Rectangle(
            (col, row),
            1,
            1,
            linewidth=1,
            edgecolor='none',
            facecolor='cyan'  # Цвет башни
        )
        ax.add_patch(center_rect)

    def _draw_tower(self, ax, row, col, tower_range):
        # Отрисовка зоны покрытия башни
        coverage_rect = patches.Rectangle(
            (col - tower_range, row - tower_range),
            2 * tower_range + 1,
            2 * tower_range + 1,
            linewidth=1,
            edgecolor='b',
            facecolor='b',
            alpha=0.3  # Прозрачность для видимости покрытия
        )
        ax.add_patch(coverage_rect)

    def optimize_tower_placement(self):
        # Оптимизация размещения башен для покрытия всех непокрытых блоков
        while not self.all_blocks_covered():
            row, col = self.get_max_uncovered_neighbors()
            self.place_tower(row, col, 1)

    def all_blocks_covered(self):
        # Проверка, покрыты ли все непрегражденные блоки
        return all(self.is_covered(i, j) or self.grid[i][j] == 1 for i in range(self.rows) for j in range(self.cols))

    def is_covered(self, row, col):
        # Проверка, покрыта ли заданная ячейка какой-либо башней
        return any(
            abs(tower_row - row) <= tower_range and abs(tower_col - col) <= tower_range
            for tower_row, tower_col, tower_range in self.towers
        )

    def get_max_uncovered_neighbors(self):
        # Получение координат непокрытой ячейки с наибольшим количеством непокрытых соседей
        max_uncovered_position = max(
            ((i, j) for i in range(self.rows) for j in range(self.cols) if self.grid[i][j] == 0 and not self.is_covered(i, j)),
            key=lambda pos: self.count_uncovered_neighbors(*pos)
        )
        return max_uncovered_position

    def count_uncovered_neighbors(self, row, col):
        # Подсчет количества непокрытых соседей для заданной ячейки
        return sum(
            1 for i in range(max(0, row - 1), min(self.rows, row + 2))
            for j in range(max(0, col - 1), min(self.cols, col + 2))
            if self.grid[i][j] == 0 and not self.is_covered(i, j)
        )


class CityWithReliability(CityGrid):
    def find_most_reliable_path(self, start_tower, end_tower):
        # Поиск наиболее надежного пути между двумя башнями
        start_node, end_node = start_tower[:2], end_tower[:2]

        if self.grid[start_node[0]][start_node[1]] == 1 or self.grid[end_node[0]][end_node[1]] == 1:
            # Проверка, возможен ли путь между заданными башнями
            print("Cannot find a reliable path between obstructed towers.")
            return []

        queue = deque([(start_node, 0)])
        visited = set([start_node])
        path_weights = {start_node: 0}
        predecessor = {start_node: None}

        while queue:
            current_node, current_weight = queue.popleft()

            for neighbor in self.get_neighbors(*current_node):
                if neighbor not in visited:
                    queue.append((neighbor, current_weight + 1))
                    visited.add(neighbor)
                    path_weights[neighbor] = current_weight + 1
                    predecessor[neighbor] = current_node

        try:
            return self.construct_path(predecessor, start_node, end_node)
        except KeyError:
            # Обработка случая, когда невозможно построить путь между башнями
            print("Cannot find a reliable path between the towers.")
            return []

    def get_neighbors(self, row, col):
        # Получение соседей для заданной ячейки
        return [
            (i, j) for i, j in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
            if 0 <= i < self.rows and 0 <= j < self.cols and self.grid[i][j] == 0
        ]

    def construct_path(self, predecessor, start_node, end_node):
        # Построение пути на основе предшественников
        path = []
        current_node = end_node

        while current_node is not None:
            path.append(current_node)
            current_node = predecessor[current_node]

        return list(reversed(path))

    def visualize_reliable_path(self, reliable_path):
        # Визуализация наиболее надежного пути
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)

        self._draw_obstructions(ax)
        self._draw_towers_and_coverage(ax)

        for i in range(len(reliable_path) - 1):
            node1, node2 = reliable_path[i], reliable_path[i + 1]
            self._draw_reliable_path(ax, node1, node2)

        plt.show()

    def _draw_reliable_path(self, ax, node1, node2):
        # Отрисовка надежного пути›
        row1, col1 = node1
        row2, col2 = node2
        ax.plot([col1 + 0.5, col2 + 0.5], [row1 + 0.5, row2 + 0.5], linestyle='solid', color='r')


# Пример использования
city_with_reliability = CityWithReliability(5, 5)  # Создание объекта CityWithReliability с сеткой 5x5
city_with_reliability.display_grid()  # Вывод начальной сетки в консоль

# Оптимизация для покрытия всех незатрудненных блоков
city_with_reliability.optimize_tower_placement()

# Выбор двух башен для поиска пути
start_tower, end_tower = city_with_reliability.towers[:2]

# Поиск наиболее надежного пути между башнями
reliable_path = city_with_reliability.find_most_reliable_path(start_tower, end_tower)

# Визуализация надежного пути
city_with_reliability.visualize_reliable_path(reliable_path)