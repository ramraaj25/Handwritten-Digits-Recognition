import numpy as np
import pygame
import tensorflow as tf


# GLOBAL VARIABLES

WIN_WIDTH = 600
WIDTH = 420
ROWS = 28

WHITE = (255, 255, 255)
BLACK = (1, 1, 1)
DARK_GRAY = (169, 169, 169)
GRAY = (211, 211, 211)
WIN = pygame.display.set_mode((WIN_WIDTH, WIDTH))
WIN.fill(WHITE)
pygame.display.init()
pygame.init()
pygame.display.set_caption('Handwritten Digits Recognition')

# Class for a single square of the grid


class Dot():
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.width = WIDTH / ROWS
        self.x = self.width * self.col
        self.y = self.width * self.row
        self.color = WHITE

    def draw(self):
        pygame.draw.rect(
            WIN, self.color, (self.x, self.y, self.width, self.width))


def make_grid(total_rows):
    grid = [[None for i in range(total_rows)] for j in range(total_rows)]
    for i in range(total_rows):
        for j in range(total_rows):
            grid[i][j] = Dot(i, j)
    return grid


def draw_grid(grid):
    for i in range(ROWS):
        for j in range(ROWS):
            grid[i][j].draw()

    draw_grid_lines()


def draw_grid_lines():
    gap = WIDTH / ROWS
    # Vertical lines
    for i in range(ROWS + 1):
        pygame.draw.line(WIN, BLACK, (gap * i, 0), (gap * i, WIDTH))

    # Horizontal lines
    for i in range(ROWS + 1):
        pygame.draw.line(WIN, BLACK, (0, gap * i), (WIDTH, gap * i))


def create_img_array(grid):
    img_array = [[0 for i in range(ROWS)] for j in range(ROWS)]
    for i in range(ROWS):
        for j in range(ROWS):
            if grid[i][j].color == WHITE:
                img_array[i][j] = 1
            elif grid[i][j].color == GRAY:
                img_array[i][j] = 211
            elif grid[i][j].color == DARK_GRAY:
                img_array[i][j] = 169
            else:
                img_array[i][j] = 255

    return img_array

# Load and predict using model


def predict(grid):
    model = tf.keras.models.load_model('model1')
    img_array = create_img_array(grid)
    print(model.predict([np.array(img_array).reshape(1, 28, 28, 1)]).argmax())
    return model.predict([np.array(img_array).reshape(1, 28, 28, 1)]).argmax()

# Function to draw label and to display predicted value


def draw_prediction(prediction, font_label, font_predicted_text):
    text = font_label.render('Prediction:', True, WHITE)
    predicted_text = font_predicted_text.render(prediction, True, WHITE)
    predicted_text_rect = predicted_text.get_rect()
    predicted_text_rect.center = (505, 85)
    WIN.blit(text, (430, 30))
    WIN.blit(predicted_text, predicted_text_rect)


def draw_buttons(font_label):
    # Outer rectangle for Predict
    pygame.draw.rect(WIN, WHITE, (450, 250, 120, 50), 1)
    # Inner rect
    pygame.draw.rect(WIN, WHITE, (455, 255, 110, 40))
    text_predict = font_label.render('Predict', True, BLACK)
    WIN.blit(text_predict, (467, 257))

    # Outer rectangle for Clear
    pygame.draw.rect(WIN, WHITE, (450, 310, 120, 50), 1)
    # Inner rect
    pygame.draw.rect(WIN, WHITE, (455, 315, 110, 40))
    text_clear = font_label.render('Clear', True, BLACK)
    WIN.blit(text_clear, (477, 317))


def main():
    grid = make_grid(ROWS)
    gap = WIDTH / ROWS
    running = True
    font_label = pygame.font.SysFont('comicsansms', 32)
    font_predicted_text = pygame.font.SysFont('comicsansms', 22)
    font_buttons = pygame.font.SysFont('comicsansms', 25)
    prediction = 'None'
    clock = pygame.time.Clock()
    while running:
        clock.tick(60)
        WIN.fill(BLACK)
        draw_grid(grid)
        draw_buttons(font_buttons)
        draw_prediction(prediction, font_label, font_predicted_text)
        pygame.display.update()
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
                break

            mouse = pygame.mouse.get_pressed()

            if mouse[0]:
                x, y = pygame.mouse.get_pos()
                row = int(y // gap)
                col = int(x // gap)
                if 0 <= row < ROWS - 1 and 0 <= col < ROWS - 1:
                    grid[row][col].color = BLACK

                    if row < ROWS - 1 and grid[row + 1][col].color != BLACK:
                        grid[row + 1][col].color = DARK_GRAY
                    if col < ROWS - 1 and grid[row][col + 1].color != BLACK:
                        grid[row][col + 1].color = DARK_GRAY
                    if row < ROWS - 1 and col < ROWS - 1 and grid[row + 1][col + 1].color != BLACK:
                        grid[row + 1][col + 1].color = GRAY

                elif 455 <= x <= 565 and 255 <= y <= 295:
                    prediction = str(predict(grid))
                elif 455 <= x <= 565 and 315 <= y <= 355:
                    for i in range(ROWS):
                        for j in range(ROWS):
                            grid[i][j].color = WHITE
                            prediction = 'None'
            elif mouse[2]:
                x, y = pygame.mouse.get_pos()
                row = int(y // gap)
                col = int(x // gap)
                grid[row][col].color = WHITE

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    for i in range(ROWS):
                        for j in range(ROWS):
                            grid[i][j].color = WHITE
                    prediction = 'None'
                if event.key == pygame.K_p:
                    prediction = str(predict(grid))

    pygame.quit()


if __name__ == "__main__":
    main()
