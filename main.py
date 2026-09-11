import pygame
import numpy as np
from keras.datasets import mnist  

pygame.init()

WIDTH = 900
HEIGHT = 500
FPS = 60
SURFACE_SIZE = (700,200)

WHITE = (255, 255, 255)
GRAY = (230, 230, 230)
DARK_BLUE = (20, 30, 50)

LEARNING_RATE = 0.001

classes = ["0", "1", "2", "3", "4","5", "6", "7", "8", "9","+", "-", "*", "/"]

data = np.load("operator_dataset.npz")
operator_images = data["images"]
operator_labels = data["labels"]
operator_images = operator_images.reshape(operator_images.shape[0],784)

surface = pygame.Surface(SURFACE_SIZE)

board_matrix = np.zeros((200, 700))

(input_train, target_train), (input_test, target_test) = mnist.load_data()

input_train = input_train.reshape(input_train.shape[0], 784) / 255.0
input_test = input_test.reshape(input_test.shape[0], 784) / 255.0

input_train = np.concatenate((input_train, operator_images),axis=0)

target_train = np.concatenate((target_train, operator_labels),axis=0)

# Input = 784 Neurons
# 128 Hidden Neurons
# 14 Output Neurons 

def ReLu(num):
    return np.maximum(0,num)

def ReLu_derivative(x):
    return x > 0

def one_hot(anwser_index):
    one_hot_vector = np.zeros((14, 1))
    one_hot_vector[anwser_index] = 1
    return one_hot_vector

def loss(predictions,one_hot_vector):
    loss_per_neuron = (predictions - one_hot_vector) ** 2
    return np.sum(loss_per_neuron)

class NeuralNetwork:
    def __init__(self,input_size=784,hidden_size=128,output_size=14):
        self.Weights_hidden = np.random.randn(hidden_size,input_size) * 0.01
        self.Bias_hidden = np.zeros((hidden_size,1))

        self.Weights_output = np.random.randn(output_size,hidden_size) * 0.01
        self.Bias_output = np.zeros((output_size,1))

    def forward_prop(self,Input):
        Weighted_Sum1 = self.Weights_hidden.dot(Input) + self.Bias_hidden
        Anwser_1 = ReLu(Weighted_Sum1)
        Weighted_Sum2 = self.Weights_output.dot(Anwser_1) + self.Bias_output

        return Weighted_Sum1, Anwser_1, Weighted_Sum2

    def back_prop(self, anwser_index, Weighted_Sum1, Anwser_1, Weighted_Sum2, Input):
        one_hot_vector = one_hot(anwser_index)

        Output_Error = 2 * (Weighted_Sum2 - one_hot_vector)

        weights_output_gradient = np.dot(Output_Error,Anwser_1.T)
        Bias_output_gradient = Output_Error

        Hidden_Error = np.dot(self.Weights_output.T,Output_Error) 
        Hidden_Error *= ReLu_derivative(Weighted_Sum1)

        weights_hidden_gradient = np.dot(Hidden_Error,Input.T)
        Bias_hidden_gradient = Hidden_Error

        self.Weights_output -= LEARNING_RATE * weights_output_gradient 
        self.Bias_output -= LEARNING_RATE * Bias_output_gradient 

        self.Weights_hidden -= LEARNING_RATE * weights_hidden_gradient 
        self.Bias_hidden -= LEARNING_RATE * Bias_hidden_gradient 

    def train(self,epochs=3 ):
        training_amount = len(input_train)

        for epoch in range(epochs):
            total_loss = 0

            indices = np.random.permutation(training_amount)
            for i in indices:

                Input = input_train[i].reshape(784, 1)
                anwser_index = target_train[i]

                Weighted_Sum1, Answer_1, Weighted_Sum2 = self.forward_prop(Input)
                total_loss += loss(Weighted_Sum2,one_hot(anwser_index))

                self.back_prop(anwser_index,Weighted_Sum1, Answer_1, Weighted_Sum2,Input)
            print(total_loss / training_amount)

    def test(self, amount=1000):
        correct = 0
        for i in range(amount):
            Input = input_test[i].reshape(784, 1)
            _, _, output = self.forward_prop(Input)
            prediction = np.argmax(output)
            if prediction == target_test[i]:
                correct += 1
        accuracy = correct / amount * 100
        print("Accuracy:", accuracy, "%")

nn = NeuralNetwork()

nn.test()
nn.train()
nn.test()

class Parcer:
    def __init__(self,tokens):
        self.tokens = tokens
        self.position = 0

    def number(self):
        value = float(self.tokens[self.position])
        self.position += 1
        return value
    
    def term(self):
        value = self.number()
        while self.position < len(self.tokens):
            operator = self.tokens[self.position]
            if operator not in ["*","/"]: break
            self.position += 1
            next_number = self.number()
            if operator == '*': value *= next_number
            else: value /= next_number  
        return value
    
    def expression(self):
        value = self.term()
        while self.position < len(self.tokens):
            operator = self.tokens[self.position]
            if operator not in ["+","-"]: break
            self.position += 1
            next_term = self.term()
            if operator == '+': value += next_term
            else: value -= next_term  
        return value

def tokeniser(expression):
    temp = ''
    equation = []
    for i in range(len(expression)):
        if expression[i].isdigit():
            temp += expression[i]
        else:
            if temp != '':
                equation.append(temp)

            equation.append(expression[i])

            temp = ''
    if temp != '': equation.append(temp)
    return equation

def draw(pos,board_matrix):
    x_center, y_center = pos
    x_center -= 100
    y_center -= 50

    radius = 6

    if 0 < x_center < 700 - radius - 1 and 0 < y_center < 200:
        pygame.draw.circle(surface,(0,0,0),(x_center, y_center),radius)
        for y in range(y_center - radius,y_center + radius + 1):
            for x in range(x_center - radius,x_center + radius + 1):
                if 0 < x < 700 and 0 < y < 200:
                    board_matrix[y][x] = 1

def find_symbols_x(matrix):
    used_columns = np.any(matrix==1,axis=0)
    symbols = []

    start = None

    for x in range(700):
        if used_columns[x] and start == None:
            start = x
        elif not used_columns[x] and start != None:
            symbols.append((start,x))
            start = None
    if start != None:
        symbols.append((start,699))

    return symbols

def find_symbols_y(matrix):
    used_rows = np.any(matrix==1,axis=1)

    y1,y2 = None,None

    for x in range(len(matrix)):
        if used_rows[x] and y1 is None:
            y1 = x
        elif not used_rows[x] and y1 != None:
            y2 = x
            return y1,y2
    return y1,199

def resize_symbol(cropped):
    height = cropped.shape[0]
    width = cropped.shape[1]

    scale = 20 / max(height, width)

    new_height = max(1,int(height * scale))
    new_width = max(1,int(width * scale))

    y_positions = np.linspace(0,height - 1,new_height).astype(int)
    x_positions = np.linspace(0,width - 1,new_width).astype(int)

    resized = cropped[y_positions][:,x_positions]

    return resized

def center_symbol(matrix):
    final_image = np.zeros((28,28))

    height = matrix.shape[0]
    width = matrix.shape[1]

    start_y = (28 - height) // 2
    start_x = (28 - width) // 2

    final_image[start_y:start_y+height,start_x:start_x+width] = matrix

    return final_image

def main():
    global board_matrix
    display = pygame.display.set_mode((WIDTH,HEIGHT))

    clock = pygame.time.Clock()

    drawing = False

    display.fill((8, 10, 14))

    my_font = pygame.font.SysFont(None,30)
    text_1 = my_font.render('Solve',True,WHITE)

    my_font = pygame.font.SysFont(None,30)
    text_2 = my_font.render('Clear',True,WHITE)
    surface.fill((255,255,255))

    Running = True
    while Running:

        display.blit(surface,(100,50))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    draw(event.pos,board_matrix)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

            elif event.type == pygame.MOUSEMOTION and drawing:
                draw(event.pos,board_matrix)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                
                button_1_left = 130 
                button_1_right = 230 

                button_2_left = 280  
                button_2_right = 380
                
                if button_1_left < mouse_x < button_1_right and 270 < mouse_y < 300:
                    symbols = find_symbols_x(board_matrix)
                    predictions = []
                    for x1,x2 in symbols:
                        y1,y2 = find_symbols_y(board_matrix[:, x1:x2 + 1])
                        cropped = board_matrix[y1:y2 + 1, x1:x2 + 1]
                        resized = resize_symbol(cropped)
                        final_image = center_symbol(resized)
                        Input = final_image.reshape(784, 1)

                        _,_,output = nn.forward_prop(Input)
                        prediction = int(np.argmax(output))
                        predictions.append(classes[prediction])
                    print(predictions)

                    parcer = Parcer(tokeniser(predictions))
                    value = parcer.expression()
                    print(value)

                if button_2_left < mouse_x < button_2_right and 270 < mouse_y < 300:
                    surface.fill((255,255,255))
                    board_matrix = np.zeros((200, 700))

        pygame.draw.rect(display,(DARK_BLUE),(130,270,100,30),border_radius=10)
        pygame.draw.rect(display,(DARK_BLUE),(280,270,100,30),border_radius=10)

        display.blit(text_1,(147,278))
        display.blit(text_2,(297,278))

        display.blit(surface, (100, 50))
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
