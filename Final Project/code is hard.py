import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2
import pygame
from typing import Callable

# Load and preprocess the dataset
def load_data(data_path):
    images = []
    labels = []
    
    for mode in ['train', 'test']:
        image_dir = os.path.join(data_path, mode, 'images')
        label_dir = os.path.join(data_path, mode, 'labels')
        
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace('.png', '.txt'))
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (300, 300))
            images.append(image)
            
            with open(label_path, 'r') as file:
                image_labels = []
                for line_num, line in enumerate(file):
                    if line_num == 0:
                        continue  # Skip the header line
                    parts = line.strip().split(',')
                    image_labels.append([int(parts[0]), float(parts[1])/300.0, float(parts[2])/300.0, float(parts[3])/300.0, float(parts[4])/300.0])
                labels.append(image_labels)
    
    images = np.array(images).astype('float32') / 255.0
    return images, labels

data_path = './data/mnist_detection/'
train_images, train_labels = load_data(data_path)

# Convert labels to numpy array for training
# Padding the labels array to ensure consistent shape
max_objects = max([len(label) for label in train_labels])
padded_labels = []

for label in train_labels:
    if len(label) < max_objects:
        label += [[-1, 0, 0, 0, 0]] * (max_objects - len(label))
    padded_labels.append(label)

train_labels = np.array(padded_labels)

def create_model():
    model = keras.models.Sequential([
        Input(shape=(300, 300, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(max_objects * 5)  # Multiple bounding boxes with 5 values each
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_model()

# Train the model
model.fit(train_images, train_labels.reshape(len(train_labels), -1), epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('mnist_object_detection_model.keras')








# Load the trained model
# model = tf.keras.models.load_model('mnist_object_detection_model.keras')

def predictor(image_array: np.ndarray) -> np.ndarray:
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)
    return prediction[0]

def predictor_formatter(image_array: np.ndarray) -> np.ndarray:
    image_array = cv2.resize(image_array, (300, 300))
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array

def confidence_lister(prediction: np.ndarray) -> dict:
    label = int(prediction[0])
    confidence = prediction[1]
    return {label: confidence}

def constant_paint_program(
        window_title: str,
        window_icon_path: str,
        predictor: Callable,
        predictor_formatter: Callable,
        width: int,
        height: int,
        scale: int,
        fps: int = 60,
        graph_width: int = 280,
        graph_border_width: int = 2,
        blank_color: tuple[int, int, int] = (0, 0, 0),
        draw_color: tuple[int, int, int] = (255, 255, 255),
        graph_bg_color: tuple[int, int, int] = (64, 64, 64),
        graph_color: tuple[int, int, int] = (30, 42, 92),
        graph_text_color: tuple[int, int, int] = (32, 34, 46),
        graph_percent_text_color: tuple[int, int, int] = (25, 26, 31),
        init_pygame: bool = False,
        quit_pygame: bool = False
    ) -> None:
    """
    Opens a paint program and lets the user draw.
    LC: draw
    RC: erase
    MC: clear
    ESC/quit: save  
    Args:
        window_title (str): The window title.
        window_icon_path (str | None): The window icon's path.
        predictor (Callable[[np.ndarray],np.ndarray]): The predictor function.
        predictor_formatter (Callable[[np.ndarray],np.ndarray]): The function to format the array for the predictor.
        width (int): How many pixels on the width? (not including the prediction graph)
        height (int): How many pixels on the height?
        scale (int): What should width and height be multiplied by before being shown on the screen?
        fps (int, optional): Frames Per Second. Defaults to 60.
        graph_width (int, optional): _description_. Defaults to 280.
        graph_border_width (int, optional): _description_. Defaults to 2.
        blank_color (tuple[int,int,int], optional): The default/erase color in RGB. Defaults to (0, 0, 0).
        draw_color (tuple[int,int,int], optional): The draw color in RGB. Defaults to (255, 255, 255).
        graph_bg_color (tuple[int,int,int], optional): The background color for the graph in RGB. Defaults to (64, 64, 64).
        graph_color (tuple[int,int,int], optional): The fill-in color for the graph in RGB. Defaults to (30, 42, 92).
        graph_text_color (tuple[int,int,int], optional): The text color for the graph in RGB. Defaults to (32, 34, 46). 
        graph_percent_text_color (tuple[int,int,int], optional): The text color for the percent of the graph in RGB. Defaults to (25, 26, 31). 
        init_pygame (bool, optional): Should it run pygame.init()? Defaults to False.
        quit_pygame (bool, optional): Should it run pygame.quit()? Defaults to False.
    """
    
    if init_pygame:
        pygame.init()

    screen: pygame.surface.Surface = pygame.display.set_mode((width*scale+graph_width, height*scale))
    fpsClock = pygame.time.Clock()

    pygame.display.set_caption(window_title)
    if window_icon_path != None:
        pygame.display.set_icon(pygame.image.load(window_icon_path))

    # Track mouse button states
    mouse_draw_down: bool = False
    mouse_erase_down: bool = False

    font = pygame.font.Font('Roboto/Roboto-Regular.ttf', 30)
    percent_font = pygame.font.Font('Roboto/Roboto-Regular.ttf', 20)

    num_render: dict[int,None|pygame.surface.Surface] = {i:None for i in range(0,10)}
    for num in num_render.keys():
        num_render[num] = font.render(str(num),False,graph_text_color)

    percent_render: dict[str,None|pygame.surface.Surface] = {str(i):None for i in range(0,101)}
    for num in percent_render.keys():
        percent_render[num] = percent_font.render(f"{num}%",False,graph_percent_text_color)

    slot_centers: list[tuple[int,int]] = []
    for slot_y in range(0,screen.get_height()-10,int((screen.get_height()-10)/10)):
        slot_centers.append((int(((graph_width-np.floor(graph_width/100)*100)*0.75)/2)+width*scale,slot_y+20))

    percent_centers: list[tuple[int,int]] = []
    for slot_y in range(0,screen.get_height()-10,int((screen.get_height()-10)/10)):
        percent_centers.append((int(((graph_width-np.floor(graph_width/100)*100)*0.75)+((np.floor(graph_width/100)*100)/2))+width*scale,slot_y+20))

    # Game loop
    running: bool = True
    screen.fill(blank_color)

    pixels: np.ndarray = pygame.surfarray.array3d(screen)[0:width*scale]
    grid_pixels: np.ndarray = np.swapaxes(pixels[::scale, ::scale],0,1)
    prediction: np.ndarray = predictor(predictor_formatter(grid_pixels))


    # showdict: bool = False

    while running:
        for event in pygame.event.get():


            # if event.type == pygame.KEYDOWN:
            #     showdict: bool = True
            # if event.type == pygame.KEYUP:
            #     showdict: bool = False



            if event.type == pygame.QUIT:
                running: bool = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running: bool = False
            
            # Check for mouse button press/release
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_draw_down: bool = True
                elif event.button == 2:
                    screen.fill(blank_color)
                elif event.button == 3:
                    mouse_erase_down: bool = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_draw_down: bool = False
                if event.button == 3:
                    mouse_erase_down: bool = False
                if mouse_draw_down is False and mouse_erase_down is False:
                    pixels: np.ndarray = pygame.surfarray.array3d(screen)[0:width*scale]
                    grid_pixels: np.ndarray = np.swapaxes(pixels[::scale, ::scale],0,1)
                    prediction: np.ndarray = predictor(predictor_formatter(grid_pixels))
                    print(np.argmax(prediction))
            if mouse_draw_down:
                x, y = pygame.mouse.get_pos()
                x: int = np.floor(x / scale) * scale
                y: int = np.floor(y / scale) * scale
                if x < width*scale:
                    pygame.draw.rect(screen, draw_color, (x, y, scale, scale))
            elif mouse_erase_down:
                x, y = pygame.mouse.get_pos()
                x: int = np.floor(x / scale) * scale
                y: int = np.floor(y / scale) * scale
                if x < width*scale:
                    pygame.draw.rect(screen, blank_color, (x, y, scale, scale))

        
        # conf_per: float = 0.50
        pygame.draw.rect(screen,graph_bg_color,pygame.Rect(width*scale,0,screen.get_width()-width*scale,screen.get_height()))

        # for e in range(0,screen.get_height()-10,int((screen.get_height()-10)/10)):
            # pygame.draw.rect(screen,blank_color,pygame.Rect(((graph_width-np.floor(graph_width/100)*100)*0.75)+(width*scale)-graph_border_width,(10-graph_border_width)+e,np.floor(graph_width/100)*100+graph_border_width*2,20+graph_border_width*2),graph_border_width)
            # pygame.draw.rect(screen,graph_color,pygame.Rect(((graph_width-np.floor(graph_width/100)*100)*0.75)+(width*scale),10+e,(np.floor(graph_width/100)*100)*conf_per,20))

        for i,(k,v) in enumerate(confidence_lister(prediction[0]).items()):
            text = num_render[k]
            assert text != None
            textRect = text.get_rect()
            textRect.center = slot_centers[i]
            screen.blit(text, textRect)

            bar_y_offset = int((screen.get_height()-10)/10) * i

            pygame.draw.rect(screen,blank_color,pygame.Rect(((graph_width-np.floor(graph_width/100)*100)*0.75)+(width*scale)-graph_border_width,(10-graph-border_width)+bar_y_offset,np.floor(graph_width/100)*100+graph_border_width*2,20+graph-border_width*2),graph_border_width)
            pygame.draw.rect(screen,graph_color,pygame.Rect(((graph_width-np.floor(graph_width/100)*100)*0.75)+(width*scale),10+bar_y_offset,(np.floor(graph_width/100)*100)*v,20))

            percent = percent_render[str(int(round(v,2)*100))]
            assert percent != None
            textRect = percent.get_rect()
            textRect.center = percent_centers[i]
            screen.blit(percent, textRect)

        # if showdict:
        #     display_dict(confidence_lister(prediction[0]),round_val=5)


        pygame.display.flip()
        fpsClock.tick(fps)



    if quit_pygame:
        pygame.quit()

