import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("mnist.h5")

# Tkinter window
root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack()

image = Image.new("L", (200, 200), color=255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
    draw.ellipse([x1, y1, x2, y2], fill=0)

canvas.bind("<B1-Motion>", paint)

def predict_digit():
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    result_label.config(text=f"Prediction: {digit}")

btn = tk.Button(root, text="Predict", command=predict_digit)
btn.pack()

result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

root.mainloop()
