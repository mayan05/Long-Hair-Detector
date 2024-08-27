import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model 

# Load models
age_model = load_model("C:\\Users\\mayan\\Desktop\\CODES\\Null Codes\\Long Hair Detection\\Age_Detector.h5")
gen_model = load_model("C:\\Users\\mayan\\Desktop\\CODES\\Null Codes\\Long Hair Detection\\Gender_Detector.h5")
hair_model = load_model("C:\\Users\\mayan\\Desktop\\CODES\\Null Codes\\Long Hair Detection\\Hair_Classifier.h5")

# GUI
win = tk.Tk()
win.title("Age, Gender, and Hair Length Detector")
win.geometry('800x600')

# Initializing the Labels
age_label = tk.Label(win, font=('arial', 15, "bold"))
gen_label = tk.Label(win, font=('arial', 15, 'bold'))
hair_label = tk.Label(win, font=('arial', 15, "bold"))
image_label = tk.Label(win)

def Detect(file_path): # will do the predictions
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array([image])
    
    pred_age = int(age_model.predict(image).reshape(-1))
    pred_gen = gen_model.predict(image).reshape(-1)
    pred_hair = hair_model.predict(image).reshape(-1)

    age_label.configure(foreground="#011638", text=f'Age: {pred_age}')

    if (pred_age >= 20) and (pred_age <= 30):
        if pred_hair == 1:  # Long Hair
            gen_label.configure(foreground="#011638", text='Gender: Female')
            hair_label.configure(text="Hair Length: Long Hair", foreground="#011638")
        else:  # Short hair
            gen_label.configure(foreground="#011638", text='Gender: Male')
            hair_label.configure(text="Hair Length: Short Hair", foreground="#011638")

    else:
        if pred_gen < 0.5:  # Male
            gen_label.configure(foreground="#011638", text='Gender: Male')
            if pred_hair == 1:  # Long Hair
                hair_label.configure(text="Hair Length: Long Hair", foreground="#011638")
            else:
                hair_label.configure(text="Hair Length: Short Hair", foreground="#011638")

        else:  # Female
            gen_label.configure(foreground="#011638", text='Gender: Female')
            if pred_hair == 1:  # Long Hair
                hair_label.configure(text="Hair Length: Long Hair", foreground="#011638")
            else:
                hair_label.configure(text="Hair Length: Short Hair", foreground="#011638")
    

def show_Detect_button(file_path):
    Detect_b = tk.Button(win, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(foreground='white', background='#011638', font=('arial', 10, 'bold'))
    Detect_b.pack(side='bottom', pady=10)

def upload_image(): # to upload the image
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((win.winfo_width()/2.25), (win.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        image_label.configure(image=im) 
        image_label.image = im 
        age_label.configure(text='')
        gen_label.configure(text='')
        hair_label.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error uploading image: {e}")

# GUI Layout
heading = tk.Label(win, text="Age, Gender, and Hair Length Detector", pady=20, font=('arial', 20, "bold"))
heading.pack()

upload = tk.Button(win, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(foreground='white', background='#011638', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=10)

image_label.pack(expand=True)
age_label.pack(expand=True)
gen_label.pack(expand=True)
hair_label.pack(expand=True)

win.mainloop()