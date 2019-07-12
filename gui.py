from tkinter import *
import tkinter as tk
import os
from PIL import ImageTk, Image


def callback():
    os.system("python camera.py")


root = Tk()
canvas = Canvas(root, width=5000, height=6000)
canvas.pack(expand = True, fill = "both")
btn = tk.Button(root, command=callback,
                activebackground = "#33B5E5")
btn.place(height=100, width=100, x=20, y=20)
img1 = ImageTk.PhotoImage(Image.open("images/camera.png"))
btn.config(image=img1, width=200, height=100)
img2 = ImageTk.PhotoImage(Image.open("images/yolo_edit.jpg"))
canvas.create_image(20, 10, anchor=NW, image=img2)
canvas.create_window(720, 580,height=140, width=250, anchor='s', window=btn)
root.mainloop()
