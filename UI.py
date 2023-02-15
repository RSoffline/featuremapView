
import tkinter as tk
import tkinter.ttk as ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np


class Event:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args, **kwds):
        if self.callback is None :return 
        self.callback(*args, **kwds)
    def setCallback(self, callback):
        self.callback = callback
    
class App(ttk.Frame):
    def __init__(self, master, *args, **kwds):
        self.master = master
        super().__init__(master, *args, **kwds)
        self.loadImageCallback = Event(None)
        self.predCallback = Event(None)
        self.scaleCallback = Event(None)
        self.comboboxCallback = Event(None)
        self.createUI()

    def createUI(self):
        imageFrame = ttk.Frame(self)
        imageFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cm = plt.cm.get_cmap("bwr")
        cm.set_under("black")
        cm.set_over("silver")
        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)
        self.imageData = ax.imshow(
            np.zeros((512,512), np.uint8),
            cmap=cm,
            interpolation="none",
            vmin=-1, vmax=1
        )
        fig.colorbar(self.imageData, ax=ax, extend="both")
        self.canvas = FigureCanvasTkAgg(fig, imageFrame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, imageFrame)
        self.scale = tk.Scale(
            imageFrame,
            orient=tk.HORIZONTAL,
            from_=0,
            to=255,
            resolution=1,
            command=self.scaleCallback
        )
        self.scale.pack(fill=tk.X)
        toolFrame = ttk.Frame(self)
        toolFrame.pack(side=tk.LEFT)
        self.combobox = ttk.Combobox(toolFrame, width=50)
        self.combobox.bind("<<ComboboxSelected>>", self.comboboxCallback)
        self.combobox.pack()
        ttk.Button(toolFrame, text="Load image", command=self.loadImageCallback).pack()
        ttk.Button(toolFrame, text = "Pred", command=self.predCallback).pack()

    def setImage(self, img, vmin=None, vmax=None):
        if vmin is None:
            vmin = img.mean() - 3*img.std()
        if vmax is None:
            vmax = img.mean() + 3*img.std()
        self.imageData.set(array=img, clim=(vmin,vmax))
        self.canvas.draw()

    def setModuleList(self, modules):
        self.combobox.configure(values=modules)

    def setImageNum(self, num):
        self.scale.configure(to=num)
