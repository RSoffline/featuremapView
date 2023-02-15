import tkinter as tk
from tkinter import filedialog

import torch
from torchvision import transforms
from PIL import Image

from model import ModuleRecoder, getModuleList
from UI import App


class Controller:
    def __init__(self, model, modelName, device, preprocess=None, recoder=None, masterUI=None):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        moduleList = getModuleList(self.model, modelName)
        self.moduleList = dict(moduleList)
        if recoder is None:
            self.recoder = ModuleRecoder(self.model)
        else:
            self.recoder = recoder

        if preprocess is None:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.preprocess = preprocess

        if masterUI is None:
            self.root = tk.Tk()
        else:
            self.root = masterUI
        self.ui = App(self.root)
        self.ui.pack(fill=tk.BOTH, expand=True)
        self.ui.loadImageCallback.setCallback(self.loadImage)
        self.ui.predCallback.setCallback(self.pred)
        self.ui.scaleCallback.setCallback(self.selectImage)
        self.ui.comboboxCallback.setCallback(self.selectModule)
        self.ui.setModuleList(list(self.moduleList.keys()))

    def run(self):
        self.root.mainloop()
    
    def loadImage(self):
        imgPath = filedialog.askopenfilename()
        if imgPath == "":
            return
        self.img = Image.open(imgPath).convert("RGB")

    def pred(self):
        imgs = self.preprocess(self.img).unsqueeze(0)
        with torch.no_grad():
            imgs = imgs.to(self.device)
            output = self.model(imgs)
        del output
        del imgs
        self.ui.setImageNum(self.recoder.outputs.shape[1]-1)
        self.ui.scale.set(0)

    def selectImage(self, val):
        val = int(val)
        self.ui.setImage(self.recoder.outputs[0, val])

    def selectModule(self, event):
        widget = event.widget
        selectedModule = widget.get()
        self.recoder.clear()
        self.recoder.detach()
        self.recoder.setModule(self.moduleList[selectedModule])
