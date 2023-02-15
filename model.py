from collections import OrderedDict


def getModuleList(model, name):
    children = list(model.named_children())
    if len(children) == 0:
        return [(name, model)]
    moduleList = [(name, model)]
    for n, c in children:
        module = getModuleList(c, name+"."+n)
        moduleList += module
    return moduleList

class ModuleRecoder:
    def __init__(self, module):
        self.setModule(module)
        self.outputs = None
        self.inputs = None

    def hook(self, module, inputs, outputs):
        if isinstance(outputs, OrderedDict):
            self.outputs = outputs["out"].detach().cpu().numpy()
        else:
            self.outputs = outputs.detach().cpu().numpy()
        self.inputs = inputs[0].detach().cpu().numpy()

    def clear(self):
        del self.outputs
        del self.inputs
        self.outputs = None
        self.inputs = None
    
    def detach(self):
        self.removeHandle.remove()
    
    def setModule(self, module):
        self.module = module
        self.removeHandle = self.module.register_forward_hook(self.hook)

