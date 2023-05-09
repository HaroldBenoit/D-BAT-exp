import torch
import torchvision


class InputNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized
    

class RobustModel(torch.nn.Module):

    def __init__(self, model):
        super(RobustModel, self).__init__()
        self.normalizer = InputNormalize(torch.zeros((3,1,1)), torch.zeros((3,1,1)))
        self.model = model

    def forward(self, inp):
        normalized_inp = self.normalizer(inp)
        output = self.model(normalized_inp)
        return output
    
def get_robust_resnet50():
    model = RobustModel(torchvision.models.resnet50())
    return model
