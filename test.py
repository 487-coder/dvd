import torch
from models import DVDNet
model = DVDNet()
model.load_state_dict(torch.load('model-old.pth'))
model.eval()