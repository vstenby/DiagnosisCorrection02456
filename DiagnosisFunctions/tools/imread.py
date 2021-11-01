from PIL import Image
import torchvision.transforms.functional as TF

def imread(path):
    #Reads in an image.
    im = Image.open(path)
    x  = TF.to_tensor(im)
    return x