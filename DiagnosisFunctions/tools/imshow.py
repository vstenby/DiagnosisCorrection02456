import matplotlib.pyplot as plt
def imshow(x):
    plt.imshow(x.permute(1,2,0))