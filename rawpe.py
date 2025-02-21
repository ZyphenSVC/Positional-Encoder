import numpy as np

def pe1D(p,L):
    p = np.asarray(p)
    frequency = 2.0 ** np.arange(L)
    encoded = np.concatenate([np.sin(np.pi * frequency * p), np.cos(np.pi * frequency * p)],
                             axis=-1)
    return encoded

def pe2D(coords, L):
    x,y = np.asarray(coords)
    encodingX = pe1D(x,L)
    encodingY = pe1D(y,L)
    encodedCoords = np.concatenate([encodingX, encodingY], axis=-1)
    return encodedCoords

pe1dTest = 0.1
L = 1023
encoded1D = pe1D(pe1dTest, L)

coords = (1,1)
encoded2D = pe2D(coords, L)

print("1D PE: ", encoded1D)
print("2D PE: ", encoded2D)