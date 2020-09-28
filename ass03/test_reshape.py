import numpy as np
if __name__=='__main__':
    a = np.array([[1,2,3], [4,5,6]])
    a = np.reshape(a, (-1, 1))
    print(a)
    