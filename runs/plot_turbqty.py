import numpy as numpy
import matplotlib.pyplot as plt
import h5py

def plot_data(file_path):

    f = h5py.File(file_path,'r')
    
    n = len(f['Turbulence/TurbQty'])    














if __name__ == "__main__":  

    if len(sys.argv) > 1:
        file_path = sys.argv[1] 
    else:

    
    plot_data(file_path)