import numpy as numpy
import matplotlib.pyplot as plt
import h5py
import re, sys
from collections import defaultdict 
import numpy as np

def natural_keys(text):
    atoi = lambda text: int(text) if text.isdigit() else text
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    

def plot_turbqty(file_path):

    f = h5py.File(file_path,'r')
    
    iterations = list(f['Turbulence/TurbQty'].keys())
    iterations.sort(key=natural_keys)
    
    data = defaultdict(list) 
    for it in iterations:
        dict_it = eval(f['Turbulence/TurbQty'][it][()])
        for key in dict_it: 
            data[key].append(dict_it[key]) 
    f.close()

    idx = [int(it) for it in iterations]
    
    fig = plt.figure()
    for key in data.keys():
        plt.semilogy(idx,data[key],label=key)

    plt.legend()
    plt.savefig(file_path[:-3]+'_turbqty.pdf')

    return

def plot_spectrum(file_path,leave_out=0):


    f = h5py.File(file_path,'r')
    
    iterations = list(f['Turbulence/Ek'].keys())
    iterations.sort(key=natural_keys)

    bins = f['Turbulence/bins'][()]+0.5
    Ek = np.zeros([len(iterations),len(bins)])
    eps = eval(f['Turbulence/TurbQty'][iterations[0]][()])['eps_forcing']

    for i,it in enumerate(iterations):
        Ek[i,:] = f['Turbulence/Ek'][it][()]
    f.close()

    Ek_avg = np.mean(Ek[leave_out:,:],axis=0)
    
    fig, ax = plt.subplots(1,1)
    ax.loglog(bins,Ek_avg) 
    ax.loglog(bins,eps**(2/3.0)*bins**(-5/3.0),color='k', linestyle='--') 
    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.savefig(file_path[:-3]+'_spectrum.pdf')
    plt.close()

    return


if __name__ == "__main__":  

    file_path = sys.argv[1] 
    
    plot_turbqty(file_path)
    plot_spectrum(file_path)