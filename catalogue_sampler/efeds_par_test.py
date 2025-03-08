from efeds_sample_generator import generate_sample
import numpy as np
from multiprocessing import Process, Queue, Value, Lock
from efeds_uniform_prior_sampler import uniform_prior
from time import time
from tqdm import tqdm
import json
import os

n = 20
times = 15

path = r'C:\Users\Артем\Documents\GitHub\NN-Clusters\catalogue_sampler\samples/'

def sample_data(counter=None, lock=None):
    pars = uniform_prior()
    sample = generate_sample(pars)
    name = str(counter.value)
    with lock:
        counter.value += 1
    os.mkdir(path + name)
    np.save(path + name + f"/{name}.npy", sample)
    with open(path + name + f"/{name}.json", 'w') as f:
        f.write(json.dumps(pars))
    
    
def run_n_times(n, queue=None, counter=None, lock=None):
    for _ in range(n):
        sample_data(counter, lock)
        queue.put(1)
        
if __name__ == "__main__":
    t1 = time()
    processes = []
    queue = Queue()
    counter = Value('i', 0)
    lock = Lock()
    for _ in range(times):
        p = Process(target=run_n_times, args=(n, queue, counter, lock))
        processes.append(p)
        p.start()
        
    with tqdm(total=n*times) as pbar:
        for _ in range(n*times):
            queue.get()
            pbar.update(1)
        
    for p in processes:
        p.join()
    
    t2 = time()
    print(t2-t1, 'sec')
    print((t2-t1)/(n*times), 'sec/sample')
    