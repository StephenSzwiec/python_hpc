#!/usr/env/bin python 
# -*- coding: utf-8 -*-

'''
Title: Parallel Computing Example
Author: Stephen Szwiec
Date: 2024-08-14

Description: This code demonstrates the following:
    - Pythonic loops are slow 
    - List comprehensions are faster 
    - vectorized operations like numpy are faster than list comprehensions
    - parallel computing is faster than vectorized operations for large datasets
    - Just-In-Time (JIT) compilation can speed up Python code 

Task 1: 
    - Create a list of n integers between 0 and 100
    - Filter out the even numbers
    - Square each odd number
    - Compute the sum of the squared numbers

Approaches:
    - Loop-based approach
    - List comprehension approach
    - NumPy approach
    - JIT approach
    - Multiprocessing approach
    - Parallel computing approach

'''

'''
Region Metadata
'''
__author__ = "Stephen Szwiec"
__email__ = "stephen.szwiec@ndsu.edu"
__status__ = "Development"
__license__ = "GPLv3"

'''
Region: import libraries
'''
import os
import shutil
import time
import random
import math
import numpy as np
import itertools
import multiprocessing as mp
from multiprocessing import Pool
import numba
from numba import jit
from joblib import Parallel, delayed
from argparse import ArgumentParser

'''
Region: function definitions
'''
#Generate a list of n integers between 0 and 100
@jit 
def generate_random_integers(n):
    random.seed(42)
    assert n > 0
    # np broadcasting is faster than list comprehension
    return np.random.randint(0, 100, n)
    
# Loop-based approach 
def loop_approach(data):
    total = 0
    for i in data:
        if i % 2 == 1:
            total += i ** 2
    return total

# List comprehension approach
def list_comprehension_approach(data):
   return sum([i ** 2 for i in data if i % 2 == 1])

# NumPy approach
def numpy_approach(data):
    np_data = np.array(data)
    return np.sum(np.square(np_data[np_data % 2 == 1]))

# multiprocessing approach; uses all available CPU cores using subprocesses
def parallel_approach(data):
    num_cores = mp.cpu_count()
    chunks = np.array_split(data, num_cores)
    with Pool(num_cores) as pool:
        results = pool.map(numpy_approach, chunks)
    return sum(results)

# JIT approach -- uses OpenMP for parallelization on the CPU, CUDA for GPU
@jit
def numba_approach(data):
    total = 0
    for i in numba.prange(len(data)):
        if data[i] % 2 == 1:
            total += data[i] ** 2
    return total

# Parallel computing approach -- uses joblib to parallelize the computation
def parallel_computing_approach(data):
    return sum(Parallel(n_jobs=-1,prefer="threads")(delayed(numba_approach)(i) for i in np.array_split(data, mp.cpu_count())))

'''
Region: argument parser
'''
def parse_args():
    parser = ArgumentParser(
            description="Parallel Computing Example")
    parser.add_argument("-n", "--size", dest="size",
                        type=int, help="Size of the dataset 10^x", 
                        action="store", required=False)
    return parser.parse_args()

'''
Region: main function
'''
def main():
    good_input = False
    args = parse_args()
    # check if the user entered a dataset size
    if args.size:
        dataset_size = args.size
        good_input = True
    # otherwise, prompt the user for the dataset size
    print("Hello, Parallel Computing Example!")
    print("Task 1: Compute the sum of the squared odd numbers")
    print("---------------------------------------------------")
    # use a case statement to determine the size of the dataset
    while not good_input:
        dataset_size = input("Enter the size of the dataset 10^x: ")
        try:
            dataset_size = int(dataset_size)
            good_input = True
        except ValueError:
            print("Please enter a valid integer")
    n = 10 ** dataset_size
    print(f"Dataset size: {n}")
    data = generate_random_integers(n)
    
    # Loop-based approach
    start = time.time()
    loop_approach(data)
    loop_time = time.time() - start
    print(f"Loop-based approach time: {loop_time} seconds")

    # List comprehension approach
    start = time.time()
    list_comprehension_approach(data)
    list_comp_time = time.time() - start
    print(f"List comprehension approach: {list_comp_time} seconds")

    # NumPy approach
    start = time.time()
    numpy_approach(data)
    numpy_time = time.time() - start
    print(f"NumPy approach: {numpy_time} seconds")

    # JIT approach
    start = time.time()
    numba_approach(data)
    numba_time = time.time() - start
    print(f"JIT approach: {numba_time} seconds")

    # Multiprocessing approach
    start = time.time()
    parallel_approach(data)
    parallel_time = time.time() - start
    print(f"Multiprocessing approach: {parallel_time} seconds")

    # Parallel computing approach
    start = time.time()
    parallel_computing_approach(data)
    parallel_computing_time = time.time() - start
    print(f"Parallel computing approach: {parallel_computing_time} seconds")
    print("---------------------------------------------------")
    print("")
    print("Summary")
    print("---------------------------------------------------")
    # print the speedup factor for each approach compared to the loop-based approach
    print(f"List comprehension speedup: {loop_time / list_comp_time}")
    print(f"NumPy speedup: {loop_time / numpy_time}")
    print(f"JIT speedup: {loop_time / numba_time}")
    print(f"Multiprocessing speedup: {loop_time / parallel_time}")
    print(f"Parallel computing speedup: {loop_time / parallel_computing_time}")
    print("---------------------------------------------------")
    # do we have gnuplot installed?
    # use exit status of which gnuplot 
    if shutil.which("gnuplot"):
        print("Generating plot...")
        with open("speedup.dat", "w") as f:
            f.write("Approach Speedup\n")
            f.write("Loop 1\n")
            f.write("List {}\n".format(loop_time / list_comp_time))
            f.write("NumPy {}\n".format(loop_time / numpy_time))
            f.write("JIT {}\n".format(loop_time / numba_time))
            f.write("MP {}\n".format(loop_time / parallel_time))
            f.write("Parallel {}\n".format(loop_time / parallel_computing_time))
        # plot to the terminal as a bar chart and scale the x-axis, and remove the legend
        os.system("gnuplot -e \"set terminal dumb; set style data histogram; set style fill solid; set xtics rotate by -45; unset key; plot 'speedup.dat' using 2:xtic(1)\"")
        # delete the data file
        os.remove("speedup.dat")

    print("---------------------------------------------------")
    print("Goodbye, Parallel Computing Example!")

if __name__ == "__main__":
    main()

