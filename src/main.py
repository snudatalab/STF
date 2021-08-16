"""
Accurate Online Tensor Factorization for Temporal Tensor Streams with Missing Values (CIKM 2021)
Authors:
- Dawon Ahn (dawon@snu.ac.kr), Seoul National University
- Seyun Kim (kim79@cooper.edu), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""


from stf import *

def main():
    
    datasets = ['condition', 'beijing', 'madrid', 'sensor', 'radar', 'chicago']
    
    for name in datasets:
        print(f"START STF on {name} streams...!")
        start_stf(name)
        

if __name__ == '__main__':
    main()
