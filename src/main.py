


from stf import *

def main():
    
    datasets = ['condition', 'beijing', 'madrid', 'sensor', 'radar', 'chicago']
    
    for name in datasets:
        print(f"START STF on {name} streams...!")
        start_stf(name)
        

if __name__ == '__main__':
    main()
