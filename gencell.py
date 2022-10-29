"""This Module Generate Seed Cells"""

import os
import argparse
import numpy as np

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default="gencell")
    parser.add_argument("--n",type=int,default=1000)
    parser.add_argument("--num_atoms",type=int,default=28)
    parser.add_argument("--varvols",nargs="+",type=float,default=[3.0,10.0])
    parser.add_argument("--minseps",nargs="+",type=float,default=[1.0,2.0])
    ## TODO: add more arguments
    args=parser.parse_args()
    
    ## prepare seed.cell 
    os.system("rm -r "+args.path)
    os.system("mkdir "+args.path)
    os.system("cp B.cell "+args.path)
    
    ## build cell
    for i in range(args.n):
        origin_seed_file=open(args.path+"/B.cell","r")
        seed_file=open(args.path+"/B28_seed.cell","w")
        for line in origin_seed_file.readlines():
            if "#VARVOL" in line:
                seed_file.write("#VARVOL={}\n".format(np.random.uniform(args.varvols[0],args.varvols[1])))
            elif "B 0.0 0.0 0.0 # B1 % NUM=" in line:
                seed_file.write("B 0.0 0.0 0.0 # B1 % NUM={}\n".format(args.num_atoms))
            else:
                seed_file.write(line)
        os.system("buildcell < {}/B28_seed.cell | cabal cell res > {}/B28_{}.res".format(args.path,args.path,i))
        os.system("rm -rf B28_seed.cell")
    os.system("rm -rf {}/B28.cell".format(args.path))
    