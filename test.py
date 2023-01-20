"""Run VASP to Compute the Energy, Forces, Stress of a Structure"""
import argparse
import os
import pickle
import time
import wandb
import numpy as np
from utils import zipDir
from ase.io import read, write
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.vasp import Vasp

parser=argparse.ArgumentParser()
parser.add_argument("--path",type=str,default=None)
parser.add_argument("--n_core",type=int,default=6)
parser.add_argument("--interleave",nargs="+",type=int,default=[0,1500])  ## interleave [low,high)
parser.add_argument("--encut",type=float,default=400)
parser.add_argument("--ediff",type=float,default=1e-4)
parser.add_argument("--ismear",type=int,default=0)
parser.add_argument("--sigma",type=float,default=0.02)
parser.add_argument("--amplitude",type=float,default=0.2)
parser.add_argument("--kspacing",type=float,default=0.2)
parser.add_argument("--gamma",type=bool,default=True)
parser.add_argument("--nelm",type=int,default=200)
parser.add_argument("--wandb",action="store_true")
parser.add_argument("--wandb_key",type=str,default="456bb4bf23bb4ed5c90d46c282e58d933ddbe068")
parser.add_argument("--print_log",action="store_true")
args=parser.parse_args()
ASE_VASP_COMMAND="mpirun -np "+str(args.n_core)+" vasp_std"

assert(os.path.exists(os.getcwd()+"/VASP_PP"))  ## VASP_PP must be under the working directory

if args.wandb:
    wandb.login(key=args.wandb_key)
    wandb.init(project="vasp compute",name=args.path.split("/")[-1]+" ["+str(args.interleave[0])+","+str(args.interleave[1])+"]",config=args)
    
## set VASP_PP_PATH
os.environ["VASP_PP_PATH"]=os.path.join(os.getcwd(),"VASP_PP")
## prepare vasp_run directory
if not os.path.exists("vasp_run"):
    os.mkdir("vasp_run")
else:
    os.system("rm -rf vasp_run/*")
## prepare results directory
if not os.path.exists("results"):
    os.mkdir("results")
else:
    os.system("rm -rf results/*")
    
calc = Vasp(xc='PBE',
            encut=args.encut,
            ediff=args.ediff,
            ismear=args.ismear,
            sigma=args.sigma,
            kspacing=args.kspacing,
            gamma=args.gamma,
            nelm=args.nelm,
            restart=None,            
            command=ASE_VASP_COMMAND,
            directory="./vasp_run",
            txt="-" if args.print_log else None)

if __name__=="__main__":
    system="15_B28"
    save_path="results/{}".format(system)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i,file in enumerate(os.listdir("./datasets/{}".format(system))):
        if file.endswith(".res"):
            atoms=read("./datasets/{}/".format(system)+file)
            start_time=time.time()
            atoms.set_calculator(calc)
            energy=atoms.get_potential_energy()
            energy_per_atom=energy/len(atoms)
            end_time=time.time()
            write(save_path+"/"+file.replace(".res",".xyz"),atoms)
            print("compute <"+file+"> in "+str(end_time-start_time)+"s")
            print("energy: {},  energy_per_atom: {}".format(energy, energy_per_atom))
            print("{} / {}".format(i+1,len(os.listdir("./datasets/{}".format(system)))))
            print("===============================================")
        os.system("rm -rf vasp_run/*")
            