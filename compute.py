"""Run VASP to Compute the Energy, Forces, Stress of a Structure"""
import argparse
import os
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
parser.add_argument("--gamma",action="store_true")
parser.add_argument("--nelm",type=int,default=200)
parser.add_argument("--wandb",action="store_true")
parser.add_argument("--print_log",action="store_true")
args=parser.parse_args()
ASE_VASP_COMMAND="mpirun -np "+str(args.n_core)+" vasp_std"

assert(args.path is not None)
assert(os.path.exists(args.path))
assert(os.path.exists(os.getcwd()+"/VASP_PP"))  ## VASP_PP must be under the working directory

if args.wandb:
    wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
    wandb.init(project="vasp compute",name="compute "+args.path+" ["+str(args.interleave[0])+","+str(args.interleave[1])+"]",config=args)
    
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
    files=os.listdir(args.path)
    files.sort()
    print("Total number of structures: ",len(files))
    files=files[args.interleave[0]:args.interleave[1]]
    traj=TrajectoryWriter("results/result["+str(args.interleave[0])+","+str(args.interleave[1])+"].traj",\
                            mode="a",properties=["energy","forces","stress"])
    time_list=[]
    for i,file in enumerate(files):
        if file.endswith(".res"):
            atoms=read(os.path.join(args.path,file))
            start_time=time.time()
            atoms.set_calculator(calc)
            traj.write(atoms)
            end_time=time.time()
            print("compute "+file+" in "+str(end_time-start_time)+"s")
            print("energy: ",atoms.get_potential_energy())
            print("{} / {}".format(i+1,len(files)))
            print("===============================================")
            time_list.append(end_time-start_time)
        os.system("rm -rf ./vasp_run/*")
    traj.close()
    print("average time: "+str(np.mean(time_list)))
    if args.wandb:
        zipDir("results","results.zip")
        wandb.save("results.zip")
            