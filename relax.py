import ase
from ase.io import read, write,Trajectory
from ase.calculators.vasp import Vasp
from ase.constraints import ExpCellFilter
from ase.optimize.precon import PreconLBFGS, Exp
import argparse
import os
import time
import wandb
import numpy as np
from utils import zipDir

parser=argparse.ArgumentParser()
parser.add_argument("--path",type=str,default=None)  ## path and path/data must exist
parser.add_argument("--n_core",type=int,default=8)
parser.add_argument("--relax_steps",type=int,default=1000)
parser.add_argument("--shake_steps",type=int,default=0)     
parser.add_argument("--sample_freq",type=int,default=1)
parser.add_argument("--interleave",nargs="+",type=int,default=[0,1500])  ## interleave [low,high)
parser.add_argument("--fmax",type=float,default=0.001)
parser.add_argument("--smax",type=float,default=0.02)
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

assert(args.path is not None)
assert(os.path.exists(args.path))
assert(args.relax_steps>0)
assert(args.shake_steps>=0)
assert(os.path.exists(os.getcwd()+"/VASP_PP"))  ## VASP_PP must be under the working directory

if args.wandb:
    wandb.login(key=args.wandb_key)
    wandb.init(project="vasp run",name="relax "+args.path+" ["+str(args.interleave[0])+","+str(args.interleave[1])+"]",config=args)

## set VASP_PP_PATH
os.environ["VASP_PP_PATH"]=os.path.join(os.getcwd(),"VASP_PP")

## relax trajectories are stored in path/relax
if not os.path.exists(os.path.join(args.path,"relax")):
    os.mkdir(os.path.join(args.path,"relax"))
if not os.path.exists(os.path.join(args.path,"vasp_run")):
    os.mkdir(os.path.join(args.path,"vasp_run"))
    
# input_path=os.path.join(args.path,"data")
input_path=os.path.join(args.path)
relax_path=os.path.join(args.path,"relax")
shake_path=os.path.join(args.path,"shake")
input_files=os.listdir(input_path)
input_files=[f for f in input_files if f.endswith(".res")]
input_files.sort()
input_files=input_files[args.interleave[0]:args.interleave[1]]


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
            directory=args.path+"/vasp_run",
            txt="-" if args.print_log else None)

if __name__=="__main__":
    os.system("rm -rf "+args.path+"/vasp_run/*")
    os.system("rm -rf "+args.path+"/relax/*")
    time_list=[]
    step_list=[]
    print(len(input_files))
    for file in input_files:
        start_time=time.time()
        try:
            ## relax begins:
            atoms=read(os.path.join(input_path,file))
            atoms.set_calculator(calc)
            ecf=ExpCellFilter(atoms)
            traj = Trajectory(relax_path+"/traj_"+file.replace(".res",".traj"), 'w', atoms, properties=["energy","forces","stress"])
            optimizer = PreconLBFGS(ecf, precon=Exp(3), use_armijo=True, master=True)
            optimizer.attach(traj.write, interval=args.sample_freq)
            optimizer.run(fmax=args.fmax, smax=args.smax, steps=args.relax_steps)
            traj.close()
            end_time=time.time()
            ## relax ends
            print("relax finished for {} in {} seconds".format(file,end_time-start_time))
            time_list.append(end_time-start_time)
            step_list.append(len(traj))
            if args.wandb:
                wandb.log({"relaxed energy":atoms.get_potential_energy()})
        except:
            end_time=time.time()
            print("relax failed for {} in {} seconds".format(file,end_time-start_time))
            time_list.append(end_time-start_time)
            step_list.append(len(traj))
        ## clean up the vasp_run directory
        os.system("rm -rf "+args.path+"/vasp_run/*")
    if args.wandb:
        zipDir(args.path+"/relax","relax_"+args.path+".zip")
        wandb.save("relax_"+args.path+".zip")
    print("average time per traj: {}".format(sum(time_list)/len(time_list)))
    print("average time per step: {}".format(np.sum(time_list)/np.sum(step_list)))