from ase.io.trajectory import Trajectory

traj=Trajectory("results/result[0,10].traj")

for atoms in traj:
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
    print(atoms.get_stress())