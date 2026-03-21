from Definition_Drone import Drone,MakeGrid
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    world_grid = MakeGrid(fire=3,gridSize=5,charging=1,water = 2)
    map= world_grid.generate()
    world_grid.print_Grid()

if rank<=2:
    # recon
    my_drone = Drone(id=rank, battery=500, batterymax=500, water=0, watermax=0, x=0, y=0,typeofDrone=0)
    print("Hello this is :"+str(rank)+" is a recon drone")


else:
    # water
    my_drone = Drone(id=rank, battery=100, batterymax=100, water=50, watermax=50, x=0, y=0,typeofDrone=1)

    print("Hello this is :"+str(rank)+" is a water drone")
