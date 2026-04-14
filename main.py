import copy
import time
from Drone.Definition_Drone import Drone
from Grid.MakeGrid import MakeGrid

from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Added this!
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    world_grid = MakeGrid(fire=5, gridSize=10, charging=4, water=3)
    grid_data = world_grid.generate()
    world_grid.place(0, 0, 50)

    
    arr = world_grid.findWater()
    arr1 = world_grid.findFire() 
    source = world_grid.findDrone(50)

    if source and arr:
        dest = arr[0]
        dest1 = arr1[0]
        drone_id = 4
        my_drone = Drone(drone_id, 100, 100, 5, 5, source[0], source[1], 1, world_grid.grid)

        world_grid.place(source[0], source[1], drone_id)
        path = my_drone.route((my_drone.x, my_drone.y), dest)
        path1 = my_drone.route((dest[0],dest[1]),dest1)
        print(path1)

        if path:
            # 1. Setup the Visual Window
            plt.ion() 
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # 0=Green (Forest), 1=Red (Fire), 2=White (Empty), 3=Blue (Water), 4=Black (Drone)
            cmap = mcolors.ListedColormap(['green', 'red', 'white', 'blue', 'black'])
            bounds = [0, 1, 2, 3, 4, 5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            print("Launching Pop-up Slideshow...")
            
            for step in path:
                if not my_drone.can_use_battery():
                    print("Battery Empty!")
                    break

                # Update Logic
                world_grid.place(my_drone.x, my_drone.y, 0)
                my_drone.x, my_drone.y = step
                world_grid.place(my_drone.x, my_drone.y, 4) 
                
                # 2. Update the "Slide"
                ax.clear()
                # Use 'origin=lower' or 'upper' depending on how you want (0,0) displayed
                ax.imshow(world_grid.grid, cmap=cmap, norm=norm)
                
                # Dynamic grid lines based on actual grid size
                size = world_grid.grid_size
                ax.set_xticks(np.arange(-.5, size, 1), minor=True)
                ax.set_yticks(np.arange(-.5, size, 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
                
                plt.title(f"Drone Mission - Step: {step} | Battery: {my_drone.battery}")
                plt.draw()
                plt.pause(0.1) 
                
                my_drone.battery_usage()

            plt.ioff()
            
        else:
            print("Target unreachable!")

        if path1:

            # 0=Green (Forest), 1=Red (Fire), 2=White (Empty), 3=Blue (Water), 4=Black (Drone)
           
            
            for step in path1:
                if not my_drone.can_use_battery():
                    print("Battery Empty!")
                    break

                # Update Logic
                world_grid.place(my_drone.x, my_drone.y, 0)
                my_drone.x, my_drone.y = step
                world_grid.place(my_drone.x, my_drone.y, 4) 
                
                # 2. Update the "Slide"
                ax.clear()
                # Use 'origin=lower' or 'upper' depending on how you want (0,0) displayed
                ax.imshow(world_grid.grid, cmap=cmap, norm=norm)
                
                # Dynamic grid lines based on actual grid size
                size = world_grid.grid_size
                ax.set_xticks(np.arange(-.5, size, 1), minor=True)
                ax.set_yticks(np.arange(-.5, size, 1), minor=True)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
                
                plt.title(f"Drone Mission - Step: {step} | Battery: {my_drone.battery}")
                plt.draw()
                plt.pause(0.1) 
                
                my_drone.battery_usage()

            plt.ioff()
            plt.show() # Keeps the window open at the end
        else:
            print("Target unreachable!")
        
       
    else:
        print("Initialization Error: Drone or Water missing.")