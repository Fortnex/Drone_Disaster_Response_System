from mpi4py import MPI
import time 
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
n = comm.Get_size()
workers = comm.Split(color=0 if rank > 0 else MPI.UNDEFINED, key=rank)
def balancer(personalLoad):
    data = workers.allreduce(personalLoad,op=MPI.SUM)
    splitValue = (data)/(n-1)
    print("Balancing Complete for:  "+str(rank))
    return splitValue-personalLoad
def allocate(x,y):
    for i in range(1,n):
        if i==y:
            comm.send(x,dest=y)
        else:
            comm.send(0,dest=i)
def split(x):
    for i in range(1,n):
        comm.send(x/(n-1),dest=i)
def TakeInput():
    InputUser = input("Enter : "
        " 1. allocate(x,y)"
        " 2. split(x)"
        " 3. End \n")
    return InputUser
if rank == 0:
    print("Hi this is "+str(rank),flush=True)
    while(True):
        time.sleep(0.5)
        inp = TakeInput()
        if inp[0]=='s':
            split(int(inp[inp.find("(")+1 : inp.find(")")]))
        elif inp[0]=='a':
            allocate(int(inp[inp.find("(")+1 : inp.find(",")]),int(inp[inp.find(",")+1 : inp.find(")")]))
        else:
            print("Will end now",flush=True)
            for i in range(1,n):
                comm.send("kill",dest=i)
            break
        time.sleep(2)
else:
    data = 0
    print("Hi this is "+str(rank),flush=True)
    while(True):
        info = comm.recv(source=0)
        if info=="kill":
            break
        data+=info
        print("Hi this is "+str(rank)+" my load is(before Balancing) "+str(data),flush=True)
        balanceValue = balancer(data)
        data+=balanceValue
        print("Hi this is "+str(rank)+" my load is(after Balancing) "+str(data),flush=True)