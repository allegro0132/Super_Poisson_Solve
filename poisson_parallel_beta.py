import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import numpy as np

Pi = math.pi
# Poisson solver for parallel-plate capacitor


# Grid spacing
h = 0.001
# SOR modules
Ni = 10   #total iterations
n_core = 12
#ndp = 2
ndp = int( np.sqrt(2*n_core) ) # the number of partition for axis

# Calculation
#Vs = copy.deepcopy(Vs)
    # Function definition
def SORkernel(j,i,Vji,BCji,RHOji,omega):
    EPS_0 = 8.854e-12
    Nyi = np.size(BCji,0)
    Nxi = np.size(BCji,1)
    # b-matrix for SOR 
    b = RHOji*pow(h,2)/EPS_0 #surface charge density
    maxR = 0
    for jj in range(Nyi):
        for ii in range(Nxi):
            nn = jj +1
            mm = ii +1
            # Neumann boundary
            if (j == 0 and jj == 0):
                Vji[nn,mm] = Vji[nn+1,mm]
                continue
            if (j == ndp -1 and jj == Nyi -1):
                Vji[nn,mm] = Vji[nn-1,mm]
                continue
            if (i == 0 and ii == 0):
                Vji[nn,mm] = Vji[nn,mm+1]
                continue
            if (i == ndp -1 and ii == Nxi -1):
                Vji[nn,mm] = Vji[nn,mm-1]
                continue
            if (BCji[jj,ii] == 1): # Dirichlet boundary
                continue
            # none of the boundary conditions being trigged, then calculate the residual using 5-point star
            R = 0.25*( Vji[nn+1,mm]+Vji[nn-1,mm]+Vji[nn,mm+1]+Vji[nn,mm-1]+b[jj,ii] ) - Vji[nn,mm]
            Vji[nn,mm] = Vji[nn,mm] + omega*R  # update
            maxR = max(maxR,abs(R))
    return maxR

def Process_TypeA(j,i,qs,Vji,BCji,RHOji,omega):
    Vji = np.insert(Vji,0,qs[j][i][0].get(),axis=0) #Top Add-on
    if (j == ndp - 1):
        Vji = np.insert(Vji,-1,qs[0][i][0].get(),axis=0)
    else:
        Vji = np.insert(Vji,-1,qs[j+1][i][0].get(),axis=0) # Bottom Add-on
    Vji = np.insert(Vji,0,np.concatenate(([0],qs[j][i][1].get(),[0])),axis=1) # Left Add-on
    if (i == ndp - 1):
        Vji = np.insert(Vji,-1,np.concatenate(([0],qs[j][0][1].get(),[0])),axis=1)
    else:
        Vji = np.insert(Vji,-1,np.concatenate(([0],qs[j][i+1][1].get(),[0])),axis=1) # Right Add-on
    maxR = SORkernel(j,i,Vji,BCji,RHOji,omega)
    print('TypeA task %s, %s (%s) finished...maxR=%s' % (j,i, os.getpid(),maxR))
    Vji = np.delete(Vji,0,axis=0)
    Vji = np.delete(Vji,-1,axis=0)
    Vji = np.delete(Vji,0,axis=1)
    Vji = np.delete(Vji,-1,axis=1)
    # set queue for calculations of TypeB
    qs[j][i][2].put(Vji[-1,:])
    qs[j][i][3].put(Vji[:,-1])
    qs[j-1][i][2].put(Vji[0,:])
    qs[j][i-1][3].put(Vji[:,0])
    return [ maxR,Vji ]

def Process_TypeB(j,i,qs,Vji,BCji,RHOji,omega):
    # Pulse when scanning to TypeB blocks. If the nearby TypeA's calculation is completed, the process will continue.
    #while qs[j-1][i][2].empty() or qs[j][i][2].empty() or qs[j][i-1][3].empty() or qs[j][i][3].empty():
    Vji = np.insert(Vji,0,qs[j-1][i][2].get(True),axis=0) #Top Add-on
    Vji = np.insert(Vji,-1,qs[j][i][2].get(True),axis=0) # Bottom Add-on
    Vji = np.insert(Vji,0,np.concatenate(([0],qs[j][i-1][3].get(True),[0])),axis=1) # Left Add-on
    Vji = np.insert(Vji,-1,np.concatenate(([0],qs[j][i][3].get(True),[0])),axis=1) # Right Add-on
    maxR = SORkernel(j,i,Vji,BCji,RHOji,omega)
    print('Type-B task %s, %s (%s)finished...maxR=%s !' % (j,i, os.getpid(),maxR))
    Vji = np.delete(Vji,0,axis=0)
    Vji = np.delete(Vji,-1,axis=0)
    Vji = np.delete(Vji,0,axis=1)
    Vji = np.delete(Vji,-1,axis=1)
    # set queue for calculations of TypeA
    qs[j][i][0].put(Vji[0,:])
    qs[j][i][1].put(Vji[:,0])
    if (j == ndp-1):
        qs[0][i][0].put(Vji[-1,:])
    else:
        qs[j+1][i][0].put(Vji[-1,:])
    if (i == ndp-1):
        qs[j][0][1].put(Vji[:,-1])
    else:
        qs[j][i+1][1].put(Vji[:,-1])
    return [ maxR,Vji ]

# Main program
maxError = 1e-6
# [ta,tb] = V.shape
# intial processes pool
if __name__ == '__main__' :
    mgr = mp.Manager()
    # Simulation parameters
    Lx = 2
    Ly = 1
    x = np.arange(-1/2*Lx, 1/2*Lx, h)
    y = np.arange(-1/2*Ly, 1/2*Ly, h)
    # build axes matrix
    X,Y = np.meshgrid(x,-y)

    # number of voltage samples
    Nx = x.size
    Ny = y.size

    # Parallel plate parameters
    Vtop = 1
    Vbot = -1
    L = 1
    W = 0.2
    x1 = -L/2
    x2 = L/2
    y1 = -W/2
    y2 = W/2

    # SOR iteration parameter
    t = math.cos(Pi/Nx) + math.cos(Pi/Ny)
    omega = ( 8-math.sqrt(64-16*pow(t,2)) )/pow(t,2)

    #####################################

    # Initialize simulation parameters matrices

    #####################################

    #  Initialize matrices.
    V   = np.zeros((Ny,Nx))         #  Potential function (V).
    BC  = np.zeros((Ny,Nx))         #  Boundary conditions (boolean).
    RHO = np.zeros((Ny,Nx))         #  Charge distrubution (C/m^2).

    # simulation boundary
    BC[0,:]  = 1
    BC[-1,:] = 1
    BC[:,0]  = 1
    BC[:,-1] = 1

    # Pick up points on plate
    bP=(X <= x2)*(X >= x1)*(Y <= y1+h/2)*(Y >= y1-h/2)  #Bottom Plate
    V[bP] = Vbot
    BC[bP] = 1
    tP=(X <= x2)*(X >= x1)*(Y <= y2+h/2)*(Y >= y2-h/2)  #Top Plate
    V[tP] = Vtop
    BC[tP] = 1

    # partitioned matrix
    
    Nyp = int( Ny/ndp )
    Nxp = int( Nx/ndp )
    Vs = [[]for j in range(ndp)]
    BCs = [[]for j in range(ndp)]
    RHOs = [[]for j in range(ndp)]
    for j in range(ndp):
        for i in range(ndp):
            # Using bigger matrix for boundary partition.
            if (i == ndp - 1):
                if (j == ndp -1 ):
                    Vs[j].append(V[j*Nyp:, i*Nxp:])
                    BCs[j].append(BC[j*Nyp:, i*Nxp:])
                    RHOs[j].append(RHO[j*Nyp:, i*Nxp:])
                    continue
                Vs[j].append(V[j*Nyp:(j+1)*Nyp, i*Nxp:])
                BCs[j].append(BC[j*Nyp:(j+1)*Nyp, i*Nxp:])
                RHOs[j].append(RHO[j*Nyp:(j+1)*Nyp, i*Nxp:])
                continue
            if (j == ndp - 1):
                Vs[j].append(V[j*Nyp:, i*Nxp:(i+1)*Nxp])
                BCs[j].append(BC[j*Nyp:, i*Nxp:(i+1)*Nxp])
                RHOs[j].append(RHO[j*Nyp:, i*Nxp:(i+1)*Nxp])
                continue
            Vs[j].append(V[j*Nyp:(j+1)*Nyp, i*Nxp:(i+1)*Nxp])
            BCs[j].append(BC[j*Nyp:(j+1)*Nyp, i*Nxp:(i+1)*Nxp])
            RHOs[j].append(RHO[j*Nyp:(j+1)*Nyp, i*Nxp:(i+1)*Nxp])

    # set communication channel between processes.
    qs = [ [[]for i in range(ndp)]for j in range(ndp) ]
    #Vs = copy.deepcopy(Vs)
    flag = 1 # 1 for typeA, -1 for TypeB
    for j in range(ndp): #initial queue for iteraion
        flag = -flag
        for i in range(ndp):
            flag = -flag
            for k in range(4):
                qs[j][i].append(mgr.Queue(1))
            if (flag == 1):
                qs[j][i][0].put(Vs[j-1][i][-1,:]) # if i,j is equal to 0, use Vs[-1] as temporary value, and we will drop it by boundary condition.
                qs[j][i][1].put(Vs[j][i-1][:,-1])
            else:
                qs[j][i][0].put(Vs[j][i][0,:])
                qs[j][i][1].put(Vs[j][i][:,0])
    for k in range(Ni):
        p = mp.Pool(4*n_core)
        result = [] # [ [maxR,V[j][i]], ... ]
        flag = 1
        for j in range(ndp):
            flag = -flag
            for i in range(ndp):
                flag = -flag
                if (flag == 1):
                    result.append( p.apply_async(Process_TypeA,args=(j,i,qs,Vs[j][i],BCs[j][i],RHOs[j][i],omega,)) )
                else:
                    result.append( p.apply_async(Process_TypeB,args=(j,i,qs,Vs[j][i],BCs[j][i],RHOs[j][i],omega,)) )
        p.close()
        p.join()
        maxRt = 0
        for j in range(ndp):
            for i in range(ndp):
                [maxR,Vs[j][i]] = result[ndp*j+i].get()
                maxRt = max(maxRt,maxR)
        print("Iteration:"+ str(k)+"  MaxError:"+str(maxRt)) # log
        if (maxRt < maxError):
            break
    
    # Concatenate matrix
    for j in range(ndp):
        for i in range(ndp):
            if i == 0:
                rVj = Vs[j][i]
            else:
                rVj = np.concatenate((rVj,Vs[j][i]),axis=1)
        if j == 0:
            rV = rVj
        else:
            rV = np.concatenate((rV,rVj),axis=0)
    



    # Render potential V's image
    plt.pcolor(X,Y,rV)
    plt.show()

    # Numerical gradient
    dh = 1 # Default step for numerical gradient

    tEx = np.zeros((Ny, Nx-1))
    tEy = np.zeros((Ny-1, Nx))
    Ex = np.zeros((Ny-1, Nx-1))
    Ey = np.zeros((Ny-1, Nx-1))

    for i in range(Ny):
        tEx[i,:] = (1/h)*( rV[i,1:]-rV[i,:-1] )
    for j in range(Nx):
        tEy[:,j] = (1/h)*( rV[1:,j]-rV[:-1,j] )
        if (j == Nx-1):
            continue
        Ex[:,j] = 0.5*( tEx[:-1,j]+tEx[1:,j] ) # Average along y-axis
    for i in range(Ny-1):
        Ey[i,:] = 0.5*( tEy[i,:-1]+tEy[i,1:] ) #Average along x-axis

    Emag = np.sqrt(pow(Ex,2)+pow(Ey,2))

#    plt.pcolor(X,Y,Emag)
#    plt.show()