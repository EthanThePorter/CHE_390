from scipy import optimize
import numpy as np
import pandas as pd

x_e = 0.6           #wt% ethulene glycol
x_w = 0.4           #wt% water
rho_e = 1113        #kg/m^3
rho_w = 998         #kg/m^3
visc_e = 0.0161     #Pa.s
visc_w = 0.0010     #Pa.s

rho = x_e*rho_e + x_w*rho_w
mui = x_e*visc_e + x_w*visc_w
roughness = 5.01*10**(-5)       #m

D=1.0*2.54/100      #pipe diameter (m)

#pipe lenghts (m)

L = np.array([260, 450, 600, 450, 400, 400, 450, 400, 400, 450])

qef = 78/1000/60    # end flow of nodes

m = len(L)              # number of pipes

def FlowRate(q,D):
    
    fp = np.zeros(m)
    
    fp[0] = q[0] - q[1] - q[2]
    fp[1] = q[1] - q[3] - q[4]
    fp[2] = q[3] - q[5] - qef
    fp[3] = q[2] + q[4] - q[6] - q[7]
    fp[4] = q[5] + q[6] - q[8] - qef
    fp[5] = q[7] - q[9] - qef
    fp[6] = q[8] - q[9] - qef

    P = Pressure(q,L)
    
    P12 = P[1]
    P14 = P[2]
    P23 = P[3]
    P24 = P[4]
    P35 = P[5]
    P45 = P[6]
    P46 = P[7]
    P57 = P[8]
    P67 = P[9]
    
    fp[7] = P12 + P24 - P14
    fp[8] = P23 + P35 -P24 - P45
    fp[9] = P45 + P57 - P46 - P67
    
    return fp


def Pressure(q,L):
    
    Re = np.zeros(m)
    f  = np.zeros(m)
    
    Re  =4*rho*q/np.pi/mui/D
    #print(Re)
    
    A = (2.457*np.log(1/((Re)**0.9+0.27*(roughness/D))))
    B = (37530/Re)**16
    
    f = 2*((8/Re)**12+1/(A + B)**(3/2))**(1/12)
    P = 2*f*rho*(4*q/np.pi/D**2)**2*L/D
    #print(f)
    
    return(P)

#initial guesses
q0 = [0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]

#q = optimize.root(FlowRate,q0, method='excitingmixing',tol= 1e-15)
q = optimize.fsolve(FlowRate,q0,args=(D), xtol= 1e-8)

print(q)

DP = Pressure(q,L)/1000     #pressure drop (kPa)

Results1 = np.zeros((m,3))
Results1[:,0] = L[:]
Results1[:,1] = q[:]
Results1[:,2] = DP[:]

Results1 = np.round(Results1,5)
aa = Results1.shape
print(aa)
index = np.arange(1,m+1)
pd.set_option("display.max_rows", None, "display.max_columns", None)
columns = ['Pipe length (m)' , 'Flow rate (m3/s)', 'Pressure drop (kPa)']
Results11=pd.DataFrame(Results1,index,columns)
ab = Results11.shape
print(Results11)

