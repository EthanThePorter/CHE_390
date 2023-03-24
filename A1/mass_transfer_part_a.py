from scipy.integrate import solve_ivp,solve_bvp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np



#part a is in this file 
#-------------------------------------------

#functions first!

#DEs function
def dfdx(x,f):

	#DE for the mass transfer 

	k=5e-6#/s
	D=1.5e-6#cm^2/s

	#Dd^2A/dx^2-kA=0 
	#f=[A ,dA/dx]
	#dfdx=[dAdx,k/D A]
	dfdx1=f[1]
	dfdx2=k/D*f[0]
	
	return np.array([dfdx1,dfdx2])

#residual function for fsolve
def resid(u_guess):
	#residual function needed for the shooting method
	#since this is a linear DE, I could have used the interpolation method too

	sol=solve_ivp(dfdx,[0,4],[0.1,u_guess[0]])#solve de based on guess

	res=sol.y[0,-1] 
	#we want the concentration at the end to be zero 
	#end of the tube so col=-1, concentration=0 so row 0				
	return res


#function for solve_bvp
def bc(ca,cb):

	#bc function for the bvp4c method

	#bcs are C(0)=0.1; C(end)=0
	#both are concentration and not the derivative so both are bc of type [0]
	return np.array([ca[0]-0.1,cb[0]])


#since the derivative is zero at the end of the tube, we should expect to
#see the same concentration profile and plots from part b as well! 
#lets see if this is the case!

if __name__=="__main__":
	#shooting method 
	uguess=0.1#guess the initial value for the derivative 

	ucorrect=fsolve(resid,uguess)

	sol_c=solve_ivp(dfdx,[0,4],[0.1,ucorrect[0]],t_eval=np.linspace(0,4,100))
	
	#bvp solver method
	x_vals=np.linspace(0,4,100)
	sol_bv=solve_bvp(dfdx,bc,x_vals,np.zeros((2,x_vals.size)))
	
	#using twinx for two y axis!see https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.twinx.html
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('x (cm)')
	ax1.set_ylabel('c (M) ', color=color)
	ax1.plot(sol_c.t, sol_c.y[0,:], color=color,linewidth=2,label='shooting method')
	ax1.plot(sol_bv.x,sol_bv.y[0],'k--',label='solve_bvp')
	ax1.tick_params(axis='y', labelcolor=color)
	ax1.legend(loc=4)
	
	
	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('dc/dx', color=color)  # we already handled the x-label with ax1
	ax2.plot(sol_c.t, sol_c.y[1,:], color=color,linewidth=2,label='shooting method')
	ax2.plot(sol_bv.x,sol_bv.y[1],'k--',label='solve_bvp')
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.legend(loc=1)
	
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()
 
