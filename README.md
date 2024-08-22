# DamBreak-ML
#Numerical Model of 1D dam break
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:53:03 2023

@author: SB106
"""



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'darkgrid')
import timeit
start = timeit.default_timer()

#Your statements here
#k=9 for 22.5-14
  
h=2
us=15.2
ds=6
L=1072
x = np.arange(0,L+h,h)
# S = 0.0008
S = 0.00
N=0.0
# fk=7.03
k=[]
mm=[]
tt=0
check=0
b=145
ss=0.0000001 #side slope
kk=5

g=981
ft=5
tg=[]
terminate=0
termi=0
value=0


Cn=0.5
# bc = [5.79]
# ic = [5.79]
n= len(x)
m=5000

Y= np.zeros((n,m))
Y1= np.zeros((n,m)) #Y*
Y2= np.zeros((n,m))# Y**
V= np.zeros((n,m)) 
D= np.zeros((n,m)) #HYDRAULIC DEPTH (A/T)
V1= np.zeros((n,m)) # V*
V2= np.zeros((n,m)) #V** 
R1= np.zeros((n,m)) #R*
Sf= np.zeros((n,m)) # SLOPE OF EGL
Sf1= np.zeros((n,m)) # S*
R= np.zeros((n,m)) #HYDRAULIC RADIUS(A/P)
A= np.zeros((n,m)) #HYDRAULIC RADIUS(A/P)
A1= np.zeros((n,m)) #A*
A2= np.zeros((n,m)) #A**
yc= np.zeros((n,m)) #CENTROID FROM FWS
yc1= np.zeros((n,m)) #CENTROID FROM FWS


# yus=3.815

XX=ss*us
tp1=b+2*XX
a11=us*(b+XX)
D11=a11/(tp1)
rup=(us**2+XX**2)**0.5
ru=a11/(2*rup+b)
yus=(us/3)*((2*b+tp1)/(tp1+b))

#-------------------------------------------- ds conditions

XX2=ss*ds
tp2=b+2*XX2
a22=ds*(b+XX2)
D22=a22/(tp2)
rds=(ds**2+XX2**2)**0.5
yds=(ds/3)*((2*b+tp2)/(tp2+b))
rd=a11/(2*rds+b)


Y[0,:]= us
Y1[0,:]= us
Y2[0,:]=us


D[0,:]= D11
Y[:,0] = ds
Y1[0,:]= us
Y2[:,0] = ds
D[:,0] = D22
R1[0,:]=ru
R1[:,0]=rd
R[0,:]=ru
R[:,0]=rd
A[0,:]=a11
A[:,0]=a22
A1[0,:]=a11
A1[:,0]=a22
A2[0,:]=a11
A2[:,0]=a22
yc[:,0]=yds
yc[0,:]=yus
yc1[:,0]=yds
yc1[0,:]=yus

nuy1= np.zeros((n,m))
nuy1[0,:]= us
nuy1[:,0] = ds
nuy2= np.zeros((n,m))
nuy2[0,:]= us
nuy2[:,0] = ds
nuyb1= np.zeros((n,m))
nuyb1[0,:]= us
nuyb1[:,0] = ds
Ynuy= np.zeros((n,m))
Ynuy[0,:]= us
Ynuy[:,0] = ds
nuv= np.zeros((n,m))
nuv[0,:]= 0
nuv[:,0] =0
Vnuv= np.zeros((n,m))
Vnuv[0,:]= 0
Vnuv[:,0] =0

ll=int((0.427*L)/h)
for i in range(0,ll): #over writing the initial condition for upstream.
    Y[i,0] = us
    Y1[i,0] = us
    Y2[i,0] = us
    D[i,0]=D11
    R1[i,0]=rup
    R[i,0]=rup
    A[i,0]=a11
    A1[i,0]=a11
    A2[i,0]=a11
    yc[i,0]=yus
    yc1[i,0]=yus
    nuy1[i,0] = us
    # nuy2[i,0] = us
    # nuy3[i,0] = us
    Ynuy[:,0] = us
    
    
V[:,0] = 0
V1[:,0]= 0
V1[n-1,:]= 0
V2[:,0] = 0
V2[n-1,:]= 0

def nu1(i,j): #nui
    nuy1[i,j]=abs(Y[i+1,j]-2*Y[i,j]+Y[i-1,j])/(abs(Y[i+1,j])+2*abs(Y[i,j])+abs(Y[i-1,j]))
    return(nuy1[i,j])
def nu2(i,j): #nui
    i=i+1
    nuy1[i,j]=abs(Y[i+1,j]-2*Y[i,j]+Y[i-1,j])/(abs(Y[i+1,j])+2*abs(Y[i,j])+abs(Y[i-1,j]))
    return(nuy1[i,j])
def nu3(i,j): #nui
    i=i-1
    nuy1[i,j]=abs(Y[i+1,j]-2*Y[i,j]+Y[i-1,j])/(abs(Y[i+1,j])+2*abs(Y[i,j])+abs(Y[i-1,j]))
    return(nuy1[i,j])
def nub1(i,j):
    nuy1[i,j]= (abs(Y[i+1,j]-Y[i,j]))/(abs(Y[i,j]+abs(Y[i+1,j])))
    return(nuy1[i,j])
def nub2(i,j):
    nuy1[i,j]= (abs(Y[i,j]-Y[i-1,j]))/(abs(Y[i,j]+abs(Y[i-1,j])))
    return(nuy1[i,j])
        




def printt(Y,cnt,tt):
    
    fig, ax = plt.subplots()
    ax.plot(Y[:, cnt],label="Time = {}".format(np.round(tt,2)))

    # ax.text(40, 6,13, "Time = {}".format(np.round(tt)), fontsize=15   ,fontweight = 'bold',
            # fontname = 'Times New Roman', verticalalignment='bottom')
    ax.set_title("MacCormak")
    ax.set_ylabel("Depth of water")
    ax.set_xlabel("distance x 100m")
    # ax.plt.title(pad='Time = {}'.format(np.round(tt))
    plt.grid(True)
    plt.legend()
    
    # fig, vx = plt.subplots()
    # vx.plot(V[:, cnt], label="Time = {}".format(np.round(tt,2)))

    # # ax.set_ylabel("Depth of water")
    # # ax.set_xlabel("distance")
    # vx.set_title("MaCcorack")
    # vx.set_ylabel("velocity")
    # vx.set_xlabel("distance x 100m")
    # plt.grid(True)
    # plt.legend()
    # # # plt.savefig("shiva_hm.png")
    # plt.show()
    




for j in range(1,m):
    if (tt<ft):
        dummy=[]
        k=list()

        for i in range(0,n): 
            dummy=h*Cn/((V[i,j-1]+(g*D[i,j-1])**0.5)) #HYDRAULIC DEPTH
            k.append(dummy)
            fk=np.amin(k)  
        for i in range(0,n): # Reservoir node
            if(i==0):
                V[i,j]=0
                Y[i,j]=Y[i+1,j-1]-V[i+1,j-1]/((g/Y[i+1,j-1])**0.5)
                A[i,j]=(b+ss*Y[i,j])*Y[i,j]
                D[i,j]= (Y[i,j]*(b+ss*Y[i,j]))/(b+2*ss*Y[i,j])

            if ((i>0) and (i<=n-2)): # Interior nodes

                A1[i,j] = A[i,j-1]-(fk/h)*(V[i,j-1]*A[i,j-1]-V[i-1,j-1]*A[i-1,j-1])
                Y1[i,j] = (-b+((b**2)+4*ss*A1[i,j])**0.5)/(2*ss)
    #  print(j)
    #  print(i)
    #  print(A1[i,j])
                yc[i,j-1] = ((Y[i,j-1]/3))*(2*b+b+2*ss*Y[i,j-1])/(b+(b+2*ss*Y[i,j-1]))
                yc[i-1,j-1] = (Y[i-1,j-1]/3)*(2*b+b+2*ss*Y[i-1,j-1])/(b+(b+2*ss*Y[i-1,j-1]))
                yc[i+1,j-1] = (Y[i+1,j-1]/3)*(2*b+b+2*ss*Y[i+1,j-1])/(b+(b+2*ss*Y[i+1,j-1]))
                V1[i,j] = (1/A1[i,j])*((V[i,j-1]*A[i,j-1])-(fk/h)*((V[i,j-1]**2)*A[i,j-1]+g*A[i,j-1]*yc[i,j-1]-(V[i-1,j-1]**2)*A[i-1,j-1]-g*A[i-1,j-1]*yc[i-1,j-1])+g*A[i,j-1]*(S-Sf[i,j-1])*fk)
    #  print(V1[i,j])
     #Corrector

                A1[i+1,j] = A[i+1,j-1]-(fk/h)*(V[i+1,j-1]*A[i+1,j-1]-V[i,j-1]*A[i,j-1])
                Y1[i+1,j] = (-b+((b**2)+4*ss*A1[i+1,j])**0.5)/(2*ss)
    #  print(Y1[i,j])
                V1[i+1,j] = (1/A1[i+1,j])*((V[i+1,j-1]*A[i+1,j-1])-(fk/h)*((V[i+1,j-1]**2)*A[i+1,j-1]+g*A[i+1,j-1]*yc[i+1,j-1]-(V[i,j-1]**2)*A[i,j-1]-g*A[i,j-1]*yc[i,j-1])+(g*A[i+1,j-1]*(S-Sf[i+1,j-1])*fk))
                
                A2[i,j] = A[i,j-1]-(fk/h)*(V1[i+1,j]*A1[i+1,j]-V1[i,j]*A1[i,j])
                Y2[i,j] =  (-b+(((b**2)+4*ss*A2[i,j])**0.5))/(2*ss)
                
                R1[i,j]= ((b+ss*Y1[i,j])*Y1[i,j])/(b+2*(((Y1[i,j])**2+(1.5*Y1[i,j])**2)**0.5))
                Sf1[i,j]= ((N**2)*(V1[i,j]**2))/(R1[i,j]**(4/3))
                yc1[i,j]= (Y1[i,j]/3)*((2*b+(b+2*ss*Y1[i,j]))/(b+b+2*ss*Y1[i,j]))
                yc1[i+1,j]= ((Y1[i+1,j]/3))*((2*b+(b+2*ss*Y1[i+1,j]))/(b+(b+2*ss*Y1[i+1,j])))
                V2[i,j] = (1/A2[i,j])*(V[i,j-1]*A[i,j-1]-(fk/h)*((V1[i+1,j]**2)*A1[i+1,j]+g*A1[i+1,j]*yc1[i+1,j]-(V1[i,j]**2)*A1[i,j]-g*A1[i,j]*yc1[i,j])+(g*A1[i,j]*(S-Sf1[i,j])*fk))
                Y[i,j]=0.5*(Y1[i,j]+Y2[i,j])
                V[i,j]=0.5*(V1[i,j]+V2[i,j])
                D[i,j]= (Y[i,j]*(b+ss*Y[i,j]))/(b+2*ss*Y[i,j]) #HYDRAULIC DEPTH (A/T)
                R[i,j]= ((b+ss*Y[i,j])*Y[i,j])/(b+2*(((Y[i,j])**2+(ss*Y[i,j])**2)**0.5)) # (A/P)
                Sf[i,j]= ((N**2)*(V[i,j]**2))/(R[i,j]**(4/3))
                A[i,j]= (b+ss*Y[i,j])*Y[i,j]
    #  A[i+1,j]=(b+ss*Y[])
            if ((i==n-1) and (j>=1)):#Sluice Gate
                V[i,j]=0
                Y[i,j]=Y[i-1,j-1]+(g**(0.5))*(Y[i-1,j-1]**(0.5))*(S-Sf[i-1,j-1])*fk+V[i-1,j-1]*((Y[i-1,j-1]/g)**0.5)
                A[i,j]=(b+ss*Y[i,j])*Y[i,j]
                D[i,j]= (Y[i,j]*(b+ss*Y[i,j]))/(b+2*ss*Y[i,j])
        # for i in range(0,n): # Reservoir node
        #     if (i==0):
        #         Y[i,j]=Y[i+1,j]
        #         V[i,j]=0
        #  print(i)
        #  print(Y[i,j])

        if(tt<ft):
            for i in range(0,n):
              if(i==1):
                  nuf1= kk*(max(nu1(i,j),nu2(i,j)))
                  nuf2=kk*(max(nu1(i,j),nub1(i,j)))
                  Y[i,j]=Y[i,j]+nuf1*(Y[i+1,j]-Y[i,j])-nuf2*(Y[i,j]-Y[i-1,j])
                  V[i,j]=V[i,j]+nuf1*(V[i+1,j]-V[i,j])-nuf2*(V[i,j]-V[i-1,j])
        #         nuy[i,j]= (abs(Y[i+1,j]-Y[i,j]))/(abs(Y[i,j]+abs(Y[i+1,j])))
        #         Ynuy[i,j]= Y[i,j]+nuy[i,j]*(Y[i+1,j]-Y[i,j])
              if ((i>=2) and (i<n-2)):   
        #         nuy[i,j]=abs(Y[i+1,j]-2*Y[i,j]+Y[i-1,j])/(abs(Y[i+1,j])+2*abs(Y[i,j])+abs(Y[i-1,j]))
                nuf1= kk*(max(nu1(i,j),nu2(i,j)))
                nuf2=kk*(max(nu1(i,j),nu3(i,j)))
                Y[i,j]=Y[i,j]+nuf1*(Y[i+1,j]-Y[i,j])-nuf2*(Y[i,j]-Y[i-1,j])
                V[i,j]=V[i,j]+nuf1*(V[i+1,j]-V[i,j])-nuf2*(V[i,j]-V[i-1,j])

              if(i==n-2):
                  nuf1= kk*(max(nu1(i,j),nub2(i,j)))
                  nuf2=kk*(max(nu1(i,j),nu3(i,j)))
                  Y[i,j]=Y[i,j]+nuf1*(Y[i+1,j]-Y[i,j])-nuf2*(Y[i,j]-Y[i-1,j])
                  V[i,j]=V[i,j]+nuf1*(V[i+1,j]-V[i,j])-nuf2*(V[i,j]-V[i-1,j])
                  
              for i in range(0,n):
                  if i==0:
                      Y[i,j]=Y[i+1,j]
                  if i==n-1:
                      Y[i,j]=Y[i-1,j]
                  
        mm.append(fk)
        tt= np.sum(mm)
        # printt(Y,j,tt)
        value=Y[0,0]-Y[0,j-1]

        tg.append(tt)
        cnt=len(mm)
  
ap=Y[int(0.744*L/h),:]
ap1=Y[int(0.8*L/h),:]

  # mm1.append(ap)
  # if (m % 5 ==0):
  #  printt(Y,cnt,tt)
fig, yx = plt.subplots()
yx.plot(Y[int(n-1),:], label="5 km")
yx.plot(Y[int((n-1)/2),:], label="mid section")
# yx.plot(Y[int(0.8*(n-1)),:], label="8 m")
yx.plot(Y[int(0.8*(n-1)),:], label="8 m")
# yx.plot(Y[10,:], label="1 km")
yx.plot(Y[0,:], label="at reservoir km")
plt.xlim(-1,250)
# plt.ylim(4.5,7.5)
yx.set_title("MaCcormack")
yx.set_ylabel("Depth of water")
yx.set_xlabel("time")
plt.grid(True)
plt.legend()    





app=ap[ap>0]
app1=ap1[ap1>0]
tg = np.insert(tg, 0, 0)
cc=len(app)
for i in range(0,cc):
    if terminate==0:
        math=app[1+i]-app[i]

        if math>0.005:
            wave=tg[i]
            terminate = terminate +1
for i in range(0,cc):
    if (termi==0):
        mmm1=app1[1+i]-app1[i]
        if mmm1>0.005:
            wave1=tg[i]
            termi = termi+1     
wspeed= 60/(wave1-wave)        
print(wspeed)
# m1=app1[i+1]-app1[i]     
fig, yx = plt.subplots()
yx.plot(tg,app)

# plt.xlim(-1,250)
# plt.ylim(4.5,7.5)
yx.set_title("MaCcormack")
yx.set_ylabel("Depth of water")
yx.set_xlabel("time")
plt.grid(True)
plt.legend()    
# app=ap[ap>0]










# # plt.savefig("shiva_hm.png")
# plt.show()
# fig, vx = plt.subplots()
# vx.plot(V[:, 0], label="0 S")
# vx.plot(V[:, 39], label="500 S")
# vx.plot(V[:, 77], label="1000 S")
# # vx.plot(V[:, 111], label="1500 S")
# vx.plot(V[:, 143], label="2000 S")
# vx.plot(V[:, 207], label="3000 S")
# ax.set_title("Lax Diffusion")
# ax.set_ylabel("Depth of water")
# ax.set_xlabel("distance")
# vx.set_title("Lax Diffusion")
# vx.set_ylabel("velocity")
# vx.set_xlabel("distance x 100m")
# plt.grid(True)
# plt.legend()
# # # plt.savefig("shiva_hm.png")
# plt.show()


stop = timeit.default_timer()

print('Time: ', stop - start)
