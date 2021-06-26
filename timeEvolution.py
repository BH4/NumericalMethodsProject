#import mpmath
import numpy as np

import matplotlib
IREMEMBEREDTOCHANGEIT=False
#IREMEMBEREDTOCHANGEIT=True
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.animation as manimation
matplotlib.rcParams.update({'font.size': 22})

from EigValAndFunc import changeOfVariables,lookAtTheFunction,hamiltonian,residuals,usingHO
from SpectralCode import SpectralChebyshevExterior
#import scipy

###################################################################################
#Initial function
###################################################################################

def gaussianEnergy(b,c):
    return (3*c**6+12*b**2*c**4+4*b**4*c**2+4)/(16*c**2)

def FWHM(c):
    return 2*np.sqrt(2*np.log(2))*c

#E30,~xmax/10,0
#tol allows the energy to be lower than the minimum energy by that amount.
#does not transform continuously with its inputs. return type 1 is drastically different (much wider) than return type 2,3
def initalFunction(minE,maxWidth,offset,tol=.1,output=True):
    minE=float(minE)
    maxWidth=float(maxWidth)
    offset=float(offset)

    #polynomial I want roots of where the roots are values of c^2
    p=[3.0,12.0*offset**2,4.0*offset**4-16.0*minE,4.0]
    r=np.roots(p)
    posCVals=[np.sqrt(x) for x in r if x>0 and FWHM(np.sqrt(x))<maxWidth]
    if len(posCVals)==0:
        E=gaussianEnergy(offset,maxWidth)
        if E>minE:
            b=offset
            c=maxWidth
            if output:
                print("method",1)
        else:
            print("No gaussian satisfying the criteria found.")
            return None
    else:
        b=offset
        posdE=[gaussianEnergy(b,c)-minE for c in posCVals]
        positive=[x for x in posdE if x>=0]
        alldE=[x for x in posdE if x>=-1*tol]

        if len(positive)>0:
            ind=positive.index(min(positive))
            if output:
                print("method",2)
        elif len(alldE)>0:
            ind=alldE.index(max(alldE))
            if output:
                print("method",3)
        else:
            print("No gaussian satisfying the criteria found.")
            return None

        c=posCVals[ind]

    if output:
        print('b='+str(b)+' ,c='+str(c)+' ,energy='+str(gaussianEnergy(b,c)))

    if isinstance(b, complex) or isinstance(c, complex):
        print("complex nums...")
        return None
    gaussian=lambda x:(1/np.sqrt(np.sqrt(np.pi)*c))*np.exp(-1*(x-b)**2/(2.0*c**2))


    return gaussian


###################################################################################
#Helpers
###################################################################################

def mapFiniteToInf(xi,L):
    if xi==1:
        return np.inf
    if xi==-1:
        return -1*np.inf
    return L*xi/np.sqrt(1-xi**2)

def mapFiniteToInfArray(xgrid,L):
    ygrid=[mapFiniteToInf(x,L) for x in xgrid]
    return ygrid

#nCutoff mostly insensitive to changes in mult
#tol=10**-3,L=5,mult=5 (2 past 500)
#nCutoffPreComp={100:4,200:32,300:63,400:96,500:129,600:165,700:195,800:227,900:260,1000:289}
#after switching to eigenvalue subtraction
#tol=10**-8,L=5,mult=5 (smaller mult past 500, 2000 is a guess) (2 for the ones ending in 50)
nCutoffPreComp={100:11,150:28,200:47,250:65,300:83,350:102,400:120,450:138,500:157,550:175,600:192,650:209,700:227,750:243,800:260,850:277,900:294,950:311,1000:327,1500:485,2000:654}

def preComputeCutoff(N,mult,L,tol):
    Nprime=mult*N
    rList,_=residuals(L,N,Nprime)
    rListMax=[max(abs(np.array(x))) for x in rList]

    #print rListMax
    nCutoff=0
    while rListMax[nCutoff]<=tol:
        nCutoff+=1

    print(nCutoff)

#attempts to quickly print the ith energy level for each i in inds
#inds does not need to be an array
def printEnergyLevels(inds,getRidOf2and4=False):
    if usingHO:
        L=6.2
    else:
        L=5.0

    try:
        lenInds=len(inds)
        if lenInds==0:
            return None
        if max(inds)>=nCutoffPreComp[1000]:
            print("I refuse to find energies that large.")
        inds=[x for x in inds if x<nCutoffPreComp[1000]]
    except:
        if inds>nCutoffPreComp[1000]:
            print("I refuse to find energies that large.")
            return None
        lenInds=1
        inds=[inds]

    #smallest N i can get away with that hopefully wont give a wrong answer.
    m=max(inds)
    keyList=list(nCutoffPreComp.keys())
    j=0
    key=keyList[j]
    while nCutoffPreComp[key]<=m:
        j+=1
        key=keyList[j]
    N=key

    
    eigen,_,grid,_,_=changeOfVariables(N,L)
    for i in inds:
        if getRidOf2and4:
            print(eigen[i][0]*(2**(4.0/3)))
        else:
            print(eigen[i][0])


def setupSuperpositionOfEigenStates(N,L,inds,output=False):
    

    eigen,card,grid,d,dd=changeOfVariables(N,L)
    ygrid=mapFiniteToInfArray(grid,L)

    #initFVec=(1/np.sqrt(2))*(eigen[i][1]+eigen[j][1])
    initFVec=(1/np.sqrt(len(inds)))*sum([eigen[ii][1] for ii in inds])


    #energy=0.5*(eigen[i][0]+eigen[j][0])
    energy=(1.0/len(inds))*(sum([eigen[ii][0] for ii in inds]))
    if len(inds)==2:
        period=2*np.pi/abs(eigen[inds[0]][0]-eigen[inds[1]][0])
    else:
        period=-1

    if output:
        xqmax=abs(meanPositionPower(initFVec,ygrid,L,1))
        print("energy="+str(energy))
        if period>0:
            print("period="+str(period))
        print('"classical period"='+str(4*np.sqrt(2)*1.31103/xqmax))

    return eigen,card,grid,d,dd,initFVec,period,energy

#assumes function is defined from -inf to inf
def funcToVec(func,grid,L):
    vec=[]
    for i,xi in enumerate(grid):
        if i==0 or i==len(grid)-1:
            vec.append(0)
        else:
            yi=L*xi/np.sqrt(1-xi**2)
            vec.append(func(yi))
    return vec

###################################################################################
#Analyze position and momentum
###################################################################################

#approximately finds the expected value of position of a state using Chebyshev-gauss quadrature
#finds the mean position on the -inf to inf variable
#assumes grid is given in the same variable.
def meanPositionPower(state,grid,L,power):
    assert grid[-2]>1.0#check to make sure I put in the correct grid.
    N=len(grid)

    total=0
    #takes the values 1,2,...,N-3,N-2   -> no endpoints
    for i in range(1,N-1):
        weight=(np.pi/(N-1))

        yi=grid[i]
        xi=yi/np.sqrt(L**2+yi**2)
        g=(L/(1-xi**2))*(yi**power)*abs(state[i])**2

        total+=weight*g

    return total

#for a gaussian it should be c/sqrt(2)
def rmsPosition(state,grid,L):
    meanX=meanPositionPower(state,grid,L,1)
    meanX2=meanPositionPower(state,grid,L,2)
    return np.sqrt(meanX2-meanX**2)

def meanMomentumPower(state,grid,L,d,dd,power):
    assert power==1 or power==2
    if power==1:
        return meanMomentum1(state,grid,L,d)
    if power==2:
        return meanMomentum2(state,grid,L,d,dd)

def meanMomentum1(state,grid,L,d):
    assert grid[-2]>1.0#check to make sure I put in the correct grid.
    N=len(grid)

    #gridCopy,d,dd,card=SpectralChebyshevExterior(-1,1,len(grid))

    dState=np.dot(d,state)

    total=0
    #takes the values 1,2,...,N-3,N-2   -> no endpoints
    for i in range(1,N-1):
        weight=(np.pi/(N-1))

        yi=grid[i]
        xi=yi/np.sqrt(L**2+yi**2)

        g=np.sqrt(1-xi**2)*np.conj(state[i])*dState[i]

        total+=weight*g


    total=total*(-1j)
    #print total
    assert abs(np.imag(total))<10**(-8)#since p is hermitian any expectation value of any power should be real.

    return np.real(total)

def meanMomentum2(state,grid,L,d,dd):
    assert grid[-2]>1.0#check to make sure I put in the correct grid.
    N=len(grid)

    #gridCopy,d,dd,card=SpectralChebyshevExterior(-1,1,len(grid))

    dState=np.dot(d,state)
    ddState=np.dot(dd,state)

    total=0
    #takes the values 1,2,...,N-3,N-2   -> no endpoints
    for i in range(1,N-1):
        weight=(np.pi/(N-1))

        yi=grid[i]
        xi=yi/np.sqrt(L**2+yi**2)

        g=((1-xi**2)/L)*np.conj(state[i])*(3*xi*dState[i]-(1-xi**2)*ddState[i])

        total+=weight*g


    return np.real(total)

def rmsMomentum(state,grid,L):
    meanP=meanMomentumPower(state,grid,L,1)
    meanP2=meanMomentumPower(state,grid,L,2)
    return np.sqrt(meanP2-meanP**2)



CVNR=6
VALNAMES=["Mean Position","RMS Position","Mean Momentum","RMS Momentum","Energy","Classical Energy"]
def checkValues(vec,ygrid,L,d,dd,r=4,output=False):
    meanX=meanPositionPower(vec,ygrid,L,1)
    meanX2=meanPositionPower(vec,ygrid,L,2)
    rmsX=np.sqrt(meanX2-meanX**2)
    
    meanP=meanMomentumPower(vec,ygrid,L,d,dd,1)
    meanP2=meanMomentumPower(vec,ygrid,L,d,dd,2)
    rmsP=np.sqrt(meanP2-meanP**2)
    
    hu=rmsX*rmsP
    if usingHO:
        energy=(meanP2)/2+(meanX2)/2
        classicalEnergy=meanP**2/2+meanX**2/2
    else:
        meanX4=meanPositionPower(vec,ygrid,L,4)
        energy=(meanP2)/2+(meanX4)/4
        classicalEnergy=meanP**2/2+meanX**4/4

    if output:
        print("The mean position="+str(round(meanX,r))+" and the rms position="+str(round(rmsX,r)))
        print("and the mean Momentum="+str(round(meanP,r))+" and the rms Momentum="+str(round(rmsP,r)))
        print("and Heisenberg's uncertanty(>.5)="+str(round(hu,r))+" and Energy="+str(round(energy,r)))
        print("classical energy?="+str(classicalEnergy))
    return meanX,rmsX,meanP,rmsP,energy,classicalEnergy

def quickCheckValuesW(N,L,func):
    _,_,grid,d,dd=changeOfVariables(N,L)
    fVec=funcToVec(func,grid,L)
    ygrid=mapFiniteToInfArray(grid,L)
    checkValues(fVec,ygrid,L,d,dd,output=True)

###################################################################################
#Time Evolution
###################################################################################

#approximately find the overlap of a state with an eigenvector using Chebyshev-gauss quadrature
#<n|Psi(0)>
#assumes f is the functional form of the initial state which can be evaluated on the entire real line.
#https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
def innerProduct(fVec,eigenstate,grid,L):
    N=len(grid)

    total=0
    #takes the values 1,2,...,N-3,N-2   -> no endpoints
    for i in range(1,N-1):
        weight=(np.pi/(N-1))

        xi=grid[i]
        #yi=L*xi/np.sqrt(1-xi**2)
        g=(L/(1-xi**2))*eigenstate[i]*fVec[i]

        total+=weight*g

    return total

#Time evolve a state f through a time t using energy eigenstates up to nCutoff.
#Will have to determine nCutoff by calculating the residuals of the eigenvectors. If the overlap of previous eigenvectors
#is not getting small by the time the residual tolerance of the eigenvectors is passed then an warning will be printed.
def timeEvolve(fVec,t,L,eigen,grid):
    N=len(grid)

    nCutoff=nCutoffPreComp[N]

    try:
        lenT=len(t)
        if len(t)==0:
            return None
        t=np.array(t)
    except:
        lenT=1
    

    if lenT>1:
        PsiT=np.zeros((len(t),len(grid)),dtype=np.complex128)
    else:
        PsiT=np.zeros(len(grid),dtype=np.complex128)


    lastOverlap=10**5
    almostLastO=10**5
    for i in range(0,nCutoff):
        eigenE,eigenV=eigen[i]


        overlap=innerProduct(fVec,eigenV,grid,L)#<n|f>
        almostLastO=lastOverlap
        lastOverlap=abs(overlap)

        evoOperator=np.exp(-1j*eigenE*t)

        #PsiT=[PsiT[i]+evoOperator*eigenV[i]*overlap for i in xrange(len(grid))]
        if lenT>1:
            PsiT+=np.outer(evoOperator,eigenV)*overlap
        else:
            nextVals=evoOperator*eigenV*overlap
            ind=(len(grid)/2)+1
            PsiT+=nextVals

    tol=10**(-8)#arbitrary 8
    if lastOverlap>tol or almostLastO>tol:
        print("warning: overlap was not small before nCutoff reached")
        print(max(lastOverlap,almostLastO))

    return PsiT


"""
#higher accuracy time evolution
def timeEvolve(fVec,t,L,eigen,grid,card):
    mpmath.mp.dps=50

    N=len(grid)

    nCutoff=nCutoffPreComp[N]

    try:
        lenT=len(t)
        if len(t)==0:
            return None
        t=np.array(t)
    except:
        lenT=1
    

    if lenT>1:
        #PsiT=np.zeros((len(t),len(grid)),dtype=np.complex128)
        PsiT=np.array([[mpmath.mpc(0) for i in xrange(len(grid))] for j in xrange(len(t))])
    else:
        #PsiT=np.zeros(len(grid),dtype=np.complex128)
        PsiT=np.array([mpmath.mpc(0) for i in xrange(len(grid))])


    lastOverlap=10**5
    almostLastO=10**5
    for i in xrange(0,nCutoff):
        eigenE,eigenV=eigen[i]
        eigenV=np.array([mpmath.mpc(x) for x in eigenV])


        overlap=mpmath.mpc(innerProduct(fVec,eigenV,grid,L))#<n|f>
        almostLastO=abs(lastOverlap)
        lastOverlap=abs(overlap)

        if lenT==1:
            evoOperator=mpmath.exp(-1j*eigenE*t)
        else:
            evoOperator=np.array([mpmath.exp(x) for x in -1j*eigenE*t])

        #PsiT=[PsiT[i]+evoOperator*eigenV[i]*overlap for i in xrange(len(grid))]
        if lenT>1:
            PsiT+=np.outer(evoOperator,eigenV)*overlap
        else:
            nextVals=evoOperator*eigenV*overlap
            ind=(len(grid)/2)+1
            print "curr value of random mid:"+str(PsiT[ind])
            print "Adding value: "+str(nextVals[ind])
            print "-"*50
            PsiT+=nextVals

    tol=10**(-8)#arbitrary 8
    if lastOverlap>tol or almostLastO>tol:
        print "warning: overlap was not small before nCutoff reached"
        print max(lastOverlap,almostLastO)

    PsiT=PsiT.astype(np.complex128)
    return PsiT
"""

#assumes functions goes to zero at infinity
#assumes the function is defined on -inf to inf
#N is the largest eigenvector I am willing to calculate. Not all eigenvectors will be used.
def timeEvolveWrapper(f,N,t,L):
    eigen,card,grid,d,dd=changeOfVariables(N,L)

    fVec=funcToVec(f,grid,L)

    fVec=np.array(fVec)

    timeEvoF=timeEvolve(fVec,t,L,eigen,grid)

    return timeEvoF,card,grid,d,dd,fVec

###################################################################################
#Residual
###################################################################################


#first construct the time evolved state then calculate its residual
def timeResidual(initFVec,L,eigen,card,grid,Nprime,t):

    N=len(grid)

    nCutoff=nCutoffPreComp[N]

    finalFVecList=timeEvolve(initFVec,t,L,eigen,grid)

    #bigger stuff
    bigGrid,dB,ddB,_=SpectralChebyshevExterior(-1,1,Nprime)
    bigH=hamiltonian(L,bigGrid,dB,ddB)

    g=lambda x: sum([initFVec[i]*card[i](x) for i in range(len(initFVec))])
    interpInitVec=g(bigGrid)

    interpEigVList=[]
    for i in range(0,nCutoff):
        eigenE,eigenV=eigen[i]
        eigenFunc=lambda x: sum([eigenV[i]*card[i](x) for i in range(len(eigenV))])
        interpEigV=eigenFunc(bigGrid)
        interpEigVList.append(interpEigV)

    RList=[]
    for j,finalFVec in enumerate(finalFVecList):
        print(j)
        f=lambda x: sum([finalFVec[i]*card[i](x) for i in range(len(finalFVec))])
        interpFinalVec=f(bigGrid)

        R=bigH.dot(interpFinalVec)
        for i in range(0,nCutoff):
            eigenE,eigenV=eigen[i]


            interpEigV=interpEigVList[i]

            evoOperator=np.exp(-1j*eigenE*t[j])
            overlap=innerProduct(interpInitVec,interpEigV,bigGrid,L)

            R-=eigenE*evoOperator*interpEigV*overlap
        RList.append(R)

    return RList,bigGrid,interpFinalVec

def timeResidualWrapper(initFunc,L,N,Nprime,t):
    eigen,card,grid,_,_=changeOfVariables(N,L)

    fVec=funcToVec(initFunc,grid,L)

    fVec=np.array(fVec)

    return timeResidual(fVec,L,eigen,card,grid,Nprime,t)

#uses finite difference methods to determine the residual
#write out the time derivative analytically but do the H Psi(x,t) with finite difference?
def finiteDifferenceTimeResidual(initFVec,L,eigen,grid,t):
    N=len(grid)

    #t=np.arange(0,tmax,dt)

    finalFVec=timeEvolve(initFVec,t,L,eigen,grid)


###################################################################################
#show residuals
###################################################################################

def residualOfEigSuperposition(N,L,i,j,t):
    Nprime=N*2
    eigen,card,grid,_,_=changeOfVariables(N,L)
    initFVec=(1/np.sqrt(2))*(eigen[i][1]+eigen[j][1])

    R,bigGrid,interpFinalVec=timeResidual(initFVec,L,eigen,card,grid,Nprime,t)

    y=mapFiniteToInfArray(bigGrid,L)
    plt.plot(y,abs(R))
    plt.xlim(-5,5)
    plt.ylim(0,1)
    plt.show()

def residualOfGaussian(N,L,t,mult,minE,maxWidth,offset):
    initFunc=initalFunction(minE,maxWidth,offset)

    Nprime=N*mult

    R,bigGrid,interpFinalVec=timeResidualWrapper(initFunc,L,N,Nprime,t)

    y=mapFiniteToInfArray(bigGrid,L)
    #y=np.array(y)
    plt.plot(y,abs(R),color='red')
    plt.plot(y,abs(interpFinalVec))
    plt.xlim(-50,50)
    plt.ylim(-1,1)
    plt.show()

def maxResidualOfGaussianPlot(N,L,t,mult,minE,maxWidth,offset):
    initFunc=initalFunction(minE,maxWidth,offset)

    Nprime=N*mult

    R,_,_=timeResidualWrapper(initFunc,L,N,Nprime,t)

    
    maxRs=[]
    for curr in R:
        maxRs.append(max(abs(curr)))

    plt.plot(t,maxRs,color='red')
    plt.title("Maximum Residual as Gaussian is time evolved\n Number of Grid Points="+str(N)+". Larger matrix is "+str(mult)+" times larger.")
    plt.xlabel("Time")
    plt.ylabel("Max Residual")
    plt.show()

###################################################################################
#"Movies"
###################################################################################

#if cardinal funcitons are supplied then the plots will be interpolated functions of 2000 evenly spaced points
def makeMovie(finalFVecs,grid,times,initFVec=None,card=None,d=None,dd=None,printValues=False,showGraph=True):
    if not IREMEMBEREDTOCHANGEIT:
        print("you forgot to change the thing at the top")
        return None

    plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=30)

    ygrid=mapFiniteToInfArray(grid,L)

    if (not showGraph) and (not printValues):
        print("This function has no purpose if you do nothing.")
        return None

    fig=plt.figure()

    dpi=100
    with writer.saving(fig, "writer_test.mp4", dpi):
        for i,finalFVec in enumerate(finalFVecs):
            
            if printValues and (not d is None) and (not dd is None):
                print("At t="+str(round(times[i],8)))
                checkValues(finalFVec,ygrid,L,d,dd,r=5,output=True)
                print('-'*50)
            
            if showGraph:
                if not initFVec is None:
                    plt.plot(ygrid,abs(initFVec)**2)
                if not card is None:
                    finalF=lookAtTheFunction(L,finalFVec,card)
                    y=np.linspace(-5,5,2000)
                    plt.plot(y,abs(finalF(y))**2)
                else:
                    plt.plot(ygrid,abs(finalFVec)**2)
                
                if usingHO:
                    potential=lambda y:(y**2)/2
                else:
                    potential=lambda y:(y**4)/4

                y=np.linspace(-7,7,100)
                l,=plt.plot(y,potential(y))
                plt.xlim(-7,7)
                plt.ylim(0,3)
                #plt.show()
                writer.grab_frame()
                plt.clf()


#interesting vals (1000,5.0,80,.426,5.0)
def GaussianTimeEvoMovie(N,L,minE,maxWidth,offset,totTime,numFrames):
    initF=initalFunction(minE,maxWidth,offset)
    #t=np.linspace(0.0,15*np.pi,16)
    #t=np.linspace(0.0,2*np.pi,15)
    t=np.linspace(0.0,totTime,numFrames)
    finalFVecs,card,grid,d,dd,initFVec=timeEvolveWrapper(initF,N,t,L)

    makeMovie(finalFVecs,grid,t,initFVec=initFVec,card=None,d=d,dd=dd,printValues=False,showGraph=True)


def superPositionEigenFuncMovie(N,L,i,j):
    eigen,_,grid,d,dd,initFVec,period,_=setupSuperpositionOfEigenStates(N,L,[i,j],output=True)

    t=np.linspace(0,period,11)
    finalFVecs=timeEvolve(initFVec,t,L,eigen,grid)

    makeMovie(finalFVecs,grid,t,initFVec=initFVec,card=None,d=d,dd=dd,printValues=True,showGraph=True)
    

def superPositionManyEigenFuncMovie(N,L,i):
    eigen,_,grid,d,dd,initFVec,_,_=setupSuperpositionOfEigenStates(N,L,i,output=True)

    t=np.linspace(0,5,10)
    finalFVecs=timeEvolve(initFVec,t,L,eigen,grid)

    makeMovie(finalFVecs,grid,t,initFVec=initFVec,card=None,d=d,dd=dd,printValues=True,showGraph=True)


def printOverlapWithTime(N,L,i,j):
    eigen,_,grid,_,_=changeOfVariables(N,L)
    if i==-1 or j==-1:
        initF=initalFunction(0.0,.4,6.0)
        fVec=[]
        for i,xi in enumerate(grid):
            if i==0 or i==len(grid)-1:
                fVec.append(0)
            else:
                yi=L*xi/np.sqrt(1-xi**2)
                fVec.append(initF(yi))

        initFVec=np.array(fVec)
        t=np.linspace(0.0,2000.0,20)
    else:
    
        period=2*np.pi/abs(eigen[i][0]-eigen[j][0])
        print(period)
        initFVec=(1/np.sqrt(2))*(eigen[i][1]+eigen[j][1])

        t=np.linspace(0,period,11)
    
    finalFVecs=timeEvolve(initFVec,t,L,eigen,grid)


    y=mapFiniteToInfArray(grid,L)

    originalOverlaps=[]
    for finalFVec in finalFVecs:
        overlaps=[]
        for k in range(0,nCutoffPreComp[len(grid)]):
            eigenE,eigenV = eigen[k]
            o=innerProduct(finalFVec,eigenV,grid,L)
            overlaps.append(o)

        overlaps=np.array(overlaps)
        if len(originalOverlaps)==0:
            originalOverlaps=abs(overlaps)
            print(originalOverlaps)
        else:
            diff=originalOverlaps - abs(overlaps)
            print(max(abs(diff)))
            #print diff
        #print "this should be 1="+str(sum(abs(overlaps)**2))


###################################################################################
#Other checks
###################################################################################

def checkEhrenfest(N,L,initFVec,eigen,grid,d,dd,tinit,dt):

    ygrid=mapFiniteToInfArray(grid,L)

    #t=[tinit+0*dt,tinit+1*dt,tinit+2*dt]
    t0=tinit
    t1=tinit+1*dt
    t2=tinit+2*dt
    t=np.concatenate((t0,t1,t2))
    finalFVecs=timeEvolve(initFVec,t,L,eigen,grid)
    meanX=[]
    meanP=[]
    meanX3=[]
    for i,finalFVec in enumerate(finalFVecs):
        meanXt,_,meanPt,_,energyt,clEnergyt=checkValues(finalFVec,ygrid,L,d,dd,output=False)
        meanX3.append(meanPositionPower(finalFVec,ygrid,L,3))
        meanX.append(meanXt)
        meanP.append(meanPt)
    #print energyt


    WL=len(tinit)
    for i in range(WL):
        #print str((meanX[2]-meanX[0])/(2*dt))+'='+str(meanP[1])
        print(((meanX[i+2*WL]-meanX[i])/(2*dt))-meanP[i+WL])
        if usingHO:
            #print str((meanP[2]-meanP[0])/(2*dt))+'='+str(-1*meanX[1])
            print(((meanP[i+2*WL]-meanP[i])/(2*dt))-(-1*meanX[i+WL]))
        else:
            #print str((meanP[2]-meanP[0])/(2*dt))+'='+str(-1*meanX3[1])
            print(((meanP[i+2*WL]-meanP[i])/(2*dt))-(-1*meanX3[i+WL]))

        print('-'*44)

def checkEhrenfestWrapper(initFunc,L,N,tinit,dt):
    eigen,_,grid,d,dd=changeOfVariables(N,L)

    initFVec=funcToVec(initFunc,grid,L)

    initFVec=np.array(initFVec)

    return checkEhrenfest(N,L,initFVec,eigen,grid,d,dd,tinit,dt)

#see what the period times xqmax is for a superposition of i and j
def checkPeriodTimesXqmax(L,eigen,grid,i,j):
    
    ygrid=mapFiniteToInfArray(grid,L)

    energy=0.5*(eigen[i][0]+eigen[j][0])
    period=2*np.pi/abs(eigen[i][0]-eigen[j][0])

    initFVec=(1/np.sqrt(2))*(eigen[i][1]+eigen[j][1])
    xqmax=abs(meanPositionPower(initFVec,ygrid,L,1))
    print(period*xqmax)
    return period,xqmax

#Plot should be independent of L
def plotGaussianCvsRequiredN(L,tol=10**(-8)):
    N=2000

    points=100

    eigen,card,grid,d,dd=changeOfVariables(N,L)
    print("got eigen")

    cList=np.linspace(.06,1,points)
    requiredNVals=[]
    for j,c in enumerate(cList):
        if j%(points/10)==0 and j>0:
            print(cList[j-1],requiredNVals[-1])
            print(float(j)/points)


        f=lambda x:(1/np.sqrt(np.sqrt(np.pi)*c))*np.exp(-1*(x)**2/(2.0*c**2))
        fVec=funcToVec(f,grid,L)
        fVec=np.array(fVec)

        nCutoff=nCutoffPreComp[N]



        lastOverlap=10**5
        almostLastO=10**5
        i=0
        while (lastOverlap>tol or almostLastO>tol) and i<nCutoff:
            eigenE,eigenV=eigen[i]

            overlap=innerProduct(fVec,eigenV,grid,L)#<n|f>
            almostLastO=lastOverlap
            lastOverlap=abs(overlap)

            i+=1

        if i==nCutoff:
            print("Inaccurate graph...",c)

        requiredNVals.append(i)

    plt.plot(cList,requiredNVals)
    plt.title("Number of eigenvectors to represent Gaussian at the origin.\nTolerance="+str(tol)+" L="+str(L))
    plt.ylabel("Number of eigenvectors")
    plt.xlabel("Standard deviation of Gaussian")
    plt.ylim(0,700)
    plt.show()
                
def testGaussianCvsRequiredN(L,c,tol=10**(-8)):
    N=1000

    eigen,card,grid,d,dd=changeOfVariables(N,L)
    print("got eigen")

    f=lambda x:(1/np.sqrt(np.sqrt(np.pi)*c))*np.exp(-1*(x)**2/(2.0*c**2))
    fVec=funcToVec(f,grid,L)
    fVec=np.array(fVec)

    nCutoff=nCutoffPreComp[N]


    lastOverlap=10**5
    almostLastO=10**5
    i=0
    overlapList=[]
    while (lastOverlap>tol or almostLastO>tol) and i<nCutoff:
        eigenE,eigenV=eigen[i]

        overlap=innerProduct(fVec,eigenV,grid,L)#<n|f>
        almostLastO=lastOverlap
        lastOverlap=abs(overlap)

        overlapList.append(abs(overlap))

        i+=1

    if i==nCutoff:
        print("Inaccurate graph...",c)

    print(i)
    print(overlapList[:i])


###################################################################################
#Presentation/Report graphs
###################################################################################

#Optional inputs are other lines that can be plotted.
#soar=size of allowed region (determined by initial <x>)
#car=classically allowed region
#cp=classical period (based on initial position) ASSUMES initFVec is a real wavefunction! and times start at 0
colors=['b','g','r','c','m','y','k']
def plotValues(N,L,initFVec,eigen,grid,d,dd,totTime,NumSteps,ValsToPlot,soar=False,cp=False,car=False,title=None):
    ygrid=mapFiniteToInfArray(grid,L)

    if usingHO:
        totTime=5*2*np.pi
    t=np.linspace(0,totTime,NumSteps)
    finalFVecs=timeEvolve(initFVec,t,L,eigen,grid)

    #arrays of mean position,rms position,mean P, rms P,energy,"classicalEnergy"
    arraysOfVals=[[] for i in range(CVNR)]
    for i,finalFVec in enumerate(finalFVecs):
        if i%(int(NumSteps/10))==0:
            print(float(i)/NumSteps)

        vals=checkValues(finalFVec,ygrid,L,d,dd)
        for i,A in enumerate(arraysOfVals):
            A.append(vals[i])


    for i in range(CVNR):
        if ValsToPlot[i]:
            plt.plot(t,arraysOfVals[i],label=VALNAMES[i],color=colors[i])

    xmax=arraysOfVals[0][0]
    xqmax=0
    if soar or cp:
        if usingHO:
            clPeriod=2*np.pi
        else:
            clPeriod=4*np.sqrt(2)*1.31103/arraysOfVals[0][0]

        if soar:
            plt.plot((t[0], t[-1]), (xmax, xmax), 'k-')
            plt.plot((t[0], t[-1]), (-xmax, -xmax), 'k-')
        if cp:
            for i in range(int(totTime/clPeriod)+1):
                plt.plot((i*clPeriod, i*clPeriod), (-xmax, xmax), 'k-')
    
    if car:
        if usingHO:
            xqmax=(2.0*arraysOfVals[4][0])**(.5)
        else:
            xqmax=(4.0*arraysOfVals[4][0])**(.25)
        
        plt.plot((t[0], t[-1]), (xqmax, xqmax), 'k-')
        plt.plot((t[0], t[-1]), (-xqmax, -xqmax), 'k-')


    if not title is None:
        plt.title(title)

    #plt.xlabel("Time")
    #plt.ylabel("Units of Position")
    ym=max([xmax,xqmax])+4
    plt.ylim(-ym,ym)
    plt.legend(loc=4)
    plt.show()

#ValsToPlot is a boolean array of length equal to the number of items returned by "checkValues" (stored in CVNR variable)
def plotValsW(N,L,initFunc,totTime,NumSteps,ValsToPlot,soar=False,cp=False,car=False,title=None):
    assert len(ValsToPlot)==CVNR
    eigen,_,grid,d,dd=changeOfVariables(N,L)

    initFVec=funcToVec(initFunc,grid,L)

    plotValues(N,L,initFVec,eigen,grid,d,dd,totTime,NumSteps,ValsToPlot,soar=soar,cp=cp,car=car,title=title)



###################################################################################
#Main
###################################################################################

#Q: Is there a time beyond which the wavefunction no longer resembles a wavepacket 
#whos mean position and momentum follow classical evolution?
#A: Yes

#Q: Is there a time beyond which the mean position remains small 
#(compared to the initial position or the size of the allowed region) throughout a classical period?
#A: Yes, the mean position never reaches the initial position again after starting there and its
#maximum value within an oscillation gets progressively smaller. 
#The rate of decrease in maximum mean position decreases more slowly for smaller starting positions (method 1 Gaussian,c=0.426)


#Q: Is there a time beyond which the rms deviation is comparable to the size of the allowed region?
#A: Not as far as I can find. Certainly not for method 1 gaussians

#Q: Is there a time beyond which your calculation loses accuracy?
#Q: How do these answers depend on the energy and width of the initial state?


#Decreasing c will force me to need more eigenvectors (sharper point means more energy eigenfunction).
#As b increases The energy drastically increases but the number of eigenfunctions to represent it does
#not increase quickly.


#Problem Values: minEnergy 82.637, xmax=4.2639, maxWidth=xmax/10
#0=large initial potential but still well represented. Very close to classical evolution.
goodInitFunctions=[(0.0,.426,6.0)]
def getInitFunc(i):
    vals=goodInitFunctions[i]
    return initalFunction(vals[0],vals[1],vals[2])

if __name__=="__main__":
    #L=6.2#good for HO
    L=5.0#good for quartic
    N=1000
    #i=0
    #j=1
    

    #["Mean Position","RMS Position","Mean Momentum","RMS Momentum","Energy","Classical Energy"]
    initFunc=initalFunction(0.0,0.056,1.0)
    plotValsW(2000,9.0,initFunc,12,400,[1,1,0,0,0,0],soar=True,cp=True,car=True,title="Initial Gaussian with b=1 and c=0.056")

    #initFunc=initalFunction(0.0,0.4,6.0)
    #plotValsW(N,L,initFunc,12,400,[1,1,0,0,0,0],soar=True,cp=True,car=False,title="Initial Gaussian with b=6 and c=0.4")
    

    #testGaussianCvsRequiredN(5,.1)
    #testGaussianCvsRequiredN(9,.1)
    #plotGaussianCvsRequiredN(9.0)
    #plotGaussianCvsRequiredN(5.0)
    
    """
    plt.plot(sorted(nCutoffPreComp.keys()),sorted(nCutoffPreComp.values()))
    plt.xticks(range(0,2001,200))
    plt.title("Largest Trusted Eigenvalues with L=5.0")
    plt.xlabel("Number of Grid Points")
    plt.ylabel("Cutoff eigenvalue")
    plt.show()
    """
    
    #maxResidualOfGaussianPlot(1000,L,np.linspace(0,12,100),2,0,.4,6.0)
    #maxResidualOfGaussianPlot(1000,9.0,np.linspace(0,12,100),2,0,.056,1.0)

    #######################
    #Example function calls
    #######################

    
    ##quickly print off some energy levels
    #printEnergyLevels([326],getRidOf2and4=False)

    ##precompute cutoffs at different values of N,mult,L,tol
    #test=[50,150,250,350,450,550,650,750,850,950,1500]
    #for t in test:
    #    preComputeCutoff(t,2,5.0,10**(-8))
    #500 166 7
    #preComputeCutoff(1000,2,9,10**(-8))
    #preComputeCutoff(500,2,8.0,10**(-8))

    ##Residual of gaussian
    #t=20000000000.0
    #residualOfGaussian(N,L,t,5,0,.4,1.0)

    ##Residual of superposition of eigenFunctions at specified time
    #residualOfEigSuperposition(N,L,0,1,10.0)
    

    ##Gaussian time evolve movie. x^4
    #numFrames=300
    #totTime=4.0
    #GaussianTimeEvoMovie(1000,5.0,80,.4,6.0,totTime,numFrames)
    ##Gaussian time evolve movie. x^2
    #numFrames=300
    #totTime=np.pi*4
    #GaussianTimeEvoMovie(1000,5.0,80,.4,6.0,totTime,numFrames)
    #numFrames=600
    #totTime=4.0
    #GaussianTimeEvoMovie(2000,9.0,0,.056,1.0,totTime,numFrames)



    ##Time evolve 'video' of superposition of states
    #superPositionEigenFuncMovie(N,L,0,1)
    ##Time evolve 'video' of superposition of many states
    #superPositionManyEigenFuncMovie(N,L,[0,1,2])



    ##print overlaps
    #printOverlapWithTime(N,L,-1,-1)

    #two different check methods
    #plotMeanPositionAndMomentum(N,L,initFVec,eigen,card,grid,2*period)

    #dt=.000001
    #initT=np.array([0,1,2,3,4,5,6,10])
    #checkEhrenfestWrapper(initFunc,9.0,N,initT,dt)
