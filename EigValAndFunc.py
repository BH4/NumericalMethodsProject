#use a change of variables (mapping y elem of -inf,inf to x elem of -1,1) to get the eigenfunctions and eigenvalues for the Hamiltonian
#H=-(1/2)d^2/dx^2+x^4/4

#cord change x=y/sqrt(L^2+y^2)


import numpy as np
import matplotlib.pyplot as plt
from SpectralCode import SpectralChebyshevExterior
import scipy.integrate

###################################################################################
#Visual testing functions. Don't use these for real math. 
###################################################################################


#basically multiply the cardinal functions by the vector.
#changes variable back to entire real line variable. Don't use this for actual math.
def lookAtTheFunction(L,vec,card):
    f=lambda y: sum([vec[i]*card[i](y/np.sqrt(L**2+y**2)) for i in xrange(len(vec))])
    return f

#returns the square root of the squared integral. Useful to check if a function is really normalized.
#assumes function is defined on the entire real line.
def checkNormalization(func):
    squaredFunc=lambda x:abs(func(x))**2
    normSquared,err=scipy.integrate.quad(squaredFunc,-np.inf,np.inf)

    return np.sqrt(normSquared)


###################################################################################
#Quick helper functions/also some testing for accuracy.
###################################################################################

#dot numpy array with numpy matrix A*M
def dotAM(A,M):
    #assert len(A.shape)==1
    N=np.copy(M)
    for i,a in enumerate(A):
        N[i,:]*=a
    return N
#dot numpy matrix with numpy array M*A
def dotMA(M,A):
    #assert len(A.shape)==1
    N=np.copy(M)
    for i,a in enumerate(A):
        N[:,i]*=a
    return N

#Construct the Hamiltonian matrix with changed coordinates
usingHO=False
def hamiltonian(L,grid,d,dd):
    #gives zero at singularities but those rows are zeroed out by BC anyway.
    V=np.diag([(L*y)**4/(4*(1-y**2)**2) if abs(y)!=1 else 0 for y in grid])

    a=np.diag([(1-y**2)**3/(2*L**2) for y in grid])
    b=np.diag([3*y*(1-y**2)**2/(2*L**2) for y in grid])
    H=-np.dot(a,dd)+np.dot(b,d)+V
    #a=[(1-y**2)**3/(2*L**2) for y in grid]
    #b=[3*y*(1-y**2)**2/(2*L**2) for y in grid]
    #H=-dotAM(a,dd)+dotAM(b,d)+V

    H[0]=[0]*len(grid)
    H[-1]=[0]*len(grid)

    return H

"""
#harmonic oscillator
usingHO=True
def hamiltonian(L,grid,d,dd):
    #gives zero at singularities but those rows are zeroed out by BC anyway.
    V=np.diag([(L*y)**2/(2*(1-y**2)) if abs(y)!=1 else 0 for y in grid])

    a=np.diag([(1-y**2)**3/(2*L**2) for y in grid])
    b=np.diag([3*y*(1-y**2)**2/(2*L**2) for y in grid])
    H=-np.dot(a,dd)+np.dot(b,d)+V

    H[0]=[0]*len(grid)
    H[-1]=[0]*len(grid)

    return H
"""

#I think these versions might be better numerically. [it doesn't seem to do better. also uses more memory because of the
#extra matrix]
"""
usingHO=False
def hamiltonian(L,grid,d,dd):
    #gives zero at singularities but those rows are zeroed out by BC anyway.
    V=np.diag([(L*x)**4/(4*(1-x**2)**2) if abs(x)!=1 else 0 for x in grid])

    a=[(1-x**2)**2/(2*L**2) for x in grid]
    b=[1-x**2 for x in grid]
    c=[3*x for x in grid]
    middle=dotAM(b,d)
    for i,cc in enumerate(c):
        middle[i,i]-=cc
    H=-np.dot(dotAM(a,middle),d)+V

    H[0]=[0]*len(grid)
    H[-1]=[0]*len(grid)

    return H
"""
"""
#harmonic oscillator
usingHO=True
def hamiltonian(L,grid,d,dd):
    #gives zero at singularities but those rows are zeroed out by BC anyway.
    V=np.diag([(L*x)**2/(2*(1-x**2)) if abs(x)!=1 else 0 for x in grid])

    a=np.diag([(1-x**2)**2/(2*L**2) for x in grid])
    b=np.diag([1-x**2 for x in grid])
    c=np.diag([3*x for x in grid])
    H=-np.dot(np.dot(a,np.dot(b,d)-c),d)+V

    H[0]=[0]*len(grid)
    H[-1]=[0]*len(grid)

    return H
"""



#approximately normalize eigenvectors using Chebyshev-gauss quadrature
#(The eigenfunction is normalized such that its squared integral on the infinite variable is 1)
#https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
def normalizeVector(L,vec,grid):
    N=len(grid)

    total=0
    #takes the values 1,2,...,N-3,N-2   -> no endpoints
    for i in xrange(1,N-1):
        weight=(np.pi/(N-1))
        g=(L/(1-grid[i]**2))*vec[i]**2

        total+=weight*g

    normalization=np.sqrt(total)

    return vec/normalization



###################################################################################
#Construct eigenvalues and eigenvectors. 
###################################################################################
#Each returns cardinal functions and eigen vector array and grid (in -1 to 1 coordinates)


def changeOfVariables(numpoints,L):

    grid,d,dd,card=SpectralChebyshevExterior(-1,1,numpoints)

    H=hamiltonian(L,grid,d,dd)

    vals,vecs=np.linalg.eig(H)
    eigen=[(vals[i],vecs[:,i]) for i in xrange(len(vals))]

    eigen.sort(key=lambda x:x[0])

    eigen=eigen[2:]#remove the 0.0 eigenvalues I added in.

    
    #normalize all eigenvectors on entire real line variable
    #This WILL change the residual since I am multiplying the residual of each eigenfunction by an overall number.
    #could simplify this by putting it directly in the loop above, but I kinda want to keep it separate from everything else
    for i in xrange(len(eigen)):
        newEigV=normalizeVector(L,eigen[i][1],grid)
        eigen[i]=(eigen[i][0],newEigV)
    

    return eigen,card,grid

###################################################################################
#Residuals
###################################################################################

#calculate the residual for one eigenvector. bigGrid is the more fine grid of points that the eigenfunction
#will be interpolated on. card is the set of cardinals which go with the smaller number of grid points.
#THIS METHOD IS CREATING LARGE INCORRECT RESIDUALS. I believe they are incorrect because the larger eigenvector is more similar to the interpolated eigenvector than the large residual would imply.
#The large error in this method must be coming from the derivative term in the Hamiltonian.
def calcR(bigH,bigGrid,E,vec,card):
    f=lambda x: sum([vec[i]*card[i](x) for i in xrange(len(vec))])
    interpVec=f(bigGrid)
    #print interpVec
    #print bigH.dot(interpVec)
    #print E



    return bigH.dot(interpVec)-E*interpVec

#N=original num Points. Nprime is larger number of points.
def residuals(L,N,Nprime):

    #small
    eigen,cardS,_=changeOfVariables(N,L)

    #big
    #gridB,dB,ddB,cardB=SpectralChebyshevExterior(-1,1,Nprime)
    #HB=hamiltonian(L,gridB,dB,ddB)
    eigenB,_,gridB=changeOfVariables(Nprime,L)



    rList=[]#list of residual vectors
    for i in xrange(len(eigen)):
        eigenE,eigenV=eigen[i]
        f=lambda x: sum([eigenV[i]*cardS[i](x) for i in xrange(len(eigenV))])
        interpVec=f(gridB)

        #rList.append(calcR(HB,gridB,eigenE,eigenV,cardS))
        rList.append(abs(abs(eigenB[i][1])-abs(interpVec)))


    return rList,gridB

def residualsWithoutRecalculation(L,Nprime,eigen,cardS):

    #big
    #gridB,dB,ddB,cardB=SpectralChebyshevExterior(-1,1,Nprime)
    #HB=hamiltonian(L,gridB,dB,ddB)
    eigenB,_,gridB=changeOfVariables(Nprime,L)

    rList=[]#list of residual vectors
    for i in xrange(len(eigen)):
        eigenE,eigenV=eigen[i]
        f=lambda x: sum([eigenV[i]*cardS[i](x) for i in xrange(len(eigenV))])
        interpVec=f(gridB)

        #rList.append(calcR(HB,gridB,eigenE,eigenV,cardS))
        rList.append(abs(abs(eigenB[i][1])-abs(interpVec)))

    return rList,gridB

def hardcoreResidual(L,eigen,i,grid,card):
    energy=eigen[i][0]

    func=lookAtTheFunction(L,eigen[i][1],card)

    y=np.linspace(-10,10,1000)
    R=[]
    for i,yi in enumerate(y):
        if i%100==0:
            print i
        if usingHO:
            temp=energy*func(yi)+scipy.misc.derivative(func,yi,dx=0.001,n=2,order=5)/2-((yi**2)/2.0)*func(yi)
        else:
            temp=energy*func(yi)+scipy.misc.derivative(func,yi,dx=0.001,n=2,order=5)/2-((yi**4)/4.0)*func(yi)
        R.append(abs(temp))

    plt.plot(y,R,color='red')
    plt.plot(y,func(y))
    plt.show()


###################################################################################
#Consistency checks
###################################################################################

def checkResiduals(N,L):
    Nprime=N*2

    eigen,card,gridS=changeOfVariables(N,L)
    rList,grid=residuals(L,N,Nprime)
    
    check=[400]
    #check=[80,100,120,150,180,200,250,270]
    for i in check:
        biggest=max(abs(rList[i]))
        print 'WF '+str(i)+': Max Error='+str(biggest)

        eigeni=lambda x: sum([eigen[i][1][j]*card[j](x) for j in xrange(len(eigen[i][1]))])
        plt.plot(grid,eigeni(grid))
        plt.plot(gridS,eigen[i][1])
        plt.plot(grid,abs(rList[i]),color='red')
        plt.xlim(-2.0,2.0)
        if biggest>1:
            plt.ylim(-1,biggest)
        else:
            plt.ylim(-1,1)
        plt.show()

def showEigenFunctionsAreOrthonormal(L,numpoints,i,j):
    eigen,card,grid=changeOfVariables(numpoints,L)

    func1=lookAtTheFunction(L,eigen[i][1],card)
    func2=lookAtTheFunction(L,eigen[j][1],card)

    squaredFunc=lambda x:func1(x)*func2(x)
    normSquared,err=scipy.integrate.quad(squaredFunc,-np.inf,np.inf)

    if abs(normSquared)<err:
        print "within error of zero"
    else:
        print normSquared

###################################################################################
#Main
###################################################################################

if __name__=="__main__":


    
    ##Checking residuals
    L=5.0
    N=1000
    #checkResiduals(N,L)

    eigen,card,grid=changeOfVariables(N,L)
    hardcoreResidual(L,eigen,1,grid,card)


    
    """
    i=100
    Nprime=N*2
    eigen,card,gridS=changeOfVariables(N,L)
    plt.plot(gridS,eigen[i][1],color='blue')
    print '1'
    rList,grid=residuals(L,N,Nprime)
    plt.plot(grid,rList[i],color='red')
    print '2'
    eigen,card,grid=changeOfVariables(Nprime,L)
    plt.plot(grid,eigen[i][1],color='green')
    print '3'
    plt.show()
    """

    
    #eigen,card,grid=changeOfVariables(10,3.0)
    #print eigen[0][1]
    #print eigen[1][1]
    #print eigen[0][1]+eigen[1][1]
    
    
    ##check that eigenfunctions are orthonormal
    #L=5.0
    #numpoints=100

    #showEigenFunctionsAreOrthonormal(L,numpoints,10,10)
    
    