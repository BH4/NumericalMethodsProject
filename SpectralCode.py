import numpy as np

def sinc(x):
    return np.sinc(x/np.pi)

def csc(x):
    return 1.0/np.sin(x)

#need to define the lambda's outside a for loop because otherwise I just get copies of the same function
def SpectralChebyshevExteriorCard(i,numpoints,npm1,tp,t):
    if i==0:
        f=lambda tx: np.cos(tx/2.0)*sinc(npm1*(tx - tp[i]))/sinc(.5*(tx - tp[i]))
    elif i==numpoints-1:
        f=lambda tx: np.sin(tx/2.0)*sinc(npm1*(tx - tp[i]))/sinc(.5*(tx - tp[i]))
    else:
        f=lambda tx: np.sin(tx)*csc(.5* (tx + tp[i]))*sinc(npm1*(tx - tp[i]))/sinc(.5*(tx - tp[i]))
    
    g=lambda x: f(t(x))
    return g

def SpectralChebyshevExterior(xmin,xmax,numpoints):
    xmin=float(xmin)
    xmax=float(xmax)
    assert int(numpoints)==numpoints
    numpoints=int(numpoints)

    npm1=numpoints-1#np minus 1; bad name

    b=(xmin+xmax)/2.0

    m=(xmin-xmax)/2.0

    t=lambda x: np.arccos((x-b)/m)

    tp=[np.pi*i/float(npm1) for i in xrange(numpoints)]

    sp=np.cos(tp)

    xp=m*sp+b

    p=[2.0 if i==0 or i==numpoints-1 else 1.0 for i in xrange(numpoints)]

    card=[]
    for i in xrange(numpoints):
        card.append(SpectralChebyshevExteriorCard(i,numpoints,npm1,tp,t))

    d=[]
    for i in xrange(numpoints):
        row=[]
        for j in xrange(numpoints):
            if i!=j:
                mij=(-1)**(i+j)*(p[i]/p[j])/(sp[i]-sp[j])
            elif i==0 and j==0:
                mij=(1+2*npm1**2)/6.0
            elif i==numpoints-1 and j==numpoints-1:
                mij=-1*(1+2*npm1**2)/6.0
            else:
                mij=(-sp[j]/2)/(1-sp[j]**2)

            row.append(mij)

        d.append(row)
    d=np.array(d)

    return (xp,d/m,d.dot(d)/m**2,card)


