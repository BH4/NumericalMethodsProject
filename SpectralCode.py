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

def SpectralChebyshevInteriorCard(i,numP,tp,t):
    pn1=(-1)**(i-1)
    f=lambda tx: pn1*csc(.5*(tx+tp[i]))*(sinc(numP*(tx-tp[i]))/sinc(.5*(tx-tp[i])))*csc(numP*tp[i])*np.sin(tp[i])

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

    tp=[np.pi*i/float(npm1) for i in range(numpoints)]

    sp=np.cos(tp)

    xp=m*sp+b

    p=[2.0 if i==0 or i==numpoints-1 else 1.0 for i in range(numpoints)]

    card=[]
    for i in range(numpoints):
        card.append(SpectralChebyshevExteriorCard(i,numpoints,npm1,tp,t))

    d=[]
    for i in range(numpoints):
        row=[]
        for j in range(numpoints):
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

def SpectralChebyshevInterior(xmin,xmax,numpoints):
    xmin=float(xmin)
    xmax=float(xmax)
    assert int(numpoints)==numpoints
    numpoints=int(numpoints)

    b=(xmin+xmax)/2.0

    m=(xmin-xmax)/2.0

    t=lambda x: np.arccos((x-b)/m)

    tp=[np.pi*(i-.5)/numpoints for i in range(numpoints)]

    sp=np.cos(tp)

    xp=m*sp+b

    card=[]
    for i in range(numpoints):
        card.append(SpectralChebyshevInteriorCard(i,numpoints,tp,t))

    d=[]
    for i in range(numpoints):
        row=[]
        for j in range(numpoints):
            if i!=j:
                mij=(-1)**(i+j)*np.sqrt((1-sp[j]**2)/(1 - sp[i]**2))/(sp[i] - sp[j])
            else:
                mij=(sp[j]/2)/(1-sp[j]**2)
            row.append(mij)
        d.append(row)
    d=np.array(d)

    dd=[]
    for i in range(numpoints):
        row=[]
        for j in range(numpoints):
            if i!=j:
                mij=d[i][j]*(sp[i]/(1 - sp[i]**2) - 2/(sp[i]-sp[j]))
            else:
                mij=sp[j]**2/(1 - sp[j]**2)**2 - (np**2 - 1)/3/(1 - sp[j]**2)
            row.append(mij)
        dd.append(row)
    dd=np.array(dd)

    return (xp,d/m,dd/m**2,card)