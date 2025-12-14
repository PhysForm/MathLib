# modules:
from functools import lru_cache as lcache
import sympy as smp
from decimal import Decimal as dcml, getcontext as gtctx

# modules required for Library: 
"""
functools (lru_cache)
sympy (CAS functionality)
decimal (Precision)
scipy (?)
numpy (?)
"""

# Accuracy stuffs
gtctx().prec = 999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999
SUCCESS_CODE = dcml(1)
ERROR_CODE = dcml(-1)
EXIT_CODE = dcml(0)
INTERRUPT = dcml()

# Mathematics section 
      

def abs(x: dcml):
    """Returns the Absolute value of x"""
    if x < 0:
        return -x
    else:
        return x

def IntAngleSum(n: dcml):
    """Returns the sum of all interior angles based off of the number of sides"""
    IAS = (n-2)*180
    return IAS
    
def IntAngle(n: dcml):
    """Returns the measure of the Interior Angle based off of the number of sides"""
    IA = ((n-2)*180)/n
    return IA
    
def AreaC(r: dcml,pi: dcml):
    """Returns the Area of a circle based off of a radius and a value of pi (default is pi(9999999999999999999))"""
    A = (pi*(r**2))
    return A
    
def Circum(r: dcml, pi: dcml):
    """Returns the Circumference of a circle based off of a radius r and pi inputs"""
    C = (2*r)*pi
    return C
    
def TriArea(base: dcml, height: dcml):
    A = (base*height)/2
    return A
    
def PythagC(a: dcml, b: dcml):
    c = sqrt((a**2)+(b**2))
    return c
    
def PythagB(a: dcml, c: dcml):
    b = sqrt((c**2)-(a**2))
    return b
    
def PythagA(b: dcml, c: dcml):
    a = sqrt((c**2)-(b**2))
    return a
    
def SinruleA(b: dcml, alpha: dcml, beta: dcml):
    a = (b*((m.sin(alpha))/(m.sin(beta))))
    return a
    
def SinRuleB(a: dcml,beta: dcml,alpha: dcml):
    b = a*(m.sin(beta)/m.sin(alpha))
    return b
    
def CosruleSide(a: dcml, b: dcml, gamma: dcml):
    c = sqrt(((a**2)+(b**2)-(2*a*b*(m.invcos(gamma)))))
    return c
    
def CosruleAngle(a: dcml, b: dcml, c: dcml):
    gamma = m.invcos(((a**2)+(b**2)-(c**2)/(2*a*b))/(2*a*b))
    return gamma
    
def asinh(x: dcml):
    ASh = ln(x+sqrt((x**2)+1))
    return ASh
    
def asin(x: dcml, num_terms:int): #num_terms is the level of accuracy, therefore increased number of terms is an increased accuracy
    if not (-1 <= x <= 1):
        raise ValueError("Input value must be in the range [-1, 1]")
    result = 0
    for n in range(num_terms):
        numerator = factorial(2 * n)
        denominator = (4**n) * (factorial(n)**2) * (2 * n + 1)
        term = (numerator / denominator) * (x**(2 * n + 1))
        result += term
    return result
    
def atanh(x: dcml):
    if abs(x) >= 1:
        raise ValueError("Input value must be between -1 and 1 (exclusive).")
    numerator = 1 + x
    denominator = 1 - x
    log_term = m.ln(numerator / denominator)
    result = 0.5 * log_term
    return result
    
def acos(x: dcml, num_terms:int,pi: dcml): #num_terms is the level of accuracy, therefore increased number of terms is an increased accuracy
    if not -1 <= x <= 1:
        raise ValueError("Input value must be between -1 and 1")
    result = pi / 2
    term = x
    for i in range(1, num_terms):
        if i == 1:
            term = x
        elif i == 2:
            term = (x**3) / 3
        elif i == 3:
            term = (3 * (x**5)) / 40
        elif i == 4:
           term = (5 * (x**7)) / 112
        else:
           term = ( (factorial(2 * i) * (x**(2*i + 1)) ) / ((2**i) * (factorial(i))**2 * (2*i + 1)) )
        result -= term
    return result
    
def acosh(x: dcml):
    if x < 1:
        raise ValueError("Input value out of domain (x must be >= 1)")
    acosh_x = m.ln(x + sqrt((x**2)-1))
    return acosh_x
    
def atan(x: dcml, num_terms:int): #num_terms is the level of accuracy, therefore increased number of terms is an increased accuracy
    result = 0
    for n in range(num_terms):
        term = ((-1)**n) * (x**(2*n+1)) / (2*n+1)
        result += term
    return result
    
def cosh(x: dcml,num_terms:int): #num_terms is the level of accuracy, therefore increased number of terms is an increased accuracy
    result = 0
    for n in range(num_terms):
        term = (x**(2 * n)) / factorial(2 * n)
        result += term
    return result
    
def sinh(x: dcml,num_terms: dcml): #num_terms is the level of accuracy, therefore increased number of terms is an increased accuracy
    sinh_x = 0
    n = 0
    for n in range(num_terms):
        term = (x**(2*n + 1)) / factorial(2*n + 1)
        sinh_x += term
    return sinh_x
    
def tanh(x: dcml):
    tanh_x = ((e()**x)-(e()**(-x))) / ((e()**x)+(e()**(-x)))
    return tanh_x
    
def csc(x: dcml):
    cscX = (1/(m.sin(x)))
    return cscX
    
def sec(x: dcml):
    secX = (1/(m.cos(x)))
    return secX
    
def cot(x: dcml):
    cotX = (1/(m.tan(x)))
    return cotX
    
def Rectarea(s1: dcml,s2: dcml):
    RA = s1*s2
    return RA
    
def TrapArea(b1: dcml, b2: dcml, h: dcml):
    TA = (((b1+b2)/2)*h)
    return TA
    
def RhombusArea(p: dcml, q: dcml): #p & q are  diagonals
    RhA = ((p*q)/2)
    return RhA
    
def PentArea(d: dcml,a,float):
    Ap = (1/4)*(sqrt(5*(5+(2*sqrt(5)))))*(a**2)
    return Ap
    
def HexArea(side: dcml):
    Ha = ((3*sqrt(3))/2)*(side**2)
    return Ha
    
def HeptArea(side: dcml):
    Ah = (7/4)*(side**2)*plg.cot(180/7)
    return Ah
    
def OctArea(side: dcml):
    Oa = 2*(1+sqrt(2))*(side**2)
    return Oa
    
def EnnArea(side: dcml):
    EnA = (9/4)*(side**2)*plg.cot(180/9)
    return EnA
    
def DecArea(side: dcml):
    Da = (5/2)*(side**2)*sqrt(5+(2*sqrt(5)))
    return Da
    
def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
def Φ(n):
        count = 0
        for i in range(1, n + 1):
            if gcd(n, i) == 1:
                count += 1
        return count

def cbrt(n: dcml,tolerance=0.0000000000001,maxiter = 1000000000000000):
    if n == 0:
        return 0
    guess = n/2
    for _ in range(maxiter):
        next_guess = (2 * guess + n / (guess**2)) / 3
        if abs(next_guess - guess) < tolerance:
            return next_guess
        guess = next_guess
    return guess
    
def sqrt(number: dcml, tolerance=0.0000000000001, max_iterations=1000000000000000):
    if number < 0:
        number = abs(number)
    if number == 0:
        return 0
    if number == 1:
        return 1
    guess = number / 2  # Initial guess
    for _ in range(max_iterations):
        new_guess = 0.5 * (guess + number / guess)
        if abs(new_guess - guess) < tolerance:
            return new_guess
    guess = new_guess
    return guess
    # Return the last guess if tolerance not reached

def Quadrat(a:int, b:int, c:int):
    QP = (((0-b)+(sqrt(((b**2)-(4*a*c)))))/(2*a))
    QM = (((0-b)-(sqrt(((b**2)-(4*a*c)))))/(2*a))
    return QP,QM
def DegRad(x: dcml,pi: dcml):
    t = ((x*pi)/(180))
    return t
def RadDeg(t: dcml, pi: dcml):
    x = ((180*t)/pi)
    return x
def DegGrad(x: dcml):
    g = x*(200/180)
    return g
def GradDeg(g: dcml):
    x = (g*180/200)
    return x
def GradRad(g: dcml, pi: dcml):
    t = g *(pi/200)
    return t
def RadGrad(t: dcml, pi: dcml):
    g = t *(200/pi)
    return g
def Longavg(sum: dcml, number_of_numbers: dcml):
    avgX = (sum/number_of_numbers)
    return avgX
def ListMean(dataset:list):
    Mean = sum(dataset)/len(dataset)
    return Mean
def TwoPointSlope(x1: dcml,y1: dcml,x2: dcml,y2: dcml):
    m = ((y2-y1)/(x2-x1))
    return m
def TwoPointSlope(xy[3]):
    m = ((xy[1]-xy[3])/(xy[0]-xy[2]))
def TwoPointDistance(x1: dcml,y1: dcml,x2: dcml,y2: dcml):
    D = sqrt(((x2-x1)**2) + ((y2-y1)**2))
    return D
def ln(a: dcml, precision = 1000000000000000000000000000000,first_guess = x + 1, e = e(999999)):
    x0 = first_guess
    while precision > 0:
        x1 = x0 - ((e-a)/(e)
        precision = precision - 1
    return x1
    
       
def log(x: dcml,base: dcml):
    if x <= 0 or base <= 0 or base == 1:
        raise ValueError("Arguments x and base must be a positive number. The base must be greater than 1")
    return log(x) / log(base) # Using the natural logarithm (base e)
def factorial(n:int):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


#constants
def pi():
    pi = 3.1415926535897932384
    return pi
def phi():
    phi = 1.618033988749895
    return phi
def e():
    e = 2.71828182845904523536028747135266249775724709369995
    return e
def tau():
    tau = 2*pi()
    return tau
def PythConst():
    PC = sqrt(2)
    return PC
def TheoConst():
    TC = sqrt(3)
    return TC
def SilvRatio():
    SR = sqrt(2)+1
    return SR
def SuperGoldRatio():
    SGR = ((1+cbrt((29+(3*sqrt(93)))/2)+cbrt((29-(3*sqrt(93)))/2))/3)
    return SGR
def i():
    i = sqrt(-1)
    return i
def ConnConst():
    CC = sqrt(2+sqrt(2))
    return CC
def GoldenAngleRad():
    GAR = ((2*pi())/((phi())**2))
    GARConf = (pi()*(3-sqrt(5)))
    if GAR !=GARConf:
        raise ValueError("SYSTEM ERROR")
    else:
        return GAR
def GoldenAngleDeg():
    GAD = 180*(3-sqrt(5))
    GADConf1 = (180/pi())*((2*pi())/((phi())**2))
    GADConf2 = (180/pi())*(pi()*(3-sqrt(5)))
    if GAD != GADConf1 or GAD != GADConf2 or GADConf1 != GADConf2:
        raise ValueError("SYSTEM ERROR")
    else:
        return GAD
def GoldenAngleGrad():
    GAG = (180*(3-sqrt(5)))*(180/200)
    GAGConf =  ((2*pi())/((phi())**2))*(pi()/200)
    if GAG == GAGConf:
        return GAG
    else: 
        raise "SYSTEM ERROR: UNIDENTIFIED ERROR"
def RamanujConst():
    RamConst = (e()**(pi()*sqrt(163)))
    return RamConst
def UnivParaConst():
    UPC = ln(1+sqrt(2))+sqrt(2) 
    UPCConf= asinh(1)+sqrt(2)
    if UPC != UPCConf:
        raise ValueError("SYSTEM ERROR")
    else:
        AnswerSet =  UPC,UPCConf
    return AnswerSet
def GelfSchneidConst():
    GSC = (2**sqrt(2))
    return GSC
def GelfConst():
    GC = e()**pi()
    return GC
def FavardConst2():
    SFC = (pi()**2)/8
    return SFC
def  LochConst():
    LC = ((6*m.ln(2*m.ln(10)))/(pi()**2))
    return LC


# Trauma for the survivors:
def Powinteg(x: dcml, n: dcml): # n = exponent
    IntegX = ((x**(n+1))/(n+1))
    return IntegX
def ScalarInteg(x: dcml, n: dcml, multiple: dcml):
    SI = multiple * m.Powinteg(x,n)
    return SI
def limit(f:vars, x_limit: dcml, step: dcml): # function f(x) must be either along the lines of or exactly defined as def f(x): \nf_x = (operations upon that x) \n return f_x
    leftx = [x_limit - step * (1/2)**i for i in range(1,10)]
    rightx = [x_limit + step * (1/2)**i for i in range(1, 10)]
    lefty = [f(x) for x in leftx]
    righty = [f(x) for x in rightx]
    if len(set(round(val, 6) for val in lefty)) == 1 and len(set(round(val, 6) for val in righty)) == 1:
        return (lefty[0] + righty[0]) / 2
    else:
        raise ValueError("Limit non-existent (None is returned)")

#sequence
def AritSeq(first_term: dcml, Term_number:int, common_difference: dcml):
    An = first_term + ((Term_number-1)*common_difference)
    return An
def GeoSeq(first_term: dcml, Term_number:int, ratio: dcml):
    Gn = (first_term * (ratio**(Term_number-1)))
    return Gn
def HarSeq(first_term: dcml, Term_number:int, common_difference: dcml):
    Hn = 1/(first_term + ((Term_number-1)*common_difference))
    return Hn
def FiboSeq(Term_number:int):
    Fn = (((phi())**Term_number)-((1-phi())**Term_number)/sqrt(5))
    return Fn
def TriSeq(n:int): # n = Term_number
    Tn = n*(n+1)/2
    return Tn
def PentSeq(n:int): # n = Term_number
    Pn = n*((3*n)-1)/2
    return Pn
def HexSeq(n:int): # n = Term_number
    Hn = n*((2*n) - 1)
    return Hn
def HeptSeq(n:int): 
    Sn = n * ((5*n) - 3) / 2
    return Sn
def OctSeq(n:int):
    On = n*((3*n)-2)
    return On
def EnneaSeq(n:int):
    Nn = ((n*((7*n)-5))/2)
    return Nn
def DekaSeq(n:int):
    Dn = n * ((4*n) - 3) 
    return Dn
def Σ(n:int,first_i:int, coefficient:int, added_value: dcml) -> float:
    a1 = (coefficient*first_i) + added_value
    an = (coefficient*n) + added_value
    Sum = ((n*(a1+an))/2)
    return Sum

# Statistics
def CapPi(seq:list):
    Res = 1
    for x in seq:
        Res *= x
    return Res
def StdDeviant(data:list):
    sqdiffsum = sum([(x-ListMean(data))**2 for x in data])
    variance = sqdiffsum / len(data)
    std_deviant = sqrt(variance)
    return std_deviant
#Geometry Library
class solg():
    def CircSA(r: dcml,pi: dcml):
        SA = 4*(pi*(r**2))
        return SA
    def CircVol(r: dcml,pi: dcml):
        V = (4/3)*(pi*(r**3))
        return V
    def ConeLSA(r: dcml,pi: dcml,l: dcml):
        LSA = pi*r*l
        return LSA
    def ConeSA(pi: dcml,r: dcml,l: dcml):
        LSA = pi*r*l
        base = pi*(r**2)
        SA = LSA+base
        return SA
    def ConeVol(pi: dcml,r: dcml):
        CoVo = ((1/3)*(pi*(r**2)))

        return CoVo


