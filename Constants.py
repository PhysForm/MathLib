# Modules:
 import MathLib as ml

# required modules:
"""
MathLib.py
sympy ?
"""

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
        raise ValueError("UNKNOWN ERROR")
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
 
def LochConst():
    LC = ((6*m.ln(2*m.ln(10)))/(pi()**2))
    return LC
