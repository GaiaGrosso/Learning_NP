import ROOT
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import pandas as pd

def DeltaPhi(phi1, phi2):
    result  = phi1 - phi2;
    result -= 2*math.pi*(result >  2*math.pi)
    result += 2*math.pi*(result <= 0*math.pi)
    return result
  
