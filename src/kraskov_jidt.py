import numpy as np
import pandas as pd
import jpype
import os

def start_JVM():
    jarLocation = "./src/kraskov/infodynamics.jar"
    if not jpype.isJVMStarted():
        jvm_path = os.path.join(os.environ['JAVA_HOME'], 'bin', 'server', 'jvm.dll')
        jpype.startJVM(jvm_path, "-ea", "-Xmx4096m", "-Djava.class.path=" + jarLocation)
        print(jpype.java.lang.System.getProperty("java.class.path"))

def calc_mi_gaussian(a, b):
    start_JVM()
    if(len(np.shape(a))==1):
        a = a.reshape((-1,1))
    if(len(np.shape(b))==1):
        b = b.reshape((-1,1))
    _,d1 = np.shape(a)
    _,d2 = np.shape(b)
    mi_calc = jpype.JPackage("infodynamics.measures.continuous.gaussian").MutualInfoCalculatorMultiVariateGaussian()
    mi_calc.initialise(d1,d2)
    mi_calc.setObservations(a.tolist(),b.tolist())
    mi_val = mi_calc.computeAverageLocalOfObservations()
    return mi_val

def calc_mi_kraskov(df, asset1, asset2):
    start_JVM()
    
    a = df[asset1]
    b = df[asset2]

    a = np.array([tuple(x) if isinstance(x, list) else x for x in a.values]) if isinstance(a, pd.Series) else a
    b = np.array([tuple(x) if isinstance(x, list) else x for x in b.values]) if isinstance(b, pd.Series) else b
    
    if(len(np.shape(a))==1):
        a = a.reshape((-1,1))
    if(len(np.shape(b))==1):
        b = b.reshape((-1,1))
    _,d1 = np.shape(a)
    _,d2 = np.shape(b)
    mi_calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1()
    mi_calc.initialise(d1,d2)
    mi_calc.setObservations(a.tolist(),b.tolist())
    mi_val = mi_calc.computeAverageLocalOfObservations()
    return mi_val

def calc_oinfo_kraskov(df):
    data = df.to_numpy()
    
    start_JVM()
    if(len(np.shape(data))==1):
        data = data.reshape((-1,1))
    _,d = np.shape(data)
    oinfo_calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").OInfoCalculatorKraskov()
    oinfo_calc.initialise(d)
    oinfo_calc.setObservations(data.tolist())
    oinfo_val = oinfo_calc.computeAverageLocalOfObservations()
    return oinfo_val

def calc_te_kraskov(a,b,k,l,delay):
    start_JVM()
    if(len(np.shape(a))==1):
        a = a.reshape((-1,1))
    if(len(np.shape(b))==1):
        b = b.reshape((-1,1))
    _,d1 = np.shape(a)
    _,d2 = np.shape(b)
    te_calc = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov()
    te_calc.initialise(d1,d2,k,1,l,1,delay)
    te_calc.setObservations(a.tolist(),b.tolist())
    te_val = te_calc.computeAverageLocalOfObservations()
    return te_val
