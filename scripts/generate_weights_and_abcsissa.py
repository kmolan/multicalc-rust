import numpy
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

MAX_LENGTH = 31


f_abcissa_laguerre = open("laguerre_abcissa.txt", "w")
f_weights_laguerre = open("laguerre_weights.txt", "w")

for i in range(1, MAX_LENGTH):
    idx = str(i)
    
    cur_laguerre_abs = laggauss(i)[0]
    cur_laguerre_weights = laggauss(i)[1]
    
    f_abcissa_laguerre.write("const LAGUERRE_ABSCISSA_"+idx+": [f64; "+idx+"] = [")
    f_weights_laguerre.write("const LAGUERRE_WEIGHT_"+idx+": [f64; "+idx+"] = [")
    
    for j in range(len(cur_laguerre_abs)-1):
        f_abcissa_laguerre.write("\n" + str(cur_laguerre_abs[j])+", ")
        f_weights_laguerre.write("\n" + str(cur_laguerre_weights[j])+", ")
        
    f_abcissa_laguerre.write("\n" + str(cur_laguerre_abs[len(cur_laguerre_abs) - 1]) + "];\n")
    f_weights_laguerre.write("\n" + str(cur_laguerre_weights[len(cur_laguerre_weights) - 1]) + "];\n")

f_abcissa_laguerre.close()
f_weights_laguerre.close()


f_abcissa_hermite = open("hermite_abcissa.txt", "w")
f_weights_hermite = open("hermite_weights.txt", "w")

for i in range(1, MAX_LENGTH):
    idx = str(i)

    cur_hermite_abs = hermgauss(i)[0]
    cur_hermite_weights = hermgauss(i)[1]

    f_abcissa_hermite.write("const HERMITE_ABSCISSA_"+idx+": [f64; "+idx+"] = [")
    f_weights_hermite.write("const HERMITE_WEIGHT_"+idx+": [f64; "+idx+"] = [")

    for j in range(len(cur_hermite_abs)-1):
        f_abcissa_hermite.write("\n"+ str(cur_hermite_abs[j])+", ")
        f_weights_hermite.write("\n" + str(cur_hermite_weights[j])+", ")

    f_abcissa_hermite.write("\n" + str(cur_hermite_abs[len(cur_hermite_abs) - 1]) + "];\n")
    f_weights_hermite.write("\n" + str(cur_hermite_weights[len(cur_hermite_weights) - 1]) + "];\n")

f_abcissa_hermite.close()
f_weights_hermite.close()


f_abcissa_legendre = open("legendre_abcissa.txt", "w")
f_weights_legendre = open("legendre_weights.txt", "w")

for i in range(1, MAX_LENGTH):
    idx = str(i)

    cur_legendre_abs = leggauss(i)[0]
    cur_legendre_weights = leggauss(i)[1]

    f_abcissa_legendre.write("const LEGENDRE_ABSCISSA_"+idx+": [f64; "+idx+"] = [")
    f_weights_legendre.write("const LEGENDRE_WEIGHT_"+idx+": [f64; "+idx+"] = [")

    for j in range(len(cur_legendre_abs)-1):
        f_abcissa_legendre.write("\n" + str(cur_legendre_abs[j])+", ")
        f_weights_legendre.write("\n" + str(cur_legendre_weights[j])+", ")

    f_abcissa_legendre.write("\n" + str(cur_legendre_abs[len(cur_legendre_abs) - 1]) + "];\n")
    f_weights_legendre.write("\n" + str(cur_legendre_weights[len(cur_legendre_weights) - 1]) + "];\n")

f_abcissa_legendre.close()
f_weights_legendre.close()
