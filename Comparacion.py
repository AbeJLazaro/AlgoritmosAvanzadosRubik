from sys import path
path.append("../")
path.append("../problemasBase")
path.append("../BusquedaInformada")
# problema
from BusquedaInformada.AEstrella import AEstrella,h1
from BusquedaInformada.Voraz import Voraz
from BusquedaNoInformada.UCS import UCS
from problemasBase import puzzle15 
from problemasBase.puzzle15 import trayectoria, Puzzle
from Manhattan import DistanciaManhattan
# utiles
from random import seed
from functools import partial
import timeit 

# compararemos los algoritmos que consideran costos

# generamos g(s)
g = lambda s: s.get_profundidad()
# generamos la heuristica h(s)
h = lambda s: h1(Puzzle(),s)
# funcion de paro
paro = lambda s: s==Puzzle()
# rompecabezas desordenado
seed(2019)
desordenado = Puzzle()
# no desordenar más o muere la compu :c
desordenado.shuffle(23)

print("HEURISTICA 1 cantidad de fichas fuera de lugar*************")
# Tomamos el tiempo para AEstrella
t=timeit.timeit(partial(AEstrella,origen=desordenado,funcionParo=paro,g=g,h=h),number=1)
print("A* tomó {0:4f} segundos".format(t))
# Tomamos el tiempo para UCS
t=timeit.timeit(partial(UCS,origen=desordenado,funcionParo=paro,g=g),number=1)
print("UCS tomó {0:4f} segundos".format(t))
# Tomamos el tiempo para GBFS o Voraz
t=timeit.timeit(partial(Voraz,origen=desordenado,funcionParo=paro,h=h),number=1)
print("Voraz tomó {0:4f} segundos".format(t))

print("HEURISTICA 2 distancia de las fichas a su lugar*************")
# UTILIZAREMOS la distancia Manhattan que es una mejor heuristica
m = DistanciaManhattan()
h2 = m.distanciaObjetivo
# Tomamos el tiempo para AEstrella
t=timeit.timeit(partial(AEstrella,origen=desordenado,funcionParo=paro,g=g,h=h2),number=1)
print("A* tomó {0:4f} segundos".format(t))
# Tomamos el tiempo para UCS
t=timeit.timeit(partial(UCS,origen=desordenado,funcionParo=paro,g=g),number=1)
print("UCS tomó {0:4f} segundos".format(t))
# Tomamos el tiempo para GBFS o Voraz
t=timeit.timeit(partial(Voraz,origen=desordenado,funcionParo=paro,h=h2),number=1)
print("Voraz tomó {0:4f} segundos".format(t))
