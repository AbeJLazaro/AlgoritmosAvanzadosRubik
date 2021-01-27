'''
Autores:      Cabrera Gaytán Jazmín Andrea
              Lázaro Martínez Abraham Josué

Versión:      1.1
Fecha:        26 de enero de 2021
Nombre:       Proyecto.py
'''
# estructuras de datos
from collections import deque
from heapq import heappush as push
from heapq import heappop as pop
# problema
from RubikPuzzle import RubikPuzzle, trayectoria 
from HeuristicaPatrones import HeuristicaBasadaEnPatrones
# convenientes
from random import seed
import matplotlib
import matplotlib.pyplot as plt
from math import inf
from functools import partial
import timeit 
import numpy as np

historial=list()
# función para A*
def AEstrella(origen,funcionParo,g,h):
  '''Función que implementa el algoritmo A*

  Parámetros
  origen: estado desde el que se inicia
  fucionParo: función que nos indica si llegamos a un estado meta
  g: función de costo acumulado
  h: función heuristica

  return solución o nada
  '''
  historia = [(0,0)]
  # solución trivial
  if funcionParo(origen):
    return trayectoria(origen),historia

  # inicializamos al agenda
  agenda = []

  # inicializamos el conjunto de expandidos
  expandidos = set()

  # generamos la función f(s) = g(s) + h(s)
  # f(s) representa el costo total a un nodo
  # costo acumulado más el costo de la heuristica a ese nodo
  # esto nos representará la prioridad para cada nodo
  f = lambda s: g(s)+h(s)

  # agregamos el primer nodo a la agenda
  push(agenda,(f(origen),origen))

  # mientras existan elementos en la agenda
  while agenda:
    historia.append((len(agenda),len(expandidos)))
    # sacamos un nodo de la agenda
    nodo = pop(agenda)[1] # pos 1 para obtener el estado de la tupla

    # lo agregamos al conjunto de expandidos
    expandidos.add(nodo)

    # comparamos si este nodo cumple con la función de paro
    if funcionParo(nodo):
      return trayectoria(nodo),historia
    '''Comparamos si el nodo cumple con la función de paro aquí dado que el 
    algoritmo termina cuando el nodo objetivo o estado meta se encuentra en 
    la cabeza de la cola de prioridad, esto ocurrirá cuando el nodo sea el 
    siguiente en salir, por eso se compará afuera del ciclo'''
    # expandimos el nodo y comparamos a sus hijos
    for hijo in nodo.expandir():

      # agregamos el hijo a la cola de prioridad
      # siempre y cuando no esté en el conjunto de expandidos
      if hijo not in expandidos:
          push(agenda,(f(hijo),hijo))

  # si no hay ruta, regresamos un vacio
  return 

def IDAEstrella(origen,funcionParo,g,h):
  '''
  Implementación del algoritmo A* con profundidad iterada

  Parámetros
  origen: estado inicial
  funcionParo: nos indica que estado es meta
  g: función de costos acumulados
  h: función heuristica

  return solución o nada
  '''
  historia = [(0,0)]
  # checamos la solución trivial
  if funcionParo(origen):
    return trayectoria(origen),historia

  # cota de costo, la inicializamos como h(origen)
  c = h(origen)

  # definimos la función de costo total
  f = lambda s: g(s)+h(s)

  # bucle infinito
  while True:

    # el "minimo" por ahora es infinito
    minimo = inf

    # inicializamos la agenda
    agenda = list()

    # agregamos el estado inicial a la agenda
    agenda.append(origen)

    # inicializamos el conjunto de expandidos
    expandidos = set()

    # mientras existan elementos en la agenda
    while agenda:
      historia.append((len(agenda),len(expandidos)))
      # checamos el tope de la pila, es este caso es el último agregado
      nodo = agenda[-1]

      # comparamos, si el nodo no se ha expandido
      if nodo not in expandidos:

        # agregamos al nodo al conjunto de expandidos
        expandidos.add(nodo)

        # encontramos sus sucesores y los ordenamos de acuerdo a 
        # su costo total
        hijos = nodo.expandir()
        hijos.sort(key=f)

        # checamos cada hijo/sucesor
        for hijo in hijos:

          # verificamos si es un estado meta
          if funcionParo(hijo):
            return trayectoria(hijo),historia

          # comparamos si su costo total es menor a la cota
          # sí es menor, lo agregamos a la agenda solo si no ha sido expandido
          if f(hijo)<=c:
            if hijo not in expandidos:
              agenda.append(hijo)
          # si el costo total es mayor a la cota
          else:
            # cambiaremos el valor de "minimo" siempre y cuando el costo de
            # este nodo sea menor al valor actual de "minimo"
            if f(hijo)<minimo:
              minimo = f(hijo)

      # si el nodo ya esta expandido, lo sacamos tanto de la agenda como 
      # del conjunto de expandidos
      else:
        agenda.pop()
        expandidos.discard(nodo)

    # Sí la agenda está vacía
    # actualizamos la cota de profundidad como el valor de "minimo"
    c = minimo

    # Sí el valor de "minimo" permaneció en infinito, no hay solución
    if minimo == inf:
      return None

def Bidireccional(edoInicial,edoFinal):
  '''
  Implementación del algoritmo de busqueda bidireccional

  Parámetros
  edoInicial: Estado inicial
  edoFinal: Estado final 

  return: ruta del estado inicial al estado final
  '''
  historia = [(0,0)]
  # solución trivial
  if edoInicial == edoFinal:
    return trayectoria(edoInicial),hisotria

  # Inicializamos la frontera (agenda) hacia delante
  fronteraDelante = {edoInicial:edoInicial}

  # Inicializamos la frontera (agenda) hacia atras
  fronteraAtras = {edoFinal:edoFinal}

  # Inicializamos el conjunto de estados expandidos
  expandidos = {}

  # iniciamos el ciclo while que iterará hasta que las dos fronteras 
  # se toquen 
  while fronteraDelante and fronteraAtras:
    historia.append((len(fronteraDelante)+len(fronteraAtras),len(expandidos)))
    # fronteras temporales vacías
    fronteraDelanteAux = {}
    fronteraAtrasAux = {}

    # expansión de la frontera progresiva 
    # expandimos la frontera hacia delante
    for nodo in fronteraDelante:

      # agregamos el nodo visitado al diccionario de expandidos
      expandidos[nodo] = nodo

      # expandimos el nodo para encontrar sus hijos
      for hijo in nodo.expandir():

        # comparamos sí el hijo se encuentra en la otra frontera
        # (frontera hacia atras)
        if hijo in fronteraAtras:

          # obtenemos la ruta del inicio a este hijo
          a = trayectoria(hijo)

          # obtenemos la ruta del final a este hijo con la otra agenda
          #  o frontera (frontera hacia atras), lo volteamos
          b = trayectoria(fronteraAtras[hijo])
          b.reverse()
          print("Encontrada hacia delante")
          #regresamos la ruta quitando el nodo redundante
          return a+b[1:],historia

        # sí no está en la otra frontera, comparamos sí no se encuentran
        # ya expandidos, de esta manera los agregamos a la frontera temporal
        elif hijo not in expandidos:

          fronteraDelanteAux[hijo] = hijo

    fronteraDelante = fronteraDelanteAux

    # expandimos la frontera hacia atras
    for nodo in fronteraAtras:

      # agregamos el nodo visitado al diccionario de expandidos
      expandidos[nodo] = nodo

      # expandimos el nodo para encontrar sus hijos
      for hijo in nodo.expandir():

        # comparamos sí el hijo se encuentra en la otra frontera
        # (frontera hacia delante)
        if hijo in fronteraDelante:

          # obtenemos la ruta del inicio a este hijo con la otra agenda
          # o frontera (frontera hacia delante)
          a = trayectoria(fronteraDelante[hijo])

          # obtenemos la ruta del final a este hijo, lo volteamos
          b = trayectoria(hijo)
          b.reverse()
          print("Encontrada hacia atras")
          #regresamos la ruta quitando el nodo redundante
          return a+b[1:],historia

        # sí no está en la otra frontera, comparamos sí no se encuentran
        # ya expandidos, de esta manera los agregamos a la frontera temporal
        elif hijo not in expandidos:

          fronteraAtrasAux[hijo] = hijo

    fronteraAtras = fronteraAtrasAux

def AEstrellaModificado(origen,final,f1,f2):
  '''Función que implementa el algoritmo A*

  Parámetros
  origen: estado desde el que se inicia
  final: estado final

  return solución o nada
  '''

  historia = [(0,0)]
  # solución trivial
  if origen==final:
    return trayectoria(origen),historia

  # Inicializamos la frontera (agenda) hacia delante
  fronteraDelante = []

  # Inicializamos la frontera (agenda) hacia atras
  fronteraAtras = []

  # inicializamos el conjunto de expandidos
  expandidos = set()

  # agregamos el primer nodo a la agenda
  push(fronteraDelante,(f1(origen),origen))
  push(fronteraAtras,(f2(final),final))

  # mientras existan elementos en la agenda
  while len(fronteraAtras)>0 and len(fronteraDelante)>0:
    historia.append((len(fronteraDelante)+len(fronteraAtras),len(expandidos)))

    #auxiliares
    fronteraAtrasAux = []
    fronteraDelanteAux = []

    # sacamos un nodo de la agenda
    nodo = pop(fronteraDelante)[1] # pos 1 para obtener el estado de la tupla
    # lo agregamos al conjunto de expandidos
    expandidos.add(nodo)

    # comparamos si este nodo cumple con la función de paro
    fronteraAtrasAux = list(map(lambda s:s[1],fronteraAtras))
    if nodo in fronteraAtrasAux:
      # obtenemos la ruta del inicio a este hijo
      a = trayectoria(nodo)

      # obtenemos la ruta del final a este hijo con la otra agenda
      #  o frontera (frontera hacia atras), lo volteamos
      b = trayectoria(fronteraAtrasAux.pop(fronteraAtrasAux.index(nodo)))
      b.reverse()
      return a+b[1:],historia

    # expandimos el nodo y comparamos a sus hijos
    for hijo in nodo.expandir():

      # agregamos el hijo a la cola de prioridad
      # siempre y cuando no esté en el conjunto de expandidos
      if hijo not in expandidos:
          push(fronteraDelante,(f1(hijo),hijo))

    # sacamos un nodo de la agenda
    nodo = pop(fronteraAtras)[1] # pos 1 para obtener el estado de la tupla
    # lo agregamos al conjunto de expandidos
    expandidos.add(nodo)

    # comparamos si este nodo cumple con la función de paro
    fronteraDelanteAux = list(map(lambda s:s[1],fronteraDelante))
    if nodo in fronteraDelanteAux:
      # obtenemos la ruta del inicio a este hijo con la otra agenda
      # o frontera (frontera hacia delante)
      a = trayectoria(fronteraDelanteAux.pop(fronteraDelanteAux.index(nodo)))

      # obtenemos la ruta del final a este hijo, lo volteamos
      b = trayectoria(nodo)
      b.reverse()
      #regresamos la ruta quitando el nodo redundante
      return a+b[1:],historia

    # expandimos el nodo y comparamos a sus hijos
    for hijo in nodo.expandir():

      # agregamos el hijo a la cola de prioridad
      # siempre y cuando no esté en el conjunto de expandidos
      if hijo not in expandidos:
        push(fronteraAtras,(f2(hijo),hijo))

  # si no hay ruta, regresamos un vacio
  return [],historia 

# definimos una función para graficar el consumo de recursos del algoritmo
def dibujar_grafica(historia,algoritmo):
  n_agenda, n_expandidos = zip(*historia)
  n_total =  [a+b for a,b in historia]
  a,=plt.plot(n_agenda,'blue')
  b,=plt.plot(n_expandidos,'red')
  c,=plt.plot(n_total,'green')
  plt.legend([a,b,c],['agenda','expandidos','total'])
  plt.xlabel('iteración')
  plt.ylabel('estados')
  plt.title('Memoria consumida por '+algoritmo)
  plt.show()

def aplicandoAEstrella(origen,paro,g,h,heuristica):
  # APLICANDO EL ALGORITMO A* ****************************************************
  print("\n\n"+"*"*70)
  print("SOLUCIÓN POR A*",heuristica)
  # invocando a A*
  solución,historia = AEstrella(origen,paro,g,h)
  global historial
  historial = historia
  print("Profundidad de la solución:",solución[-1].get_profundidad())
  print("Memoria consumida:",historia[-1][0]+historia[-1][1])

def aplicandoBidireccional(origen,final):
  # APLICANDO EL ALGORITMO BIDIRECCIONAL ***************************************
  print("\n\n"+"*"*70)
  print("SOLUCIÓN POR BIDIRECCIONAL")
  # invocando al algoritmo bidireccional
  solución,historia = Bidireccional(origen,final)
  global historial
  historial = historia
  print("Profundidad de la solución:",len(solución)-1)
  print("Memoria consumida:",historia[-1][0]+historia[-1][1])

def aplicandoIDAEstrella(origen,paro,g,h,heuristica):
  # APLICANDO EL ALGORITMO IDA* ************************************************
  print("\n\n"+"*"*70)
  print("SOLUCIÓN POR IDA*",heuristica)
  # invocando a A*
  solución,historia = IDAEstrella(origen,paro,g,h)
  global historial
  historial = historia
  print("Profundidad de la solución:",solución[-1].get_profundidad())
  print("Memoria consumida:",max(historia,key= lambda s: s[0])[0]+max(historia,key= lambda s: s[1])[1])

def aplicandoAEstrellaM(origen,final,f1,f2):
  # APLICANDO EL ALGORITMO A* MODIFICADO*****************************
  print("\n\n"+"*"*70)
  print("SOLUCIÓN POR A* MODIFICADO")
  # invocando al algoritmo bidireccional con cota de profundidad
  solución,historia = AEstrellaModificado(origen,final,f1,f2)
  global historial
  historial = historia
  print(solución)
  print("Profundidad de la solución:",len(solución)-1)
  print("Memoria consumida:",historia[-1][0]+historia[-1][1])

def graficarTiempo(tiempos,labels=['A*', 'IDA*']):

  primeros = tiempos[0]
  segundos = tiempos[1]
  terceros = tiempos[2]
  x = np.arange(len(labels))  # the label locations
  width = 0.45  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/3, primeros, width/3, label='Esquinas')
  rects2 = ax.bar(x , segundos, width/3, label='Cruz')
  rects3 = ax.bar(x + width/3, terceros, width/3, label='Mix')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Tiempo')
  ax.set_title('Tiempo por algoritmo')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()
  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')
  autolabel(rects1)
  autolabel(rects2)
  autolabel(rects3)

  fig.tight_layout()

  plt.show()

def graficarTiempoTresAlgoritmos(tiempos,labels=['A*', 'IDA*','Bidireccional']):

  x = np.arange(len(labels))  # the label locations
  width=0.45
  fig, ax = plt.subplots()
  rects = ax.bar(x , tiempos, width, label='tiempo')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Tiempo')
  ax.set_title('Tiempo por algoritmo')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()

  plt.show()

def graficarMemoria(tamaños,labels=['A*', 'IDA*','Bidireccional']):

  x = np.arange(len(labels))  # the label locations
  width=0.45
  fig, ax = plt.subplots()
  rects = ax.bar(x , tamaños, width)

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Memoria consumida')
  ax.set_title('Memoria consumida por algoritmo')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  fig.tight_layout()

  plt.show()

def parte1():
  seed(1)
  # creamos el cubo y le damos 10 vueltas aleatorias
  cubo = RubikPuzzle()
  cubo.shuffle(10)
  print(cubo)
  global historial 
  # generamos g(s)
  g = lambda s: s.get_profundidad()
  # funcion de paro
  paro = lambda s: s==RubikPuzzle()
  
  # implementación de las heuristicas por bases de datos
  db1 = HeuristicaBasadaEnPatrones(profundidad=5)
  #db2 = HeuristicaBasadaEnPatrones(profundidad=5,patron="BDEFHKUVWhNXYZkPabcnSdefpsuvwxy")
  
  # generamos las heuristicas
  h1 = db1.heuristica
  #h2 = db2.heuristica
  #h3 = lambda s: max(h1(s),h2(s))
  
  # tiempos
  tiempos = []

  # aplicamos A estrella
  t=timeit.timeit(partial(aplicandoAEstrella,origen=cubo,
    paro=paro,g=g,h=h1,heuristica="Patrones esquinas"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos.append(round(t,2))
  dibujar_grafica(historial,"A* ")

  '''
  # aplicamos A estrella
  t=timeit.timeit(partial(aplicandoAEstrella,origen=cubo,
    paro=paro,g=g,h=h2,heuristica="Patrones cruz"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos[1].append(round(t,2))

  # aplicamos A estrella
  t=timeit.timeit(partial(aplicandoAEstrella,origen=cubo,
    paro=paro,g=g,h=h3,heuristica="Combinados"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos[2].append(round(t,2))
  '''
  # aplicamos IDA estrella
  t=timeit.timeit(partial(aplicandoIDAEstrella,origen=cubo,
    paro=paro,g=g,h=h1,heuristica="Patrones Esquinas"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos.append(round(t,2))
  dibujar_grafica(historial,"IDA* ")
  '''
  # aplicamos IDA estrella
  t=timeit.timeit(partial(aplicandoIDAEstrella,origen=cubo,
    paro=paro,g=g,h=h2,heuristica="Patrones cruz"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos[1].append(round(t,2))

  # aplicamos IDA estrella
  t=timeit.timeit(partial(aplicandoIDAEstrella,origen=cubo,
    paro=paro,g=g,h=h3,heuristica="Combinados"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos[2].append(round(t,2))
  '''
  # aplicamos bidireccional
  
  t=timeit.timeit(partial(aplicandoBidireccional,origen=cubo,
    final=RubikPuzzle()),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos.append(round(t,2))
  dibujar_grafica(historial,"Bidireccional ")
  print(tiempos)
  graficarTiempoTresAlgoritmos(tiempos)

def parte2():
  seed(1)
  # creamos el cubo y le damos 10 vueltas aleatorias
  cubo = RubikPuzzle()
  cubo.shuffle(10)
  print(cubo)
  global historial 
  # generamos g(s)
  g = lambda s: s.get_profundidad()
  # funcion de paro
  paro = lambda s: s==RubikPuzzle()
  
  # implementación de las heuristicas por bases de datos
  db1 = HeuristicaBasadaEnPatrones(profundidad=5)
  db2 = HeuristicaBasadaEnPatrones(objetivo=cubo,profundidad=5)

  # generamos las heuristicas
  h1 = db1.heuristica
  h2 = db2.heuristica

  # generamos las funciones de costo total
  f1 = lambda s: h1(s)+g(s)
  f2 = lambda s: h2(s)+g(s)
  # tiempos
  tiempos = []
  # tamaño
  tamaño = []
  # aplicamos A estrella
  t=timeit.timeit(partial(aplicandoAEstrella,origen=cubo,
    paro=paro,g=g,h=h1,heuristica="Patrones esquinas"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos.append(round(t,2))
  dibujar_grafica(historial,"IDA* ")
  tamaño.append(max(historial,key=lambda x: x[0])[0]+max(historial,key=lambda x: x[1])[1])

  # aplicamos IDA estrella
  t=timeit.timeit(partial(aplicandoIDAEstrella,origen=cubo,
    paro=paro,g=g,h=h1,heuristica="Patrones Esquinas"),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos.append(round(t,2))
  dibujar_grafica(historial,"IDA* ")
  tamaño.append(max(historial,key=lambda x: x[0])[0]+max(historial,key=lambda x: x[1])[1])

  # aplicamos A estrella modificada
  t=timeit.timeit(partial(aplicandoAEstrellaM,origen=cubo,
    final=RubikPuzzle(),f1=f1,f2=f2),number=1)
  print("Tiempo consumido: {0:4f} segundos".format(t))
  tiempos.append(round(t,2))
  dibujar_grafica(historial,"A* Modificiado")
  tamaño.append(max(historial,key=lambda x: x[0])[0]+max(historial,key=lambda x: x[1])[1])

  graficarTiempoTresAlgoritmos(tiempos,labels=["A*","IDA*","A* modificado"])
  graficarMemoria(tamaño,labels=["A*","IDA*","A* modificado"])

if __name__ == '__main__':
  parte1()
  parte2()
