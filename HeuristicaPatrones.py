'''
Autores:      Lázaro Martínez Abraham Josué

Versión:      1.1
Fecha:        26 de enero de 2021
Nombre:       HeuristicaPatrons.py
'''
# estructuras de datos
from collections import deque
from heapq import heappush as push
from heapq import heappop as pop
from sys import path
path.append("../")
path.append("../problemasBase")
path.append("../BusquedaNoInformada")
# problema
from problemasBase import RubikPuzzle
from problemasBase.RubikPuzzle import RubikPuzzle 
from problemasBase.puzzle15 import trayectoria
# convenientes
from random import seed

class HeuristicaBasadaEnPatrones:
  '''
  Implementación de la función heuristica para el cubo rubik
  basada en una base de datos de patrones 
  '''
  def __init__(self,objetivo=None,profundidad=6,patron=None):
    '''
    Constructor de la base de datos de patrones
    
    Parámetros
    objetivo: Estado meta
    profundidad: profundidad máxima de los estados de la base
    patron: patron con el que se forma la base
    '''
    print("Calculando base de datos de patrones...")
    
    # si no se establece un objetivo, se ocupa el cubo ordenado
    if objetivo==None:
      objetivo = RubikPuzzle()

    # para generar la base de datos ocupamos una búsqueda BFS
    agenda = deque()
    self.explorados = set()
    self.profundidad = profundidad

    # agregamos el estado objetivo como nodo inicial
    agenda.append(objetivo)

    # Inicializamos la base de datos como un diccionario
    self.patrones = {}

    # si no se especifica el patrón, usaremos las esquinas
    if patron == None:
      patron = 'ACGIJLgiMÑjlOQmñRToqrtxz'
    self.patron = patron

    # obteniendo la mascara para ese patrón
    self.mascaraPatron = RubikPuzzle.get_mascaraPatron(patron)
    '''
    a=bin(self.mascaraPatron)[2:]
    b=bin(self.mascaraPatron&objetivo.configuracion)[2:]
    for i in range(0,len(a),3):
      print(a[i:i+3]," - ",b[i:i+3])
    '''
    # mientras haya elementos en la agenda
    while agenda:
      # sacamos el frente de la cola
      nodo = agenda.popleft()
      # lo agregamos a expandidos
      self.explorados.add(nodo)

      # obtenemos la configuración del nodo
      conf = self.mascaraPatron&nodo.configuracion

      # agregamos la subconfiguración a la base de datos
      # si es la primera vez que la descubrimos le asociamos
      # la profundidad
      if conf not in self.patrones:
        self.patrones[conf] = nodo.profundidad

      for hijo in nodo.expandir():
        if hijo.profundidad>profundidad:
          # terminamos
          print("Listo!")
          return
        elif hijo not in self.explorados:
          # agregamos al hijo en caso de que o se haya 
          # expandido
          agenda.append(hijo)

  def heuristica(self,puzzle):
    '''
    Calcua la heuristica usando la base de datos
    '''
    key = self.mascaraPatron&puzzle.configuracion
    return (self.patrones[key] if key in self.patrones else self.profundidad+1)

if __name__ == '__main__':
  otro = HeuristicaBasadaEnPatrones()
  