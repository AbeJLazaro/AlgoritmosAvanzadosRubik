from functools import reduce
from termcolor import colored
from random import choice
from itertools import product
from estadoBase import EstadoProblema
# estructuras de datos
from collections import deque
from heapq import heappush as push
from heapq import heappop as pop

# función para obtener la trayectoria desde un estado hasta su origen
def trayectoria(final):
  # Ocupamos una cola 
  ruta = deque()
  # agregamos el estado final o meta
  ruta.append(final)
  # mientras exista un padre (el que ya no tendrá padre será el
  # nodo principal raíz)
  while final.get_padre():
      # movemos el apuntador al nodo padre
      final = final.get_padre()
      # lo agregamos a la lista
      ruta.append(final)
  # invertimos el orden de la secuencia
  ruta.reverse()
  # lo regresamos como una lista
  return list(ruta)

# Códigos de colores, se ocupa cyan ya que no hay naranja :c
# mantendremos los nombres en inglés dado que en español algunos 
# chocan como amarillo y azul
# Blanco
W = 0
# Verde
G = 1
# Rojo
R = 2
# Azul
B = 3
# Azul cielo
C = 4
# Amarillo
Y = 5

# Diccionario con los nombres para los códigos
mapaColor = {
    0:"white",
    1:"green",
    2:"red",
    3:"blue",
    4:"cyan",
    5:"yellow"}

'''
Modelaremos el problema como una secuencia de
letras
Cada letra será una posición relativa a los
colores del centro de cada cara ya que estos
no se mueven

Para modelarlo, imaginamos el cubo desdoblado

    ABC
    DEF
    GHI
JKL MNÑ OPQ RST
UVW XYZ abc def
ghi jkl mnñ opq
    rst
    uvw
    xyz

Las letras que no se mueven son:
  E,V,Y,b,e,v

El cubo esta codificado como una secuencia de bits
Cada cuadrito esta configurado como un conjunto de 3
bits para representar su color
Por ejemplo, el código de colores con 3 bits es

000 blanco (white)
001 verde (green)
010 rojo (red)
011 azul (blue)
100 cian (cyan)
101 amarillo (yellow)
'''

'''
El siguiente código representa a que letra
le toca que color
Se espacian de 3 en 3 el primer elemento de la tupla
referente a la letra por los 3 bits que necesita
cada letra para representar su color
'''
codigo ={
  'A' : (0,W),
  'B' : (3,W),
  'C' : (6,W),
  'D' : (9,W),
  'E' : (12,W),
  'F' : (15,W),
  'G' : (18,W),
  'H' : (21,W),
  'I' : (24,W),
  'J' : (27,G),
  'K' : (30,G),
  'L' : (33,G),
  'M' : (36,R),
  'N' : (39,R),
  'Ñ' : (42,R),
  'O' : (45,B),
  'P' : (48,B),
  'Q' : (51,B),
  'R' : (54,C),
  'S' : (57,C),
  'T' : (60,C),
  'U' : (63,G),
  'V' : (66,G),
  'W' : (69,G),
  'X' : (72,R),
  'Y' : (75,R),
  'Z' : (78,R),
  'a' : (81,B),
  'b' : (84,B),
  'c' : (87,B),
  'd' : (90,C),
  'e' : (93,C),
  'f' : (96,C),
  'g' : (99,G),
  'h' : (102,G),
  'i' : (105,G),
  'j' : (108,R),
  'k' : (111,R),
  'l' : (114,R),
  'm' : (117,B),
  'n' : (120,B),
  'ñ' : (123,B),
  'o' : (126,C),
  'p' : (129,C),
  'q' : (132,C),
  'r' : (135,Y),
  's' : (138,Y),
  't' : (141,Y),
  'u' : (144,Y),
  'v' : (147,Y),
  'w' : (150,Y),
  'x' : (153,Y),
  'y' : (156,Y),
  'z' : (159,Y) 
}

# espacios en blanco, nos servira cuando
# hagamos las impresiones de los cubos
BLANK = " "*6

# caracter de llenado chr(FILL)= █, con
# este caracter de llenado crearemos nuestros
# cubos, solo pintandolo y concatenandolo
LLENO = 9608

# cuantas veces aparecerá el caracter de
# llenado
K = 2

# ACCIONES POSIBLES
'''
Esta parte es complicada
Las acciones son una lista de listas de listas de tuplas
      [
        [
          [(),(),... ],
          ... 
        ],
        ... 
      ]
Estas acciones representan giros de 90 grados en todas
las caras del cubo
'''
# la lista más exterior representa el conjunto de 
# todos los movimientos
acciones = [
  # la primer lista representa al eje x
  [
    # La primera lista de la lista que representa al eje x
    # Es la lista de tuplas que indican como 
    # se intercambiaran las letras al aplicar la acción
    # acción: Giro de 90° en dirección de las manecillas del reloj
    # en la cara inferior visto desde arriba
    # Por ejemplo, ("A","g") indica que "g" tomará la posición
    # de "A" al rotar este eje
    [('A','g'),('B','U'),('C','J'),('Q','A'),('c','B'),
     ('ñ','C'),('z','Q'),('y','c'),('x','ñ'),('g','z'),
     ('U','y'),('J','x'),('R','T'),('S','f'),('T','q'),
     ('d','S'),('f','p'),('o','R'),('p','d'),('q','o')],
    # La segunda lista de la lista que representa al eje x
    # Es la lista de tuplas que indican como se 
    # intercambiaran las letras al aplicar la acción
    # acción: Giro de 90° en dirección a las manecillas del reloj
    # en la cara superior visto desde arriba
     [('G','i'),('H','W'),('I','L'),('O','G'),('a','H'),
     ('m','I'),('t','O'),('s','a'),('r','m'),('i','t'),
     ('W','s'),('L','r'),('M','j'),('N','X'),('Ñ','M'),
     ('Z','N'),('l','Ñ'),('k','Z'),('j','l'),('X','k')]
  ],

  # la segunda lista representa el eje y
  [
    # Giro de 90 grados hacia el frente de la cara
    # que queda del lado izquierdo
    [('A','q'),('D','f'),('G','T'),('M','A'),('X','D'),
     ('j','G'),('r','M'),('u','X'),('x','j'),('T','x'),
     ('f','u'),('q','r'),('J','g'),('K','U'),('L','J'),
     ('U','h'),('W','K'),('g','i'),('h','W'),('i','L')],
    # Giro de 90 grados hacia el frente de la cara
    # que queda del lado derecho
    [('C','o'),('F','d'),('I','R'),('Ñ','C'),('Z','F'),
     ('l','I'),('t','Ñ'),('w','Z'),('z','l'),('o','t'),
     ('d','w'),('R','z'),('O','Q'),('P','c'),('Q','ñ'),
     ('a','P'),('c','n'),('m','O'),('n','a'),('ñ','m')]
  ],

  # la tercera lista representa al eje z
  [    
    # Giro de 90 grados hacia la derecha de la cara
    # que se encuentra arriba en la figura 2D
    [('J','R'),('K','S'),('L','T'),('M','J'),('N','K'),
     ('Ñ','L'),('O','M'),('P','N'),('Q','Ñ'),('R','O'),
     ('S','P'),('T','Q'),('G','A'),('H','D'),('I','G'),
     ('F','H'),('C','I'),('B','F'),('A','C'),('D','B')], 
    # Giro de 90 grados hacia la derecha de la cara
    # que se encuentra abajo en la figura 2D
    [('g','o'),('h','p'),('i','q'),('j','g'),('k','h'),
     ('l','i'),('m','j'),('n','k'),('ñ','l'),('o','m'),
     ('p','n'),('q','ñ'),('r','x'),('s','u'),('t','r'),
     ('u','y'),('w','s'),('x','z'),('y','w'),('z','t')]
  ]
]

'''
Las listas de movimientos se pueden ver de la siguiente
jerarquía

visto desde arriba:
acciones
  eje x
    giro de la cara inferior(la que esta atrás) 90° manecillas del reloj
    giro de la cara superior(la que esta de frente) 90° manecillas del reloj
  eje y
    giro de la cara izquierda 90° hacia enfrente
    giro de la cara derecha 90° hacia enfrente
  eje z
    giro de la cara que queda arriba 90° derecha
    giro de la cara que quda abajo 90° derecha

Solo consideramos giros hacia una dirección en cada eje, para llegar a dar
un giro hacia el otro sentido en cualquiera de los 3 ejes, necesitamos
invertir el orden de las tuplas de cualquier lista

También consideramos para cada eje movimientos en dos líneas nada más, esto
con la intención de siempre permanecer con los centros inmovibles
'''

# calcula la configuración ordenada del cubo
'''
En esta parte se genera la configuración inicial de un cubo de rubik ordenado
Al igual que para el Puzzle15, esta basada cada posición en un grupo de bits,
en este caso, en grupos de 3 que son los necesarios para representar los colores
definidos arriba
Cada 3 bits representan un cuadrito de color y el color de este
'''
configuracionInicial = reduce(lambda x,y:(0,x[1]|(y[1]<<y[0])), \
[(0,0)]+[v for k,v in codigo.items()])[1]

# Clase RubikPuzzle
class RubikPuzzle(EstadoProblema):
  '''Cubo de Rubik de 3x3
  Implementación con todos los cuadritos/subcubos
  Cada subcubo es un conjunto de 3 bits (terna de bits)
  que codifica su color'''
  def __init__(self, padre=None,accion=None,profundidad=0,patron=None):
    '''
    Constructor del estado-nodo cubo rubik

    Parámetros
    padre: padre de este nodo
    accion: acción que toma el padre para llegar a este nodo
    profundidad: profundidad del nodo
    patron: diccionario con la configuración a establecer en el nodo
    '''
    self.padre = padre
    self.profundidad = profundidad
    if padre!=None and accion!=None:
      # si existe un padre y una acción, se crea el cubo
      # rubik/nodo a partir de esta información
      self.configuracion = padre.configuracion
      # Se aplica la acción que el padre hace para llegar a este
      # nodo para representar el nodo con ese movimiento
      self.aplicar(accion)
    # si no tiene padre, pero se pasó una configuración base
    elif patron!=None:
      self.configuracion = self.inicializar(patron)
    # si no tiene padre ni patron de construcción, se crea uno ordenado
    else:
      self.configuracion = configuracionInicial

  def inicializar(self,patron):
    '''
    Establece una configuración al cubo por medio de un patrón

    Parámetros
    patron: configuración a establecer

    return configuración codificada en bits
    '''

    # la configuración a establecer esta en un diccionario
    # en forma de {letra:código de color}
    return reduce(lambda x,y: x|y,\
      [val<<(codigo[key][0]) for key,val in patron.items()])

  def cubo(self,simbolo):
    '''
    Un cuadrito o subcubo que se mostrará

    Parámetros
    simbolo: letra de la posición a mostrar 

    return cadena como un subcubo o cuadrito
    '''
    # se obtiene el número de la posición de la letra
    # en base al diccionario codigo
    n = codigo[simbolo][0]
    '''colored cambia el color de un caracter
    chr(LLENO) nos arroja el cuadrito visto anteriormente
    Para encontrar el color con el que pintaremos el cuadrito será
    con el diccionario mapaColor, aplicaremos una mascara
    para encontrar los 3 bits de un cuadrito en la configuración
    y con esos 3 bits encontraremos el número de color al que se refiere,
    lo pintaremos y lo repetiremos dos veces, lo retornamos
    '''
    return colored(chr(LLENO),\
      mapaColor[(((7<<n)&self.configuracion)>>n)])*K
  
  def aplicar(self,accion):
    '''
    aplica una acción a la configuración del nodo
    
    Parámetros
    accion: acción que se aplicará
    '''

    # la acción se representa con una tupla
    # (eje, renglon, dirección)
    
    # checamos si la dirección esta como 0, de izquierda a derecha
    if accion[2]==0:
      '''reduce nos arrojará dos resultados
      con una función lambda que regresa una tupla de dos elementos
        el primer elemento representa un or entre el primer elemento de
          dos tuplas
        el segundo elemento representa un or entre el segundo elemento de
          dos tuplas
      la secuencia que tomará reduce es una secuencia de movimientos
      para cada valor en la lista de tuplas de acciones, seleccionada por
      medio de los primeros dos valores de la tupla parámetro accion
      '''
      mover,mascara=reduce(lambda x,y:(x[0]|y[0],x[1]|y[1]),\
            [self.mover(x) for x in acciones[accion[0]][accion[1]]])
    # checamos si la dirercción esta en otro valor representando un giro de
    # derecha a izquierda
    else:
      '''
      En esta parte hacemos lo mismo, solo que invertimos el orden de la
      tupla de movimiento, para hacer un cambio de dirección
      '''
      mover,mascara = reduce(lambda x,y:(x[0]|y[0],x[1]|y[1]),\
            [self.mover((b,a)) for a,b in acciones[accion[0]][accion[1]]])

    # teniendo la máscara de bits movidos y el bloque movido, se refleja el 
    # cambio en la configuración del bloque
    self.configuracion = mover | ((((2<<162)-1)^mascara)&self.configuracion)

  def mover(self,lugares):
    '''
    Mueve los cuadritos de lugar

    Parámetros
    lugar: tupla que indica a donde mover que color

    return tupla con el bloque movido y la máscara de bits
    '''

    # de la posición i a la j
    i = codigo[lugares[0]][0]
    j = codigo[lugares[1]][0]
    #regresa tanto el bloque movido como la máscara
    return (((((7<<i)&self.configuracion)>>i)<<j),(7<<i)|(7<<j))

  def __str__(self):
    '''
    El cubo en forma de cadena
    return: representación del cubo en texto
    '''
    return ('\n'+
    BLANK+self.cubo('A')+self.cubo('B')+self.cubo('C')+'\n'+
    BLANK+self.cubo('D')+self.cubo('E')+self.cubo('F')+'\n'+
    BLANK+self.cubo('G')+self.cubo('H')+self.cubo('I')+'\n'+
    self.cubo('J')+self.cubo('K')+self.cubo('L')+
    self.cubo('M')+self.cubo('N')+self.cubo('Ñ')+
    self.cubo('O')+self.cubo('P')+self.cubo('Q')+  
    self.cubo('R')+self.cubo('S')+self.cubo('T')+'\n'+
    self.cubo('U')+self.cubo('V')+self.cubo('W') +
    self.cubo('X')+self.cubo('Y')+self.cubo('Z') +
    self.cubo('a')+self.cubo('b')+self.cubo('c')+ 
    self.cubo('d')+self.cubo('e')+self.cubo('f') +'\n'+        
    self.cubo('g')+self.cubo('h')+self.cubo('i')+
    self.cubo('j')+self.cubo('k')+self.cubo('l') +
    self.cubo('m')+self.cubo('n')+self.cubo('ñ')+ 
    self.cubo('o')+self.cubo('p')+self.cubo('q') +'\n'+                
    BLANK+self.cubo('r')+self.cubo('s')+self.cubo('t')+'\n'+
    BLANK+self.cubo('u')+self.cubo('v')+self.cubo('w')+'\n'+
    BLANK+self.cubo('x')+self.cubo('y')+self.cubo('z')+'\n' )

  def __repr__(self):
    '''return representación visual del cubo'''
    return self.__str__()

  def __eq__(self,otro):
    '''Nos ayuda a comparar un nodo/cubo con otro

    Parámetros
    otro: cubo o nodo con el que se compara

    return True o False si son iguales en la comparación'''

    return (isinstance(otro, self.__class__)) and \
        (self.configuracion==otro.configuracion)

  def __ne__(self,otro):
    '''Comparación de desigualdad

    Parámetros
    otro: cubo o nodo con el que se compara

    return True si son diferentes, False si son iguales'''

    return not self.__eq__(otro)

  def __lt__(self,otro):
    '''
    less than determina si la profundidad del nodo es menor que 
    la de otro

    Paŕametros
    otro: cubo o nodo con el que se compara

    return True si la profundidad del nodo es menor a la de otro
    '''

    return self.profundidad<otro.profundidad

  def __hash__(self):
    """
    Función de hash para un cubo
    :return: un entero hash 
    """
    return hash(self.configuracion)

  def igualPatron(self,patron,objetivo=configuracionInicial):
    """
    Determina si el cubo es parte de un patrón

    Parámetros
    patron: patrón a verificar
    objetivo(opcional): meta

    return: True si el patrón incluye la configuración del cubo
    """
    mascara = RubikPuzzle.get_mascaraPatron(patron)
    return ((mascara&self.configuracion)^(mascara&objetivo))==0

  @staticmethod
  def get_mascaraPatron(patron):
    '''
    Calcula la máscara de bits para extraer los patrones 
    que se pasan como argumento

    Parámetros
    patron: patrón que define la mascara

    return mascara de bits
    '''

    return reduce(lambda x,y:x|y,[(7<<codigo[letra][0])\
        for letra in patron])

  def get_padre(self):
    return self.padre

  def get_profundidad(self):
    return self.profundidad

  def shuffle(self,n):
    '''
    Desordena el cubo

    Parámetros
    n: cantidad de movimientos aleatorios
    '''

    for i in range(n):
      # tupla que representa la acción
      self.aplicar((choice([0,1,2]),choice([0,1]),choice([0,1])))

  def expandir(self):
    # La función lambda comprueba que no aparezca el padre
    # Es fácil generar los movimientos ya que se puede mover a 
    # cualquier lado y escoger cualquier acción de acciones
    return list(filter(lambda x: (x!=self.padre), \
      [RubikPuzzle(self,accion,self.profundidad+1) \
      for accion in product([0,1,2],[0,1],[0,1])]))

if __name__ == '__main__':
  cubo = RubikPuzzle()
  print(cubo)
  # cada cubo tiene 12 hijos posibles máximo, arborescencia o
  # ramificación
  sucesores = cubo.expandir()
  #print(sucesores)
  print("Cantidad de hijos:",len(sucesores))
  cubo.shuffle(5)
  print(cubo)