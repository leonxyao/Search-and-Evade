from util import manhattanDistance
from game import Grid
import os
import random

from collections import defaultdict

VISIBILITY_MATRIX_CACHE = {}
pacmanStartLoc = (-1,-1)
ghostStartLoc = (-1,-1)

class Layout:
  """
  A Layout manages the static information about the game board.
  """
  
  def __init__(self, layoutText, roomText):
    self.width = len(layoutText[0])
    self.height= len(layoutText)
    self.walls = Grid(self.width, self.height, False)
    self.doors = Grid(self.width, self.height, False)
    self.food = Grid(self.width, self.height, False)
    self.capsules = []
    self.agentPositions = []
    self.numGhosts = 0
    self.layoutText = layoutText
    self.roomText = roomText

    self.room = Grid(self.width, self.height, '%')
    self.rooms_mapping = defaultdict(list)

    self.processLayoutText(layoutText,roomText)

    #print self.rooms_mapping
    # self.initializeVisibilityMatrix()
    
  def getNumGhosts(self):
    return self.numGhosts
    
  def initializeVisibilityMatrix(self):
    global VISIBILITY_MATRIX_CACHE
    if reduce(str.__add__, self.layoutText) not in VISIBILITY_MATRIX_CACHE:
      from game import Directions
      vecs = [(-0.5,0), (0.5,0),(0,-0.5),(0,0.5)]
      dirs = [Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
      vis = Grid(self.width, self.height, {Directions.NORTH:set(), Directions.SOUTH:set(), Directions.EAST:set(), Directions.WEST:set(), Directions.STOP:set()})
      for x in range(self.width):
        for y in range(self.height):
          if self.walls[x][y] == False:
            for vec, direction in zip(vecs, dirs):
              dx, dy = vec
              nextx, nexty = x + dx, y + dy
              while (nextx + nexty) != int(nextx) + int(nexty) or not self.walls[int(nextx)][int(nexty)] :
                vis[x][y][direction].add((nextx, nexty))
                nextx, nexty = x + dx, y + dy
      self.visibility = vis      
      VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)] = vis
    else:
      self.visibility = VISIBILITY_MATRIX_CACHE[reduce(str.__add__, self.layoutText)]
      
  def isWall(self, pos):
    x, col = pos
    return self.walls[x][col]

  def isDoor(self,pos):
    x, col = pos
    return self.walls[x][col]
  
  def getRandomLegalPosition(self):
    x = random.choice(range(self.width))
    y = random.choice(range(self.height))
    while self.isWall( (x, y) ):
      x = random.choice(range(self.width))
      y = random.choice(range(self.height))
    return (x,y)

  def getRandomCorner(self):
    poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
    return random.choice(poses)

  def getFurthestCorner(self, pacPos):
    poses = [(1,1), (1, self.height - 2), (self.width - 2, 1), (self.width - 2, self.height - 2)]
    dist, pos = max([(manhattanDistance(p, pacPos), p) for p in poses])
    return pos
  
  def isVisibleFrom(self, ghostPos, pacPos, pacDirection):
    row, col = [int(x) for x in pacPos]
    return ghostPos in self.visibility[row][col][pacDirection]
  
  def __str__(self):
    return "\n".join(self.layoutText)
    
  def deepCopy(self):
    return Layout(self.layoutText[:],self.roomText[:])
    
  def processLayoutText(self, layoutText,roomText):
    """
    Coordinates are flipped from the input format to the (x,y) convention here
    
    The shape of the maze.  Each character  
    represents a different type of object.   
     % - Wall                               
     . - Food
     o - Capsule
     G - Ghost
     P - Pacman
     | - Door
    Other characters are ignored.
    """
    # x = 1
    # assert(x==2)
    maxY = self.height - 1
    for y in range(self.height):       
      for x in range(self.width):
        layoutChar = layoutText[maxY - y][x]  
        roomChar = roomText[maxY - y][x]
        self.processLayoutChar(x, y, layoutChar)
        self.processRoomChar(x, y, roomChar)
    self.agentPositions.sort()
    self.agentPositions = [ ( i == 0, pos) for i, pos in self.agentPositions]
    for y in range(self.height):
      for x in range(self.width):
        room = self.room[x][y]
        if room != '%' and room != '|': 
          self.rooms_mapping[room].append((x,y))
        elif room == '|':
          self.rooms_mapping['door'].append((x,y))
        elif room == '%':
          self.rooms_mapping['wall'].append((x,y))
  
  def processLayoutChar(self, x, y, layoutChar):
    global pacmanStartLoc
    global ghostStartLoc
    if layoutChar == '%':      
      self.walls[x][y] = True
    elif layoutChar == '|':
      self.doors[x][y] = True
    elif layoutChar == '.':
      self.food[x][y] = True 
    elif layoutChar == 'o':    
      self.capsules.append((x, y))   
    elif layoutChar == 'P':    
      self.agentPositions.append( (0, (x, y) ) )
      pacmanStartLoc = (x,y)
    elif layoutChar in ['G']:    
      self.agentPositions.append( (1, (x, y) ) )
      ghostStartLoc = (x,y)
      self.numGhosts += 1
    elif layoutChar in  ['1', '2', '3', '4']:
      self.agentPositions.append( (int(layoutChar), (x,y)))
      self.numGhosts += 1 

  def processRoomChar(self, x, y, roomChar):
    if roomChar == '%' or roomChar == '|':
      self.room[x][y] = roomChar
    elif roomChar.isalpha() and roomChar.islower:
      self.room[x][y] = roomChar
    elif roomChar == 'H':
      self.room[x][y] = 'H'

def getLayout(name, back = 2):
  if name.endswith('.lay'):
    name = name[:len(name)-4]
    layout = tryToLoad('layouts/' + name)
    if layout == None: layout = tryToLoad(name)
  else:
    layout = tryToLoad('layouts/' + name)
    if layout == None: layout = tryToLoad(name)
  if layout == None and back >= 0:
    curdir = os.path.abspath('.')
    os.chdir('..')
    layout = getLayout(name, back -1)
    os.chdir(curdir)
  return layout

# def getRoom(name, back = 2):
#   if name.endswith('.lay'):
#     layout = tryToLoad('layouts/' + name + 'room')
#     if layout == None: layout = tryToLoad(name + 'room')
#   else:
#     layout = tryToLoad('layouts/' + name + 'room' + '.lay')
#     if layout == None: layout = tryToLoad(name + '.lay')
#   if layout == None and back >= 0:
#     curdir = os.path.abspath('.')
#     os.chdir('..')
#     layout = getRoom(name, back -1)
#     os.chdir(curdir)
#   return layout

def tryToLoad(fullname):
  if(not os.path.exists(fullname + '.lay')): return None
  f = open(fullname + '.lay')
  f_room = open(fullname + 'room.lay')
  try: return Layout([line.strip() for line in f],[line.strip() for line in f_room] )
  finally: f.close()
