from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

from game import Grid
import heapq

class GhostAgent( Agent ):
  def __init__( self, index ):
    self.index = index

  def getAction( self, state ):
    dist = self.getDistribution(state)
    if len(dist) == 0: 
      return Directions.STOP
    else:
      return util.chooseFromDistribution( dist )
    
  def getDistribution(self, state):
    "Returns a Counter encoding a distribution over actions from the provided state."
    util.raiseNotDefined()

class RandomGhost( GhostAgent ):


  class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left...

    def isEmpty(self):
      return len(self.heap) == 0
  
  class node:
    def __init__(self,x,y):
      self.loc = (x,y)
      self.path = []
      self.cost = 0
      #self.gameState = gameState
    def __eq__(self,other):
      self.loc == other.loc
    def __hash__(self):
      return hash(self.loc)


  def heuristic(self,startLoc,endLoc):
    #print startLoc,endLoc
    return abs(startLoc[0]-endLoc[0]) + abs(startLoc[1]-endLoc[1])

  def getActions(self,gameState,node):
    locX = node.loc[0]
    locY = node.loc[1]
    layout = gameState.getLayout()
    layoutRoom = gameState.getLayout().room
    #print len(layoutText),len(layoutText[0])

    legalActions = []
    if layoutRoom[locX-1][locY] != '%':
      legalActions.append('West')
    if layoutRoom[locX+1][locY] != '%':
      legalActions.append('East')
    if layoutRoom[locX][locY-1] != '%':
      legalActions.append('South')
    if layoutRoom[locX][locY+1] != '%':
      legalActions.append('North')
    #print legalActions
    return legalActions

  def A_star(self,startLoc,endLoc,heuristic,gameState):
    layout = gameState.getLayout()
    already_visited = Grid(layout.width,layout.height,False)
    pq = self.PriorityQueue()
    endNode = self.node(endLoc[0],endLoc[1])
    startNode = self.node(startLoc[0],startLoc[1])
    #print 'start actions: ', self.getActions(gameState,startNode)
    pq.update(startNode,self.heuristic(startNode.loc,endNode.loc))
    while not pq.isEmpty():
      currTotal = pq.removeMin()
      currNode = currTotal[0]
      if already_visited[currNode.loc[0]][currNode.loc[1]]: continue
      else: already_visited[currNode.loc[0]][currNode.loc[1]] = True
      if currNode.loc == endNode.loc:
        #print 'OLD',currNode.path
        #currNode.path.reverse()
        #print 'NEW',currNode.path
        #print already_visited
        if len(currNode.path) == 0 : return ('Stop',0)
        return (currNode.path[0] , currTotal[1])
      for action in self.getActions(gameState,currNode):
        newLoc = (-1,-1)
        if action == 'North':
          newLoc = (currNode.loc[0],currNode.loc[1]+1)
        elif action == 'South':
          newLoc = (currNode.loc[0],currNode.loc[1]-1)
        elif action == 'West':
          newLoc = (currNode.loc[0]-1,currNode.loc[1])
        elif action == 'East':
          newLoc = (currNode.loc[0]+1,currNode.loc[1])
        newNode = self.node(newLoc[0],newLoc[1])
        newNode.cost = currNode.cost + 1
        newNode.path = list(currNode.path)
        newNode.path.append(action)
        pq.update(newNode,newNode.cost+self.heuristic(newNode.loc,endNode.loc))
    #print already_visited


  "A ghost that chooses a legal action uniformly at random."
  def getDistribution( self, state ):
    def getActions(gameState,loc):
      locX = int(loc[0])
      locY = int(loc[1])
      layout = gameState.getLayout()
      layoutRoom = gameState.getLayout().room
      legalActions = []
      if layoutRoom[locX-1][locY] != '%':
        legalActions.append('West')
      if layoutRoom[locX+1][locY] != '%':
        legalActions.append('East')
      if layoutRoom[locX][locY-1] != '%':
        legalActions.append('South')
      if layoutRoom[locX][locY+1] != '%':
        legalActions.append('North')
      #print legalActions
      return legalActions
    dist = util.Counter()
    #for a in state.getLegalActions( self.index ): dist[a] = 1.0
    dist[state.getLegalActions(self.index)[0]] = 1.0
    #print dist
    pacmanLoc = state.getPacmanPosition()
    ghostLoc = state.getGhostPosition(self.index)
    best_action = Directions.STOP
    best_priority = -1
    #print state.getLegalActions(self.index)
    for a in state.getLegalActions( self.index ): 
      if a == 'North':
        newGhostLoc = (ghostLoc[0],ghostLoc[1]+1)
      elif a == 'South':
        newGhostLoc = (ghostLoc[0],ghostLoc[1]-1)
      elif a == 'West':
        newGhostLoc = (ghostLoc[0]-1,ghostLoc[1])
      elif a == 'East':
        newGhostLoc = (ghostLoc[0]+1,ghostLoc[1])
      priority = self.A_star(pacmanLoc,newGhostLoc,self.heuristic,state)
      #print priority[0] , priority[1]
      dist[a] = priority[1] * len(getActions(state,newGhostLoc))
    #   if priority[1] > best_priority:
    #     best_priority = priority[1]
    #     best_action = a
    # dist[best_action] = 1.0
    #print dist
    if len(dist) > 0:
      dist.normalize()
    dist[Directions.STOP] = 0.1
    dist.normalize()
    return dist

class DirectionalGhost( GhostAgent ):
  "A ghost that prefers to rush Pacman, or flee when scared."
  def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
    self.index = index
    self.prob_attack = prob_attack
    self.prob_scaredFlee = prob_scaredFlee
      
  def getDistribution( self, state ):
    # Read variables from state
    ghostState = state.getGhostState( self.index )
    legalActions = state.getLegalActions( self.index )
    pos = state.getGhostPosition( self.index )
    isScared = ghostState.scaredTimer > 0
    
    speed = 1
    if isScared: speed = 0.5
    
    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
      bestScore = max( distancesToPacman )
      bestProb = self.prob_scaredFlee
    else:
      bestScore = min( distancesToPacman )
      bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
    
    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist
