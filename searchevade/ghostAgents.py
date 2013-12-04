from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

from game import Grid
from random import choice 
import heapq

import MDP
import MDPUtil
from sets import Set

policy = []
haveCalculated = False

possiblePacmanStates = util.Counter()
possiblePacmanStates[(1,11)] = 1.0
frontierStates = Set()
frontierStates.add((1,11))

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
    locX = int(node.loc[0])
    locY = int(node.loc[1])
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
      if already_visited[int(currNode.loc[0])][int(currNode.loc[1])]: continue
      else: already_visited[int(currNode.loc[0])][int(currNode.loc[1])] = True
      if currNode.loc == endNode.loc:
        if len(currNode.path) == 0 : return ('Stop',0)
        #print currNode.path[0]
        return (currNode.path[0],currTotal[1])
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
    print already_visited

  def findFurthestLoc(self,state,pacmanLoc,ghostLoc):
    layout = state.getLayout()
    room = layout.room
    bestLocs = [((-1,-1),-1)]
    for x in range(room.width):
      for y in range(room.height):
        if room[x][y] == '%': continue
        dist = self.heuristic(pacmanLoc,(x,y))
        #print x,y,dist
        if len(bestLocs) > 10:
          if dist > bestLocs[len(bestLocs)-1][1]:
              bestLocs[len(bestLocs)-1] = ((x,y),dist)
              bestLocs = sorted(bestLocs,key = lambda loc: loc[1],reverse = True)
        else:
          bestLocs.append(((x,y),dist))
    # closestDistance = 100000000000
    # closestLoc = (-1,-1)
    # for loc in bestLocs:
    #   ghostDist = self.heuristic(loc[0],ghostLoc)
    #   if ghostDist < closestDistance:
    #     closestLoc = loc[0]
    terminalState = ((-1,-1),-1)
    if terminalState in bestLocs:
      bestLocs.remove(terminalState)
    closestLoc = choice(bestLocs)[0]
    #print bestLocs, closestLoc
    return closestLoc

  def getCloseDistribution( self, state ):
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
      return legalActions
    dist = util.Counter()
    pacmanLoc = state.getPacmanPosition()
    ghostLoc = state.getGhostPosition(self.index)
    best_action = Directions.STOP
    best_priority = -1
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
      if priority[1] > best_priority:
        best_priority = priority[1]
        best_action = a

    dist[best_action] = 1.0
    if len(dist) > 0:
      dist.normalize()
    # dist[Directions.STOP] = 0.1
    # dist.normalize()
    return dist

  "A ghost that chooses a legal action uniformly at random."
  def getFarDistribution( self, state ):
    dist = util.Counter()
    pacmanLoc = state.getPacmanPosition()
    ghostLoc = state.getGhostPosition(self.index)
    #print pacmanLoc, ghostLoc
    furthestLoc = self.findFurthestLoc(state,pacmanLoc,ghostLoc)
    action = self.A_star(ghostLoc,furthestLoc,self.heuristic,state)[0]
    #print action
    dist[action] = 1.0
    # dist[Directions.STOP] = 0.1
    dist.normalize()
    #print dist
    return dist


  def getPolicyDistribution(self,state):
    global haveCalculated
    global policy
    dist = util.Counter()
    if not haveCalculated:
      algorithm = MDP.PolicyIteration()
      mdp = MDPUtil.SearchEvadeMDP(state)
      algorithm.solve(mdp,0.001)
      mdp.computeStates()
      policy = algorithm.pi
      haveCalculated = True
      print 'MDP calculated'
      for pair in policy.keys():
        print pair[0],pair[1],pair[2],policy[pair]
    pacmanLoc = state.getPrevPacmanPosition()
    ghostLoc = state.getGhostPosition(self.index)
    print 'getPolicyDist: ', pacmanLoc,ghostLoc,'False'
    action = policy[(pacmanLoc,ghostLoc,False)]
    if action == (-1,0):
      dist['West'] = 1.0
    elif action == (1,0):
      dist['East'] = 1.0
    elif action == (0,-1):
      dist['South'] = 1.0
    elif action == (0,1):
      dist['North'] = 1.0
    elif action == (0,0):
      dist['Stop'] = 1.0
    print 'Distribution: ', dist
    return dist

  def convertAction(self, action): 
        if action == "West":
            return(-1,0)
        elif action == "East":
            return (1,0)
        elif action == "South":
            return (0,-1)
        elif action == "North":
            return (0,1)

  def getDarkRoomDistribution(self,state):
    ghostLoc = state.getGhostPosition(self.index)
    dist = util.Counter()
    onRooms = state.data.roomsOn
    room = state.getLoctoRoom()
    locationQ = [state.getGhostPosition(self.index)]
    while len(locationQ)>0:
      currLoc = locationQ.pop(0)
      # print 'currLoc: ',currLoc
      roomLetter = room[int(currLoc[0])][int(currLoc[1])]
      if (roomLetter not in onRooms) and (roomLetter != '|'):
        action = self.A_star(ghostLoc,currLoc,self.heuristic,state)[0]
        dist[action] = 1.0
        return dist
      else:
        currNode = self.node(currLoc[0],currLoc[1])
        possibleActions = self.getActions(state,currNode)
        for action in possibleActions:
          actionTuple = self.convertAction(action)
          newLoc = (currLoc[0]+actionTuple[0],currLoc[1]+actionTuple[1])
          if newLoc in locationQ: continue
          locationQ.append(newLoc)
    return self.getFarDistribution(state)




  def getRandomDistribution(self,state):
    dist = util.Counter()
    for action in state.getLegalActions( self.index ):
      dist[action] = 1.0
    dist.normalize()
    return dist

  def getPartiallyObservaleLoc(self,state):
    global possiblePacmanStates
    global frontierStates
    pacmanLoc = state.getPacmanPosition()
    ghostLoc = state.getGhostPosition(self.index)
    room = state.getLoctoRoom()
    if room[int(pacmanLoc[0])][int(pacmanLoc[1])] in state.data.roomsOn:
      print pacmanLoc, ' in roomOn: ', room[int(pacmanLoc[0])][int(pacmanLoc[1])]
      possiblePacmanStates.clear()
      possiblePacmanStates[pacmanLoc] = 1.0
      frontierStates = [pacmanLoc]
      return pacmanLoc

    else:
      print pacmanLoc, ' in roomOff: ', room[int(pacmanLoc[0])][int(pacmanLoc[1])]

      predictLoc = util.chooseFromDistribution(possiblePacmanStates)
      while (room[int(predictLoc[0])][int(predictLoc[1])] in state.data.roomsOn) and len(possiblePacmanStates.keys())!=1:
        possiblePacmanStates[predictLoc] = 0
        predictLoc = util.chooseFromDistribution(possiblePacmanStates)
        print 'NEW PREDICT LOC: ', predictLoc,possiblePacmanStates

      tempFrontierStates = Set()
      for frontierLoc in frontierStates:
        tempNode = self.node(int(frontierLoc[0]),int(frontierLoc[1]))
        possibleActions = self.getActions(state,tempNode)

        for frontierAction in possibleActions:
          actionTuple = self.convertAction(frontierAction)
          newFrontierState = (frontierLoc[0]+actionTuple[0],frontierLoc[1]+actionTuple[1])
          
          # if newFrontierState in possiblePacmanStates.keys(): continue
          if room[int(newFrontierState[0])][int(newFrontierState[1])] in state.data.roomsOn: continue
          tempFrontierStates.add(newFrontierState)
          oldProb = possiblePacmanStates[frontierLoc]
          possiblePacmanStates[newFrontierState] += oldProb/float(len(possibleActions))

      possiblePacmanStates.normalize()
      frontierStates = tempFrontierStates
      print 'PACMAN STATES: ', possiblePacmanStates
      print "ghostLoc: ", ghostLoc, "predictLoc: ", predictLoc, "pacmanLoc: ", pacmanLoc
      pacmanLoc = predictLoc
      return pacmanLoc

  def getDistribution( self, state ):
    pacmanLoc = state.getPacmanPosition()
    ghostLoc = state.getGhostPosition(self.index)

    #uncomment out for partially Observable
    pacmanLoc = self.getPartiallyObservaleLoc(state)

    dist = self.heuristic(pacmanLoc,ghostLoc)
    if dist > 4:
      #uncomment for go to closest Dark Room Policy or Far Distribution Policy

      #return self.getDarkRoomDistribution(state)
      return self.getFarDistribution(state)
    else:
      return self.getCloseDistribution(state)

    # comment out above and uncomment getPolicyDistribution for MDP
    # return self.getPolicyDistribution(state)

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
