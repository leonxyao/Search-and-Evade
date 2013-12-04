from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Grid
from game import Agent
import heapq

import MDPUtil,util

import layout
from sets import Set

possibleGhostStates = util.Counter()
possibleGhostStates[layout.ghostStartLoc] = 1.0
frontierStates = Set()
frontierStates.add(layout.ghostStartLoc)
print 'GLOBAL Ghost: ', layout.ghostStartLoc

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best




    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getBestMinimaxValue(self,gameState,currDepth,agentNumber):
    newDepth = currDepth
    newAgentNumber = agentNumber
    if agentNumber == gameState.getNumAgents():
      newDepth+=1
      newAgentNumber=0
    if gameState.isWin() or gameState.isLose() or newDepth == self.depth: #depth start at 0
      return (self.evaluationFunction(gameState),Directions.STOP)

    elif newAgentNumber == 0:
      return self.getMaxValue(gameState,newDepth,newAgentNumber) #just agent Number

    elif newAgentNumber >= 1:
      return self.getMinValue(gameState,newDepth,newAgentNumber)
    else:
      assert False

  def getMaxValue(self,gameState,currDepth,agentNumber):
    maxValue = -sys.maxint
    bestAction = 0
    assert agentNumber < gameState.getNumAgents()
    assert agentNumber == 0
    for action in gameState.getLegalActions(agentNumber):
      if action == Directions.STOP:
        continue
      newGameState = gameState.generateSuccessor(agentNumber,action)
      currMax = self.getBestMinimaxValue(newGameState,currDepth,agentNumber+1)[0]
      if currMax > maxValue:
        maxValue = currMax
        bestAction = action
    return (maxValue,bestAction)

  def getMinValue(self,gameState,currDepth,agentNumber):
    minValue = sys.maxint
    bestAction = 0
    assert agentNumber < gameState.getNumAgents()
    assert agentNumber != 0
    for action in gameState.getLegalActions(agentNumber):
      if action == Directions.STOP:
        continue
      newGameState = gameState.generateSuccessor(agentNumber,action)
      currMin = self.getBestMinimaxValue(newGameState,currDepth,agentNumber+1)[0]
      if currMin < minValue:
        minValue = currMin
        bestAction = action
    return (minValue,bestAction)

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
      currNode = pq.removeMin()[0]
      if already_visited[currNode.loc[0]][currNode.loc[1]]: continue
      else: already_visited[currNode.loc[0]][currNode.loc[1]] = True
      if currNode.loc == endNode.loc:
        return currNode.path[0]
      
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

  def convertAction(self, action): 
        if action == "West":
            return(-1,0)
        elif action == "East":
            return (1,0)
        elif action == "South":
            return (0,-1)
        elif action == "North":
            return (0,1)
  def convertTuple(self,action):
        if action == (-1,0):
            return "West"
        elif action == (1,0):
            return "East"
        elif action == (0,-1):
            return "South"
        elif action == (0,1):
            return "North"
  def fullyObservableAction(self,gameState):
    pacmanLoc = gameState.getPacmanPosition()
    ghostLoc = gameState.getGhostPosition(1)
    if (pacmanLoc, ghostLoc) in MDPUtil.AstarPolicy.keys():
        action = self.convertTuple(MDPUtil.AstarPolicy[(pacmanLoc,ghostLoc)])
    else:
      action = self.A_star(pacmanLoc,ghostLoc,self.heuristic,gameState)
      MDPUtil.AstarPolicy[(pacmanLoc,ghostLoc)] = self.convertAction(action)
    return action

  def partiallyObservableAction(self,gameState):
    global possibleGhostStates
    global frontierStates
    pacmanLoc = gameState.getPacmanPosition()
    ghostLoc = gameState.getGhostPosition(1)
    room = gameState.getLoctoRoom()
    if room[int(ghostLoc[0])][int(ghostLoc[1])] in gameState.data.roomsOn:
      print ghostLoc, ' in roomOn: ', room[int(ghostLoc[0])][int(ghostLoc[1])]
      possibleGhostStates.clear()
      possibleGhostStates[ghostLoc] = 1.0
      frontierStates = [ghostLoc]
      if (pacmanLoc, ghostLoc) in MDPUtil.AstarPolicy.keys():
        action = self.convertTuple(MDPUtil.AstarPolicy[(pacmanLoc,ghostLoc)])
      else:
        action = self.A_star(pacmanLoc,ghostLoc,self.heuristic,gameState)
        MDPUtil.AstarPolicy[(pacmanLoc,ghostLoc)] = self.convertAction(action)
    else:
      print ghostLoc, ' in roomOff: ', room[int(ghostLoc[0])][int(ghostLoc[1])]
      predictLoc = (-1,-1)
      while True:
        predictLoc = util.chooseFromDistribution(possibleGhostStates)
        while (room[int(predictLoc[0])][int(predictLoc[1])] in gameState.data.roomsOn) and len(possibleGhostStates.keys())!=1:
          possibleGhostStates[predictLoc] = 0
          predictLoc = util.chooseFromDistribution(possibleGhostStates)
          print 'WHILE LOOOOOOP: ',predictLoc, possibleGhostStates
        if predictLoc != pacmanLoc: break
      tempFrontierStates = Set()
      for frontierLoc in frontierStates:
        tempNode = self.node(int(frontierLoc[0]),int(frontierLoc[1]))
        possibleActions = self.getActions(gameState,tempNode)

        for frontierAction in possibleActions:
          actionTuple = self.convertAction(frontierAction)
          newFrontierState = (frontierLoc[0]+actionTuple[0],frontierLoc[1]+actionTuple[1])
          
          if newFrontierState in possibleGhostStates.keys(): continue
          if room[int(newFrontierState[0])][int(newFrontierState[1])] in gameState.data.roomsOn: continue
          tempFrontierStates.add(newFrontierState)
          oldProb = possibleGhostStates[frontierLoc]
          possibleGhostStates[newFrontierState] += oldProb/float(len(possibleActions))

      possibleGhostStates.normalize()
      frontierStates = tempFrontierStates
      #print 'FRONTIER STATES: ',frontierStates
      print 'GHOST STATES: ', possibleGhostStates
      print "ghostLoc: ", ghostLoc, "predictLoc: ", predictLoc, "pacmanLoc: ", pacmanLoc

      if (pacmanLoc, predictLoc) in MDPUtil.AstarPolicy.keys():
        action = self.convertTuple(MDPUtil.AstarPolicy[(pacmanLoc,predictLoc)])
      else:
        action = self.A_star(pacmanLoc,predictLoc,self.heuristic,gameState)
        MDPUtil.AstarPolicy[(pacmanLoc,predictLoc)] = self.convertAction(action)
    return action


  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
	
      gameState.isWin():
        Returns True if it's a winning state
	
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    # BEGIN_YOUR_CODE (around 68 lines of code expected)
    action = self.partiallyObservableAction(gameState)
    #action = self.fullyObservableAction(gameState)
    return action

    # END_YOUR_CODE

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """
  def getBestAlphaBetaValue(self,gameState,currDepth,agentNumber,alpha,beta):
    newDepth = currDepth
    newAgentNumber = agentNumber
    if agentNumber == gameState.getNumAgents():
      newDepth+=1
      newAgentNumber=0
    if gameState.isWin() or gameState.isLose() or newDepth == self.depth: #depth start at 0
      return (self.evaluationFunction(gameState),Directions.STOP)

    elif newAgentNumber == 0:
      return self.getMaxValue(gameState,newDepth,newAgentNumber,alpha,beta) #just agent Number

    elif newAgentNumber >= 1:
      return self.getMinValue(gameState,newDepth,newAgentNumber,alpha,beta)
    else:
      assert False

  def getMaxValue(self,gameState,currDepth,agentNumber,alpha,beta):
    maxValue = -sys.maxint
    bestAction = 0
    assert agentNumber < gameState.getNumAgents()
    assert agentNumber == 0
    for action in gameState.getLegalActions(agentNumber):
      if action == Directions.STOP:
        continue
      newGameState = gameState.generateSuccessor(agentNumber,action)
      currMax = self.getBestAlphaBetaValue(newGameState,currDepth,agentNumber+1,alpha,beta)[0]
      if currMax >= beta:
        return (currMax,action)
      alpha = max(alpha,currMax)
      if currMax > maxValue:
        maxValue = currMax
        bestAction = action
    return (maxValue,bestAction)

  def getMinValue(self,gameState,currDepth,agentNumber,alpha,beta):
    minValue = sys.maxint
    bestAction = 0
    assert agentNumber < gameState.getNumAgents()
    assert agentNumber != 0
    for action in gameState.getLegalActions(agentNumber):
      if action == Directions.STOP:
        continue
      newGameState = gameState.generateSuccessor(agentNumber,action)
      currMin = self.getBestAlphaBetaValue(newGameState,currDepth,agentNumber+1,alpha,beta)[0]
      if currMin <=alpha:
        return (currMin,action)
      beta = min(currMin,beta)
      if currMin < minValue:
        minValue = currMin
        bestAction = action
    return (minValue,bestAction)
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (around 69 lines of code expected)
    score,action = self.getBestAlphaBetaValue(gameState,0,0,-float('inf'),float('inf'))
    return action
    # END_YOUR_CODE

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """
  def getBestExpectimaxValue(self,gameState,currDepth,agentNumber):
    newDepth = currDepth
    newAgentNumber = agentNumber
    if agentNumber == gameState.getNumAgents():
      newDepth+=1
      newAgentNumber=0
    if gameState.isWin() or gameState.isLose() or newDepth == self.depth: #depth start at 0
      return (self.evaluationFunction(gameState),Directions.STOP)

    elif newAgentNumber == 0:
      return self.getMaxValue(gameState,newDepth,newAgentNumber) #just agent Number

    elif newAgentNumber >= 1:
      return self.getMinValue(gameState,newDepth,newAgentNumber)
    else:
      assert False

  def getMaxValue(self,gameState,currDepth,agentNumber):
    maxValue = -sys.maxint
    bestAction = 0
    assert agentNumber < gameState.getNumAgents()
    assert agentNumber == 0
    for action in gameState.getLegalActions(agentNumber):
      if action == Directions.STOP:
        continue
      newGameState = gameState.generateSuccessor(agentNumber,action)
      currMax = self.getBestExpectimaxValue(newGameState,currDepth,agentNumber+1)[0]
      if currMax > maxValue:
        maxValue = currMax
        bestAction = action
    return (maxValue,bestAction)

  def getMinValue(self,gameState,currDepth,agentNumber):
    minValue = sys.maxint
    bestAction = 0
    assert agentNumber < gameState.getNumAgents()
    assert agentNumber != 0
    totalValue = 0.0
    totalActions = len(gameState.getLegalActions(agentNumber))
    if Directions.STOP in gameState.getLegalActions(agentNumber):
      totalActions-=1
    for action in gameState.getLegalActions(agentNumber):
      if action == Directions.STOP:
        continue
      newGameState = gameState.generateSuccessor(agentNumber,action)
      currMin = self.getBestExpectimaxValue(newGameState,currDepth,agentNumber+1)[0]
      totalValue+=currMin
    average = totalValue/totalActions
    return (average,'None')
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (around 70 lines of code expected)
    score,action = self.getBestExpectimaxValue(gameState,0,0)
    return action
    # END_YOUR_CODE

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (problem 4).

    DESCRIPTION: I used the distance to the closest ghost (1 over that value) and the distance to the closest food and the original getScore function
    I weighted more heavily the ghost because dying is the worst possible thing, but to avoid it from trashing in the corner I also had to find a good weight
    for the food so it will actually have incentive to go to it. The original getScore was just a good index to start from.
  """

  # BEGIN_YOUR_CODE (around 69 lines of code expected)
  #currEvaluationFunction = self.evaluationFunction(currentGameState)
  distancesToGhosts = 0
  pacPos = currentGameState.getPacmanPosition()
  foodPos = currentGameState.getFood()
  distanceToClosestGhost=float('inf')
  for ghostIndex in range(1,currentGameState.getNumAgents()):
    distance = manhattanDistance(pacPos,currentGameState.getGhostPosition(ghostIndex))
    if distance < distanceToClosestGhost:
      distanceToClosestGhost = distance
    distancesToGhosts+=distance

  totalFoodDistances=0
  closestFood = float('inf')
  for x in range(foodPos.width):
    for y in range(foodPos.height):
      if foodPos[x][y] == True:
        distance = manhattanDistance((x,y),pacPos)
        totalFoodDistances+= distance
        if closestFood>distance:
          closestFood = distance
  #currEvalFnWeight = 1
  ghostWeight = 1000
  foodWeight = 1
  scoreWeight = 100
  if distanceToClosestGhost == 0:
    return -10000
  score =  -ghostWeight*1/distanceToClosestGhost + foodWeight*closestFood + scoreWeight * currentGameState.getScore()
  return  score

  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction


