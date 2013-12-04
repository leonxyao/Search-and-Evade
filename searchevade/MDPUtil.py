import collections, random, submission
from game import Grid
from game import Agent
import heapq
from game import Directions

import layout

AstarPolicy = {}


# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        print 'initial states: ',self.states
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                #print 'action: ', state, action
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        #print 'new state: ',newState
                        self.states.add(newState)
                        queue.append(newState)
        global AstarPolicy
        print 'ASTARPOLICY: ',AstarPolicy, len(AstarPolicy)
        # print "%d states" % len(self.states)
        # print self.states

############################################################

# A simple example of an MDP where states are integers in [-n, +n].
# and actions involve moving left and right by one position.
# We get rewarded for going to the right.
class NumberLineMDP(MDP):
    def __init__(self, n=5): self.n = n
    def startState(self): return 0
    def actions(self, state): return [-1, +1]
    def succAndProbReward(self, state, action):
        return [(state, 0.4, 0), (min(max(state + action, -self.n), +self.n), 0.6, state)]
    def discount(self): return 0.9

############################################################
class Search():
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

  
class SearchEvadeMDP(MDP):
    def __init__(self,gameState): self.gameState = gameState
    def getActions(self,gameState,loc):
          locX = int(loc[0])
          locY = int(loc[1])
          layout = gameState.getLayout()
          layoutRoom = gameState.getLayout().room
          legalActions = []#[(0,0)]
          if layoutRoom[locX-1][locY] != '%':
            legalActions.append((-1,0)) #West
          if layoutRoom[locX+1][locY] != '%':
            legalActions.append((1,0)) #East
          if layoutRoom[locX][locY-1] != '%':
            legalActions.append((0,-1)) #South
          if layoutRoom[locX][locY+1] != '%':
            legalActions.append((0,1)) #North
          return legalActions
    
    def convertAction(self, action): 
        if action == "West":
            return(-1,0)
        elif action == "East":
            return (1,0)
        elif action == "South":
            return (0,-1)
        elif action == "North":
            return (0,1)
        # else:
        #     return (0,0)

    def startState(self):
        pacmanLoc = layout.pacmanStartLoc#(1,11)
        layout = self.gameState.getLayout()

        ghostLoc = self.gameState.getGhostPosition(1)

        return (pacmanLoc,ghostLoc,False)
    
    def actions(self,state):
        legalActions = self.getActions(self.gameState,state[1])
        return legalActions

    def succAndProbReward(self,state,action):
        global AstarPolicy
        if state[2]:
            return []
        pacmanLoc = state[0]
        ghostLoc = state[1]
        searcher = Search()
        if (pacmanLoc,ghostLoc) not in AstarPolicy.keys():
          pacmanAction = self.convertAction(searcher.A_star(pacmanLoc,ghostLoc,searcher.heuristic,self.gameState))
          AstarPolicy[(pacmanLoc,ghostLoc)] = pacmanAction
        else:
          pacmanAction = AstarPolicy[(pacmanLoc,ghostLoc)]
        tuples = []
        newPacmanLoc = (pacmanLoc[0] + pacmanAction[0], pacmanLoc[1] + pacmanAction[1])
        newGhostLoc = (ghostLoc[0] + action[0], ghostLoc[1]+action[1])
        #print 'pacman: ', newPacmanLoc, 'ghost: ', newGhostLoc
        if newPacmanLoc == newGhostLoc or (newPacmanLoc == ghostLoc and newGhostLoc == pacmanLoc):
            terminalState = (newPacmanLoc,newGhostLoc,True)
            nextTuple = (terminalState,1.0,-100000)
        else: 
            reward = searcher.heuristic(newPacmanLoc,newGhostLoc)
            # if searcher.heuristic(newPacmanLoc,newGhostLoc) > 5:
            #   reward = 1
            newState = (newPacmanLoc,newGhostLoc,False)
            nextTuple = (newState,1.0,reward)
        tuples.append(nextTuple)
        return tuples

    def discount(self):
        return 1




############################################################

# An algorithm that solves an MDP (i.e., computes the optimal
# policy).
class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")

############################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class FixedRLAlgorithm(RLAlgorithm):
    def __init__(self, pi): self.pi = pi

    # Just return the action given by the policy.
    def getAction(self, state): return self.pi[state]

    # Don't do anything: just stare off into space.
    def incorporateFeedback(self, state, action, reward, newState): pass

############################################################

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
             sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)
    return totalRewards
