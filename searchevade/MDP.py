import collections, MDPUtil, math, random, pdb, sys

############################################################
# Problem 1a

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    sum = 0
    #print 'computeQ init: ', state, action
    for case in mdp.succAndProbReward(state,action):
        newState = case[0]
        #print 'computeQ newState: ', newState
        prob = case[1]
        reward = case[2]
        # if newState not in V.keys(): 
        #     print 'computeQ continue: ', newState
        #     continue
        sum += prob*(reward+mdp.discount()*V[newState])
    return sum
    # END_YOUR_CODE

############################################################
# Problem 1b

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    allStates = mdp.states
    Vprev = {}
    Vcurr = V
    underErrorTolerance = False
    while underErrorTolerance == False:
        newVpi = {}
        for state in allStates:
            newVpi[state] = computeQ(mdp,Vcurr,state,pi[state])
        Vprev = Vcurr 
        Vcurr = newVpi

        underErrorTolerance = True
        for state in allStates:
            if abs(Vcurr[state] - Vprev[state])> epsilon:
                underErrorTolerance = False
    return Vcurr

    # END_YOUR_CODE

############################################################
# Problem 1c

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    pi = {}
    print 'states: ',mdp.states, len(mdp.states)
    print 'actions: ', mdp.actions
    for state in mdp.states:
        # if len(mdp.actions(state)) == 0: 
        #     continue
        bestAction = [mdp.actions(state)[0]]
        bestQ = computeQ(mdp,V,state,bestAction[0])
        for action in mdp.actions(state):
            currQ = computeQ(mdp,V,state,action)
            if currQ > bestQ:
                bestQ = currQ
                bestAction = [action]
            elif currQ == bestQ:
                bestAction.append(action)
        pi[state] = max(bestAction)
    return pi
    # END_YOUR_CODE

############################################################
# Problem 1d

class PolicyIteration(MDPUtil.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        states = mdp.states
        # compute V and pi
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        Vinit = {}
        for state in states:
            Vinit[state] = 0
        pi = 0
        Vprev = Vinit
        underErrorTolerance = False
        while underErrorTolerance == False:
            pi = computeOptimalPolicy(mdp,Vprev)
            Vcurr = policyEvaluation(mdp,Vprev,pi)

            underErrorTolerance = True
            for state in states:
                if abs(Vcurr[state] - Vprev[state])> epsilon:
                    underErrorTolerance = False
            Vprev = Vcurr

        # END_YOUR_CODE
        self.pi = pi
        self.V = Vprev

############################################################
# Problem 1e

def singlePolicyEvaluation(mdp, V, pi):
    states = mdp.states
    Vcurr = V
    newVpi = {}
    for state in states:
        newVpi[state] = computeQ(mdp,Vcurr,state,pi[state])
    return newVpi

class ValueIteration(MDPUtil.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        states = mdp.states
        # BEGIN_YOUR_CODE (around 13 lines of code expected)
        Vinit = {}
        for state in states:
            Vinit[state] = 0
        pi = 0
        Vprev = Vinit
        Vcurr = Vprev
        underErrorTolerance = False
        while underErrorTolerance == False:
            pi = computeOptimalPolicy(mdp,Vprev)
            Vcurr = singlePolicyEvaluation(mdp,Vprev,pi)

            underErrorTolerance = True
            for state in states:
                if abs(Vcurr[state] - Vprev[state])> epsilon:
                    underErrorTolerance = False
            Vprev = Vcurr
        # END_YOUR_CODE
        self.pi = pi
        self.V = Vprev

############################################################
# Problem 1f

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(MDPUtil.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        pass
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return [1]
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        #print self.n
        tuples = []
        if state!=0:
            return []
        else:
            prob = 0.05
            tuples.append((-1,prob,1000))
            tuples.append((1,1-prob,1))
            return tuples
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 1
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return 0.9
    # END_YOUR_CODE

############################################################
# Problem 2a

class BlackjackMDP(MDPUtil.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.  The second element is the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.  The final element
    # is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to (0,).
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 50 lines of code expected)
        handValue = state[0]
        topCard = state[1]
        deck = state[2]
        newdeck = deck
        tuples = []
        if sum(deck)==0:
            return []
        if action == 'Quit':
            nextState = (handValue,None,(0,))
            reward = handValue
            nextTuple = (nextState,1.0,reward)
            tuples.append(nextTuple)

        elif action == 'Take':
            for card in range(len(self.cardValues)):
                if deck[card] == 0:
                    continue
                if topCard != None and topCard!=card:
                    continue
                currCard = self.cardValues[card]
                if topCard!=None:
                    prob = 1.0
                else:
                    prob = float(deck[card])/sum(deck)
                newdeck = list(deck)
                newdeck[card]-=1
                newdeck = tuple(newdeck)
                reward = 0
                if handValue + currCard <= self.threshold:
                    if sum(newdeck)==0:
                        reward = handValue+currCard
                    nextState = (handValue+currCard,None,newdeck)
                    nextTuple = (nextState,prob,reward)
                    tuples.append(nextTuple)
                else:
                    nextState = (handValue+currCard,None,(0,))
                    nextTuple = (nextState,prob,0)
                    tuples.append(nextTuple)
        elif action == 'Peek':
            if topCard == None:
                for card in range(len(self.cardValues)):
                    if deck[card] > 0:
                        currCard = self.cardValues[card]
                        prob = float(deck[card])/sum(deck)
                        nextState = (handValue,card,deck)
                        nextTuple = (nextState,prob,-self.peekCost)
                        tuples.append(nextTuple)
            else:
                return []

        return tuples
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 2b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    blackJack = BlackjackMDP(cardValues = [4,5,20], multiplicity=2, threshold = 20, peekCost = 1)
    return blackJack
    # END_YOUR_CODE

############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(MDPUtil.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # Your algorithm will be asked to produce an action given a state.
    # You should use an epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        rand = random.uniform(0.0,1.0)
        if rand < self.explorationProb:
            randIndex = random.randint(0,len(self.actions(state))-1)
            return self.actions(state)[randIndex]
        else:
            bestActions = [self.actions(state)[0]] #-sys.maxint-1
            bestQ = self.getQ(state,bestActions[0]) #-sys.maxint-1
            for action in self.actions(state):
                if self.getQ(state,action) > bestQ:
                    bestQ = self.getQ(state,action)
                    bestActions = [action]
                elif self.getQ(state,action) == bestQ:
                    bestActions.append(action)
            return max(bestActions)
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (around 12 lines of code expected)
        Qlist = []
        if newState != None:
            for actionBlah in self.actions(newState):
                Qlist.append(self.getQ(newState,actionBlah))
            maxQ = max(Qlist)
        else:
            maxQ = 0.0
        r = (reward+self.discount*maxQ)-self.getQ(state,action)
        for feature,value in self.featureExtractor(state,action):
            self.weights[feature] = self.weights[feature]+self.getStepSize()*r*value

        # END_YOUR_CODE

############################################################
# Problem 3b: convergence of Q-learning

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

############################################################
# Problem 3c: features for Q-learning.

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action(1 feature).
# - indicator on the number of cards for each card type and the action (len(counts) features).
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    featureVector = []
    featureVector.append( ((total,action), 1) )
    presence = [0]*len(counts)
    for index in range(len(counts)):
        if counts[index] != 0:
            presence[index] = 1
    featureVector.append( ((tuple(presence),action),1) )
    for index in range(len(counts)):
        featureVector.append( ((index,counts[index],action),1) )
    return featureVector



    # END_YOUR_CODE

############################################################
# Problem 3d: changing mdp

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)
