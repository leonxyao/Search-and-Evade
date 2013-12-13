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
    minList = [0.0]
    for case in mdp.succAndProbReward(state,action):
        newState = case[0]
        prob = case[1]
        reward = case[2]
        minList.append(prob*(reward+mdp.discount()*V[newState]))
    return min(minList)
    # return sum
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
    # print 'states: ',mdp.states, len(mdp.states)
    # print 'actions: ', mdp.actions
    for state in mdp.states:
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

