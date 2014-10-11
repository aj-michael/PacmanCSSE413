# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    if successorGameState.isLose():
        return -1*float("inf")
    elif successorGameState.isWin() or newPos in currentGameState.getCapsules():
        return float("inf")
    
    gPos = currentGameState.getGhostPosition(1)    
    gDist = manhattanDistance(gPos, newPos)
    if len(currentGameState.getCapsules()) > len(successorGameState.getCapsules()):
        return float("inf")
    closestFood = min(newFood.asList(), key=lambda fPos : manhattanDistance(newPos, fPos))
    if gDist <= 2:
        return -1*float("inf")
    score = max(gDist, 4) * 6
    if currentGameState.getNumFood() > successorGameState.getNumFood():
        return float("inf")
    score = score - 10*manhattanDistance(newPos, closestFood)
    if Directions.REVERSE[action] == currentGameState.getPacmanState().getDirection() or action == Directions.STOP:
        score = score - 5
    return score
        


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
    Your minimax agent (question 2)
    """


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

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
        """
        "*** YOUR CODE HERE ***"
        value, action = self.maxi(gameState,1)
        return action
    
    def maxi(self,state,depth):
        value = None
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        for legalAction in state.getLegalActions(0):
            successor = state.generateSuccessor(0,legalAction)
            minValue = self.mini(successor,depth,1)[0]
            if value == None or minValue > value:
                value, action = minValue, legalAction
        return value, action

    def mini(self,state,depth,agent):
        value = None
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        for legalAction in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent,legalAction)
            if agent + 1 == state.getNumAgents() and depth == self.depth:
                maxValue = self.evaluationFunction(successor)
            elif agent + 1 == state.getNumAgents():
                maxValue = self.maxi(successor,depth+1)[0]
            else:
                maxValue = self.mini(successor,depth,agent+1)[0]
            if value == None or maxValue < value:
                value, action = maxValue, legalAction
        return value, action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value, action = self.maxi(gameState,1,float("-inf"),float("inf"))
        return action
        
    def maxi(self,state,depth,alpha,beta):
        value = None
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        for legalAction in state.getLegalActions(0):
            successor = state.generateSuccessor(0,legalAction)
            minValue = self.mini(successor,depth,1,alpha,beta)[0]
            if value == None or minValue > value:
                value, action = minValue, legalAction
            if value >= beta:
                return value, action
            alpha = max(alpha, value)
        return value, action

    def mini(self,state,depth,agent,alpha,beta):
        value = None
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        for legalAction in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent,legalAction)
            if agent + 1 == state.getNumAgents() and depth == self.depth:
                maxValue = self.evaluationFunction(successor)
            elif agent + 1 == state.getNumAgents():
                maxValue = self.maxi(successor,depth+1,alpha,beta)[0]
            else:
                maxValue = self.mini(successor,depth,agent+1,alpha,beta)[0]
            if value == None or maxValue < value:
                value, action = maxValue, legalAction
            if value <= alpha:
                return value, action
            beta = min(beta, value)
        return value, action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        value, action = self.maxi(gameState,1)
        return action

    def maxi(self,state,depth):
        value = None
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        for legalAction in state.getLegalActions(0):
            successor = state.generateSuccessor(0,legalAction)
            minValue = self.expecti(successor,depth,1)
            if value == None or minValue > value:
                value, action = minValue, legalAction
        return value, action

    def expecti(self,state,depth,agent):
        value = None
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        total = 0 ; count = 0
        for legalAction in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent,legalAction)
            if agent + 1 == state.getNumAgents() and depth == self.depth:
                maxValue = self.evaluationFunction(successor)
            elif agent + 1 == state.getNumAgents():
                maxValue = self.maxi(successor,depth+1)[0]
            else:
                maxValue = self.expecti(successor,depth,agent+1)
            total = total + maxValue ; count = count + 1
        return total / float(count)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 9999999
    elif currentGameState.isLose():
        return float("-inf")
    score = 0

    # less food is good
    score = score - currentGameState.getNumFood()

    ppos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    closestG = None
    for g in ghosts:
        dist = manhattanDistance(ppos, g.getPosition())
        if g.scaredTimer:
            pass
        else:
            closestG = max(closestG, dist)
    
    # far from normal ghosts is good
    score = score + min(max(closestG, 0),3)
   
    # less capsules is good 
    score = score - len(currentGameState.getCapsules()) * 5

    # deterministic algorithms are boring
    score = score + random.randrange(0,2)

    closestF = None
    for f in currentGameState.getFood().asList():
        dist = manhattanDistance(ppos, f)
        closestF = dist if closestF == None else min(dist, closestF)
    
    # far from food is bad
    score = score - closestF / 10.0

    return score


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

