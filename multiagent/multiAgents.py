# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random

import util
from game import Agent
from util import manhattanDistance


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.

        RUN COMMANDS:
        python pacman.py -p ReflexAgent -l testClassic
        python pacman.py --frameTime 0 -p ReflexAgent -k 1 -q -n 100
        python pacman.py --frameTime 0 -p ReflexAgent -k 2 -q -n 100
        python autograder.py -q q1 --no-graphics
        """

        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()  # pacman position
        newGhostStates = childGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        total_score = 0.0  # our score
        old_food = currentGameState.getFood()  # Grid of boolean food indicator variables.

        # Guide pacman in positions closer to food
        # Iterates old_food matrix
        for x in range(old_food.width):
            for y in range(old_food.height):
                if old_food[x][y]:  # check if in position (x,y) of the matrix has a food
                    # compute distance between food and new pacman position
                    d = manhattanDistance((x, y), newPos)
                    if d == 0:
                        total_score += 100
                    else:
                        total_score += 1.0 / (d * d)

        # Function to calculate distance between ghost
        for ghost in newGhostStates:
            # compute distance between ghost and new pacman position
            d = manhattanDistance(ghost.getPosition(), newPos)
            if d <= 1:
                if ghost.scaredTimer != 0:
                    total_score += 2000
                else:
                    total_score -= 200

        # Function to get the capsule
        for capsule in currentGameState.getCapsules():
            d = manhattanDistance(capsule, newPos)
            if d == 0:
                total_score += 1000
            else:
                total_score += 1.0 / (d * d)

        return total_score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def terminalTest(self, gameState, depth):
        return depth == 0 or gameState.isWin() or gameState.isLose()


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

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        RUN COMMANDS:
        python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3 -q -n 100
        python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
        python pacman.py -p MinimaxAgent -l mediumClassic depth=4
        python autograder.py -q q2 --no-graphics
        """

        value = float("-inf")
        actions = []

        for action in gameState.getLegalActions(0):
            u = self.min_value(
                game_state=gameState.getNextState(0, action),
                agent=1,
                depth=self.depth
            )
            if u == value:
                actions.append(action)
            elif u >= value:
                value = u
                actions = [action]

        return random.choice(actions)

    def min_value(self, game_state, agent, depth):
        if self.terminalTest(game_state, depth):
            return self.evaluationFunction(game_state)

        value = float("inf")

        for action in game_state.getLegalActions(agent):
            succ = game_state.getNextState(agent, action)
            if agent == game_state.getNumAgents() - 1:
                value = min(value, self.max_value(succ, agent=0, depth=depth - 1))
            else:
                # You are in the same level because there are more than one ghost
                # so they perform as a team (same depth).
                value = min(value, self.min_value(succ, agent=agent + 1, depth=depth))
        return value

    def max_value(self, game_state, agent, depth):
        if self.terminalTest(game_state, depth):
            return self.evaluationFunction(game_state)

        value = float("-inf")

        for action in game_state.getLegalActions(agent):
            value = max(value, self.min_value(
                game_state.getNextState(agent, action),
                agent=1,
                depth=depth
            ))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)

    RUN COMMANDS:
    python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
    python autograder.py -q q3 --no-graphics
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = []
        value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            u = self.min_value(gameState.getNextState(0, action), 1, self.depth, alpha, beta)
            if u == value:
                actions.append(action)
            elif u > value:
                value = u
                actions = [action]

            alpha = max(alpha, value)
        return random.choice(actions)

    def min_value(self, gameState, agent, depth, alpha, beta):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float("inf")
        for action in gameState.getLegalActions(agent):
            if agent == gameState.getNumAgents() - 1:
                value = min(value, self.max_value(gameState.getNextState(agent, action), 0, depth - 1, alpha, beta))
            else:
                value = min(value, self.min_value(gameState.getNextState(agent, action), agent + 1, depth, alpha, beta))

            if value < alpha:
                return value
            beta = min(beta, value)

        return value

    def max_value(self, gameState, agent, depth, alpha, beta):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float("-inf")

        for action in gameState.getLegalActions(agent):
            value = max(value, self.min_value(gameState.getNextState(agent, action), 1, depth, alpha, beta))
            if value > beta:
                return value  # prunning because min(beta, v) always will be beta
            alpha = max(alpha, value)  # otherwise, update

        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.

        RUN COMMANDS:
        python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
        python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10
        python autograder.py -q q4 --no-graphics
        """

        value = float("-inf")
        actions = []
        for action in gameState.getLegalActions(0):
            u = self.expectation_value(gameState.getNextState(0, action), 1, self.depth)
            if u == value:
                actions.append(action)
            elif u > value:
                value = u
                actions = [action]

        return random.choice(actions)

    def expectation_value(self, gameState, agent, depth, ):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)

        values = []
        for action in gameState.getLegalActions(agent):
            if agent == gameState.getNumAgents() - 1:
                values.append(self.max_value(gameState.getNextState(agent, action), 0, depth - 1))
            else:
                values.append(self.expectation_value(gameState.getNextState(agent, action), agent + 1, depth))
        return sum(values) / float(len(values))

    def max_value(self, gameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.evaluationFunction(gameState)

        value = float("-inf")
        for action in gameState.getLegalActions(agent):
            value = max(value, self.expectation_value(gameState.getNextState(agent, action), 1, depth))
        return value


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    RUN COMMANDS:
    python autograder.py -q q5
    python autograder.py -q q5 --no-graphics
    """
    total_score = scoreEvaluationFunction(currentGameState)

    if currentGameState.isWin() or currentGameState.isLose():  # check if the game end
        return total_score

    # Food
    food_distance = [1.0 / manhattanDistance(food_pos, currentGameState.getPacmanPosition()) for food_pos in
                     currentGameState.getFood().asList()]

    if len(food_distance):
        total_score += max(food_distance)

    # Ghost
    for ghost in currentGameState.getGhostStates():
        ghost_distance = manhattanDistance(ghost.getPosition(), currentGameState.getPacmanPosition())
        if ghost_distance <= 1:
            if ghost.scaredTimer != 0:
                total_score += max(60.0 / ghost_distance, 0)
            else:
                total_score -= max(1.0 / ghost_distance, 0)

    # Capsule
    capsule_disance = [1.0 / manhattanDistance(capsule, currentGameState.getPacmanPosition()) for capsule in
                       currentGameState.getCapsules()]

    if len(capsule_disance):
        total_score += max(capsule_disance)

    return total_score


# Abbreviation
better = betterEvaluationFunction
