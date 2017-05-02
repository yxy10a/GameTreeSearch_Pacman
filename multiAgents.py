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


from util import manhattanDistance
from game import Directions
import random, util
import datetime

from game import Agent

def scoreEvaluationFunction(currentGameState):
   """
     This default evaluation function just returns the score of the state.
     The score is the same one displayed in the Pacman GUI.

     This evaluation function is meant for use with adversarial search agents
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

    def minimax(self,state,agent,depth):
        val = 0
        if agent == state.getNumAgents():
            agent = self.index
        if depth == (self.depth*state.getNumAgents()) or state.isWin() or state.isLose():
            val =  self.evaluationFunction(state)
        elif agent == self.index:
            val = self.maxval(state,agent,depth)
        else:
            val = self.minval(state,agent,depth)
        return val
  
    def maxval(self,state,agent,depth):
        v = float("-inf")
        actions = state.getLegalActions(agent)
        for action in actions:
            v = max(v,self.minimax(state.generateSuccessor(agent,action),agent+1,depth+1))          
        return v
  
    def minval(self,state,agent,depth):
        v = float("inf")
        for action in state.getLegalActions(agent):
            v = min(v,self.minimax(state.generateSuccessor(agent,action),agent+1,depth+1))          
        return v
    
    def getAction(self, gameState):
        depth = 0
        agent = self.index
        actionDict = {}
        for action in gameState.getLegalActions(agent):
            val = self.minimax(gameState.generateSuccessor(agent,action),agent+1,depth+1)
            actionDict[val] = action
        return actionDict[max(actionDict)]
        util.raiseNotDefined()

class abVal():
    def __init__(self,value,action):
        self.action = action
        self.value = value
    
    def __cmp__(self,other):
        if self.value == other.value:
            return 0
        elif self.value > other.value:
            return 1
        else:
            return -1

class AlphaBetaAgent(MultiAgentSearchAgent):

    def alphabeta(self,state,agent,depth,action,alpha,beta):
        retval = []
        if agent == state.getNumAgents():
            agent = self.index
        if depth == (self.depth*state.getNumAgents()) or state.isWin() or state.isLose():
            retval = abVal(self.evaluationFunction(state),action)
        elif agent == self.index:
            retval = self.maxval(state,agent,depth,alpha,beta)
        else:
            retval = self.minval(state,agent,depth,alpha,beta)
        return retval
  
    def maxval(self,state,agent,depth,alpha,beta):
        v = abVal(float("-inf"),Directions.STOP)
        actions = state.getLegalActions(agent)
        for action in actions:
            tempv = self.alphabeta(state.generateSuccessor(agent,action),agent+1,depth+1,action,alpha,beta)
            tempv.action = action
            v = max(v,tempv)
            if v.value >= beta:
                return v
            alpha = max(alpha,v.value)          
        return v
  
    def minval(self,state,agent,depth,alpha,beta):
        v = abVal(float("inf"),Directions.STOP)
        for action in state.getLegalActions(agent):
            tempv = self.alphabeta(state.generateSuccessor(agent,action),agent+1,depth+1,action,alpha,beta) 
            tempv.action = action
            v = min(v,tempv)
            if v.value <= alpha:
                return v
            beta = min(beta,v.value)          
        return v

    def getAction(self, gameState):
        depth = 0
        agent = self.index
        alpha = float("-inf")
        beta = float("inf")
        action = Directions.STOP
        v = self.alphabeta(gameState,agent,depth,action,alpha,beta)
        return v.action

class ExpectimaxAgent(MultiAgentSearchAgent):
    
    def expectedvalue(self, gameState, agentindex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        num_ghosts = gameState.getNumAgents() - 1
        legalActions = gameState.getLegalActions(agentindex)
        num_actions = len(legalActions)
        val = 0
        for action in legalActions:
            nextState = gameState.generateSuccessor(agentindex, action)
            if (agentindex == num_ghosts):
                val += self.maxvalue(nextState, depth - 1)
            else:
                val += self.expectedvalue(nextState, agentindex + 1, depth)
        return val / num_actions
      
    def maxvalue(self, gameState, depth):      
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        score = -(float("inf"))
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            score = max(score, self.expectedvalue(nextState, 1, depth))
        return score
      
    def getAction(self, gameState):      
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        score = -(float("inf"))
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prev_score = score
            score = max(score, self.expectedvalue(nextState, 1, self.depth))
            if score > prev_score:
                bestAction = action
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
   """
    Commentary:
    
    dist_ghost is the distance from pacman to the closest ghost. Score is rewarded by staying "safe" if there is a ghost within a distance of 3.
    
    closest_food is the distance from pacman to the closest food. Score is taken away if pacman's action gets away from the nearest food.
    
    One point is rewarded as an incentive to gather food.
    
    Two points is rewarded as an incentive to gather a capsule, so that pacman would favor a capsule over food when both are available.

   """
   if currentGameState.isWin():
      return float("inf")
   if currentGameState.isLose():
      return - float("inf")
   score = scoreEvaluationFunction(currentGameState)
   food_array = currentGameState.getFood()
   food_list = food_array.asList()
   closest_food = float("inf")
   for item in food_list:
      cur_dist = util.manhattanDistance(item, currentGameState.getPacmanPosition())
      if (dist < closest_food):
          closest_food = cur_dist
   num_ghosts = currentGameState.getNumAgents() - 1
   i = 1
   dist_ghost = float("inf")
   while i <= num_ghosts:
      cur_dist = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(i))
      dist_ghost = min(dist_ghost, cur_dist)
      i += 1
   score += max(dist_ghost, 3)
   score -= closest_food
   rebel_locations = currentGameState.getCapsules()
   score -= len(food_list)
   score -= 2 * len(rebel_locations)
   return score
   util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

