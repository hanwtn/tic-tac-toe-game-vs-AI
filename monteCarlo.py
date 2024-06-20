
import copy
import random
import time
import sys
import math
from collections import namedtuple
import numpy as np

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support

class MCTS: #Monte Carlo Tree Search implementation
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild



    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def isTerminalState(self, utility, moves):
        return utility != 0 or len(moves) == 0

import copy
import random
import time
import sys
import math
from collections import namedtuple
#import numpy as np

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support

class MCTS: #Monte Carlo Tree Search implementation
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild



    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def isTerminalState(self, utility, moves):
        return utility != 0 or len(moves) == 0
    def monteCarloPlayer(self, timelimit = 4):
        """Entry point for Monte Carlo search"""
        start = time.perf_counter()
        end = start + timelimit
        """Use timer above to apply iterative deepening"""
        while time.perf_counter() < end:
        #count = 100  # use this and the next line for debugging. Just disable previous while and enable these 2 lines
        #while count >= 0:
            #count -= 1
            # SELECT stage use selectNode()
            node = self.selectNode(self.root)

            # EXPAND stage
            if not self.game.terminal_test(node.state):
                self.expandNode(node)

            # SIMULATE stage using simuplateRandomPlay()
            result = self.simulateRandomPlay(node)

            # BACKUP stage using backPropagation
            self.backPropagation(node, result)

        winnerNode = self.root.getChildWithMaxScore()
        assert(winnerNode is not None)
        return winnerNode.state.move

    """selection stage function. walks down the tree using findBestNodeWithUCT()"""
    def selectNode(self, nd):
        node = nd
        while node.children:
            node, _ = self.findBestNodeWithUCT(node)
        return node

    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. Parse nd's children and use uctValue() to collect uct's for the
        children....."""
        childUCT = []
        bestChild = None
        bestUCT = -np.inf 

        for child in nd.children:
            uct = self.uctValue(nd.visitCount, child.winScore, child.visitCount)
            childUCT.append(uct)  # Collect UCT value for each child
            if uct > bestUCT:
                bestUCT = uct
                bestChild = child

        return bestChild, childUCT


    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """compute Upper Confidence Value for a node"""
        if nodeVisit == 0:
            return 0 if self.exploreFactor == 0 else sys.maxsize
        return (nodeScore / nodeVisit) + self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)

    def expandNode(self, nd):
        """generate the child nodes and append them to nd's children"""
        stat = nd.state
        tempState = GameState(to_move=stat.to_move, move=stat.move, utility=stat.utility, board=stat.board, moves=stat.moves)
        for a in self.game.actions(tempState):
            childNode = self.Node(self.game.result(tempState, a), nd)
            nd.children.append(childNode)


    def simulateRandomPlay(self, nd):
        # first check win possibility for the current node:
        winStatus = self.game.compute_utility(nd.state.board, nd.state.move, nd.state.board[nd.state.move])
        if winStatus == self.game.k:  #means it is opponent's win
            assert(nd.state.board[nd.state.move] == 'X')
            if nd.parent is not None:
                nd.parent.winScore = -sys.maxsize
            return ('X' if winStatus > 0 else 'O')

        """now roll out a random play down to a terminating state. """

        tempState = copy.deepcopy(nd.state) # to be used in the following random playout
        to_move = tempState.to_move
        
        while not self.game.terminal_test(tempState):
            actions = self.game.actions(tempState)
            if not actions:
                break
            action = random.choice(actions)
            tempState = self.game.result(tempState, action)
        
        winStatus = self.game.utility(tempState, to_move)  

        return ('X' if winStatus > 0 else 'O' if winStatus < 0 else 'N') # 'N' means tie


    def backPropagation(self, nd, winningPlayer):
        """propagate upword to update score and visit count from
        the current leaf node to the root node."""
        tempNode = nd
        while tempNode is not None:
            tempNode.visitCount += 1
            if winningPlayer == 'X':  # 'X' is the maximizing player
                tempNode.winScore += 1
            elif winningPlayer == 'O':  # 'O' is the minimizing player
                tempNode.winScore -= 1
            tempNode = tempNode.parent

