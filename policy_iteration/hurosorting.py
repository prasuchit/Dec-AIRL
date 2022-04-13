import numpy as np
from scipy import full, sparse
import copy
import math as m
import time
import logging
import os
from operator import mod
import pickle
from mdptoolbox.mdp import PolicyIterationModified
logger = logging.getLogger(__name__)

class HuRoSorting():
    """
    This environment is a slightly modified version of sorting env in this paper: MMAP-BIRL(https://arxiv.org/pdf/2109.07788.pdf).
    This is a multi-agent sparse interactions Dec-MDP with 2 agents - a Human and a Robot. The Human agent is always given preference 
    while sorting and allowed to choose if there's a conflict. The Robot performs an independent sort while accounting for the Human's 
    local state and action. In order to learn this behavior, the Robot observes two Humans do the sort as a team where one of them 
    assume the role of the Robot.
    ------------------------------------------------------------------------------------------------------------------------
    Global state S - (s_rob, s_hum)
    Global action A - (a_rob, a_hum)
    Transitions self.T = Pr(S' | S, a_rob, a_hum)
    Joint Reward R(S,A) - Common reward that both agents get.
    Boolean variable eta - 1 if S is interactive state else 0.
    R(S,A) = eta*R_int + (1-eta)*R_non_int
    ------------------------------------------------------------------------------------------------------------------------
    State and action space are the same for both agents. 'agent' subscript below could mean either robot/human. 
    s_agent - (Onion_location, End_eff_loc, Onion_prediction)
    a_agent - (Noop, Detect, Pick, Inspect, PlaceOnConveyor, PlaceInBin)
    ------------------------------------------------------------------------------------------------------------------------
    Onion_location - Describes where the onion in focus currently is - (Unknown, OnConveyor, AtHome, InFront)
    End_eff_loc - Describes where the end effector currently is - (OnConveyor, AtHome, InFront, AtBin)
    Prediction - Provides the classification of the onion in focus - (Unknown, Good, Bad)
    NOTE: Onion location turns unknown when it's successfully placed on conv or in bin; 
    Until detect is done, both Prediction and Onion_location remain unknown.
    -------------------------------------------------------------------------------------------------------------------------
    Detect - Uses a classifier NN and CV techniques to find location and class of onion - (Onion_location, Initial_Prediction)
    Pick - Dips down, grabs the onion and comes back up - (Onion_location: AtHome, End_eff_loc: AtHome)
    PlaceInBin - If Initial_Prediction is bad, the onion is directly placed in the bad bin - (Onion_location: Unknown, End_eff_loc: AtBin). 
    Inspect - If Initial_Prediction is good, we need to inspect again to make sure, onion is shown close to camera - (Onion_location: InFront, End_eff_loc: InFront). 
    PlaceOnConveyor - If Prediction turns out to be good, it's placed back on the conveyor and liftup - (Onion_location: Unknown, End_eff_loc: OnConv).
    Noop - No action is done here - Meant to be used by Robot when both agents want to detect/placeonconveyor at the same time.
    -------------------------------------------------------------------------------------------------------------------------
    Episode starts from one of the valid start states where eef is anywhere, onion is on conv and pred is unknown.
    Episode ends when one onion is successfully chosen, picked, inspected and placed somewhere.
    """
    def __init__(self, noise=0.05, discount = 0.9, useSparse = False, full_observable=True, 
                    verbose = False, debug = False, setTransition = True, setReward = True):

        global ACTION_MEANING, ONIONLOC, EEFLOC, PREDICTIONS, AGENT_MEANING
        self.n_agents = len(AGENT_MEANING)
        self.full_observable = full_observable
        self.verbose = verbose
        self.nOnionLoc = len(ONIONLOC)
        self.nEEFLoc = len(EEFLOC)
        self.nPredict = len(PREDICTIONS)
        self.nSAgent = self.nOnionLoc*self.nEEFLoc*self.nPredict
        self.nAAgent = len(ACTION_MEANING)
        self.nSGlobal = self.nSAgent**self.n_agents
        self.nAGlobal = self.nAAgent**self.n_agents
        self.start = np.zeros((self.n_agents, self.nSAgent))
        self.update_start()
        self.prev_obsv = [None]*self.n_agents
        self.reward = 0
        self._full_obs = None
        self.noise = noise
        self.discount = discount
        self.T_global = np.zeros((self.nSGlobal, self.nSGlobal, self.nAGlobal)) 
        self.T_agent = np.zeros((self.nSAgent, self.nSAgent, self.nAAgent))    # state transition probability
        self.R_global = np.zeros((self.nSGlobal, self.nAGlobal))
        self.debug = debug
        self.setTransition = setTransition
        self.setReward = setReward

    def update_start(self):
        '''
        @brief - Sets the initial start state distrib. Currently, it's uniform b/w all valid states.
        '''
        for i in range(self.n_agents):
            for j in range(self.nSAgent):
                o_l, eef_loc, pred = self.sid2vals(j)
                if self.isValidState(o_l, eef_loc, pred):
                    self.start[i][j] = 1
            self.start[i][:] = self.start[i][:] / \
                np.count_nonzero(self.start[i][:])
            assert np.sum(self.start[i]) == 1, "Start state distb doesn't add up to 1!"

    def get_action_meanings(self, action):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return ACTION_MEANING[action]
    
    def get_state_meanings(self, o_loc, eef_loc, pred):
        '''
        @brief - Just a translator to make it human-readable.
        '''
        return ONIONLOC[o_loc], EEFLOC[eef_loc], PREDICTIONS[pred]

    def get_init_obs(self):
        '''
        @brief - Samples from the start state distrib and returns a joint one hot obsv.
        '''
        sample_r = np.random.multinomial(
            n=1, pvals=np.reshape(self.start[0], (self.nSAgent)))
        sample_h = np.random.multinomial(
            n=1, pvals=np.reshape(self.start[1], (self.nSAgent)))
        s_r = int(np.squeeze(np.where(sample_r == 1)))
        s_h = int(np.squeeze(np.where(sample_h == 1)))
        self.set_prev_obsv(0, s_r)
        self.set_prev_obsv(1, s_h)
        if self.custom:
            return np.concatenate((self.get_one_hot(s_r, self.nSAgent), self.get_one_hot(s_h, self.nSAgent)), axis = 0)
        else:
            return [self.get_one_hot(s_r, self.nSAgent), self.get_one_hot(s_h, self.nSAgent)]
    
    def sid2vals(self, s):
        '''
        @brief - Given state id, this func converts it to the 3 variable values. 
        '''
        sid = s
        onionloc = int(mod(sid, self.nOnionLoc))
        sid = (sid - onionloc)/self.nOnionLoc
        eefloc = int(mod(sid, self.nEEFLoc))
        sid = (sid - eefloc)/self.nEEFLoc
        predic = int(mod(sid, self.nPredict))
        return onionloc, eefloc, predic

    def vals2sid(self, nxtS):
        '''
        @brief - Given the 3 variable values making up a state, this converts it into state id 
        '''
        ol = nxtS[0]
        eefl = nxtS[1]
        pred = nxtS[2]
        return (ol + self.nOnionLoc * (eefl + self.nEEFLoc * pred))

    def vals2sGlobal(self, oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h):
        return (oloc_r + self.nOnionLoc * (eefloc_r + self.nEEFLoc * (pred_r + self.nPredict * (oloc_h + self.nOnionLoc * (eefloc_h + self.nEEFLoc * pred_h)))))

    def vals2aGlobal(self, a_r, a_h):
        return a_r + self.nAAgent * a_h

    def sGlobal2vals(self, s_global):
        s_g = s_global
        oloc_r = int(mod(s_g, self.nOnionLoc))
        s_g = (s_g - oloc_r)/self.nOnionLoc
        eefloc_r = int(mod(s_g, self.nEEFLoc))
        s_g = (s_g - eefloc_r)/self.nEEFLoc
        pred_r = int(mod(s_g, self.nPredict))
        s_g = (s_g - pred_r)/self.nPredict
        oloc_h = int(mod(s_g, self.nOnionLoc))
        s_g = (s_g - oloc_h)/self.nOnionLoc
        eefloc_h = int(mod(s_g, self.nEEFLoc))
        s_g = (s_g - eefloc_h)/self.nEEFLoc
        pred_h = int(mod(s_g, self.nPredict))
        return oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h

    def aGlobal2vals(self, a_global):
        a_g = a_global
        a_r = int(mod(a_g, self.nAAgent))
        a_g = (a_g - a_r)/self.nAAgent
        a_h = int(mod(a_g, self.nAAgent))
        return a_r, a_h

    def isValidAction(self, onionLoc, eefLoc, pred, action):
        '''
        @brief - For each state there are a few invalid actions, returns only valid actions.
        '''
        assert action <= 5, 'Unavailable action. Check if action is within num actions'
        if action == 0: # Noop, this can be done from anywhere.
            return True
        elif action == 1:   # Detect
            if pred == 0 or onionLoc == 0:  # Only when you don't know about the onion
                return True
            else: return False
        elif action == 2:   # Pick
            if onionLoc == 1 and eefLoc != 1:   # As long as onion is on conv and eef is not
                return True
            else: return False
        elif action == 3 or action == 4 or action == 5:   # Inspect # Placeonconv # Placeinbin
            if pred != 0 and onionLoc != 0: # Pred and onion loc both are known. 
                if onionLoc == eefLoc and eefLoc != 1:    # Onion is in hand and hand isn't on conv
                    return True
                else: return False
            else: return False
        else: 
            logger.error(f"Trying an impossible action are we? Better luck next time!")
            return False
        
    def isValidGlobalState(self, oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h):
        return (self.isValidState(oloc_r, eefloc_r, pred_r) and self.isValidState(oloc_h, eefloc_h, pred_h))
        
    def findNxtStates(self, onionLoc, eefLoc, pred, a):
        ''' 
        @brief - Returns the valid nextstates. Currently it's deterministic transition, could be made stochastic.
                This function assumes that you're doing the action from the appropriate current state.
                eg: If (onionloc - unknown, eefloc - onconv, pred - unknown), that's still a valid
                current state but an inappropriate state to perform inspect action and you shouldn't
                be able to transition into the next state induced by inspect. (Thanks @YikangGui for catching this.)
                Therefore inappropriate actions are filtered out by getValidActions method now. 

        Onionloc: {0: 'Unknown', 1: 'OnConveyor', 2: 'InFront', 3: 'AtHome'}
        eefLoc = {0: 'InBin', 1: 'OnConveyor', 2: 'InFront', 3: 'AtHome'}
        Predictions = {0:  'Unknown', 1: 'Bad', 2: 'Good}
        Actions: {0: 'Noop', 1: 'Detect', 2: 'Pick', 3: 'Inspect', 4: 'PlaceOnConveyor', 5: 'PlaceInBin}
        '''
        if a == 0:
            ''' Noop '''
            return [[onionLoc, eefLoc, pred]]
        elif a == 1:
            ''' Detect '''
            '''NOTE: While doing detect the eef has to come back home.
                    Detect is done after placing somewhere and
                    if it stays at bin or on conv after detect,
                    that takes the transition to an invalid state.'''
            n_states = [[1, 3, 1], [1, 3, 2]]
            return n_states            
        elif a == 2:
            ''' Pick '''
            return [[3, 3, pred]]
        elif a == 3:
            ''' Inspect '''
            n_states = [[2, 2, 1], [2, 2, 2]]
            # choice_index = np.random.choice(len(n_states))
            # return n_states[choice_index]
            return n_states
        elif a == 4:
            ''' PlaceOnConv '''
            return [[0, 1, 0]]
        elif a == 5:
            ''' PlaceInBin '''
            return [[0, 0, 0]]

    def isValidState(self, onionLoc, eefLoc, pred):
        '''
        @brief - Checks if a given state is valid or not.

        '''
        if (onionLoc == 2 and eefLoc != 2) or (onionLoc == 3 and eefLoc != 3) or \
            (onionLoc == 0 and pred != 0) or (onionLoc != 0 and pred == 0):
            return False
        return True

    def getKeyFromValue(self, my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key
        return "key doesn't exist"


    def setModelVals(self):

        if self.setTransition:
            for s in range(self.nSAgent):
                onionLoc, eefLoc, pred = self.sid2vals(s)
                for a in range(self.nAAgent):
                    if self.isValidState(onionLoc, eefLoc, pred):
                        if self.isValidAction(onionLoc, eefLoc, pred, a):
                            nextStates = self.findNxtStates(onionLoc, eefLoc, pred, a)
                            for nxtS in nextStates:
                                if self.isValidState(nxtS[0], nxtS[1], nxtS[2]):
                                    ns = self.vals2sid(nxtS)
                                    if self.T_agent[s, s, a] == 0:
                                        self.T_agent[s, s, a] += self.noise # Noise must only be added once
                                    # Succeding in intended action with high prob
                                    if (pred == 2 and a == 3) or (pred == 0 and a == 1): # During detect and Inspect
                                        if nxtS[2] == 2:  # We give higher prob to good coz bad onions are less
                                            self.T_agent[ns, s, a] += (1 - self.noise) * 0.6
                                        else:
                                            self.T_agent[ns, s, a] += (1 - self.noise) * 0.4/(len(nextStates) - 1)
                                    else:
                                        self.T_agent[ns, s, a] += (1 - self.noise)/len(nextStates)
                                else:
                                    self.T_agent[2, s, a] = 1   # If next state is invalid send it to the sink. State 2 is (infront, inbin, unknown)
                        else:
                            self.T_agent[2, s, a] = 1   # If action is invalid send it to the sink. State 2 is (infront, inbin, unknown)
                    else:
                        self.T_agent[2, s, a] = 1   # If current state is invalid send it to the sink. State 2 is (infront, inbin, unknown)
            
            if self.debug:
                # Check transition probability
                for a in range(self.nAAgent):
                    for s in range(self.nSAgent):
                        err = abs(sum(self.T_agent[:, s, a]) - 1)
                        if err > 1e-6 or np.any(self.T_agent[:, s, a]) > 1 or np.any(self.T_agent[:, s, a]) < 0:
                            print(f"self.T(:,{s},{a}) = {self.T_agent[:, s, a]}")
                            print('ERROR: \n', s, a, np.sum(self.T_agent[:, s, a]))

        for sg in range(self.nSGlobal):
            oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h = self.sGlobal2vals(sg)
            s_r = self.vals2sid([oloc_r, eefloc_r, pred_r])
            s_h = self.vals2sid([oloc_h, eefloc_h, pred_h])
            for nsg in range(self.nSGlobal):
                n_oloc_r, n_eefloc_r, n_pred_r, n_oloc_h, n_eefloc_h, n_pred_h = self.sGlobal2vals(nsg)
                ns_r = self.vals2sid([n_oloc_r, n_eefloc_r, n_pred_r])
                ns_h = self.vals2sid([n_oloc_h, n_eefloc_h, n_pred_h])
                for ag in range(self.nAGlobal):
                    a_r, a_h = self.aGlobal2vals(ag)
                    if self.setTransition:
                        self.T_global[nsg, sg, ag] = (self.T_agent[ns_r, s_r, a_r] * self.T_agent[ns_h, s_h, a_h])  

                    if self.setReward:
                        # Bad onion, placeinBin, any agent
                        if (pred_r == 1 and a_r == 5) or (pred_h == 1 and a_h == 5):
                            self.R_global[sg, ag] = 1
                        # Bad onion, placeinBin, both agents
                        if (pred_r == 1 and a_r == 5) and (pred_h == 1 and a_h == 5):
                            self.R_global[sg, ag] = 2
                        # Good onion, placeonConv, any agent
                        if (pred_r == 2 and a_r == 4) or (pred_h == 2 and a_h == 4):
                            self.R_global[sg, ag] = 1 
                        # Good onion, placeonConv, both agents
                        if (pred_r == 2 and a_r == 4) and (pred_h == 2 and a_h == 4):
                            self.R_global[sg, ag] = 2
                        # Currently picked, either find good, inspect
                        if ((oloc_r == 3 and pred_r == 2 and a_r == 3) or (oloc_h == 3 and pred_h == 2 and a_h == 3)):
                            self.R_global[sg, ag] = 1 
                        # Currently picked, both find good, inspect
                        if ((oloc_r == 3 and pred_r == 2 and a_r == 3) and (oloc_h == 3 and pred_h == 2 and a_h == 3)):
                            self.R_global[sg, ag] = 2
                        # Bad onion, placeonConv, any agent
                        if (pred_r == 1 and a_r == 4) or (pred_h == 1 and a_h == 4):
                            self.R_global[sg, ag] = -1
                        # Bad onion, placeonConv, both agents
                        if (pred_r == 1 and a_r == 4) and (pred_h == 1 and a_h == 4):
                            self.R_global[sg, ag] = -2 
                        # Good onion, PlaceinBin, any agent
                        if (pred_r == 2 and a_r == 5) or (pred_h == 2 and a_h == 5):
                            self.R_global[sg, ag] = -1 
                        # Good onion, PlaceinBin, both agents
                        if (pred_r == 2 and a_r == 5) and (pred_h == 2 and a_h == 5):
                            self.R_global[sg, ag] = -2
                        # Both have good, onions already picked, human tries to placeonconv and robot doesn't wait
                        if (pred_r == pred_h == 2 and oloc_h != 1 and oloc_r != 1 and (a_h != 4 or a_r != 0)):
                            self.R_global[sg, ag] = -3 
                        # Currently unknown, human is not detecting or robot doesn't wait
                        if (oloc_h == oloc_r == 0 and (a_h != 1 or a_r != 0)):
                            self.R_global[sg, ag] = -3 
        
        if self.debug and self.setTransition:
            # Check transition probability
            for a in range(self.nAGlobal):
                for s in range(self.nSGlobal):
                    err = abs(sum(self.T_global[:, s, a]) - 1)
                    if err > 1e-6 or np.any(self.T_global[:, s, a]) > 1 or np.any(self.T_global[:, s, a]) < 0:
                        print(f"self.T(:,{s},{a}) = {self.T_global[:, s, a]}")
                        print('ERROR: \n', s, a, np.sum(self.T_global[:, s, a]))



AGENT_MEANING = {
    0: 'Robot',
    1: 'Human'
}

ACTION_MEANING = {
    0: 'Noop',
    1: 'Detect',
    2: 'Pick',
    3: 'Inspect',
    4: 'PlaceOnConveyor',
    5: 'PlaceinBin'
}

ONIONLOC = {
    0: 'Unknown',
    1: 'OnConveyor',
    2: 'InFront',
    3: 'AtHome'
}
EEFLOC = {
    0: 'InBin',
    1: 'OnConveyor',
    2: 'InFront',
    3: 'AtHome'
}

PREDICTIONS = {
    0: 'Unknown',
    1: 'Bad',
    2: 'Good'
}


if __name__ == "__main__":

    examineTraj = True
    examinePolicy = False
    test = False
    setTransition = False
    setReward = False
    debug = False
    setModelVals = False

    if os.path.exists(os.getcwd()+ f'/gail-airl-ppo/policy_iteration/models/hurosorting/mdp.pkl'):
        picklefile = open(os.getcwd()+ f'/gail-airl-ppo/policy_iteration/models/hurosorting/mdp.pkl', 'rb')
        #unpickle the dataframe
        mdp = pickle.load(picklefile)
        if setModelVals:
            newmdp = HuRoSorting(debug = debug, setReward=setReward, setTransition=setTransition)
            newmdp.setModelVals()
            if setTransition: 
                mdp.T_agent = copy.deepcopy(newmdp.T_agent)
                mdp.T_global = copy.deepcopy(newmdp.T_global)
            if setReward:
                mdp.R_global = copy.deepcopy(newmdp.R_global)
            #pickle the dictionary and write it to file
            pickle.dump(mdp, picklefile)
        #close the file
        picklefile.close()
    else:
        mdp = HuRoSorting(debug = debug)
        mdp.setModelVals()
        #create a pickle file
        picklefile = open(os.getcwd()+ f'/gail-airl-ppo/policy_iteration/models/hurosorting/mdp.pkl', 'wb')
        #pickle the dictionary and write it to file
        pickle.dump(mdp, picklefile)
        #close the file
        picklefile.close()
    
    
    if os.path.exists(os.getcwd()+ f'/gail-airl-ppo/policy_iteration/results/hurosorting/policy.csv') and not setModelVals:
        piL = np.loadtxt(os.getcwd()+ f'/gail-airl-ppo/policy_iteration/results/hurosorting/policy.csv')
    else:
        MAX_ITERS = 10000
        EPS = 1e-12
        SHOW_MSG = False
        pi = PolicyIterationModified(np.transpose(
            mdp.T_global), mdp.R_global, mdp.discount, max_iter=MAX_ITERS, epsilon=EPS)
        pi.run()
        # Q = utils.QfromV(pi.V, mdp)
        piL = np.reshape(pi.policy, (mdp.nSGlobal, 1))
        # H = evalToolbox(piL, mdp)
        np.savetxt(os.getcwd()+ f'/gail-airl-ppo/policy_iteration/results/hurosorting/policy.csv', piL, fmt='%i')
        # return piL, pi.V, Q, H

    if examinePolicy:
        for i in range(mdp.nSGlobal):
            oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h = mdp.sGlobal2vals(i)
            a_r, a_h = mdp.aGlobal2vals(piL[i])
            if mdp.isValidState(oloc_r, eefloc_r, pred_r) and mdp.isValidState(oloc_h, eefloc_h, pred_h):
                print(f'Agent 0 state: {mdp.get_state_meanings(oloc_r, eefloc_r, pred_r)} | Agent 1 state: {mdp.get_state_meanings(oloc_h, eefloc_h, pred_h)}')
                print(f'Agent 0 action: {mdp.get_action_meanings(a_r)} | Agent 1 action: {mdp.get_action_meanings(a_h)}\n')

    if examineTraj:
        oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h = 0,3,0,0,3,0
        sid_gl = mdp.vals2sGlobal(oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h)
        for _ in range(25):
            a_r, a_h = mdp.aGlobal2vals(piL[sid_gl])
            # if a_r == 4 or a_h == 4:
            print(f'Agent 0 state: {mdp.get_state_meanings(oloc_r, eefloc_r, pred_r)} | Agent 1 state: {mdp.get_state_meanings(oloc_h, eefloc_h, pred_h)}')
            print(f'Agent 0 action: {mdp.get_action_meanings(a_r)} | Agent 1 action: {mdp.get_action_meanings(a_h)}\n')
            # print("Transition: ", mdp.T_global[np.squeeze(mdp.T_global[:, sid_gl, int(piL[sid_gl])]) != 0, sid_gl, int(piL[sid_gl])])
            # ns = np.argmax(np.squeeze(mdp.T_global[:, sid_gl, int(piL[sid_gl])]))
            ns = np.random.choice(np.arange(mdp.nSGlobal), p = np.squeeze(mdp.T_global[:, sid_gl, int(piL[sid_gl])]))
            oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h = mdp.sGlobal2vals(ns)
            sid_gl = ns

    if test:
        a_g = mdp.vals2aGlobal(4,4)
        # conv = [i for i,a in enumerate(piL) if a == a_g and mdp.isValidGlobalState(mdp.sGlobal2vals(i))]
        conv = []
        for i,a in enumerate(piL):
            oloc_r, eefloc_r, pred_r, oloc_h, eefloc_h, pred_h = mdp.sGlobal2vals(i)
            print(f'Agent 0 state: {mdp.get_state_meanings(oloc_r, eefloc_r, pred_r)} | Agent 1 state: {mdp.get_state_meanings(oloc_h, eefloc_h, pred_h)}')
