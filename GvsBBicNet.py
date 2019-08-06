#print('test')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

import sys
sys.path.append("C:\\Users\\10353\\using_saida\\SAIDA_RL\\python")

import math
from collections import deque
from core.policies import MA_EpsGreedyQPolicy, MA_GreedyQPolicy, LinearAnnealedPolicy, AdvEpsGreedyPolicy
from core.common.util import *
from core.common.processor import Processor
from core.memories import SequentialMemory
from core.callbacks import DrawTrainMovingAvgPlotCallback
from core.common.callback import *
from core.policies import *
from core.common.random import *
from saida_gym.starcraft_multi.goliathVsBattlecruiser import GoliathVsBattlecruiser

"""
you need to clone git repo with link below and import this as referenced project into your pycharm project
https://github.com/openai/multiagent-particle-envs
"""
from keras.layers import Dense, Reshape, Input, Bidirectional, CuDNNLSTM, Activation, Permute, LSTM
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model
from keras.regularizers import l2

from keras.callbacks import TensorBoard
from time import time


from core.network.bicnet import actor_net, critic_net
from core.algorithm.BiCNet import BiCNetAgent

import importlib
import param

import argparse
from core.common.util import OPS


"""
====================================================
FOR hyper parameters combination parameter setting from auto launcher
====================================================  
"""
parser = argparse.ArgumentParser(description='Configuration')

parser.add_argument(OPS.NO_GUI.value, help='gui', type=bool, default=False)
parser.add_argument(OPS.DOUBLE.value, help='double dqn', default=True, action='store_true')
parser.add_argument(OPS.DUELING.value, help='dueling dqn', default=True, action='store_true')
parser.add_argument(OPS.BATCH_SIZE.value, type=int, default=32, help="batch size")
parser.add_argument(OPS.REPLAY_MEMORY_SIZE.value, type=int, default=20000, help="replay memory size")
parser.add_argument(OPS.LEARNING_RATE.value, type=float, default=0.0025, help="learning rate")
parser.add_argument(OPS.TARGET_NETWORK_UPDATE.value, type=int, default=5000, help="target_network_update_interval")
parser.add_argument(OPS.WINDOW_LENGTH.value, type=int, default=2, help="window length")
parser.add_argument(OPS.N_STEPS.value, type=int, default=2000, help="n steps for training")
parser.add_argument(OPS.TIME_WINDOW.value, type=int, default=1, help="Temporal Splice Size")
parser.add_argument(OPS.SCENARIO_NAME.value, type=str, default='simple_spread', help="scenario name")
parser.add_argument(OPS.ENABLE_MODEL_BASED.value, type=int, default=0, help="Enable Model based")
parser.add_argument(OPS.DISCOUNT_FACTOR.value, type=float, default=0.9, help="discount factor")
parser.add_argument(OPS.EGREEDY_ANNELING_STEP.value, type=int, default=150000, help="Annealing Step")


args = parser.parse_args()

dict_args = vars(args)
post_fix = ''
for k in dict_args.keys():
    if k == 'no_gui':
        continue
    post_fix += '_' + k + '_' + str(dict_args[k])

print('post_fix : {}'.format(post_fix))

CURRENT_FILE_NAME = os.path.basename(__file__).split('.')[0]
CURRENT_FILE_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
FILE_NAME_FOR_LOG = os.path.basename(__file__).split('.')[0] + "_" + yyyymmdd24hhmmss()

TIME_WINDOW = dict_args[OPS.TIME_WINDOW()]
N_STEPS = dict_args[OPS.N_STEPS()]
SCENARIO_NAME = dict_args[OPS.SCENARIO_NAME()]
EGREEDY_ANNELING_STEP = dict_args[OPS.EGREEDY_ANNELING_STEP()]
ENABLE_MODEL_BASED = dict_args[OPS.ENABLE_MODEL_BASED()]


#from starcraft
NO_GUI = dict_args[OPS.NO_GUI()]
DISCOUNT_FACTOR = dict_args[OPS.DISCOUNT_FACTOR()]
ENABLE_DOUBLE = dict_args[OPS.DOUBLE()]
ENABLE_DUELING = dict_args[OPS.DUELING()]
BATCH_SIZE = dict_args[OPS.BATCH_SIZE()]
REPLAY_BUFFER_SIZE = dict_args[OPS.REPLAY_MEMORY_SIZE()]
LEARNING_RATE = dict_args[OPS.LEARNING_RATE()]
TARGET_MODEL_UPDATE_INTERVAL = dict_args[OPS.TARGET_NETWORK_UPDATE()]
WINDOW_LENGTH = dict_args[OPS.WINDOW_LENGTH()]

WIN_REWARD = 100
DEFEAT_REWARD = -10
KILL_REWARD = 5
DEAD_REWARD = -5
ATTACK_REWARD = 0.9
ATTACK_REWARD_DROPSHIP= 0.7
DAMAGED_REWARD = -1
DAMAGED_REWARD_GOLIATH= -0.8
COOLDOWN_REWARD = -0.55
INVALID_ACTION_REWARD = -2

MOVE_ANGLE = 45

NB_AGENTS = 2
OBSERVATION_SIZE = 32
STATE_SIZE = (WINDOW_LENGTH, NB_AGENTS, OBSERVATION_SIZE)  # state_size = ( nb_agents, time_window(optional), observation)

tensorboard = TensorBoard(log_dir="tensorboard_logs/{}".format(time()))

class MyCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.per_win = 0

    def on_episode_end(self, episode, logs={}):
        importlib.reload(param)

        if logs['info']['hps'] >= 0:
            self.per_win += 1
        else:
            self.per_win = 0

        if self.agent is not None:
            if self.per_win > 10:
                self.agent.save_weights(os.path.realpath('../../save_model/' + FILE_NAME + str(episode) + '_win.h5f'), True)


class ObsProcessor(Processor):
    def __init__(self, nb_action):
        self.nb_action = nb_action
        #from starcraft
        self.last_action = [env.action_space[1] - 1, env.action_space[1] - 1, env.action_space[1] - 1]
        self.env = env
        self.accumulated_observation = deque(maxlen=2)
        self.last_my_unit_cnt = deque(maxlen=2)
        self.last_enemy_unit_cnt = deque(maxlen=2)
        self.last_enemy_unit_hp = deque(maxlen=2)

    def process_state_batch(self, batch):
        return batch

    def process_action(self, action):
        self.last_action = action
        return action

    #def process_action(self, action):
    #    action_onehot = [np.eye(self.nb_action)[x] for x in action.tolist()]
    #    return action_onehot
    def process_step(self, observation, reward, done, info):
        
        if done:
            #print("End of an episode")
            #recently edite
            

            if reward > 0:
                info['hps'] = observation.my_unit[0].hp\
                              + observation.my_unit[1].hp
            elif reward < 0:
                info['hps'] = -observation.en_unit[0].hp
            else:
                info['hps'] = 0
            #print("see the info_hps"+str(info['hps']))
        #print("observation:   ")
        #print(observation)
        state_array = self.process_observation(observation)
        #print("state array: ")
        #print(state_array)
        

        reward = self.reward_reshape(observation, reward, done, info)
        #print("reward:   ")
        #print(reward)

        if param.verbose == 1 or param.verbose == 4:
            print('action : ', self.last_action, 'reward : ', reward)
        if param.verbose == 2 or param.verbose == 4:
            print('observation : ', observation)
        if param.verbose == 3 or param.verbose == 4:
            print('state_array : ', state_array)

        #print(reward)

        return state_array, reward, done, info


    #def process_step(self, observation, reward, done, info):
    #    info = {'done': done}
    #    return observation, reward, all(done), info


    def reward_reshape(self, observation, reward, done, info):
        reward_goliath = reward
        reward_goliathb = reward

        #print("start_reward_shape")
        #print(reward)
        if done:
            if reward > 0:
                total_hp = sum(self.accumulated_observation[1][:, 0])
                total_unit = np.sum(np.array(self.accumulated_observation[1][:, 0]) > 0)
                # 승리 시 5, 체력 16 당 0.15, 유닛 갯수당 5
                reward = WIN_REWARD + total_hp * 1.5 / 160 + total_unit * KILL_REWARD
                reward_goliath = reward
                reward_goliathb = reward
                
            # 죽은 경우
            else:
                total_hp = self.last_enemy_unit_hp[1]
                #print("total_hp_here:  "+str(total_hp))
                total_unit = self.last_enemy_unit_cnt[1]
                #print("total_unit_here:  "+str(total_unit))
                # 패배 시 -5, 체력 16 당 -0.15, 유닛 갯수당 -5
                reward = DEFEAT_REWARD - total_hp * 1.5 / 160 + total_unit * DEAD_REWARD
                reward_goliath = reward
                reward_goliathb = reward
                #reward_battle_cruiser = reward
        else:
            cooldown_zero = 0
            #print("accumulated_observation")
            #print(self.accumulated_observation)
            pre_cooldown = self.accumulated_observation[0][:, 1]
            cur_cooldown = self.accumulated_observation[1][:, 1]
            pre_hp = self.accumulated_observation[0][:, 0]
            cur_hp = self.accumulated_observation[1][:, 0]
            
            # # 죽인 경우
            if self.last_enemy_unit_cnt[0] > self.last_enemy_unit_cnt[1]:
                #print("this_shall_not_start")
                reward_goliath += (self.last_enemy_unit_cnt[0] - self.last_enemy_unit_cnt[1]) * KILL_REWARD
                #reward_battle_cruiser += (self.last_enemy_unit_cnt[0] - self.last_enemy_unit_cnt[1]) * KILL_REWARD * (ATTACK_REWARD)

            # 때린 경우 적 에너지 16당 0.1
            if self.last_enemy_unit_hp[0] > self.last_enemy_unit_hp[1]:
                #print("enemy_lose_hp")
                reward_goliath += ((self.last_enemy_unit_hp[0] - self.last_enemy_unit_hp[1])) * ATTACK_REWARD
                #reward_battle_cruiser += ((self.last_enemy_unit_hp[0] - self.last_enemy_unit_hp[1]) / 8) * ATTACK_REWARD_DROPSHIP

            # 죽은 경우
            if self.last_my_unit_cnt[0] > self.last_my_unit_cnt[1]:
                #print("lose an unit")
                reward_goliath += (self.last_my_unit_cnt[0] - self.last_my_unit_cnt[1]) * DEAD_REWARD * (DAMAGED_REWARD_GOLIATH/DAMAGED_REWARD)
                #reward_battle_cruiser += (self.last_my_unit_cnt[0] - self.last_my_unit_cnt[1]) * DEAD_REWARD

            # 맞은 경우
            elif np.sum(pre_hp) > np.sum(cur_hp):
                #print("ally_lose_hp")
                reward_goliath += math.ceil((np.sum(pre_hp) - np.sum(cur_hp)) * 20) * DAMAGED_REWARD_GOLIATH
                #reward_battle_cruiser += math.ceil((np.sum(pre_hp) - np.sum(cur_hp)) * 20) * DAMAGED_REWARD
            
            # invalid action
            if np.any(info['infoMsg'].was_invalid_action): #Question: Invalid action should not be applied to both agents
                reward_goliath += info['infoMsg'].was_invalid_action[1] * INVALID_ACTION_REWARD
                #reward_battle_cruiser += info['infoMsg'].was_invalid_action[0] * INVALID_ACTION_REWARD

            # # 쿨타임 없는 경우
            elif np.any(cur_cooldown == cooldown_zero):
                # 쿨타임이 없는데 p컨을 안하면 마이너스
                reward_goliath += COOLDOWN_REWARD

        
        return [reward_goliathb,reward_goliath]

    def process_observation(self, observation):
        """
        string unit_type = 1;
        int32 hp = 2;
        int32 shield = 3;
        int32 energy = 4;
        int32 cooldown = 5;
        int32 pos_x = 6;
        int32 pos_y = 7;
        double velocity_x = 8;
        double velocity_y = 9;
        double angle = 10;
        bool accelerating = 11;
        bool braking = 12;
        bool attacking = 13;
        bool is_attack_frame = 14;
        repeated bool invalid_action = 15;
        repeated TerrainInfo pos_info = 16;

        only local observation

        hp, shield, energy, cooldown, (pos_x, pos_y,) velocity_x, velocity_y, angle, accelerating, braking, attacking, is_attack_frame,
        pos_info_for_closed_terrain(12), pos_info_for_count_of_enemies(12), pos_info_for_distance_with_cloest_enemy(12)  = 49

        global observation
    """
        #print("start_process_obs")
        processed_observation = np.zeros((NB_AGENTS, OBSERVATION_SIZE))

        goliath_type = getattr(env, 'Terran_Goliath')
        battlecruiser_type = getattr(env, 'Terran_Battlecruiser')
        '''
        goliath and battlecruiser type:
        hp_max: 125
        armor: 1
        cooldown_max: 22
        acceleration: 1
        top_speed: 4.57
        damage_amount: 12
        damage_factor: 1
        weapon_range: 192
        sight_range: 256
        seek_range: 160

            hp_max: 500
        energy_max: 200
        armor: 3
        cooldown_max: 30
        acceleration: 27
        top_speed: 2.5
        damage_amount: 25
        damage_factor: 1
        weapon_range: 192
        sight_range: 352
        '''
        #print("goliath and battlecruiser type:")
        #print(goliath_type)
        #print(battlecruiser_type)

        for i, agent in enumerate(observation.my_unit):
            if agent.hp <= 0:
                continue
            my_x = agent.pos_x
            my_y = agent.pos_y
            my_type_str = agent.unit_type
            my_type = goliath_type if my_type_str == 'Terran_Goliath' else print("error in the my_type")
            t1 = [agent.hp + agent.shield, agent.cooldown, math.atan2(agent.velocity_y, agent.velocity_x),
                  math.sqrt((agent.velocity_x) ** 2 + (agent.velocity_y) ** 2), agent.angle,
                  1 if agent.accelerating else -1 if agent.braking else 0, agent.attacking, agent.is_attack_frame]
            t2 = [self.last_action[i] / (env.action_space[1] - 1)]
            t3 = [i.nearest_obstacle_dist for i in agent.pos_info]
            t4 = []
            t5 = []
            t4_max = []
            t5_max = []
            for idx, enemy in enumerate(observation.en_unit):
                en_type_str = enemy.unit_type
                if en_type_str == 'Terran_Battlecruiser':
                    en_type = battlecruiser_type
                else:
                    continue 
                if enemy.hp <= 0:
                    t4.extend([0,0,0,0,0,0,0,0,0,0])
                else:
                    t4.extend([math.atan2(enemy.pos_y - my_y, enemy.pos_x - my_x), math.sqrt((enemy.pos_x - my_x) ** 2 + (enemy.pos_y - my_y) ** 2),
                     math.atan2(enemy.velocity_y, enemy.velocity_x), math.sqrt((enemy.velocity_x) ** 2 + (enemy.velocity_y) ** 2),
                     enemy.cooldown, enemy.hp + enemy.shield, enemy.angle, 1 if agent.accelerating else -1 if agent.braking else 0, agent.attacking, agent.is_attack_frame])
                t4_max.extend([math.pi, 320, math.pi, en_type.top_speed, en_type.cooldown_max, en_type.hp_max + en_type.shield_max, math.pi, 1, 1, 1])
            for idx, ally in enumerate(observation.my_unit):
                if i == idx:
                    continue
                if ally.hp <= 0:
                    t5.extend([0,0,0,0,0])
                else:
                    t5.extend([math.atan2(ally.pos_y - my_y, ally.pos_x - my_x), math.sqrt((ally.pos_x - my_x) ** 2 + (ally.pos_y - my_y) ** 2),
                     math.atan2(ally.velocity_y, ally.velocity_x), math.sqrt((ally.velocity_x) ** 2 + (ally.velocity_y) ** 2), ally.hp + ally.shield])
                ally_type = goliath_type
                t5_max.extend([math.pi, 320, math.pi, ally_type.top_speed, ally_type.hp_max + ally_type.shield_max])
            if my_type_str == 'Terran_Goliath':
                t1_max = [my_type.hp_max + my_type.shield_max, 1, math.pi, my_type.top_speed, math.pi, 1, 1, 1]
            else:
                t1_max = [my_type.hp_max + my_type.shield_max, my_type.cooldown_max, math.pi, my_type.top_speed, math.pi, 1, 1, 1]
            #t4_max = [math.pi, 320, math.pi, en_type.top_speed, en_type.cooldown_max, en_type.hp_max + en_type.shield_max, math.pi, 1, 1, 1]
            #t5_max = [math.pi, 320, math.pi, ally_type.top_speed, ally_type.hp_max + ally_type.shield_max]

            #t5_max = [32, 32, type.hp_max + type.shield_max, type.cooldown_max,
                      #32, 32, type.hp_max + type.shield_max, type.cooldown_max,
                      #32, 32, type.hp_max + type.shield_max, type.cooldown_max,
                      #32, 32, type.hp_max + type.shield_max, math.pi,
                      #32, 32, type.hp_max + type.shield_max, math.pi,
                      #32, 32, type.hp_max + type.shield_max, math.pi]

            t1 = np.divide(t1, t1_max) # runtime warning
            t2 = np.array(t2) / 320
            t3 = np.array(t3) / 320
            t4 = np.divide(t4, t4_max)
            t5 = np.divide(t5, t5_max)

            processed_observation[i] = np.concatenate([t1, t2, t3, t4, t5])

        self.last_my_unit_cnt.append(np.sum(np.array([u.hp+u.shield for u in observation.my_unit]) > 0))
        self.last_enemy_unit_cnt.append(np.sum(np.array([u.hp+u.shield for u in observation.en_unit]) > 0))
        self.last_enemy_unit_hp.append(sum([u.hp + u.shield for u in observation.en_unit]))
        self.accumulated_observation.append(processed_observation)


        return processed_observation

def random_policy(observation):
    action = []
    for _ in range(2):
        action.append(np.random.randint(NB_ACTIONS))
    return action

if __name__ == '__main__':
    training_mode = True

    FILE_NAME = os.path.basename(__file__).split('.')[0]
    scenario_name = SCENARIO_NAME
    env = GoliathVsBattlecruiser(version=0, frames_per_step=4, action_type=0, move_angle=MOVE_ANGLE, move_dist=4, verbose=0, no_gui=NO_GUI
                         , auto_kill=True)

    NB_ACTIONS = env.action_space[1:][0]  # action_size = ( nb_agents, nb_actions)
    ACTION_SIZE = (NB_AGENTS, NB_ACTIONS)

    try:

        env.observation_space = STATE_SIZE
        """
        i = Input(shape=env.observation_space)
        y = Permute((2, 1, 3))(i)
        y = Dense(200, activation='relu', kernel_regularizer=l2(1e-3))(y)
        y = Dense(100, activation='relu', kernel_regularizer=l2(1e-3))(y)
        y = Dense(100, activation='relu', kernel_regularizer=l2(1e-3))(y)
        y = Dense(60, activation='relu', kernel_regularizer=l2(1e-3))(y)
        y = Reshape((NB_AGENTS, 120))(y)
        y = Bidirectional(LSTM(20, return_sequences=True))(y)
        y = Activation('relu')(y)
        y = Dense(ACTION_SIZE[1], activation='linear', kernel_regularizer=l2(1e-3))(y)
    
        model = Model(i, y)

        model.summary()
        """
        memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=WINDOW_LENGTH)


        # obs_dim = (TIME_WINDOW, NB_AGENTS, env.observation_space[0].shape[0])
        obs_dim = env.observation_space
        act_dim = [NB_AGENTS, None]


        act_dim[1] = NB_ACTIONS
        action_type = 'Discrete'

        # note : here we use gumbel-softmax without using noise into action
        actor = actor_net(env, obs_dim=obs_dim, act_dim=act_dim, enable_gumbel=True, enable_mb=ENABLE_MODEL_BASED)
        critic = critic_net(env, obs_dim=obs_dim, act_dim=act_dim, enable_mb=ENABLE_MODEL_BASED)

        policy = LinearAnnealedPolicy(MA_EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.001, nb_steps=EGREEDY_ANNELING_STEP)
        #policy = LinearAnnealedPolicy(MA_EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.01, value_test=.05, nb_steps=EGREEDY_ANNELING_STEP)
        test_policy = MA_GreedyQPolicy()
        # policy = NoisePolicy(random_process=OrnsteinUhlenbeckProcess(size=1, theta=.15, mu=0., sigma=.2))
        # test_policy = NoisePolicy(
            # random_process=OrnsteinUhlenbeckProcess(size=1, theta=.15, mu=0., sigma=.1))
        print("zhelikaishilema")
        agent = BiCNetAgent(observation_space=obs_dim, nb_agents=act_dim[0], nb_actions=act_dim[1], memory=memory, actor=actor, critic=critic, action_type='discrete',critic_action_input=critic.inputs[1], train_interval=1,
                            batch_size=BATCH_SIZE, nb_steps_warmup_critic=1000, reward_factor=1, nb_steps_warmup_actor=1000, policy=policy, test_policy=test_policy, gamma=DISCOUNT_FACTOR, target_model_update=1e-3,
                            processor=ObsProcessor(nb_action=act_dim[1]))

        # agent.compile(Adam(lr=LEARNING_RATE))
        print("ready to train or already start?")
        agent.compile([SGD(lr=LEARNING_RATE), SGD(lr=LEARNING_RATE)], metrics=['mae'])
        print("see what's here")

        actor.summary()
        print("not_start_yet")
        critic.summary()
        print("mark here")

        callbacks = []

        if training_mode:
            cb_plot = DrawTrainMovingAvgPlotCallback(os.path.realpath('../../save_graph/' + FILE_NAME + '_{}' + post_fix + '.png')
                                                 , 1, 100, l_label=['episode_reward', 'hps'], save_raw_data=True)
            print("check_pos_with_process")
            callbacks.append(cb_plot)
            print(callbacks)
            my_callback = MyCallback(agent=agent)
            callbacks.append(my_callback)
            callbacks.append(tensorboard)
        else:
            h5f = 'GvsGBicNetMyTrial_double_True_dueling_True_batch_size_32_repm_size_20000_lr_0.0025_tn_u_5000_wl_2_nsteps_500000_t_w_1_sc_nm_simple_spread_mb_0_df_0.9_anl_s_300000'
            agent.load_weights(os.path.realpath('../../save_model/' + h5f + '.h5f'))

        agent.run(env, N_STEPS, nb_max_episode_steps=10000, train_mode=training_mode, verbose=2, callbacks=callbacks, nb_max_start_steps=10, random_policy=random_policy)

        if training_mode:
            agent.save_weights(os.path.realpath('../../save_model/' + FILE_NAME + post_fix + '.h5f'), True)

    finally:
        env.close()
