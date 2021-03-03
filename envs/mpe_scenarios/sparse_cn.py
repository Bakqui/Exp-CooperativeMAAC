import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

num_agents_global = 4
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0#2
        num_agents = num_agents_global
        num_landmarks = num_agents_global
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05#0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.35, 0.35])#np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        k=0
        for agent in world.agents:
            if k==0:
                agent.state.p_pos = np.array([-0.15,-0.15])#np.random.uniform(0, 0, world.dim_p)
            elif k==1:
                agent.state.p_pos = np.array([-0.15,+0.15])
            elif k==2:
                agent.state.p_pos = np.array([+0.15,-0.15])
            else:
                agent.state.p_pos = np.array([0.15,0.15])
            k+=1
            # agent.state.p_pos = np.random.uniform(-0.15, +0.15, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        k=0
        for i, landmark in enumerate(world.landmarks):
            if k==0:
                landmark.state.p_pos = np.array([-0.3,-0.3])#np.random.uniform(0, 0, world.dim_p)
            elif k==1:
                landmark.state.p_pos = np.array([-0.3,+0.3])
            elif k==2:
                landmark.state.p_pos = np.array([+0.3,-0.3])
            else:
                landmark.state.p_pos = np.array([0.3,0.3])
            k+=1
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def set_world(self, world, obs_n):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.35, 0.35])#np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        k=0
        for i,agent in enumerate(world.agents, 0):
            # if k==0:
            #     agent.state.p_pos = np.array([0.2,0.2])#np.random.uniform(-1, +1, world.dim_p)
            # else:
            #     agent.state.p_pos = np.array([-0.2,-0.2])
            # k+=1
            agent.state.p_pos = np.squeeze(obs_n[i*2:(i+1)*2])
            agent.state.p_vel = np.squeeze(obs_n[(num_agents_global*2+i*2):(num_agents_global*2+(i+1)*2)])#np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        k=0
        for i, landmark in enumerate(world.landmarks,0):
            # landmark.state.p_pos = np.squeeze(obs_n[(2*4+i*2):(2*4+(i+1)*2)])#np.random.uniform(-1, +1, world.dim_p)
            if k==0:
                landmark.state.p_pos = np.array([-0.3,-0.3])#np.random.uniform(0, 0, world.dim_p)
            elif k==1:
                landmark.state.p_pos = np.array([-0.3,+0.3])
            elif k==2:
                landmark.state.p_pos = np.array([+0.3,-0.3])
            else:
                landmark.state.p_pos = np.array([0.3,0.3])
            k+=1
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # rew = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        #
        rew = 0#-1#-1#0
        # a = agent
        #
        # dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        # if min(dists)<0.1:
        #     rew += 1
        # rew -= min(dists)
        # if agent.collide:
        #     for wa in world.agents:
        #         # if wa is agent: #for training needed (collide)
        #         #     continue
        #         # else:
        #         if self.is_collision(wa, agent):
        #             rew -= 1#10
        occupied_landmarks = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if occupied_landmarks == num_agents_global:
            rew = 1#0

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # def bound(x):
        #     if x < 0.5:
        #         return 0
        #     return 1
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew

    def observation(self, agent, world):

        # comp_pos = []
        # for comp in range(2):
        #     comp_pos.append(np.array([0,0]))
        #
        # comp_vel = []
        # for comp in range(2):
        #     comp_vel.append(np.array([0,0]))

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)

        pos = []
        vel = []
        for other in world.agents:
            # if other is agent: continue
            # comm.append(other.state.c)
            pos.append(other.state.p_pos)
            # if not other.adversary:
            vel.append(other.state.p_vel)
        # print(comm)
        return np.concatenate(pos + vel)# + entity_pos)
