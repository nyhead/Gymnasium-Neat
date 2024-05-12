from functools import partial
import multiprocessing
import os
import pickle
import random

import neat
import numpy as np
import gym
import argparse
import visualize

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str,default='MountainCarContinuous-v0',required=False)
parser.add_argument('--conf', type=str,default='configs/config-mountain-car-continuos',required=False)
parser.add_argument('--nruns', type=int,default=5,required=False)
parser.add_argument('--evolve', action='store_true',required=False)
parser.add_argument('--show', action='store_true',required=False)
parser.add_argument('--evaluate', action='store_true',required=False)
parser.add_argument('--seeds', nargs="+", type=int,default=[42],required=False)
args = parser.parse_args()

runs_per_net = args.nruns
net_type = 'feedforward'
env_name = args.env

env = gym.make(env_name)
def eval_genome(genome, config,seed):
    fitnesses = []

    if net_type == 'feedforward':
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    for runs in range(runs_per_net):
        fitness = 0.0
        observation,_ = env.reset(seed=seed)
        done = False
        steps = 0
        while not done:
            action =  net.activate(observation)
            action = np.argmax(action) if len(action) > 1 else action
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            done = terminated or truncated
        fitnesses.append(fitness)

    return np.mean(fitnesses)
def eval_genome_wrapper(genome, config, seed):
    return eval_genome(genome, config, seed)

def run(save_winner=True,seed=None):

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, args.conf)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    eval_func = partial(eval_genome_wrapper, seed=seed)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_func)    
    winner = pop.run(pe.evaluate)

    # Save the winner.
    if save_winner:
        with open(f'winner-{args.env}', 'wb') as f:
            pickle.dump(winner, f)

    print(winner)
    return stats,winner

def show_winner():
    with open(f'winner-{args.env}', 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, args.conf)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    if net_type == 'feedforward':
        net = neat.nn.FeedForwardNetwork.create(c, config)


    env = gym.make(args.env,render_mode="human")
    observation,_ = env.reset()

    done = False
    while not done:
        action =  net.activate(observation)
        action = np.argmax(action) if len(action) > 1 else action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = truncated or terminated
    env.close()
    



if __name__ == '__main__':

    if args.evolve:
        run()
        env.close()
    if args.show:
        show_winner()
    if args.evaluate:
        seeds = args.seeds
        env = gym.make(env_name)
        for s in seeds:
            random.seed(s) 
            print(s)
            stats,winner = run(False,s)
            visualize.plot_stats(stats,view=True)