#import gym
import numpy as np
from DDQNAgent import DDQNAgent
from puzzle import GameGrid
import matplotlib.pyplot as plt

def plot_learning_curve_simple(epoche, scores, filename):
    # Data for plotting

    fig, ax = plt.subplots()
    ax.plot(epoche, scores)

    ax.set(xlabel='Epoche', ylabel='Points',
           title='Points per Epoche Graph')
    ax.grid()
    fig.savefig(filename)
    plt.show()

#TODO: Die Maße anpassen
def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)




if __name__ == '__main__':
    #env = gym.make('LunarLander-v2')


    # alpha = learning rate - gamma = discount factor - epsilon - batch_size - input dimensions
    # um wv epsilon weniger wird - epsilon minimum wert, max memory size = 1 million
    # name of file - nach wv er syncen soll zwischen den 2 Networks replace target ist hyper parameter
    #def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
    #            input_dims, epsilon_decr=0.966, epsilon_end=0.01,
    #             mem_size=1000000, fname='ddqn_model.h5', replace_target=100):

    #TODO: Problem mit input Shape da bei ihm vector von 8 zahlen bei mir 4x4x1 matrix
    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
                            batch_size=32, input_dims=4)
    n_games = 400

    # load saved Model Here
    ddqn_agent.load_model()

    #TODO: Alles in numpy arrays umwandeln
    epoches = []
    ddqn_scores = []
    ddqn_biggest_tiles = []
    epsilon_history = [] # um zu sehen wie gut der score sich verbessert wenn epsilon weniger wird
    avg_scores = []

    #save output
    #will overwrite anything in the folder -> diffrent directories with diffrent models
    #env = wrappers.Nobitor(env, 'tmp/lunar-lander', video_callable=lambda episode_id True, force=True)

    #for i in range(n_games):

    #TODO: AB hier hat er schon einmal durch also fehler ausbessern das er hier das erste mal macht und nicht schon einmal davor
    for i in range(n_games):
        score = 0
        gamegrid = GameGrid(agent=ddqn_agent)


        print('\nScore: ', gamegrid.score, ' Biggest tile: ', gamegrid.biggest_tile, ' Epsilon: ', gamegrid.ddqn_agent.epsilon)
        ddqn_scores.append(gamegrid.score)
        ddqn_biggest_tiles.append(gamegrid.biggest_tile)
        epsilon_history.append(gamegrid.ddqn_agent.epsilon)

        # Um zu sehen ob der Agent besser wird und dazu lernt printen wir den avg score von den letzten 100 spielen aus
        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        avg_scores.append(avg_score)
        print('\nepisode: ', i, 'score %.2f' %gamegrid.score, 'average score %.2f' %avg_score)
        print('\n---------\n')
        if i%5 == 0 and i > 0:
            print('------------------- hier')
            #gamegrid.ddqn_agent.save_model()

        epoches.append(i)

        #TODO: Eine Liste machen und speichern wie oft welches grösste Tile erreicht wurde wie oft welches grösste Tile erreicht wurde
        #TODO: Diagramme und Modells nicht immer überschreiben sondern extra ordner für jedes einzelne machen und immer passend benennen
    filename = 'Epoch_Score.png'
    x = [i+1 for i in range(n_games)]

    plot_learning_curve(x, np.array(epoches), np.array(ddqn_scores), filename)
    #plot_learning_curve_simple(np.array(epoches), np.array(ddqn_scores), filename)
