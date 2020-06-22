import random
import math
import time
from tkinter import Frame, Label, CENTER, Tk
from DDQNAgent import DDQNAgent
import numpy as np  # For numerical fast numerical calculations
import time
import logic
import constants as c


class GameGrid(Frame):
    #TODO: VARIABLEN NAMEN IN ENGLISH UND MEHR HYPERPARAMETER HINZUFÜGEN ZUM TWEAKEN
    REWARD_KOMPONENTEN_RATIO = 0.5

    REWARD_AKTUELLES_STATE_GROSSTE_KACHEL = 0.5
    REWARD_AKTUELLES_STATE_SCORE = 0.5

    REWARD_POTENZIAL_STATE_ECKE = 0.3
    REWARD_POTENZIAL_STATE_FREIE_KACHELN = 0.3
    REWARD_POTENZIAL_STATE_MONOTONIE = 0.2
    REWARD_POTENZIAL_STATE_GLEICHMAESSIG = 0.2


    def __init__(self, agent):
        # maybe den Agent hier übernehmen weil nicht im frame ?
        Frame.__init__(self)
        self.score = 0
        self.highscore = 0
        self.biggest_tile = 0
        # Agent wird hier bei jedem Spiel übergeben
        self.ddqn_agent = agent
        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        # self.gamelogic = gamelogic
        self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                         c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right,
                         c.KEY_UP_ALT: logic.up, c.KEY_DOWN_ALT: logic.down,
                         c.KEY_LEFT_ALT: logic.left,
                         c.KEY_RIGHT_ALT: logic.right}

        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        #TODO: Hier highscore implementieren?
        #Mit dieser Schleife kann ich meinen Code neben dem Mainloop von tkinter ausühren
        done = False
        while not done:
            done = self.task()

#        self.mainloop()

#TODO: Name anpassen - Hier kommt mein code Rein
    def task(self):
        self.powers_of_two_matrix = self.translate_matrix(self.matrix)

        #TODO: WIESO WIRD CHOOSE ACION NICHT ERKANNT????????????????? !!!!!!!!!!!!!!!!!!!!!!!!!!
        reward, done_new, action = self.step(self.ddqn_agent.choose_action(self.powers_of_two_matrix))
        done = self.check_if_done_auto(done_new)
        if done != True:
            done = False

        #TODO: Score arrays befüllen hier oder draussen im main
        self.ddqn_agent.remember(self.matrix_old, action, reward, self.matrix, done)
        self.ddqn_agent.learn()

        #TODO: Wenn ich jeden schritt des agenten sehen will die kommentare auskommentieren
        if done:
            #print(self.matrix[0], '\n', self.matrix[1], '\n', self.matrix[2], '\n', self.matrix[3], '\n')
            return True
        self.update_grid_cells()
        self.update()

        # TODO: sleep nur zum visualisieren
        # time.sleep(1)
        #print(self.matrix[0], '\n', self.matrix[1], '\n', self.matrix[2], '\n', self.matrix[3], '\n')


    # TODO: Return state_new, reward, done
    # Reward Function
    #   Momentan Komponente:
    #                   Grösste kachel (größer geworden oder nicht?)
    #                   score (verdoppeln ist das maximale -> 1 zb. und nichts mehr ist 0)
    #   Potenzial:
    #                  Pin die ecke
    #                  frei kacheln
    #                  Monotonie (fängt gross an und wird kleiner) nur bei großen kacheln wichtig bei 2 und 4 und 8 egal eigentlich??
    #                  gleichmäßigkeit (gleiche kacheln neben einander)

    #TODO: Weitere Reward Kriterien noch einbauen (Potenzial)
    #TODO: HIER DIE VERTEILUGN PARAMETARISIEREN UM ZU TUNEN
    def calc_reward(self, new_points):
        #momentan + potenzial
        reward = 0
        if logic.game_state(self.matrix) == 'lose':
            return -100
        if np.array_equal(self.matrix_old, self.matrix):
            return -10

        # Wenn es eine neue größte kache gibt
        if self.biggest_tile > self.get_biggest_tile(self.matrix):
            reward += self.REWARD_AKTUELLES_STATE_GROSSTE_KACHEL

        #TODO: hier sollte mit der alten matrix gearbeitet werden
        max_points_possible = self.calc_max_points_possible(self.matrix_old)

        #TODO: Guter ansatz oder besser alle punkte geben ?
        #wenn keine Punkte möglich sind gib ihm hälfte der möglichen punkte
        if(max_points_possible == 0):
            reward = reward + self.REWARD_AKTUELLES_STATE_SCORE * 0.5
        else:
            #TODO: Arbeitet er hier mit der MAtrix nach der Aktion oder davor ? sollte mit der matrix vor der aktion arbeiten
            reward = reward + (self.REWARD_AKTUELLES_STATE_SCORE * ((new_points*100) / max_points_possible))

        # an hier potenzial Komponente ----------------------------------------------------
        return reward

    # TODO: Auslagern zu Logix diese Funktion
    # max points per step ist alle 2 gleichen mit sum aufaddieren
    def calc_max_points_possible(self, matrix):
        doppelte = []
        sum = 0
        for i in range(4):
            for j in range(4):
                if matrix[i][j] in doppelte:
                    doppelte.remove(matrix[i][j])
                    sum = sum + (matrix[i][j]*2)
                else:
                    doppelte.append(matrix[i][j])
        return sum

    def step(self, action):
        #TODO: Check hier ob die neue matrix auch wirklich die neue ist und die alte die alte und nicht das beide gleich sind
        self.matrix_old = self.copy_matrix()
        if action == 0:
            self.matrix, done, new_points = logic.up(self.matrix)
        elif action == 1:
            self.matrix, done, new_points = logic.right(self.matrix)
        elif action == 2:
            self.matrix, done, new_points = logic.down(self.matrix)
        elif action == 3:
            self.matrix, done, new_points = logic.left(self.matrix)
        self.score += new_points
        self.biggest_tile = self.get_biggest_tile(self.matrix)

        reward = self.calc_reward(new_points)
        #self.matrix =  self.matrix_nach_aktion
        #done = self.check_if_done(done)


        return (reward, done, action)

    def get_biggest_tile(self, matrix):
        biggest_tile = 0
        for i in range(4):
            for j in range(4):
                if matrix[i][j] > biggest_tile:
                    biggest_tile = matrix[i][j]
        return biggest_tile

    #TODO: Auslagern zu Logix diese Funktion
    #TODO: Falls ich andere Matrixen benutze ausser 4x4 dann hier auch ändern
    #TODO:  Feature Scaling Methode Normalisierung Einabuen zahlen von 0-1
    def translate_matrix(self, matrix):
        new_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if matrix[i][j] != 0:
                    new_matrix[i][j] = math.log2(matrix[i][j])
        return new_matrix

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return random.randint(0, c.GRID_LEN - 1)

    def init_matrix(self):
        #TODO: IST DIE 90% 10% Wahrscheinlichkeit einer 4 nicht drinnen oder was?
        self.matrix = logic.new_game(4)
        self.history_matrixs = list()
        self.matrix = logic.add_two(self.matrix)
        self.matrix = logic.add_two(self.matrix)

        #TODO: Name anpassen zb. state_
        self.matrix_old = self.matrix
        self.powers_of_two_matrix = np.zeros((4, 4))

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(
                        new_number), bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def key_down(self, event):
        key = repr(event.char)
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            #TODO: Mit score arbeiten und abspeichern
            self.matrix, done, score = self.commands[repr(event.char)](self.matrix)
            self.check_if_done(done)

#Ausgelagert weil code Zwei mal vorkommt, ein mal für die manuelle eingabe und einmal für ide automatische
#TODO: Sollte am anfang eines programms eine auswahl passieren ob man selber spielen möchte oder nicht ? oder zb. ein Tipp Butten der mit den besten tipp gibt wär auch cool
    def check_if_done(self, done):
        if done:
            self.matrix = logic.add_two(self.matrix)
            # record last move
            self.history_matrixs.append(self.matrix)
            self.update_grid_cells()
            done = False
            if logic.game_state(self.matrix) == 'win':
                #TODO: Hier den limit für 2048 löschen um zu schauen wie weit er kommen wird
                self.grid_cells[1][1].configure(
                    text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(
                    text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            if logic.game_state(self.matrix) == 'lose':
                self.quit()
                self.destroy()
                return True
                #TODO: Sollte ich hier nicht ein done boolean returnen damit er weis ok ist done wo ich die funktionen aufrufe und es dann feeden kann dem .remember?

                # self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                # self.grid_cells[1][2].configure(
                # text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    #TODO: Eventuell brauche ich das nicht es genügt ein if else um zu schauen ob manuel gedrückt oder automatisch und dann je nachdem aktuelle matrix oder die neue matrix
    def check_if_done_auto(self, done):
        if done:
            #TODO: Hier matrix nach aktion oder normale matrix?
            self.matrix = logic.add_two(self.matrix)
            # record last move
            self.history_matrixs.append(self.matrix)
            self.update_grid_cells()
            done = False
            if logic.game_state(self.matrix) == 'win':
                #TODO: Hier den limit für 2048 löschen um zu schauen wie weit er kommen wird
                self.grid_cells[1][1].configure(
                    text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                self.grid_cells[1][2].configure(
                    text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            if logic.game_state(self.matrix) == 'lose':
                self.quit()
                self.destroy()
                return True
                #TODO: Sollte ich hier nicht ein done boolean returnen damit er weis ok ist done wo ich die funktionen aufrufe und es dann feeden kann dem .remember?


    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

    def copy_matrix(self):
        new_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for i in range(4):
            for j in range(4):
                if self.matrix[i][j] != 0:
                    new_matrix[i][j] = self.matrix[i][j]
        return (new_matrix)