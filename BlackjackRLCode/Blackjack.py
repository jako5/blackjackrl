import random
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from time import time
import multiprocessing as mp
import pandas as pd
from tensordict import TensorDict
from torch import Tensor
import torch


def stringify_hand(hand):
    return " | ".join([f"{c1}-{c2}" for c1, c2, _ in hand])


def load_decision_table(path):
    df = pd.read_csv(path, index_col=0, header=0, encoding="utf-8", dtype=str)
    for a in ["J", "K", "Q"]:
        df[a] = df["10"]
    return {str(k):v for k, v in df.to_dict('index').items()}

def getval(rank):
    return 11 if (rank == "A") else (10 if rank in ['K', 'Q', 'J'] else int(rank))

class Blackjack:
    def __init__(self, DECKS=4, LOG=True, PLOT=False, DRAW_STRATEGY="Manual", BET_STRATEGY="Constant", THRESHOLD=0,
                 MAXROUNDS=np.inf, **kwargs):

        self.gameround = -1
        self.LOG = LOG
        self.PLOT = PLOT
        self.lastresult = None
        self.DRAW_STRATEGY = DRAW_STRATEGY
        self.BET_STRATEGY = BET_STRATEGY
        self.THRESHOLD = THRESHOLD
        self.maxrounds = MAXROUNDS
        self.shufflings = 0
        self.game_running = False
        self.DECKS = DECKS
        self.split_depot = None
        self.progess_balance = []
        self.progess_hilocount = []

        self.player_hand = None
        self.dealer_hand = None
        self.has_doubled_down = False
        self.has_split = False

        self.dctbl_singles = load_decision_table("./data/decisiontable_singles.csv")
        self.dctbl_aces = load_decision_table("./data/decisiontable_aces.csv")
        self.dctbl_pairs = load_decision_table("./data/decisiontable_pairs.csv")

        suits = np.array(['Hearts', 'Diamonds', 'Clubs', 'Spades'])
        ranks = np.array(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'])

        singledeck = np.array([np.array([rank, suit, getval(rank)]) for rank in ranks for suit in suits])
        self.cardbank = np.tile(singledeck, [self.DECKS, 1])
        self.cardindex = None
        self.shuffle_cardbank()
        self.balance = 0
        self.bet = 0

        self.results = {
            "win_by_blackjack": [],
            "win_by_bust": [],
            "win_by_comparison": [],
            "tie": [],
            "lose_by_blackjack": [],
            "lose_by_bust": [],
            "lose_by_comparison": [],
        }

    def shuffle_cardbank(self):
        if self.LOG: print("### RESHUFFLING DECKS ###")
        rp = np.random.permutation(self.cardbank.shape[0])
        self.cardbank = self.cardbank[rp]
        self.cardindex = -1
        self.shufflings += 1

    def draw_card(self):
        # Shuffle Cards and reset counter in case of completed deck
        if self.cardindex == (self.cardbank.shape[0] - 1):
            self.shuffle_cardbank()
            self.cardindex = -1
        self.cardindex += 1
        assert self.cardindex > -1
        return self.cardbank[self.cardindex]

    def get_card_value(self, card):
        return int(card[-1])
        # rank, suit, value = card
        # return 11 if (rank == "A") else (10 if rank in ['K', 'Q', 'J'] else int(rank))

    def calculate_hand_value(self, hand, return_aces_pairs=False):
        value = sum([self.get_card_value(card) for card in hand])
        # Check for aces and adjust value if needed
        num_aces = list(map(lambda x: x[0], hand)).count('A')
        ace_iter = copy(num_aces)
        while ace_iter > 0 and value > 21:
            value -= 10
            ace_iter -= 1

        if return_aces_pairs:
            return value, num_aces, hand[0][0]==hand[1][0]
        else:
            return value

    def display_board(self, show_all=False):
        if not self.LOG: return
        print(f"Your hand: {stringify_hand(self.player_hand)} ({self.calculate_hand_value(self.player_hand)})")
        if show_all:
            print(f"Dealer's hand: {stringify_hand(self.dealer_hand)} ({self.calculate_hand_value(self.dealer_hand)})")
        else:
            print(f"Dealer's hand: {stringify_hand([self.dealer_hand[0], ['X', 'X', None]])}")

    def get_reward(self):
        return Tensor([self.balance])

    def get_done(self):
        return torch.tensor([int(not self.game_running)], dtype=torch.int8)

    def get_state(self):
        playerhandval, playeraces, playerpair = self.calculate_hand_value(self.player_hand, return_aces_pairs=True)
        dealerhandval = self.get_card_value(self.dealer_hand[0])
        return TensorDict({
            "playerhandval": playerhandval,
            "dealerhandval": dealerhandval,
            "playerace": playeraces,
            "playerpair": int(playerpair), # TEMP Solution
        })
        
    def get_action(self):
        action = None
        permitted_actions = ["h","s","d","v"]
        can_split = (self.player_hand[0][0] == self.player_hand[1][0]) and not self.has_split and len(self.player_hand)==2 # equal rank
        permitted_actions += ["v"] if can_split else []

        if self.DRAW_STRATEGY == "Manual":
            while action is None:
                inp = input(f"Do you want to {'split (v), ' if can_split else ''}hit (h), stand (s) or double (d)? ").lower()
                if inp in permitted_actions:
                    action = inp

        elif self.DRAW_STRATEGY == "Threshold":
            assert self.THRESHOLD is not None
            hit = self.calculate_hand_value(self.player_hand) < self.THRESHOLD
            action = "h" if hit else "s"
            if self.LOG:
                print(f"{self.calculate_hand_value(self.player_hand)} {'<' if hit else '>='}\
                        {self.THRESHOLD} ---> ACTION IS {'HIT' if action else 'STAND'}")

        elif self.DRAW_STRATEGY == "DecisionTable":
            playerhandval, playeraces, playerpair = self.calculate_hand_value(self.player_hand, return_aces_pairs=True)
            dface1, _, _ = self.dealer_hand[0]
            pface1, _, _ = self.player_hand[0]
            pface2, _, _ = self.player_hand[1]


            if playeraces and len(self.player_hand)==2:
                action = self.dctbl_aces[pface1 if pface1!="A" else pface2][dface1]
            elif playerpair and len(self.player_hand)==2:
                action = self.dctbl_pairs[pface1][dface1]
            else:
                action = self.dctbl_singles[str(playerhandval)][dface1]

        # action = "h" if action in ["v","d"] else action

        if action not in permitted_actions:
            raise ValueError(action)

        return action

    def execute_bet(self):
        if self.BET_STRATEGY == "Constant":
            self.balance -= 1
            self.bet = 1

        elif self.BET_STRATEGY == "Adrenaline":
            if self.lastresult == "lose":
                self.balance -= 1
                self.bet = 1
            else:
                self.balance -= self.bet
                self.bet += self.bet

        else:
            self.bet = 0
            count = None
            if self.BET_STRATEGY == "HiLo":
                count = self.get_HiLo_Count()
            elif self.BET_STRATEGY == "Omega":
                count = self.get_Omega_Count()
            elif self.BET_STRATEGY == "Halves":
                count = self.get_Halves_count()
            elif self.BET_STRATEGY == "HiOpt":
                count = self.get_HiOpt_count()


            if count>4:
                self.balance -= 3
                self.bet += 3
            elif count>2:
                self.balance -= 2
                self.bet += 2
            else:
                self.balance -= 1
                self.bet += 1

    def finish_game(self, result, message):

        if result == "win_by_blackjack":
            self.balance += (self.bet * 5 / 2)
            self.lastresult = "win"
        elif "win" in result:
            self.balance += (self.bet * 4 / 2)
            self.lastresult = "win"
        elif "tie" in result:
            self.balance += self.bet
            self.lastresult = "tie"
        elif "lose" in result:
            self.lastresult = "lose"
            pass
        else:
            assert False

        self.progess_balance.append(self.balance)


        if self.LOG:
            print(message)
        self.results[result].append((self.gameround, self.balance))
        self.game_running = False

    def get_used_cards(self):
        return self.cardbank[:self.cardindex+1]

    def get_HiLo_Count(self):
        uc = self.get_used_cards()
        minus_cards = len([rank for rank, _, _ in uc if rank in ['10', 'J', 'Q', 'K', 'A']])
        plus_cards = len([rank for rank, _, _ in uc if rank in ['2', '3', '4', '5', '6']])
        running_count = plus_cards - minus_cards
        return self.get_TC(running_count)

    def get_Omega_Count(self):
        pass
        # uc = self.get_used_cards()
        # minus2_cards = len([rank for rank, _, _ in uc if rank in ['10', 'J', 'Q', 'K']])
        # minus1_cards = len([rank for rank, _, _ in uc if rank in ['9']])
        # plus1_cards = len([rank for rank, _, _ in uc if rank in ['2', '3', '7']])
        # plus2_cards = len([rank for rank, _, _ in uc if rank in ['4', '5', '6']])
        # running_count = plus2_cards * 2 + plus1_cards - minus1_cards - 2 * minus2_cards
        # true_count = running_count / self.DECKS
        # return true_count

    def get_HiOpt_count(self):
        pass

    def get_Halves_count(self):
        uc = self.get_used_cards()
        minus_cards = len([rank for rank, _, _ in uc if rank in ['J', 'Q', 'K', 'A']])
        minus_half_cards = len([rank for rank, _, _ in uc if rank in ['9']])
        plus_half_cards = len([rank for rank, _, _ in uc if rank in ['2', '7']])
        plus_cards = len([rank for rank, _, _ in uc if rank in ['3', '4', '6']])
        plus_oneandhalf = len([rank for rank, _, _ in uc if rank in ['5']])
        running_count = plus_oneandhalf * 1.5 + plus_cards * 1 + plus_half_cards * 0.5 - minus_half_cards * 0.5 - minus_cards
        return self.get_TC(running_count)

    def get_TC(self, RC):
        normfac = self.DECKS * (self.cardbank.shape[0] - self.cardindex - 1) / (self.cardbank.shape[0])
        normfac = np.inf if normfac==0 else normfac
        return RC / normfac
    
    def play_round(self):
        # Display current game state
        self.display_board()

        # Check for player blackjack or bust after cards have been dealt
        interrupt = self.react_to_roundstart()
        if interrupt:
            return

        # Ask player to hit or stand
        action = self.get_action() if not self.has_doubled_down else None

        # React to Players Action
        self.react_to_action(action)

    def react_to_roundstart(self):
        if self.calculate_hand_value(self.player_hand) == 21:
            self.finish_game("win_by_blackjack", "Blackjack! You win!")
            return True

        elif self.calculate_hand_value(self.player_hand) > 21:
            self.finish_game("lose_by_bust", "Bust! You lose.")
            return True

    def react_to_action(self, action):

        if action == "h":
            self.player_hand.append(self.draw_card())

        elif action == "d":
            self.player_hand.append(self.draw_card())
            self.balance -= self.bet
            self.bet *= 2
            self.has_doubled_down = True

        elif action == "v":
            self.split_depot = self.player_hand.pop()
            self.player_hand.append(self.draw_card()) # Second card is replaced
            self.has_split = True

        else:
            # Dealer's turn
            while self.calculate_hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.draw_card())
            self.display_board(show_all=True)

            # Check for dealer blackjack or bust
            if self.calculate_hand_value(self.dealer_hand) == 21:
                self.finish_game("lose_by_blackjack", "Dealer has Blackjack! You lose.")
            elif self.calculate_hand_value(self.dealer_hand) > 21:
                self.finish_game("win_by_bust", "Dealer busts! You win!")
            else:
                if self.calculate_hand_value(self.player_hand) > self.calculate_hand_value(self.dealer_hand):
                    self.finish_game("win_by_comparison", "You win!")
                elif self.calculate_hand_value(self.player_hand) < self.calculate_hand_value(self.dealer_hand):
                    self.finish_game("lose_by_comparison", "You lose.")
                else:
                    self.finish_game("tie", "It's a tie!")

    def prepare_game(self):
    
        self.gameround += 1

        self.execute_bet()

        # Deal initial cards
        self.player_hand = [self.draw_card(), self.draw_card()] if self.split_depot is None else [copy(self.split_depot), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]

        if self.LOG:
            print("CARDS DRAWN:", self.cardindex + 1, "/", len(self.cardbank), "\n")
            print( "-> SPLIT CARD ROUND" if  self.has_split else "")

        self.game_running = True
        self.has_doubled_down = False
        self.has_split = False
        self.split_depot = None
    
    
    def play_game(self):
    
        self.gameround = -1
        # Game loop
        while self.gameround < self.maxrounds:
            
            # Prepare Game
            self.prepare_game()

            # Round loop
            while self.game_running:
                self.play_round()


        if self.PLOT:
            fig, ax = plt.subplots(figsize=(30, 15))

            ax.scatter(*np.array(self.results["win_by_blackjack"]).T, marker="X", color="g", s=100)
            ax.scatter(*np.array(self.results["win_by_bust"]).T, marker="s", color="g", s=70)
            ax.scatter(*np.array(self.results["win_by_comparison"]).T, marker="o", color="g", s=50)

            ax.scatter(*np.array(self.results["lose_by_blackjack"]).T, marker="X", color="r", s=100)
            ax.scatter(*np.array(self.results["lose_by_bust"]).T, marker="s", color="r", s=70)
            ax.scatter(*np.array(self.results["lose_by_comparison"]).T, marker="o", color="r", s=50)

            ax.plot(self.progess_balance)

            # npcounts = np.array(self.counts)
            # ax.set_title(f"nice: {np.sum(npcounts > 2)}")

            plt.show()

        return {
            "roundsplayed": self.gameround,
            "results": self.results,
            "shufflings": self.shufflings,
            "balance": self.balance,
            "balanceprogress": self.progess_balance
        }


def proxylauncher(args):
    agent = Blackjack(**args)

    data = agent.play_game()

    totalwins = sum([len(data['results'][res]) for res in data['results'].keys() if 'win' in res])
    totallosses = sum([len(data['results'][res]) for res in data['results'].keys() if 'lose' in res])

    if args["LOGRESULTS"]:
        print(f"ROUNDS PLAYED: {data['roundsplayed']}")
        for res in data["results"].keys():
            print(f"/// {res}: {len(data['results'][res])}")
        print(f"\n///// WIN TOTAL {totalwins}")
        print(f"///// LOSE TOTAL {totallosses}")
        print(f"WIN/LOSE Ratio: {totalwins / totallosses:.3f}")
        print(f"Balance: {data['balance']}")

        print("\nShuffled:", data["shufflings"])

    # print("-- finished sample", args["sid"] if "sid" in args.keys() else "")

    return {
        "wlr": totalwins / totallosses,
        "balance": data['balance'],
        **{res: len(data['results'][res]) for res in data["results"].keys()},
        "balanceprogress": data['balanceprogress'],
    }


# Run the game
if __name__ == "__main__":
    st = time()

    # ### Interactive Game
    # Blackjack().play_game()

    ### Automated Serial Analysis
    SAMPLES = 10
    specs = {
        "MAXROUNDS": 1000,
        "THRESHOLD": 17,
        "DECKS": 4,
        "DRAW_STRATEGY": "DecisionTable",  # Manual, Threshold, DecisionTable
        "BET_STRATEGY": "Constant",  # Constant, HiOpt, HiLo, Omega, Halves
        "LOG": False,
        "LOGRESULTS": False,
        "PLOT": False,
    }

    # Single Game
    proxylauncher({**specs, "LOGRESULTS": True, "PLOT": True})

    # Pooled
    #pool = mp.Pool()
    #data_for_samples = pool.map(proxylauncher, [{**specs, "sid": sid + 1} for sid in range(SAMPLES)])

    print(f"{time() - st:.2f} s.")
