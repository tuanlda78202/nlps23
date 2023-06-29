from all.model.mc import build_markov_model, write_play
import random
import pandas as pd


def main(lines, mc_odr1, mc_odr2):
    words = build_markov_model(lines, mc_odr1, mc_odr2)

    play_length = 150
    hints = [random.choice(words) for x in range(play_length)]
    poem = write_play(hints, mc_odr1, mc_odr2)
    print(poem)

if __name__ == "__main__":
    df = pd.read_csv('poems_dataset.csv')

    genre_list = df['genre'].tolist()
    genre_list = list(dict.fromkeys(genre_list))

    df = df.loc[df['genre'] == 'luc bat']
    df = df.sample(frac=0.12)

    lines = df['content'].tolist()
    mc_odr1 = {}

    # This is the second order markov chain. It chains a pair
    # of words with word(s) that can follow it
    mc_odr2 = {}
    main(lines=lines, mc_odr1=mc_odr1, mc_odr2=mc_odr2)