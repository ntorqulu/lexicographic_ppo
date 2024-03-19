import numpy as np


def no_filter(probs):
    try:
        return np.random.multinomial(1, probs).argmax()
    except ValueError as e:
        probs /= probs.sum()
        return np.random.multinomial(1, probs).argmax()


def greedy(probs):
    return np.argmax(probs)


def upper_filter(probs):
    if len(probs[probs > 0.5]):
        return greedy(probs)
    else:
        return no_filter(probs)


def bottom_filter(probs):
    epsilon = 0.1
    higher = probs[probs > epsilon]
    if len(higher) != len(probs):
        # Softmax it
        softmax_values = np.exp(higher) / np.exp(higher).sum(axis=0)
        cont = 0
        for i, p in enumerate(probs):
            if p > epsilon:
                probs[i] = softmax_values[cont]
                cont += 1
            else:
                probs[i] = 0
    return no_filter(probs)


def both_filter(probs):
    if len(probs[probs > 0.5]):
        return greedy(probs)
    return bottom_filter(probs)
