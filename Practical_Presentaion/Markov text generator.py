import random

def build_markov_chain(text):
    words = text.split()
    chain={}

    for i in range(len(words)-1):
        current_word=words[i]
        next_word=words[i+1]

        if current_word not in chain:
            chain[current_word]=[]

        chain[current_word].append(next_word)
    return chain

def generate_text(chain, length=20):
    start_word = random.choice(list(chain.keys()))
    result = [start_word]

    current_word = start_word

    for i in range(length - 1):
        if current_word not in chain:
            break

        next_word = random.choice(chain[current_word])
        result.append(next_word)
        current_word = next_word

    return " ".join(result)

text = """
I love coding
I love Python
Python is easy
coding is fun
"""

markov_chain = build_markov_chain(text)
output = generate_text(markov_chain, 15)

print(output)