from sudachipy import tokenizer
from sudachipy import dictionary
from tqdm import tqdm

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

with open("ja_wiki40b_test.txt") as f, open("ja_wiki40b_test_wakati.txt", "w") as fo:
    for line in tqdm(f):
        line = line.strip()
        print(" ".join([m.surface() for m in tokenizer_obj.tokenize(line, mode)]), file=fo)
