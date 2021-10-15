import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e6

    def __init__(self, inputFileName, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.", end="\r")

        self.id2word[0] = "[PADDING]"
        self.word2id["[PADDING]"] = 0
        self.word_frequency[0] = 0
        wid = 1
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("\nTotal embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        pow_frequency[0] = 0
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)


    def getNegatives(self, target, size):  # TODO check equality with target
        response = []
        while len(response) < size:
            if self.negatives[self.negpos] not in target:
                response.append(self.negatives[self.negpos])
            self.negpos += 1
            if len(self.negatives) <= self.negpos:
                self.negpos = 0
        return np.array(response)


# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataset(Dataset):
    def __init__(self, data, window_size, model="skipgram", num_neg=5):
        self.data = data
        self.window_size = window_size
        self.input_file = [line.strip().split() for line in open(data.inputFileName, encoding="utf8") if line]
        self.model = model
        self.num_neg = num_neg

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        words = self.input_file[idx]
        word_ids = [self.data.word2id[w] for w in words if
                    w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]        

        boundary = np.random.randint(1, self.window_size)
        data = []
        for i, u in enumerate(word_ids):
            context = [v for v in word_ids[max(i - boundary, 0):i + boundary + 1] if u != v]
            pre_len = self.window_size-min(i, boundary)
            suf_len = (2*self.window_size)-(len(context)+pre_len)
            input_mask = [0]*pre_len + [1]*len(context) + [0]*suf_len
            context = [0]*pre_len + context + [0]*suf_len
            assert len(context) == 2*self.window_size, f"length error:{context}"
            assert len(input_mask) == 2*self.window_size, f"length error:{input_mask} {context}"
            if self.model == "skipgram":
                data.append((u, context, self.data.getNegatives(word_ids[max(i - boundary, 0):i + boundary + 1], self.num_neg), input_mask))
            elif self.model == "cbow":
                data.append((context, u, self.data.getNegatives(word_ids[max(i - boundary, 0):i + boundary + 1], self.num_neg), input_mask))
        return data

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v, _ in batch if len(batch) > 0]
        all_input_mask = [input_mask for batch in batches for _, _, _, input_mask in batch if len(batch) > 0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v), torch.LongTensor(all_input_mask)
