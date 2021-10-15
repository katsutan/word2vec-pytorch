from gensim.models import word2vec

model = word2vec.KeyedVectors.load_word2vec_format("test.vec")

print("パソコン", model.most_similar("パソコン", topn=5))

print("俳優 - 男 + 女", model.most_similar(positive=["俳優", "女"], negative=["男"], topn=3))
print("王 - 男 + 女", model.most_similar(positive=["王", "女"], negative=["男"], topn=3))


