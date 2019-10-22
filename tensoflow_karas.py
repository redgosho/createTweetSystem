# 一文字ずつ学習させていくスタイル。
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
from tqdm import tqdm
path = "./SAO1_01_utf8.txt"
bindata = open(path, "r",encoding="utf-8").read()
# text = bindata.decode("utf-8")
text = bindata
print("Size of text: ",len(text))
chars = sorted(list(set(text)))
print("Total chars :",len(chars))
#辞書を作成する
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))
#40文字の次の1文字を学習させる. 3文字ずつずらして40文字と1文字というセットを作る
maxlen = 40
step = 3
sentences = []
next_chars = []
# for i in range(0, len(text)-maxlen, step):
#     sentences.append(text[i:i+maxlen])
#     next_chars.append(text[i+maxlen])

# X = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
# y = np.zeros((len(sentences),len(chars)),dtype=np.bool)

# for i, sentence in enumerate(tqdm(sentences)):
#     for t ,char in enumerate(sentence):
#         X[i,t,char_indices[char]] = 1
#     y[i,char_indices[next_chars[i]]] = 1
#     #テキストのベクトル化
#     X = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
#     y = np.zeros((len(sentences),len(chars)),dtype=np.bool)



# for i, sentence in enumerate(tqdm(sentences)):
#     for t ,char in enumerate(sentence):
#         X[i,t,char_indices[char]] = 1
#     y[i,char_indices[next_chars[i]]] = 1

# np.save("XXX.npy",X)
# np.save("yyy.npy",y)

X = np.load("./XXX.npy")
y = np.load("./yyy.npy")

#LSTMを使ったモデルを作る
model = Sequential() #連続的なデータを扱う
print("うわああああああああああああああああああああ"+str(len(chars))+"うわああああああああああああああああああああ")
model.add(LSTM(128, input_shape=(maxlen,len(chars))))
model.add(Dense(len(chars)))
model.add(Activation("softmax"))
optimizer = RMSprop(lr = 0.01)
model.compile(loss="categorical_crossentropy",optimizer=optimizer)
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)


#生成する
# for iteration in range(1,7):
#     print()
#     print("-"*50)
#     print("繰り返し回数: ",iteration)
#     model.fit(X, y, batch_size=128, epochs=5)
#     start_index = random.randint(0, len(text)-maxlen-1)
#     model.save("file_save-{}.h5".format(iteration))


model = load_model("./file_save-6.h5");
start_index = random.randint(0, len(text)-maxlen-1)

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print("-----diversity", diversity)
    generated =""
    sentence = text[start_index: start_index + maxlen ]
    generated += sentence
    print("-----Seedを生成しました: " + sentence + '"')
    sys.stdout.write(generated)

    #次の文字を予測して足す
    for i in range(400):
        x = np.zeros((1,maxlen,len(chars)))
        print(maxlen,len(chars))
        for t,char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1
    
        preds = model.predict(x, verbose =9)[0] #次の文字を予測
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
model.save('sao_model.h5')
file = open('saogentext.txt','w+',encoding='utf-8').write(generated)