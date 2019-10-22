#  参考：http://cedro3.com/ai/keras-lstm-text-word/

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from janome.tokenizer import Tokenizer
import re
import numpy as np
import random
import sys
import io
from tqdm import tqdm

# mainーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

# model読み込み
model = load_model('./model/sao_model2019-07-16.h5')




# text読み込み
path = './1000sao.txt'

#ファイル読み込み
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower() #大文字を小文字に
    text = format_text(text)

    #出力textをtextとして保存
    x = text
    file = open('InputText.txt', 'w', encoding='utf-8')
    file.write(x)
    file.close()

print('corpus length:', len(text))

text =Tokenizer(mmap=True).tokenize(text, wakati=True)  # 分かち書きする
# print(text)
chars = text
count = 0
char_indices = {}  # 辞書初期化
indices_char = {}  # 逆引き辞書初期化
 
for word in tqdm(chars):
    if not word in char_indices:  # 未登録なら
       char_indices[word] = count  # 登録する      
       count +=1
    #    print(count,word)  # 登録した単語を表示
# 逆引き辞書を辞書から作成する
indices_char = dict([(value, key) for (key, value) in char_indices.items()])
 
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 5
step = 1
sentences = []
next_chars = []
for i in tqdm(range(0, len(text) - maxlen, step)):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

np.save("chars.npy",chars)
np.save("sentences.npy",sentences)
np.save("next_chars.npy",next_chars)
np.save("char_indices.npy",char_indices)
np.save("indices_char.npy",indices_char)

# chars = np.load("./chars.npy",allow_pickle=True)
# sentences = np.load("./sentences.npy",allow_pickle=True)
# next_chars = np.load("./next_chars.npy",allow_pickle=True)
# char_indices = np.load("./char_indices.npy",allow_pickle=True)
# indices_char = np.load("./indices_char.npy",allow_pickle=True)

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
 
 
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
 
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
 
 
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
 
 
def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
 
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 0  # テキストの最初からスタート
    for diversity in [0.2]: 
        print('----- diversity:', diversity)
 
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        print(sentence)
        # sentence はリストなので文字列へ変換して使用
        generated += "".join(sentence)
        print(sentence)
        
        # sentence はリストなので文字列へ変換して使用
        print('----- Generating with seed: "' + "".join(sentence)+ '"')
        sys.stdout.write(generated)
 
 
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
 
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
 
            generated += next_char
            sentence = sentence[1:]
            # sentence はリストなので append で結合する
            sentence.append(next_char)  
 
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
 
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
 
model.fit(x, y,
          batch_size=64,
          epochs=30,
          callbacks=[print_callback])

model.save('sao_model.h5')