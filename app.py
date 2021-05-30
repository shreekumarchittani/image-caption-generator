from flask import Flask, request, render_template
import string
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model

app = Flask(__name__)

token_path = "./Flickr8k.token.txt"
train_images_path = './Flickr_8k.trainImages.txt'
glove_path = './archive'

doc = open(token_path,'r').read()
print(doc[:410])

descriptions = dict()
for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
          image_id = tokens[0].split('.')[0]
          image_desc = ' '.join(tokens[1:])
          if image_id not in descriptions:
              descriptions[image_id] = list()
          descriptions[image_id].append(image_desc)


table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [w.translate(table) for w in desc]
        desc_list[i] =  ' '.join(desc)


vocabulary = set()
for key in descriptions.keys():
        [vocabulary.update(d.split()) for d in descriptions[key]]
print('Original Vocabulary Size: %d' % len(vocabulary))


lines = list()
for key, desc_list in descriptions.items():
    for desc in desc_list:
        lines.append(key + ' ' + desc)
new_descriptions = '\n'.join(lines)

doc = open(train_images_path,'r').read()
dataset = list()
for line in doc.split('\n'):
    if len(line) > 1:
      identifier = line.split('.')[0]
      dataset.append(identifier)

train = set(dataset)

train_descriptions = dict()
for line in new_descriptions.split('\n'):
    tokens = line.split()
    image_id, image_desc = tokens[0], tokens[1:]
    if image_id in train:
        if image_id not in train_descriptions:
            train_descriptions[image_id] = list()
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        train_descriptions[image_id].append(desc)


all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)


word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

print('Vocabulary = %d' % (len(vocab)))


ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1

all_desc = list()
for key in train_descriptions.keys():
    [all_desc.append(d) for d in train_descriptions[key]]
lines = all_desc
max_length = max(len(d.split()) for d in lines)

print('Description Length: %d' % max_length)


import numpy as np
embeddings_index = {} 
f = open(os.path.join(glove_path, 'glove.6B.200d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs


embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)



def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x



def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec


from keras.models import load_model
model = load_model('project.h5')

uploads_dir = os.path.join(os.path.dirname(app.instance_path), 'static')


@app.route('/', methods=['GET'])
def index():
    print(uploads_dir)
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        print(f.filename)
        return render_template('home.html') 

@app.route('/process', methods=['POST'])
def process():  
    f = request.files['file']
    f.save(os.path.join(uploads_dir, secure_filename(f.filename)))  
    filename=f.filename
    def greedySearch(photo):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                break

        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final


    def beam_search_predictions(image, beam_index = 3):
        start = [wordtoix["startseq"]]
        start_word = [[start, 0.0]]
        while len(start_word[0][0]) < max_length:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
                preds = model.predict([image,par_caps], verbose=0)
                word_preds = np.argsort(preds[0])[-beam_index:]
                # Getting the top <beam_index>(n) predictions and creating a 
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])
                        
            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]
        
        start_word = start_word[-1][0]
        intermediate_caption = [ixtoword[i] for i in start_word]
        final_caption = []
        
        for i in intermediate_caption:
            if i != 'endseq':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption
    path = uploads_dir+'/'+filename
    myimage = encode(path).reshape((1,2048))
    message=""
    message=beam_search_predictions(myimage, beam_index = 10)
    # os.remove(filename)
    return render_template('success.html',caption=message,image_path=filename)


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
    