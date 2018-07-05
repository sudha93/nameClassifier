
# This code uses a character-level BiLSTM to classify the names 
# Given a name, the network returns the probabilities of it belonging to each one of the 18 langauges 

import torch 
import torch.autograd as autograd 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.nn.utils.rnn import pad_packed_sequence

import glob
import unicodedata
from keras.preprocessing.sequence import pad_sequences
from random import shuffle

torch.manual_seed(1)
# Each character embedding size
DIM = 30
H_DIM = 40
# Batch size 
BATCH = 3   

class nameClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, vocab_size):
        super(nameClassifier, self).__init__()
        # Embedding layer  
        self.embed_layer = nn.Embedding(vocab_size,embed_dim)
        self.lstm = nn.LSTM(embed_dim,hidden_dim,bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim,output_dim)

    def forward(self,batch,idRep):
        embed = self.embed_layer(idRep)
        print embed.size()
        out,(h,c) = self.lstm(embed.view(idRep.size(1),batch,-1))
        print h.size()
        output = self.linear(h.view(batch,-1) )
        print output.size()
        scores = F.log_softmax(output,dim=1) 
        print scores.size(),'\n'
        return scores

#Testing
def testFunction(test,model):
    count = 0
    for item in test:
        sample = autograd.Variable(torch.LongTensor([item[0]])).cuda()
        op = model(1,sample)
        #print op,'\n'
        value , index = op.max(1)
        #print item[1] 
        #print int(index[0])
        if (int(index[0]) == item[1]) :
            count += 1
    print 'accuracy = ', float(100*count/len(test))
    return 
    
def unicodeToAscii(s):
    new_s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
    return new_s

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(unicode(line,'utf-8')) for line in lines] 
    #return [line for line in lines]

def return_encoding(name,idDict):
    #embeds = nn.Embedding(len(idDict),DIM)
    return [idDict[name[i]] for i in range(len(name))]
    #return  autograd.Variable(lookup_tensor)

# MAIN CODE 

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []
for filename in glob.glob('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# Creating indices for all the characters in the corpus
total_set = set() 
for key, value  in category_lines.items():
    for name in value:
        s = set(name)
        total_set = total_set | s

# Dictionary with ids for each character
idDict = {char:i for i,char in enumerate(total_set)}
vocab_size=len(idDict)

# Get a list of tuples as input 
total_in = []
for key,value in category_lines.items():
    for name in value :
        encoding = return_encoding(name,idDict)
        cat_index = all_categories.index(key)
        t = (encoding,cat_index)
        total_in.append(t)

# Shuffling the list
shuffle(total_in) 
# Dividing data into train set and test set 
train = total_in[:int(4*len(total_in)/5)]
print len(train) 
test = total_in[int(4*len(total_in)/5):]
print len(test)

# Creating instance of the class
model = nameClassifier(DIM, H_DIM,n_categories,vocab_size)
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
model.cuda()

batches = len(train)/BATCH
# Training
for epoch in range(5):
    for i in range(batches):
        #print float(i)/batches
        # list of lists of ids 
        namelist = [train[j][0] for j in range(i*BATCH,i*BATCH+BATCH)]
        classlist = [train[j][1] for j in range(i*BATCH,i*BATCH+BATCH)]
        #print batchlist,'\n'
        model.zero_grad()
        padded_names = pad_sequences(namelist,padding='pre',value=0.0)
        embed_input = autograd.Variable(torch.LongTensor(padded_names)).cuda()
        #print embed_input.size(),'start'
        op = model(BATCH,embed_input)
        #loss = loss_fn(op,item[1])
        target =  autograd.Variable(torch.LongTensor(classlist)).cuda()
        #print target.size()
        loss = loss_fn(op,target).cuda()
        loss.backward()
        optimizer.step()
    print 'epoch = ',epoch
    testFunction(test,model)










