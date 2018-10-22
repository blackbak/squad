### This will be the script for qgan with local w2v

import numpy as np
import json
import gensim
from text_prep import *
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
from torch import optim


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu:0")
torch.backends.cudnn.benchmark = True

class Discriminator(nn.Module):
    def __init__(self, embed_dimension, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.embed_dimension = embed_dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=embed_dimension, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_size, 100) # since we want bidirectional
        self.fc2 = nn.Linear(100, 1)
        self.activation_fc1 = nn.SELU()
        self.activation_fc2 = nn.Sigmoid()
        
    def forward(self, padded_input, input_lengths, batch_size):
        total_length = padded_input.shape[1] #padded_input must be ordered by size
        packed_input = pack_padded_sequence(padded_input, input_lengths, batch_first=True)
        packed_output, last_hidden = self.rnn(packed_input)
        gru_output, sequence_length = pad_packed_sequence(packed_output, total_length = total_length)
        last_output = gru_output[total_length-1, :, :]
        fc1 = self.activation_fc1(self.fc1(last_output))
        discriminator_output = self.activation_fc2(self.fc2(fc1))
        return discriminator_output
    
class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dimension, num_layers):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embed_dimension = embed_dimension
        self.hidden_size = embed_dimension
        self.num_layers = num_layers
        self.noise2hidden = nn.Linear(noise_dim, num_layers*embed_dimension)
        self.tanh = nn.Tanh()
        #batch first does not work on autoregressive rnn
        self.rnn = nn.GRU(input_size=embed_dimension, hidden_size=embed_dimension, num_layers=num_layers, batch_first=True)

    def forward(self, o, h):
        o, h = self.rnn(o, h)
        return o, h
    
    def init_hidden(self, noise):
        return self.tanh(self.noise2hidden(noise)).view([self.num_layers, 1, self.embed_dimension])

def most_similar(embedding, emb_input):
    #cos = nn.CosineSimilarity(dim=1)
    #similarity = cos(embedding.weight, emb_input.view(1, -1))
    similarity = torch.mv(embedding.weight, emb_input.squeeze())
    value, idx = torch.max(similarity, 0)
    return idx

def train(generator, discriminator, input_sequence, embedding, generator_optimizer, discriminator_optimizer, criterion, eos):
    #input sequence of shape [1,seq_len,300]
    ###Discriminator training
    #train with real data
    discriminator_optimizer.zero_grad()
    real_output = discriminator.forward(padded_input=input_sequence.to(device),
                                        input_lengths=torch.tensor([input_sequence.shape[1]],
                                                                   device=device), batch_size=1)
    real_label = torch.ones(1, device=device)
    real_error = criterion(real_output, real_label)
    real_error.backward()
    #train with fake
    #generate sequence
    generated_sequence = []
    generated_idx = []
    noise = torch.randn(generator.noise_dim).to(device)
    o_gen = torch.zeros(embedding.weight.shape[1], device=device).view(1,1,-1)
    h_gen = generator.init_hidden(noise)
    for i in range(20):
        o_gen, h_gen = generator.forward(o_gen, h_gen)
        generated_sequence.append(o_gen)
        idx = most_similar(embedding, o_gen)
        generated_idx.append(idx)
        if idx==eos:
            break
    generated_sequence = torch.cat(generated_sequence).view(1, -1, embedding.weight.shape[1])
    fake_output = discriminator.forward(padded_input=generated_sequence.detach(),
                                        input_lengths=torch.tensor([generated_sequence.shape[1]],
                                                                   device=device), batch_size=1)
    fake_label = torch.zeros(1, device=device)
    fake_error = criterion(fake_output, fake_label)
    fake_error.backward()
    discriminator_error = real_error + fake_error
    discriminator_optimizer.step()
    ###Generator training
    generator_optimizer.zero_grad()
    fake_output_gen = discriminator.forward(padded_input=generated_sequence,
                                            input_lengths=torch.tensor([generated_sequence.shape[1]],
                                                                       device=device), batch_size=1)
    fake_labels_gen = torch.ones(1, device=device)
    generator_error = criterion(fake_output_gen, fake_labels_gen)
    generator_error.backward()
    generator_optimizer.step()
    loss = discriminator_error + generator_error
    return loss.item(), generator_error.item(), discriminator_error.item()

def train_iter(generator, discriminator, dataset, embedding, eos, epochs, lr=0.001):
    total_loss = []
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for e in range(epochs):
        loss = 0
        gen_loss = 0
        dis_loss = 0
        for i, data in enumerate(dataset):
            embeds = embedding(torch.tensor(data+[eos], device=device)).view(1, -1, embedding.weight.shape[1])
            current_loss, curr_gen_loss, curr_dis_loss = train(generator=generator, discriminator=discriminator, input_sequence=embeds, 
                                 embedding=embedding, generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer, criterion=criterion, eos=eos)
            loss += current_loss
            gen_loss += curr_gen_loss
            dis_loss += curr_dis_loss
            if i%50==0:
                total_loss.append(loss)
                print("Loss at iteration {}: {}".format(i, loss))
                print("Gen loss at iteration {}: {}".format(i, gen_loss))
                print("Dis loss at iteration {}: {}".format(i, dis_loss))
                loss = 0
                gen_loss = 0
                dis_loss = 0

    #plot_loss(total_loss)
            
def generate_question(generator, embedding, model):
    generated_sequence = []
    generated_idx = []
    generated_words = []
    noise = torch.randn(generator.noise_dim).to(device)
    o_gen = torch.zeros(embedding.weight.shape[1], device=device).view(1,1,-1)
    h_gen = generator.init_hidden(noise)
    for i in range(20):
        o_gen, h_gen = generator.forward(o_gen, h_gen)
        generated_sequence.append(o_gen)
        idx = most_similar(embedding, o_gen)
        generated_idx.append(idx)
        generated_words.append(idx2word(model, idx))
        if idx==eos:
            break
    return generated_words, generated_idx

def main():
    with open("C:/Users/blackbak/Documents/github/data/squad_data/train-v2.0.json") as f:
        data = json.load(f)
    questions = []
    for i in range(len(data["data"])):
        for j in range(len(data["data"][i]["paragraphs"])):
            for k in range(len(data["data"][i]["paragraphs"][j]["qas"])):
                questions.append(data["data"][i]["paragraphs"][j]["qas"][k]["question"]+" </s>")
    w2v_model = build_local_w2v(questions)
    #w2v_model = build_w2v_model(questions)
    questions_idx = [sentence2idx(w2v_model, sentence) for sentence in questions]
    embedding = gensim2embedding(w2v_model, device)
    generator = Generator(noise_dim=10, embed_dimension=100, num_layers=3)
    generator.to(device)
    discriminator = Discriminator(embed_dimension=100, hidden_size=64, num_layers=2)
    discriminator.to(device)
    eos = word2idx(w2v_model, "</s>")
    train_iter(generator=generator, discriminator=discriminator, 
           dataset=questions_idx, embedding=embedding, eos=eos, epochs=1, lr=0.001)
    for i in range(5):
        q, idx = generate_question(generator, embedding, model)
        print(q)
        
if __name__ == "__main__":
    main()