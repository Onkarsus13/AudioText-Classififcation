import torch
import torch.nn as nn
from transformers import AutoModel
from utils import get_class_dicts


class BERT_Arch(nn.Module):

    def __init__(self, bert, num_actions, num_objects, num_locations):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)  # 768 neuron final
      

      self.lstm1 = nn.LSTM(256, 256)
      self.lstm2 = nn.LSTM(256, 512)

      self.a = nn.Linear(1024, num_actions)
      self.l = nn.Linear(1024, num_locations)
      self.o = nn.Linear(1024, num_objects)


    def forward(self, sent_id, mask, mels):

      #pass the inputs to the model  
      p = self.bert(sent_id, attention_mask=mask)
      #print(type(p[1]))
      x = self.fc1(p[1])

      x = self.relu(x)

      x = self.dropout(x)

      l = self.lstm2(self.lstm1(mels)[0])[0]

      x = torch.cat((l.mean(1), x), axis=1)

      action = self.a(x)
      objectt = self.o(x)
      location = self.l(x)
    

      return action, objectt, location

class MultiClassLoss(nn.Module):

    def __init__(self):
        super(MultiClassLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred1, pred2, pred3, y1, y2, y3):
        l1 = self.loss(pred1, y1)
        l2 = self.loss(pred2, y2)
        l3 = self.loss(pred3, y3)
        return l1 + l2 + l3


def get_model_loss_optimizer(lr, device):

    class_dict_a, class_dict_o, class_dict_l = get_class_dicts()

    bert = AutoModel.from_pretrained('bert-base-uncased')
    for params in bert.parameters():
        params.requires_grad = False

    model = BERT_Arch(bert, len(class_dict_a), len(class_dict_o), len(class_dict_l)).to(device)
    loss = MultiClassLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss, optimizer







