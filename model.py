import torch
import torch.nn as nn
from transformers import AutoModel
from utils import get_class_dicts
from transformers import Wav2Vec2Processor, HubertModel
from utils import get_class_dicts


class CMSA(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_dims, num_head):
    super(CMSA, self).__init__()
    self.msa = nn.MultiheadAttention(hidden_dims, num_head, batch_first=True)
    self.linear = nn.Sequential(
        nn.Linear(in_channels, hidden_dims),
        nn.Dropout(0.3),
        nn.GELU(),
        nn.Linear(hidden_dims, out_channels),
        nn.Dropout(0.2),
        nn.GELU()
    )

    self.norm1 = nn.LayerNorm(hidden_dims)
    self.norm2 = nn.LayerNorm(out_channels)

  
  def forward(self, q, k, v):
    x = self.norm1(self.msa(q, k, v)[0] + q)
    x = self.norm2(self.linear(x) + x)
    return x


class SLBert(nn.Module):

    def __init__(self, num_heads, dims, num_actions, num_objects, num_locations):
      
      super(SLBert, self).__init__()

      self.bert  = AutoModel.from_pretrained('bert-base-uncased')
      self.bert.config.output_hidden_states = True
      self.hubert = HubertModel.from_pretrained("superb/hubert-base-superb-ks")
      self.hubert.config.output_hidden_states = True

      self.msalist = nn.ModuleList([])

      for _ in range(13):
        self.msalist.append(CMSA(dims, dims, dims, num_heads))      

      self.a = nn.Linear(dims, num_actions)
      self.l = nn.Linear(dims, num_locations)
      self.o = nn.Linear(dims, num_objects)


    def forward(self, sent_id, mask, mels):

      #pass the inputs to the model  
      bert_hidden = self.bert(sent_id, attention_mask=mask)
      hubert_hidden = self.hubert(mels)

      init_hidden = torch.zeros_like(bert_hidden.hidden_states[0])
      for idx, (i, j) in enumerate(zip(bert_hidden.hidden_states, hubert_hidden.hidden_states)):
        i += init_hidden
        x = self.msalist[idx](i, j, j)
        init_hidden = x

      action = self.a(x.mean(1))
      objectt = self.o(x.mean(1))
      location = self.l(x.mean(1))
    

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


if __name__ == "__main__":
  m = SLBert(8, 768, 5, 5, 5)
  xs = torch.tensor([9, 18, 54, 333, 1]).to(torch.int64).unsqueeze(0)
  xa = None
  x = torch.randn((1, 7000))
  print(m(xs, xa, x).shape)

