import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def self_attention(q,k,v,mask=None):
    d_k=q.size[-1]
    scaled=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
    if mask is not None:
        scaled+=mask
    attention=F.softmax(scaled)
    out=nn.Linear(attention)
    return attention,out

class Multiheadattention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dims=d_model/num_heads
        self.qkv_layer=nn.Linear(d_model,3*d_model)
        self.linear=nn.Linear(d_model,d_model)
    def forward(self,x,mask=None):
        batch_size,sequence_len,input_dim=x.size()
        qkv=self.qkv_layer(x)
        qkv=qkv.reshape(batch_size,sequence_len,self.num_heads,3*self.d_model)
        qkv=qkv.permute(0,2,1,3)
        q,k,v=qkv.chunks(3)
        attention,values=self_attention(q,k,v)
        values=values.reshape(batch_size,sequence_len,self.num_heads*self.d_model)
        out=self.linear(values)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self,max_sequence_len,d_model):
        super().__init__()
        self.max_sequence_len=max_sequence_len
        self.d_model=d_model
    def forward(self):
        even_i=torch.arange(0,self.max_sequence_len,2).float()
        denominator=torch.pow(10000,even_i/self.d_model).float()
        pos=torch.arange(self.max_sequence_len).reshape(self.max_sequence_len,1)
        even_i=torch.sin(pos/denominator)
        odd_i=torch.cos(pos/denominator)
        stacked=torch.stack([even_i,odd_i],dim=2)
        out=stacked.flatten(stacked,start_dim=1,end_dim=1)
        return out

class LayerNormalization(nn):
    def __init__(self,parameter_shape,eps=1e-5):
        super().__init__()
        self.eps=eps
        self.parameter_shape=parameter_shape
        self.gamma=nn.Parameter(torch.ones(parameter_shape))
        self.beta=nn.Parameter(torch.zeros(parameter_shape))
    def forward(self,inputs):
        dims=[-(i+1) for i in range(len(self.parameter_shape))]
        mean=inputs.mean(dim=dims,keepdim=True)
        var=((inputs-mean)**2).mean(dim=dims,keepdim=True)
        std=(var+self.eps).sqrt()
        y=(inputs-mean)/std
        return self.gamma*y+self.beta
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,drop_prob=0.01):
        super(PositionwiseFeedForward,self).__init__()
        self.linear1=nn.Linear(d_model,hidden)
        self.linear2=nn.Linear(hidden,d_model)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=drop_prob)
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention=Multiheadattention(self,d_model,num_heads)
        self.norm1=LayerNormalization(parameter_shape=[d_model])
        self.drop1=nn.Dropout(p=drop_prob)
        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm2=LayerNormalization(parameter_shape=[d_model])
        self.dropout2=nn.Dropout(p=drop_prob)
    def forward(self,x):
        residual_x=x
        x=self.attention(x,mask=None)
        x=self.dropout1(x)
        x=self.norm1(x+residual_x)
        residual_x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+residual_x)
        return x

class Encoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers):
        super().__init__()
        self.layers=nn.Sequential(*[EncoderLayer(d_model,ffn_hidden,num_heads,drop_prob)
                                    for _ in range(num_layers)])
    def forward(self,x):
        out=self.layers(x)
        return out

if __name__=="__main__":
    batch_size=30
    num_heads=8
    num_layers=5
    d_model=512
    ffn_hidden=2048
    drop_prob=0.1
    max_sequence_len=200
    encoder=Encoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers)
    x=torch.randn((batch_size,max_sequence_len,d_model))
    out=encoder(x)
    print(x)
