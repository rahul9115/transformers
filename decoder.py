import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def self_attention(q,k,v,mask=None):
    d_k=q.size()[-1]
    scaled=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)
    if mask is not None:
        scaled+=mask
    attention=F.softmax(scaled)
    out=torch.matmul(attention,v)
    return attention,out 

class Multiheadattention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dims=d_model//num_heads
        self.qkv_layer=nn.Linear(d_model,3*d_model)
        self.linear=nn.Linear(d_model,d_model)
    def forward(self,x,mask=None):
        batch_size,sequence_length,input_dim=x.size()
        qkv=self.qkv_layer(x)
        qkv=qkv.reshape(batch_size,sequence_length,self.num_heads,3*self.head_dims)
        qkv=qkv.permute(0,2,1,3)
        q,k,v=qkv.chunk(3,dim=-1)
        attention,values=self_attention(q,k,v,mask)
        values=values.reshape(batch_size,sequence_length,self.num_heads*self.head_dims)
        return self.linear(values)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        self.kv_layer=nn.Linear(d_model,2*d_model)
        self.q_layer=nn.Linear(d_model,d_model)
        self.linear=nn.Linear(d_model,d_model)
    def forward(self,x,y,mask=None):
        batch_size,max_sequence_len,d_model=x.size()
        
        kv=self.kv_layer(x)
        print("Size of kv",kv.size())
        q=self.q_layer(y)
        print("Size of q",q.size())
        kv=kv.reshape(batch_size,max_sequence_len,self.num_heads,2*self.head_dim)
        kv=kv.permute(0,2,1,3)
        q=q.reshape(batch_size,max_sequence_len,self.num_heads,self.head_dim)
        q=q.permute(0,2,1,3)
        k,v=kv.chunk(2,dim=-1)
        print("Q size",q.size(),"KV size",k.size())
        attention,values=self_attention(q,k,v,mask)
        print("Values size",values.size())
        values=values.reshape(batch_size,max_sequence_len,d_model)
        return self.linear(values)



class PositionalEncoding(nn.Module):
    def __init__(self,max_sequence_length,d_model):
        super().__init__()
        self.d_model=d_model
        self.max_sequence_length=max_sequence_length
    def forward(self):
        even_i=torch.arange(0,self.max_sequence_length,2).float()
        denominator=torch.pow(10000,even_i/self.d_model).float()
        pos=torch.arange(self.max_sequence_length,dtype=torch.float).reshape(self.max_sequence_length,1)
        even_PE=torch.sin(pos/denominator)
        odd_PE=torch.cos(pos/denominator)
        stack=torch.stack([even_PE,odd_PE],dim=2)
        PE=torch.flatten(stack,start_dim=1,end_dim=1)
        return PE
        
class LayerNormalization(nn.Module):
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
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(DecoderLayer,self).__init__()
        self.self_attention=Multiheadattention(d_model,num_heads)
        self.norm1=LayerNormalization(parameter_shape=[d_model])
        self.dropout1=nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention=MultiHeadCrossAttention(d_model,num_heads)
        self.norm2=LayerNormalization(parameter_shape=[d_model])
        self.dropout2=nn.Dropout(p=drop_prob)
        self.ffn=PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm3=LayerNormalization(parameter_shape=[d_model])
        self.dropout3=nn.Dropout(p=drop_prob)
    
    def forward(self,x,y,decoder_mask):
        _y=y
        y=self.self_attention(y,decoder_mask)
        y=self.dropout1(y)
        y=self.norm1(y+_y)
        _y=y
        y=self.encoder_decoder_attention(x,y,mask=None)
        y=self.dropout2(y)
        y=self.norm2(y+_y)
        _y=y
        y=self.ffn(y)
        y=self.dropout3(y)
        y=self.norm3(y+_y)
        return y





"""
In sequential you cannot pass more than one parameter so if you have more than one parameter
you need to create this class
"""
class SequentialDecoder(nn.Sequential):
    def forward(self,*inputs):
        x,y,mask=inputs
        for module in self._modules.values():
            y=module(x,y,mask)
        return y

class Decoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers):
        super().__init__()
        self.layers=SequentialDecoder(*[DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob)
                                        for _ in range(num_layers)])
    def forward(self,x,y,mask):
        y=self.layers(x,y,mask)
        return y

if __name__=="__main__":
    d_model=512
    batch_size=30 #batch update happen less frequent
    ffn_hidden=2048
    num_heads=8
    num_layers=5
    drop_prob=0.1
    max_sequence_len=200
    #Mulitheadattention
    #CrossAttention
    x=torch.randn((batch_size,max_sequence_len,d_model)) #English sentence positional encoded
    #x: 30x200x512
    y=torch.randn((batch_size,max_sequence_len,d_model))# Telugu sentence positional encoded
    #y: 30x200x512

    mask=torch.full([max_sequence_len,max_sequence_len],float("-inf"))
    mask=torch.triu(mask,diagonal=1)
    '''
    The mask is a look ahead mask. We don't have information of every single word at the time and so we
    would need mask. It is to make the words more context aware of the words near to it
    ''' 
    decoder=Decoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers)
    out=decoder(x,y,mask)