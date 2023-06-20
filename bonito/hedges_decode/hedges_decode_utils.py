
import torch 
from collections import namedtuple

reverse_map={"A":"T","T":"A","C":"G","G":"C"}
def reverse_complement(seq:str)->str:
    return "".join([reverse_map[_] for _ in seq[::-1]])

def complement(seq:str)->str:
    return "".join([reverse_map[_] for _ in seq])

semiring = namedtuple('semiring', ('zero', 'one', 'mul', 'sum'))                                                                                                               
                                                                                                                                                                      
Log = semiring(zero=-1e38, one=0., mul=torch.add, sum=torch.logsumexp)                                                                                             

Max = semiring(zero=-1e38, one=0., mul=torch.add, sum=(lambda x, dim=0: torch.max(x, dim=dim)))                                                                      

                                                                                                                                                    
def dot(x, y, S=Log, dim=-1):                                                                                                                                                          
    return S.sum(S.mul(x, y), dim=dim)                                                                                                                                                 
                                   
