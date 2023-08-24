
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
                                   
def hedges_batch_scores(scores,batchsize=1): #batch scores together so hedges decoder can process together
    reads=[]
    scores_set=[]
    batch_counter=0
    for read,score in scores:
        reads.append(read)
        scores_set.append(score["scores"])
        if batch_counter%batchsize==0:
            max_score_length = max(scores_set,lambda x:x.size(0))
            scores_set = (torch.nn.functional.pad(score,(0,0,0,max_score_length-score.size(0),"constant")) for score in scores_set)
            yield (reads,{'scores':torch.stack(scores_set)})
            scores_set=[]
            reads=[]