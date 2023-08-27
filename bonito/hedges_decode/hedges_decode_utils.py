
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
    scores = list(scores)
    #scores = sorted(scores,key = lambda x:x[1].size(0))
    for read,score in scores:
        batch_counter+=1
        reads.append(read)
        if isinstance(score,dict): scores_set.append(score["scores"])
        else:
            scores_set.append(score)
        if batch_counter%batchsize==0 and batch_counter>0:
            max_score_length = max((s.size(0) for s in scores_set))
            print(max_score_length)
            scores_set = (torch.nn.functional.pad(score,(0,0,0,max_score_length-score.size(0)),value=Log.zero) for score in scores_set)
            yield (reads,{'scores':torch.stack(list(scores_set))})
            scores_set=[]
            reads=[]
            batch_counter=0
       
    
