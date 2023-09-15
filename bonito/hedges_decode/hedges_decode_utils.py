
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


def bundle_scores(reads,scores_set):
    max_score_length = max((s.size(0) for s in scores_set))
    scores_gen = (torch.nn.functional.pad(score,(0,0,0,max_score_length-score.size(0)),value=Log.zero) for score in scores_set)
    return (reads,{'scores':torch.stack(list(scores_gen))})
              

def hedges_batch_scores(scores,batchsize=1,windowsize=10000): #batch scores together so hedges decoder can process together
    reads=[]
    scores_set=[]
    window_counter=0    
    window=[]
    for read,score  in scores:
        window_counter+=1
        reads.append(read)
        if isinstance(score,dict): 
            scores_set.append(score["scores"])
        else:
            scores_set.append(score)
        if (window_counter%windowsize==0 and window_counter>0):
            assert len(reads)==len(scores_set)
            candidate_reads,candidate_scores = zip(*sorted(zip(reads,scores_set),key=lambda x: x[1].size(0)))
            yield bundle_scores(candidate_reads[:batchsize],candidate_scores[:batchsize])
            end = min(batchsize,len(candidate_reads))
            scores_set=candidate_scores[end:]
            reads=candidate_reads[end:]
            window_counter=len(reads)
    if len(reads)>0:
        assert len(reads)==len(scores_set)
        reads,scores_set =  zip(*sorted(zip(reads,scores_set),key=lambda x: x[1].size(0)))
        while len(reads)>0:
            yield bundle_scores(reads[:batchsize],scores_set[:batchsize]) #wrap up
            end=min(batchsize,len(reads))
            reads=reads[end:]
            scores_set=scores_set[end:]
    

