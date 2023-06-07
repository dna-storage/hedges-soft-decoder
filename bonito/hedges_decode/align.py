from bonito.hedges_decode.decode_ctc import * 
import torch

class Align:
    def __init__(self,alphabet:list,device="cpu") -> None:
        self._alphabet=alphabet
        self._letter_to_index = {_:i for i,_ in enumerate(self._alphabet)} #reverse map for the alphabet
        self._device=device

    def get_index_range(self,BT:torch.Tensor,F:torch.Tensor)->tuple[int,int]:
        argmax_t = torch.argmax(F[1:,-1])
        current_strand_index = BT.size(1)-1
        current_time_index = int(argmax_t)
        while current_strand_index!=-1:
            current_strand_index = int(BT[current_time_index,current_strand_index])-1
            current_time_index-=1
        return int(current_time_index+1),int(argmax_t)
    
    def align(self,scores:torch.Tensor,seq:str)->tuple[int,int,torch.Tensor]:
        T=scores.size(0)
        target_indexes = HedgesBonitoCTC.string_to_indexes(seq,self._letter_to_index,device=self._device)
        target_indexes = HedgesBonitoCTC.insert_blanks(target_indexes)
        BT = scores.new_full((T,target_indexes.size(0)),0,dtype=torch.int64) #backtrace to know where alignment ranges over T
        F = scores.new_full((T,target_indexes.size(0)+1),Log.zero) #forward calculating Trellis
        target_indexes = target_indexes[None,:].expand(T,-1)
        emission_scores = scores.gather(1,target_indexes)
        emission_scores=torch.nn.functional.pad(emission_scores,(1,0),value=Log.one)
        running_alpha = scores.new_full((emission_scores.size(1)+2,),Log.zero)
        running_alpha[2]=Log.one
        mask = target_indexes[0,:-2]==target_indexes[0,2:]
        mask=torch.nn.functional.pad(mask,(3,0),value=1)
        r = torch.arange(1,emission_scores.size(1),device=self._device)
        zeros= scores.new_full((emission_scores.size(1),),Log.zero)
        for t in torch.arange(T):
            #print(t)
            stay = Log.mul(running_alpha[2:],emission_scores[t,:])
            previous = Log.mul(running_alpha[1:-1],emission_scores[t,:])
            previous_previous = Log.mul(torch.where(mask,zeros,running_alpha[:-2]),emission_scores[t,:])
            F[t,:],arg_max = Max.sum(torch.stack([stay,previous,previous_previous]))
            running_alpha[2:]=F[t,:]
            BT[t,:]=r-arg_max[1:]
        lower,upper = self.get_index_range(BT.to("cpu"),F.to("cpu"))
        return lower,upper,torch.max(F[1:,-1])
    
    
class AlignCTC(Align):
    def __init__(self,alphabet:list,device="cpu") -> None:
        super().__init__(alphabet,device)
    def align(self, scores: torch.Tensor, seq: str) -> tuple[int, int, torch.Tensor]:
        return super().align(scores, seq)
