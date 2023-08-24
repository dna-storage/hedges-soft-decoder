from bonito.hedges_decode.decode_ctc import * 
import torch
import cupy as cp
import numpy as np

class Align:
    def __init__(self,alphabet:list,device="cpu") -> None:
        self._alphabet=alphabet
        self._letter_to_index = {_:i for i,_ in enumerate(self._alphabet)} #reverse map for the alphabet
        self._device=device

    def get_index_range(self,BT:torch.Tensor,F:torch.Tensor)->torch.Tensor:
        argmax_t = torch.argmax(F[:,1:,-1],dim=1) #Nx1 tensotr
        return_matrix = np.ndarray((argmax_t.size(0),2),dtype=np.int) #Nx2 tensor
        for i in range(argmax_t.size(0)):
            current_strand_index = BT.size(2)-1
            current_time_index = int(argmax_t[i])
            while current_strand_index!=-1:
                current_strand_index = int(BT[current_time_index,current_strand_index])-1
                current_time_index-=1
            return_matrix[i,0]=int(current_time_index+1)
            return_matrix[i,1]=int(argmax_t[i])
        return torch.from_numpy(return_matrix)
    
    def align(self,scores:torch.Tensor,seq:str)->tuple[torch.Tensor,torch.Tensor]:
        raise NotImplementedError()
    
class AlignCTC(Align):
    def __init__(self,alphabet:list,device="cpu") -> None:
        super().__init__(alphabet,device)
    def align(self, scores: torch.Tensor, seq: str) -> tuple[torch.Tensor, torch.Tensor]:
        N,T,B=scores.size
        target_indexes = HedgesBonitoCTC.string_to_indexes(seq,self._letter_to_index,device=self._device,N)
        target_indexes = HedgesBonitoCTC.insert_blanks(target_indexes)
        BT = scores.new_full((N,T,target_indexes.size(0)),0,dtype=torch.int64) #backtrace to know where alignment ranges over T
        F = scores.new_full((N,T,target_indexes.size(0)+1),Log.zero) #forward calculating Trellis
        target_indexes = target_indexes[:,None,:].expand(-1,T,-1)
        emission_scores = scores.gather(2,target_indexes)
        emission_scores=torch.nn.functional.pad(emission_scores,(1,0),value=Log.one)
        running_alpha = scores.new_full((N,emission_scores.size(2)+2),Log.zero)
        running_alpha[:,2]=Log.one
        mask = target_indexes[:,0,:-2]==target_indexes[:,0,2:]
        mask=torch.nn.functional.pad(mask,(3,0),value=1)
        r = torch.arange(1,emission_scores.size(2),device=self._device)[None,:].repeat(N,1)
        zeros= scores.new_full((N,emission_scores.size(2),),Log.zero)
        for t in torch.arange(T):
            stay = Log.mul(running_alpha[:,2:],emission_scores[:,t,:])
            previous = Log.mul(running_alpha[:,1:-1],emission_scores[:,t,:])
            previous_previous = Log.mul(torch.where(mask,zeros,running_alpha[:,:-2]),emission_scores[:,t,:])
            F[:,t,:],arg_max = Max.sum(torch.stack([stay,previous,previous_previous]))
            running_alpha[:,2:]=F[:,t,:]
            BT[:,t,:]=r-arg_max[1:]
        lower,upper = self.get_index_range(BT.to("cpu"),F.to("cpu"))
        return lower,upper,torch.max(F[:,1:,-1],dim=1)

class AlignCTCGPU(Align):
    fwd_alg_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","fwd_logspace_align",FLOAT='float',SUM2='logsumexp2',SUM='logsumexp3',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))

    def __init__(self, alphabet: list, device="cuda:0") -> None:
        super().__init__(alphabet, device)
    def align(self, scores: torch.Tensor, seq: str) -> tuple[torch.Tensor,torch.Tensor]:
        N,T,B=scores.size(0)
        scores_gpu=scores.to(self._device)
        target_indexes = HedgesBonitoCTC.string_to_indexes(seq,self._letter_to_index,device=self._device,N)
        target_indexes = HedgesBonitoCTC.insert_blanks(target_indexes)
        BT = torch.full((N,T,target_indexes.size(0)),0,dtype=torch.int64,device=self._device) #backtrace to know where alignment ranges over T
        F = torch.full((N,T,target_indexes.size(0)+1),Log.zero,device=self._device) #forward calculating Trellis
        mask = target_indexes[:,:-2]==target_indexes[0,2:]
        mask=torch.nn.functional.pad(mask,(3,0),value=1)
        mask = torch.where(mask,torch.full((N,emission_scores.size(1)),Log.zero,device=self._device),
                           torch.full((N,emission_scores.size(1)),Log.one,device=self._device))
        with cp.cuda.Device(0):
            L = target_indexes.size(2)
            target_threads = 512
            batch_per_block=(target_threads//L)+N%-(target_threads//L)
            batch_blocks = N//batch_per_block
            AlignCTCGPU.fwd_alg_kernel(grid=(batch_blocks,1,1),block=(L,batch_per_block,1),shared_mem = (2*4*(L+2))*batch_per_block,args=(scores_gpu.data_ptr(),target_indexes.data_ptr(),F.data_ptr(),BT.data_ptr(),mask.data_ptr(),
                                                                                               T,L,2,N))
        lower_upper_tensor = self.get_index_range(BT.to("cpu"),F.to("cpu"))
        return lower_upper_tensor,torch.max(F[:,1:,-1],dim=1).to("cpu")
        
