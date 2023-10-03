from bonito.hedges_decode.decode_ctc import * 
import torch
import cupy as cp
import numpy as np

class Align:
    def __init__(self,alphabet:list,device="cpu") -> None:
        self._alphabet=alphabet
        self._letter_to_index = {_:i for i,_ in enumerate(self._alphabet)} #reverse map for the alphabet
        self._device=device

    def get_index_range(self,BT:np.ndarray,F:torch.Tensor)->torch.Tensor:
        argmax_t = torch.argmax(F[:,1:,-1],dim=1) #Nx1 tensotr
        return_matrix = np.ndarray((argmax_t.size(0),2),dtype=np.int32) #Nx2 tensor
        for i in range(argmax_t.size(0)):
            current_strand_index = BT.shape[2]-1
            current_time_index = int(argmax_t[i])
            while current_strand_index>=0:
                current_strand_index = int(BT[i,current_time_index,current_strand_index])-1
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
        N,T,B=scores.size()
        target_indexes = HedgesBonitoCTC.string_to_indexes(seq,self._letter_to_index,device=self._device,batch=N)
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
        lower,upper = self.get_index_range(BT.to("cpu").numpy(),F.to("cpu"))
        return lower,upper,torch.max(F[:,1:,-1],dim=1)

class AlignCTCGPU(Align):
    fwd_alg_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","fwd_logspace_align",FLOAT='float',SUM2='logsumexp2',SUM='logsumexp3',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))

    def __init__(self, alphabet: list, device="cuda:0") -> None:
        super().__init__(alphabet, device)
    
    def get_index_range(self,BT:np.ndarray,F:torch.Tensor)->torch.Tensor:
        argmax_t = torch.argmax(F[:,:,-1],dim=1) #Nx1 tensotr
        return_matrix = np.ndarray((argmax_t.size(0),2),dtype=np.int32) #Nx2 tensor
        for i in range(argmax_t.size(0)):
            current_strand_index = BT.shape[2]-1 
            current_time_index = int(argmax_t[i])
            while current_strand_index>=0:
                current_strand_index = int(BT[i,current_time_index,current_strand_index])
                current_time_index-=1
            return_matrix[i,0]=int(current_time_index+1)
            return_matrix[i,1]=int(argmax_t[i])
        return torch.from_numpy(return_matrix)
    
    def align(self, scores: torch.Tensor, seq: str) -> tuple[torch.Tensor,torch.Tensor]:
        N,T,B=scores.size()
        scores_gpu=scores.to(self._device)
        target_indexes = HedgesBonitoCTC.string_to_indexes(seq,self._letter_to_index,device=self._device,batch=N)
        target_indexes = HedgesBonitoCTC.insert_blanks(target_indexes)
        BT = torch.full((N,T,target_indexes.size(1)),0,dtype=torch.int64,device=self._device) #backtrace to know where alignment ranges over T
        F = torch.full((N,T,target_indexes.size(1)),Log.zero,device=self._device) #forward calculating Trellis
        mask = target_indexes[:,:-2]==target_indexes[:,2:]
        mask=torch.nn.functional.pad(mask,(2,0),value=1)
        mask = torch.where(mask,torch.full((N,target_indexes.size(1)),Log.zero,device=self._device),
                           torch.full((N,target_indexes.size(1)),Log.one,device=self._device))
        with cp.cuda.Device(0):
            L = target_indexes.size(1)
            target_threads = L
            batch_per_block=1
            batch_blocks = N//batch_per_block
            AlignCTCGPU.fwd_alg_kernel(grid=(batch_blocks,1,1),block=(L,batch_per_block,1),shared_mem = (2*4*(L+2))*batch_per_block,args=(scores_gpu.data_ptr(),target_indexes.data_ptr(),F.data_ptr(),BT.data_ptr(),mask.data_ptr(),T,L,2,N))
        lower_upper_tensor = self.get_index_range(BT.to("cpu").numpy(),F.to("cpu"))
        max_score,arg_max = torch.max(F[:,:,-1],dim=1)
        return lower_upper_tensor,max_score.to("cpu")
        





class Alignment: #new return class to bundle information about alignment
    def __init__(self) -> None:
        self.ctc_encoding=""
        self.alignment_start=0
        self.alignment_end=0
        self.alignment_score=0


    @property.setter
    def alignment_start(self,start:int):self._alignment_start=start
    @property
    def alignment_start(self): return self._alignment_start

    @property.setter
    def alignment_end(self,end:int):self._alignment_end=end
    @property
    def alignment_end(self): return self._alignment_end

    @property.setter
    def alignment_score(self,score:float):self._alignment_score=score
    @property
    def alignment_score(self): return self._alignment_score

    @property.setter
    def ctc_encoding(self,encoding:str):self._ctc_encoding=encoding
    @property
    def ctc_encoding(self): return self._ctc_encoding




class LongStrandAlignCTCGPU(Align):
    fwd_alg_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","longstrand_fwd_logspace_align",FLOAT='float',SUM2='logsumexp2',SUM='logsumexp3',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))

    def __init__(self, alphabet: list, device="cuda:0") -> None:
        super().__init__(alphabet, device)
    
    def get_index_range(self,BT:np.ndarray,F:torch.Tensor,target_indexes:torch.Tensor)->tuple[np.ndarray,np.ndarray]:
        argmax_t = torch.argmax(F[:,-1],dim=1) 
        alignment_range = np.ndarray((2,),dtype=np.int32) #(2,) tensor
        alignment_indexes = []
        current_strand_index = BT.shape[1]
        current_time_index = int(argmax_t)
        while current_strand_index>=0:
            current_symbol = target_indexes[current_strand_index] #collect symbols as we go through time
            alignment_indexes.append(current_symbol)
            current_strand_index = int(BT[current_time_index,current_strand_index])
            current_time_index-=1
        alignment_range[0]=int(current_time_index+1)
        alignment_range[1]=int(argmax_t)
        return alignment_range,np.array(alignment_indexes[::-1]) #alignment indexes reversed to account for the reversed traversal
    
    def align(self, scores: torch.Tensor, seq: str, remove_end_blanks=False) -> Alignment:
        alignment=Alignment()
        #right now, long strand aligner only handles 1 strand at a time, ineffcient for GPU resources, but probably will get the job done quick enough
        T,B=scores.size()
        scores_gpu=scores.to(self._device)
        target_indexes = HedgesBonitoCTC.string_to_indexes(seq,self._letter_to_index,device=self._device,batch=N)
        target_indexes = HedgesBonitoCTC.insert_blanks(target_indexes)
        if remove_end_blanks: target_indexes=target_indexes[1:-1] #remove end blanks from the forced alignment
        BT = torch.full((T,target_indexes.size(0)),0,dtype=torch.int64,device=self._device) #backtrace to know where alignment ranges over T
        F = torch.full((T,target_indexes.size(0)),Log.zero,device=self._device) #forward calculating Trellis
        mask = target_indexes[:-2]==target_indexes[2:]
        mask=torch.nn.functional.pad(mask,(2,0),value=1)
        mask = torch.where(mask,torch.full((target_indexes.size(1)),Log.zero,device=self._device),
                           torch.full((target_indexes.size(1)),Log.one,device=self._device))
        with cp.cuda.Device(0):
            L = target_indexes.size(0)
            offset=0
            """
            problem here is that each L is dependent on the previous 2 positions (L-1,L-2), and so we can't just throw very long strands on 1 GPU kernel call by splitting
            positions across independent blocks.
            We need to call the GPU kernel multiple times to resolve the alignment, which should be possible by simple chunking of the L dimension.
            """
            while L>0:
                L_threads = min(L,1024) #make sure we don't use too many threads that will fit in a block
                LongStrandAlignCTCGPU.fwd_alg_kernel(grid=(1,1,1),block=(L_threads,1,1),shared_mem = (2*4*(L+2)),
                                                     args=(scores_gpu.data_ptr(),target_indexes.data_ptr(),F.data_ptr(),
                                                           BT.data_ptr(),mask.data_ptr(),T,L,2,offset))
                L-=L_threads
                offset+=L_threads #advance the offset 
        lower_upper,alignment = self.get_index_range(BT.to("cpu").numpy(),F.to("cpu"))
        max_score,arg_max = torch.max(F[:,-1],dim=1)
        
        alignment.alignment_start = lower_upper[0]
        alignment.alignment_end = lower_upper[1]
        alignment.alignment_score=max_score.to("cpu")
        return alignment
