from .plot import *
import bonito.hedges_decode.cuda_utils as cu
from .hedges_decode_utils import *

import torch
import dnastorage.codec.hedges_hooks as hedges_hooks
import cupy as cp
import os

#get env variables
PLOT = os.getenv("PLOT",False)

class HedgesBonitoScoreBase:
    def init_initial_state_F(self,scores:torch.Tensor)->torch.Tensor:
        raise NotImplementedError()

    def forward_step(self,scores:torch.Tensor,base_transitions:torch.Tensor,F:torch.Tensor,initial_bases:torch.Tensor,strand_index:int,
                     nbits:int)->tuple[torch.Tensor,torch.Tensor]:
        raise NotImplementedError()
    def __init__(self,full_message_length:int,H:int,fastforward_seq:str,device:str,initial_state_index:int,letter_to_index:dict) -> None:
        self._full_message_length = full_message_length
        self._H = H
        self._fastforward_seq=fastforward_seq
        self._device =device
        self._initial_state_index = initial_state_index
        self._letter_to_index=letter_to_index



class HedgesBonitoCTC(HedgesBonitoScoreBase):
    """
    @brief      Hedges decoding on Bonito CTC output

    @details    Implements the necessary methods to extract information out of Bonito scores when using the Bonito CTC model
    """
    def __init__(self, full_message_length: int, H: int, fastforward_seq: str, device: str, initial_state_index: int,
                  letter_to_index: dict, window:int) -> None:
        super().__init__(full_message_length, H, fastforward_seq, device, initial_state_index, letter_to_index)
        self._window = window #indicates size of window to use
        self._current_F_lower=0 #used for windowing

    @classmethod
    def _fwd_algorithm(cls,target_scores:torch.Tensor,mask:torch.Tensor,
                       F:torch.Tensor,lower_t_range:int,upper_t_range:int,device,
                       using_window:bool,pad:int)->torch.Tensor:
        target_scores=target_scores[:,:,:,1:]
        running_alpha_index=2
        T,H,E,L=target_scores.size()
        alpha_t = torch.full((T,H,E),Log.zero)
        running_alpha =torch.full((H,E,L+2),Log.zero)
        log_zeros = torch.full((H,E,L),Log.zero)
        results_stack=torch.full((3,running_alpha.size(0),running_alpha.size(1),running_alpha.size(2)-running_alpha_index),Log.zero)
        tmp_F=F
        if not using_window:
            r = torch.arange(lower_t_range,upper_t_range)
            F_offset = int(-1)
        else: #this is only necesssary when we are trying to store only the time range we are calculating
            r = torch.arange(0,upper_t_range-lower_t_range)
            F_offset = pad-1
            tmp_F = torch.cat([F,torch.full((pad,F.size(1)),Log.zero)],dim=0)
        for t in r:                
            running_alpha[:,:,running_alpha_index-1] = tmp_F[t+F_offset,:][:,None]
            results_stack[0,:,:,:] = running_alpha[:,:,running_alpha_index:]
            results_stack[1,:,:,:] = running_alpha[:,:,running_alpha_index-1:-1]
            results_stack[2,:,:,:] = torch.where(mask,log_zeros,running_alpha[:,:,running_alpha_index-2:-2])
            running_alpha[:,:,running_alpha_index:]= Log.mul(target_scores[t,:,:,:],Log.sum(results_stack,dim=0))
            alpha_t[t,:,:]=running_alpha[:,:,-1]
        return alpha_t

    @classmethod
    def _dot_product(cls,target_scores,alpha_t)->torch.Tensor:
        T,H,E,L = target_scores.size()
        log_prob_no_transition=torch.nn.functional.pad(
            Log.sum(torch.stack([torch.zeros((T,H,E)),target_scores[:,:,:,-1]]),dim=0), 
            (0,0,0,0,0,1),value=Log.one)[1:,:,:] #TxHx2^nbits tensor to help with final score calculation
        return dot(log_prob_no_transition,alpha_t[:,:,:],dim=0) #Hx2^nbits output

    @classmethod
    def string_to_indexes(cls,seq:str,letter_to_index:dict,device="cpu")->torch.Tensor:
        ret_tensor = torch.zeros((len(seq),),device=device)
        for i,s in enumerate(seq):
            ret_tensor[i]=letter_to_index[s]
        return ret_tensor
    
    @classmethod
    def insert_blanks(cls,seq:torch.Tensor)->torch.Tensor:
        #blanks should be index 0 in the alphabet
        L = seq.size(0)
        ret_tensor = seq.new_zeros((L*2+1,),dtype=torch.int64)
        ret_tensor[1::2]=seq
        return ret_tensor 
    
    def init_initial_state_F(self, scores:torch.Tensor) -> torch.Tensor:
        T,I = scores.size()

        if self._window>0:
            scores_per_base = T/self._full_message_length
            strand_index = len(self._fastforward_seq)-1
            lower_t_range=int(max(((strand_index)*scores_per_base)-self._window,0))
            upper_t_range=int(min((strand_index*scores_per_base)+self._window,T-self._full_message_length+strand_index+1))
            self._current_F_lower = lower_t_range
        else:
            lower_t_range =0
            upper_t_range = T
        T_range = upper_t_range-lower_t_range
        F = torch.full((T_range,self._H),Log.zero)
        #nothing to do if there is no initial strand
        if len(self._fastforward_seq)==0: return F
        strand_indexes = HedgesBonitoCTC.string_to_indexes(self._fastforward_seq,self._letter_to_index)
        padded_strand = HedgesBonitoCTC.insert_blanks(strand_indexes)[:-1] #leave off last blank due to viterbi branch path nature
        strand_index_matrix = padded_strand[None,:].expand(T_range,-1)#TxL matrix
        scores_matrix = scores[lower_t_range:upper_t_range,:].gather(1,strand_index_matrix) #get log probabilities for each base at each time point
        _,L = scores_matrix.size()
        running_alpha = torch.full((L+2,),Log.zero) #1 dimensional tensor that tracks alpha for all characters at time t
        #need a mask matrix for repeats
        mask = torch.nn.functional.pad(padded_strand[2:]==padded_strand[:-2],(2,0),value=1)
        log_zeros = torch.full((L,),Log.zero)
        #iterate over T dimension and calcualte alphas
        running_alpha[2]=Log.one
        for t in torch.arange(T_range):
            running_alpha[2:] = Log.mul(scores_matrix[t,:],Log.sum(torch.stack([running_alpha[2:],running_alpha[1:-1],torch.where(mask,log_zeros,running_alpha[0:-2])]),dim=0))
            F[t,self._initial_state_index] = running_alpha[-1]
        return F

    #@profile        
    def forward_step(self, scores: torch.Tensor, base_transitions: torch.Tensor, F: torch.Tensor, initial_bases:torch.Tensor, strand_index:int,
                     nbits:int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @brief      Calculates a forward step in the hedges ctc algorithm

        @param      scores Txlen(alphabet) tensor of state scores
        @param      base_transitions Hx2^nbits tensor of indexes representing bases outgoing from states
        @param      F TxH tensor of alphas up to the point in the codeword
        @param      initial_bases H-length tensor of last base for each trellis state

        @return     return tuple of tensors (X,Y) where X is a Hx2^nbits tensor of scores, and Y is a TxHx2^nbits tensor of outgoing alpha calculations
        """
        T,A = scores.size()
        H,E,L_trans = base_transitions.size()
        using_window=False
        if self._window and self._window>0:
            #print(self._window)
            using_window=True
            scores_per_base=torch.argmax(F[:,0],dim=0)+self._current_F_lower
            lower_t_range=int(max((scores_per_base)-self._window,strand_index+1-L_trans//2,self._current_F_lower+1))
            upper_t_range=int(min(scores_per_base+self._window,T-self._full_message_length+strand_index+1))
            T_range = upper_t_range-lower_t_range
            #need to create a Hx2^nbitsxL tensor to represent all strings we are calculating alphas for var in collection:
            targets = torch.concat([initial_bases[:,:,None],base_transitions],dim=2)
            _,_2,L = targets.size()
        else: #base, no window case
            lower_t_range=strand_index+1-L_trans//2
            upper_t_range=T-self._full_message_length+strand_index+1
            #need to create a Hx2^nbitsxL tensor to represent all strings we are calculating alphas for var in collection:
            targets = torch.concat([initial_bases[:,:,None],base_transitions],dim=2)
            _,_2,L = targets.size()
        mask = torch.nn.functional.pad(targets[:,:,2:]==targets[:,:,:-2],(1,0),value=1)
        #calculate valid ranges of t to avoid unnecessary iterations
        alpha_t,out_scores = self._fwd_algorithm(targets,scores,mask,F,lower_t_range,upper_t_range,self._device,using_window,lower_t_range-self._current_F_lower)
        #if PLOT and strand_index==(864+48): plot_scores(alpha_t,lower_t_range,upper_t_range,True,plot_list=[0])
        #out_scores=self._dot_product(targets,scores,lower_t_range,upper_t_range,alpha_t,using_window)       
        #print(torch.max(out_scores))
        self._current_F_lower=lower_t_range #keeps track of most recent lower_t_range
        return out_scores,alpha_t


class HedgesBonitoCTCGPU(HedgesBonitoCTC):
    fwd_alg_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","fwd_logspace_reduce",FLOAT='float',SUM2='logsumexp2',SUM='logsumexp3',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))
    dot_mul_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","dot_mul",FLOAT='float',SUM='logsumexp3',SUM2='logsumexp2',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))
    dot_reduce_kernel=cu.load_cupy_func("cuda/reduce.cu","dot_reduce",FLOAT='float',REDUCE="logsumexp2",ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))
   

    def __init__(self, full_message_length: int, H: int, fastforward_seq: str, device: str,
                  initial_state_index: int, letter_to_index: dict, window:int) -> None:
        super().__init__(full_message_length, H, fastforward_seq, device, initial_state_index, letter_to_index,window)
        assert torch.cuda.is_available() #make sure we have cuda for this class

    @classmethod
    def _dot_product(cls,targets,scores,lower_t_range,upper_t_range,alpha_t,using_window,strand_index=0)->torch.Tensor:
        H,E,L = targets.size()
        if using_window: 
            t_range_offset=lower_t_range
            T=upper_t_range-lower_t_range
        else:
            T=scores.size(0)
            t_range_offset=0
        z = alpha_t.new_zeros((T,H,E))
        with cp.cuda.Device(0):
            #With bigger H, blocking needs to be reformulated
            #1024 >= H*E*t_per_block
            #1024//E = H*t_per_block, E is some power of 2 
            #H is a multiple of some 2^n so pick t_per_block to be some power of 2
            t_per_block = 64
            h_per_block = 1024//(t_per_block*E)
            total_t_blocks = (T//t_per_block)+1
            assert H%h_per_block==0 #this should divide evenly
            total_h_blocks = (H//h_per_block) 
            #perform multiplication of dot product
            HedgesBonitoCTCGPU.dot_mul_kernel(grid=(total_t_blocks,total_h_blocks,1),block=(E,h_per_block,t_per_block),shared_mem = 0, args=(targets.data_ptr(),scores.data_ptr(),alpha_t.data_ptr(),z.data_ptr(),T,L,H,E,t_range_offset))
            #We can make this more efficient by reducing the number of dead threads
            stride=1 # this indicates where to find the next valid time value for reduction
            while True:
                t_per_block = 32 #all t should fit in a warp
                total_t_calculated_per_step = t_per_block*2
                h_per_block =1024//(t_per_block*E)
                assert H%h_per_block==0 #this should divide evenly
                total_h_blocks=H//h_per_block
                total_T_blocks=T//(total_t_calculated_per_step)
                if T%total_t_calculated_per_step!=0:
                    total_T_blocks +=1
                HedgesBonitoCTCGPU.dot_reduce_kernel(grid=(total_h_blocks,total_T_blocks,1),block=(t_per_block,E,h_per_block),shared_mem = 1024*4, args=(z.data_ptr(),z.data_ptr(),T,stride,H,E))
                if T<total_t_calculated_per_step: break
                T=total_T_blocks
                stride*=(total_t_calculated_per_step)
        return z[0,:,:]
    
    @classmethod
    def _fwd_algorithm(cls, targets: torch.Tensor,scores: torch.Tensor, mask: torch.Tensor,
                       F: torch.Tensor, lower_t_range: int, upper_t_range: int,device,
                       using_window,pad)->torch.Tensor:
        H,E,L=targets.size()
        L-=1
        #convert mask from bools to floats to avoid control flow in GPU kernel
        mask = torch.where(mask,torch.full(mask.size(),Log.zero,device=device),torch.full(mask.size(),Log.one,device=device))
        w=F.to(device)
        out_scores = targets.new_zeros((H,E))
        if not using_window:
            r_1 = lower_t_range
            r_2 = upper_t_range
            f_offset=int(-1)
            lower_t_range_offset=0
            #print("scores size {}".format(scores.size(0)))
            y = torch.full((scores.size(0),H,E),Log.zero,device=device) #output pointer
        else:
            r_1 = 0
            r_2 = upper_t_range-lower_t_range
            f_offset=pad-1
            lower_t_range_offset=lower_t_range
            y = torch.full((upper_t_range-lower_t_range,H,E),Log.zero,device=device) #output pointer
        with cp.cuda.Device(0):
            #Need to break down L,H,E into block sizes <=1024
            #L can't be moved because of thread sync. dependency
            #1024/L = H*E
            #E is typically small (at most 2 for now), so we cajust do 1024//(L*E) for H threads per block
            max_T_per_block=128
            h_per_block = max_T_per_block//(L*E)
            H_blocks = (H//h_per_block)+1
            #print(H_blocks)
            HedgesBonitoCTCGPU.fwd_alg_kernel(grid=(H_blocks,1,1),block=(L,h_per_block,E),shared_mem=2*4*(L+2)*h_per_block*E,args=(targets.data_ptr(),scores.data_ptr(),y.data_ptr(),
                                                                                                                                   mask.data_ptr(),w.data_ptr(),out_scores.data_ptr(),r_1,r_2,
                                                                                                                                   H,E,L,2,1,f_offset,F.size(0),lower_t_range_offset))
        return y,out_scores
        
    

