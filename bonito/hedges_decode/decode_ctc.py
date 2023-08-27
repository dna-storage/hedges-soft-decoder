from .plot import *
import bonito.hedges_decode.cuda_utils as cu
from .hedges_decode_utils import *
import numpy as np

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
    def _fwd_algorithm():
        pass

    @classmethod
    def string_to_indexes(cls,seq:str,letter_to_index:dict,device="cpu",batch=None)->torch.Tensor:
        ret_tensor = torch.zeros((len(seq),),device=device)
        for i,s in enumerate(seq):
            ret_tensor[i]=letter_to_index[s]
        if batch is None:
            return ret_tensor
        else:
            return ret_tensor[None,:].repeat(batch,1)
    
    @classmethod
    def insert_blanks(cls,seq:torch.Tensor)->torch.Tensor:
        #blanks should be index 0 in the alphabet
        N,L = seq.size()
        ret_tensor = seq.new_zeros((N,L*2+1,),dtype=torch.int64)
        ret_tensor[:,1::2]=seq
        return ret_tensor 
    
    def init_initial_state_F(self, scores:torch.Tensor,reverse:torch.Tensor) -> torch.Tensor:
        N,T,I = scores.size()

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
        F = scores.new_full((N,T_range,self._H),Log.zero)
        #nothing to do if there is no initial strand
        if len(self._fastforward_seq)==0: return F
        #For batching, account for reverse and forward indexes
        forward_strand_indexes = HedgesBonitoCTC.string_to_indexes(self._fastforward_seq,self._letter_to_index,batch=N)
        reverse_strand_indexes = HedgesBonitoCTC.string_to_indexes(complement(self._fastforward_seq),self._letter_to_index,batch=N)
        strand_indexes = torch.where(reverse[:,None].expand(-1,len(self._fastforward_seq)),reverse_strand_indexes,forward_strand_indexes)
        padded_strand = HedgesBonitoCTC.insert_blanks(strand_indexes)[:,:-1] #leave off last blank due to viterbi branch path nature
        strand_index_matrix = padded_strand[:,None,:].expand(-1,T_range,-1).to(scores.get_device())#TxL matrix
        scores_matrix = scores[:,lower_t_range:upper_t_range,:].gather(2,strand_index_matrix) #get log probabilities for each base at each time point
        N,_,L = scores_matrix.size()
        running_alpha = scores.new_full((N,L+2,),Log.zero) #1 dimensional tensor that tracks alpha for all characters at time t
        #need a mask matrix for repeats
        mask = torch.nn.functional.pad(padded_strand[:,2:]==padded_strand[:,:-2],(2,0),value=1).to(scores.get_device())
        log_zeros = scores.new_full((N,L),Log.zero)
        #iterate over T dimension and calcualte alphas
        running_alpha[:,2]=Log.one
        for t in torch.arange(T_range):
            running_alpha[:,2:] = Log.mul(scores_matrix[:,t,:],Log.sum(torch.stack([running_alpha[:,2:],running_alpha[:,1:-1],torch.where(mask,log_zeros,running_alpha[:,0:-2])]),dim=0))
            F[:,t,self._initial_state_index] = running_alpha[:,-1]
        #print(torch.max(F[:,:,self._initial_state_index],dim=1))
        #exit(0)
        return F

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
        N,T,A = scores.size()
        N,H,E,L_trans = base_transitions.size()
        using_window=False
        if self._window and self._window>0:
            using_window=True
            scores_per_base=torch.argmax(F[:,0],dim=0)+self._current_F_lower
            lower_t_range=int(max((scores_per_base)-self._window,strand_index+1-L_trans//2,self._current_F_lower+1))
            upper_t_range=int(min(scores_per_base+self._window,T-self._full_message_length+strand_index+1))
            T_range = upper_t_range-lower_t_range
            #need to create a NxHx2^nbitsxL tensor to represent all strings we are calculating alphas for var in collection:
            targets = torch.concat([initial_bases[:,:,:,None],base_transitions],dim=3)
            N,_,_2,L = targets.size()
        else: #base, no window case
            lower_t_range=strand_index+1-L_trans//2
            upper_t_range=T-self._full_message_length+strand_index+1
            #need to create a Hx2^nbitsxL tensor to represent all strings we are calculating alphas for var in collection:
            targets = torch.concat([initial_bases[:,:,:,None],base_transitions],dim=3)
            N,_,_2,L = targets.size()
        mask = torch.nn.functional.pad(targets[:,:,:,2:]==targets[:,:,:,:-2],(1,0),value=1)
        #calculate valid ranges of t to avoid unnecessary iterations
        alpha_t,out_scores = self._fwd_algorithm(targets,scores,mask,F,lower_t_range,upper_t_range,self._device,using_window,lower_t_range-self._current_F_lower)
        self._current_F_lower=lower_t_range #keeps track of most recent lower_t_range
        return out_scores,alpha_t

    


class HedgesBonitoCTCGPU(HedgesBonitoCTC):
    fwd_alg_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","fwd_logspace_reduce",FLOAT='float',SUM2='logsumexp2',SUM='logsumexp3',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))

    def __init__(self, full_message_length: int, H: int, fastforward_seq: str, device: str,
                  initial_state_index: int, letter_to_index: dict, window:int) -> None:
        super().__init__(full_message_length, H, fastforward_seq, device, initial_state_index, letter_to_index,window)
        assert torch.cuda.is_available() #make sure we have cuda for this class
    
    @classmethod
    def _fwd_algorithm(cls, targets: torch.Tensor,scores: torch.Tensor, mask: torch.Tensor,
                       F: torch.Tensor, lower_t_range: int, upper_t_range: int,device,
                       using_window,pad)->torch.Tensor:
        N,H,E,L=targets.size()
        L-=1
        #convert mask from bools to floats to avoid control flow in GPU kernel
        mask = torch.where(mask,torch.full(mask.size(),Log.zero,device=device),torch.full(mask.size(),Log.one,device=device))
        w=F.to(device)
        out_scores = targets.new_zeros((N,H,E),dtype=torch.float)
        if not using_window:
            r_1 = lower_t_range
            r_2 = upper_t_range
            f_offset=int(-1)
            lower_t_range_offset=0
            y = torch.full((N,scores.size(1),H,E),Log.zero,device=device) #output pointer
        else:
            r_1 = 0
            r_2 = upper_t_range-lower_t_range
            f_offset=pad-1
            lower_t_range_offset=lower_t_range
            y = torch.full((N,upper_t_range-lower_t_range,H,E),Log.zero,device=device) #output pointer
        with cp.cuda.Device(0):
            #Need to break down L,H,E into block sizes <=1024
            #L can't be moved because of thread sync. dependency
            #1024/L = H*E
            #E is typically small (at most 2 for now), so we cajust do 1024//(L*E) for H threads per block
            max_T_per_block=128 #max threads to have per block
            h_per_block = max_T_per_block//(L*E)
            H_blocks = (H//h_per_block)+1
            HedgesBonitoCTCGPU.fwd_alg_kernel(grid=(H_blocks,N,1),block=(L,h_per_block,E),shared_mem=2*4*(L+2)*h_per_block*E,args=(targets.data_ptr(),scores.data_ptr(),y.data_ptr(),
                                                                                                                                   mask.data_ptr(),w.data_ptr(),
                                                                                                                                   out_scores.data_ptr(),r_1,r_2,
                                                                                                                                   H,E,L,2,1,f_offset,F.size(1),scores.size(1),lower_t_range_offset))
        #print(lower_t_range)
        #print("F {}".format(F[0,-100,:]))
        #print("Y {}".format(y[0,-100,:,:]))
        #print("O {}",format(out_scores[0,:]))
        #if(lower_t_range==1720): exit(0)
        return y,out_scores
        
    

