from bonito.hedges_decode.base_decode import *
from bonito.hedges_decode.plot import *
import bonito.hedges_decode.cuda_utils as cu

import torch
import dnastorage.codec.hedges_hooks as hedges_hooks
import cupy as cp


class HedgesBonitoCTC(HedgesBonitoBase):
    """
    @brief      Hedges decoding on Bonito CTC output

    @details    Implements the necessary methods to extract information out of Bonito scores when using the Bonito CTC model
    """
    def __init__(self, hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint,alphabet,device,window=0) -> None:
        super().__init__(hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint,alphabet,device)
        self._trellis_connections,self._trellis_transition_values = self.calculate_trellis_connections(range(0,3),self._H) #list of matrices with trellis connections for different points in the codeword
        self._window = window #indicates size of window to use
        
    def get_trellis_state_length(self,hedges_param_dict,using_hedges_DNA_constraint)->int:
        return 2**hedges_param_dict["prev_bits"]
    
    def get_initial_trellis_index(self,global_hedge_state)->int:
        return 0

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
    
    def init_initial_state_F(self, scores:torch.Tensor, F: torch.Tensor) -> None:
        T,I = scores.size()
        initial_state_index = self.get_initial_trellis_index(self._global_hedge_state_init)
        #nothing to do if there is no initial strand
        if len(self._fastforward_seq)==0: return
        strand_indexes = HedgesBonitoCTC.string_to_indexes(self._fastforward_seq,self._letter_to_index)
        padded_strand = HedgesBonitoCTC.insert_blanks(strand_indexes)[:-1] #leave off last blank due to viterbi branch path nature
        strand_index_matrix = padded_strand[None,:].expand(T,-1)#TxL matrix
        scores_matrix = scores.gather(1,strand_index_matrix) #get log probabilities for each base at each time point
        _,L = scores_matrix.size()
        running_alpha = torch.full((L+2,),Log.zero) #1 dimensional tensor that tracks alpha for all characters at time t
        #need a mask matrix for repeats
        mask = torch.nn.functional.pad(padded_strand[2:]==padded_strand[:-2],(2,0),value=1)
        log_zeros = torch.full((L,),Log.zero)
        #iterate over T dimension and calcualte alphas
        running_alpha[2]=Log.one
        for t in torch.arange(T):
            running_alpha[2:] = Log.mul(scores_matrix[t,:],Log.sum(torch.stack([running_alpha[2:],running_alpha[1:-1],torch.where(mask,log_zeros,running_alpha[0:-2])]),dim=0))
            F[t,initial_state_index] = running_alpha[-1]

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
        scores_per_base = T/self._full_message_length
        H,E,L_trans = base_transitions.size()
        using_window=False
        if self._window and self._window>0:
            using_window=True
            lower_t_range=max(((strand_index-L_trans)*scores_per_base)-self._window,strand_index-L_trans)
            upper_t_range=min((strand_index*scores_per_base)+self._window,T-self._full_message_length+strand_index+1)
            T_range = upper_t_range-lower_t_range
            #need to create a Hx2^nbitsxL tensor to represent all strings we are calculating alphas for var in collection:
            targets = torch.concat([initial_bases[:,:,None],base_transitions],dim=2)
            _,_2,L = targets.size()
            targets=targets[None,:,:,:].expand(T_range,-1,-1,-1) #expand the targets along the time dimension
            target_scores=torch.gather(scores[lower_t_range:upper_t_range,None,None,:].expand(-1,H,E,-1),3,targets)#gather in the scores for the targets    
        else: #base, no window case
            lower_t_range=strand_index-L_trans
            upper_t_range=T-self._full_message_length+strand_index+1
            #need to create a Hx2^nbitsxL tensor to represent all strings we are calculating alphas for var in collection:
            targets = torch.concat([initial_bases[:,:,None],base_transitions],dim=2)
            _,_2,L = targets.size()
            targets=targets[None,:,:,:].expand(T,-1,-1,-1) #expand the targets along the time dimension
            target_scores=torch.gather(scores[:,None,None,:].expand(-1,H,E,-1),3,targets)#gather in the scores for the targets 

        mask = torch.nn.functional.pad(targets[0,:,:,2:]==targets[0,:,:,:-2],(1,0),value=1)
        #calculate valid ranges of t to avoid unnecessary iterations

        alpha_t = self._fwd_algorithm(target_scores,mask,F,lower_t_range,upper_t_range,self._device,using_window,scores_per_base)
        #if PLOT and strand_index==504: plot_scores(alpha_t,lower_t_range,upper_t_range,True)
        if PLOT and strand_index==1002: plot_scores(alpha_t,int((self._T*lower_t_range/2160)-1000),int((self._T*lower_t_range/2160)+1000),True,int(self._T*lower_t_range/2160))
        out_scores=self._dot_product(target_scores,alpha_t)
        return out_scores,alpha_t
 
        
            
         
    def calculate_trellis_connections(self, bit_range: range, trellis_states: int) -> tuple[list[torch.Tensor], ...]:
        index_list=[]
        value_list=[]
        dtype = torch_get_index_dtype(trellis_states)
        for nbits in bit_range:
            incoming_states_matrix=torch.full((trellis_states,2**nbits),0,dtype=torch.int64)
            incoming_state_value_matrix=torch.full((trellis_states,2**nbits),0,dtype=torch.int64)
            for h in range(trellis_states):
                value,incoming_states = hedges_hooks.get_incoming_states(self._global_hedge_state_init,nbits,h)
                for prev_index,s_in in incoming_states:
                    incoming_states_matrix[h,prev_index]=s_in
                    incoming_state_value_matrix[h,prev_index]=value
            index_list.append(incoming_states_matrix)
            value_list.append(incoming_state_value_matrix)
        return index_list,value_list


class HedgesBonitoCTCGPU(HedgesBonitoCTC):
    fwd_alg_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","fwd_logspace",FLOAT='float',SUM2='logsumexp2',SUM='logsumexp3',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))
    dot_mul_kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","dot_mul",FLOAT='float',SUM='logsumexp3',SUM2='logsumexp2',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))
    dot_reduce_kernel=cu.load_cupy_func("cuda/reduce.cu","dot_reduce",FLOAT='float',REDUCE="logsumexp2",ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))

    def __init__(self, hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint,alphabet,device,window=0) -> None:
        super().__init__(hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint,alphabet,device,window)
        assert torch.cuda.is_available() #make sure we have cuda for this class

    @classmethod
    def _dot_product(cls,target_scores,alpha_t,strand_index=0)->torch.Tensor:
        T,H,E,L = target_scores.size()
        z = alpha_t.new_zeros((T,H,E))
        with cp.cuda.Device(0):
            t_per_block = 1024//(H*E)
            total_blocks = (T//t_per_block)+1
            #perform multiplication of dot product
            HedgesBonitoCTCGPU.dot_mul_kernel(grid=(total_blocks,1,1),block=(E,H,t_per_block),shared_mem = 0, args=(target_scores.data_ptr(),alpha_t.data_ptr(),z.data_ptr(),T,L))
            #now we need to reduce along the time direction
            stride=1
            while True:
                total_T_blocks=T//2048
                if T%2048!=0:
                    total_T_blocks +=1
                HedgesBonitoCTCGPU.dot_reduce_kernel(grid=(H,E,total_T_blocks),block=(1024,1,1),shared_mem = 1024*4, args=(z.data_ptr(),z.data_ptr(),T,stride))
                if T<2048:
                    break
                T=total_T_blocks
                stride*=2048
        return z[0,:,:]
    @classmethod
    def _fwd_algorithm(cls, target_scores: torch.Tensor,mask: torch.Tensor,
                       F: torch.Tensor, lower_t_range: int, upper_t_range: int,device,
                       using_window,pad)->torch.Tensor:
        T,H,E,L=target_scores.size()
        L-=1
        y = torch.full((T,H,E),Log.zero,device=device)
        #convert mask from bools to floats to avoid control flow in GPU kernel
        mask = torch.where(mask,torch.full(mask.size(),Log.zero,device=device),torch.full(mask.size(),Log.one,device=device))
        w=F.to(device)
        if not using_window:
            r_1 = lower_t_range
            r_2 = upper_t_range
            f_offset=int(-1)
        else:
            r_1 = 0
            r_2 = upper_t_range-1
            f_offset=pad-1
        
        with cp.cuda.Device(0):
            h_divider=L//2
            h_per_block=H//h_divider
            H_blocks = (H//h_per_block)+1
            HedgesBonitoCTCGPU.fwd_alg_kernel(grid=(H_blocks,1,1),block=(L,h_per_block,E),shared_mem=2*4*(L+2)*h_per_block*E,args=(target_scores.data_ptr(),y.data_ptr(),
                                                                                                                                   mask.data_ptr(),w.data_ptr(),r_1,r_2,
                                                                                                                                   upper_t_range,H,E,L,T,2,1,f_offset,F.size(0)))
        return y
        

