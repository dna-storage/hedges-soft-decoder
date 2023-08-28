import os
import torch
import numpy as np
import dnastorage.codec.hedges as hedges
import dnastorage.codec.hedges_hooks as hedges_hooks
import math 
import gc
from bonito.hedges_decode.context_utils import ContextManager
import bonito.hedges_decode.context_utils as context_utils
from .decode_ctc import *
from .hedges_decode_utils import *
from bonito.hedges_decode.beam_viterbi import run_beam_1


def torch_get_index_dtype(states)->torch.dtype:
    num_bits = int(math.ceil(math.log2(states)))
    if num_bits<=8:
        return torch.uint8
    elif num_bits<=15: #int16 is signed, so assume last bit is sign
        return torch.int16
    else:
        return torch.int32

class HedgesBonitoBase:
    _trellis_connection_cache=[]
    get_new_F_kernel=cu.load_cupy_func("cuda/index_gather.cu","F_copy",FLOAT='float')

    """
    @brief      Base class for Hedges Bonito decoding

    @details    Base class for decoding CTC-type outputs of models used for basecalling nanopore signals, provides general functionality of implementing a CTC-based branch metric
    """
    
    def get_trellis_state_length(self,hedges_param_dict,using_hedges_DNA_constraint)->int:
        return 2**hedges_param_dict["prev_bits"]
    
    def get_initial_trellis_index(self,global_hedge_state)->int:
        return 0
    
    def calculate_trellis_connections_mask(self,context:ContextManager,nbits:int,dead_states:np.ndarray)->torch.Tensor|None:
        return None #for a basic trellis, not using masks

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
    
    def gather_trans_scores(self, trans_scores:torch.Tensor, H_indexes:torch.Tensor, E_indexes:torch.Tensor)->torch.Tensor:
         return trans_scores[:,H_indexes,E_indexes] #should produce N,Hx2^n matrix of scores that need to be compared
    
    @property 
    def fastforward_seq(self):
        return self._fastforward_seq

    @fastforward_seq.setter
    def fastforward_seq(self,s):
        self._fastforward_seq=s
        self._scorer._fastforward_seq=s

    @property 
    def window(self):
        pass
    @window.setter
    def window(self,w):
        self._scorer._window=w

    @property
    def is_beam(self):
        return False
    


    def fill_base_transitions(self,N:int,H:int,transitions:int,C:ContextManager,nbits:int,reverse:bool)->np.ndarray:
        """
        @brief      Fills in base_transitions with indexes representing the characters at this point in the message
        @param      base_transitions tensor holding indexes of bases
        @param      C list of contexts
        @param      nbits number of bits on this transition
        @return     None
        """
        return context_utils.fill_base_transitions(N,H,transitions,C,nbits,reverse)
    def __init__(self,hedges_param_dict:dict,hedges_bytes:bytes,using_hedges_DNA_constraint:bool,alphabet:list,device:str,
                 score:str,window:int=0) -> None:
        
        self._global_hedge_state_init = hedges_hooks.make_hedge( hedges.hedges_state(**hedges_param_dict)) #stores pointer to a hedges state object
        self._fastforward_seq=hedges_hooks.fastforward_context(bytes(hedges_bytes),self._global_hedge_state_init) #modifies the global hedge state to reflect state at end of hedges_bytes
        self._H = self.get_trellis_state_length(hedges_param_dict,using_hedges_DNA_constraint) #length of history side of matrices
        self._using_hedges_DNA_constraint = using_hedges_DNA_constraint
        self._full_message_length = hedges_hooks.get_max_index(self._global_hedge_state_init) #total message length
        self._L = self._full_message_length - len(self._fastforward_seq)#length of message-length side of matrices
        self._alphabet = alphabet #alphabet we are using
        self._letter_to_index = {_:i for i,_ in enumerate(self._alphabet)} #reverse map for the alphabet
        self._trellis_connections=[]
        self._trellis_transition_values=[]
        self._max_bits=1 #max number of bits per base
        self._device=device
         
        #initialize connections
        self._trellis_connections,self._trellis_transition_values = self.calculate_trellis_connections(range(0,self._max_bits+1),self._H) 

        #instantiate scorer class
        if score=="CTC" and "cuda" in self._device:
            self._scorer = HedgesBonitoCTCGPU(self._full_message_length,self._H,self._fastforward_seq,self._device,
                                              self.get_initial_trellis_index(self._global_hedge_state_init),
                                              self._letter_to_index,window)
        elif score=="CTC" and "cpu" in self._device:
            self._scorer = HedgesBonitoCTC(self._full_message_length,self._H,self._fastforward_seq,self._device,
                                           self.get_initial_trellis_index(self._global_hedge_state_init),
                                           self._letter_to_index,window)
        else: 
            raise ValueError("Scorer could not be instantiated")
        

        #make device versions ahead of time
        self._device_trellis_connections = [_.to(self._device) for _ in self._trellis_connections]
        self._device_trellis_transition_values= [_.to(self._device) for _ in self._trellis_transition_values]


    def string_from_backtrace(self,BT_index:np.ndarray,BT_bases:np.ndarray,start_state:int)->str:
        H,L = BT_index.shape
        current_state=torch.tensor(start_state)
        return_sequence=""
        states=[]
        for i in range(L)[::-1]:
            return_sequence+=self._alphabet[int(BT_bases[current_state,i])]
            states.append(current_state)
            
            if i==0: break
            current_state=BT_index[current_state,i]
        
        return return_sequence[::-1]
    

    #helps transfer F scores more efficiently than re-arangement
    def get_new_F(self,temp_f_outgoing:torch.Tensor,trellis_incoming_indexes:torch.Tensor,
                  trellis_incoming_value:torch.Tensor,value_of_max_scores:torch.Tensor,Fmem=None)->torch.Tensor:
        assert torch.cuda.is_available() #make sure we have cuda for this class
        if Fmem is None:
            return_F = torch.full((temp_f_outgoing.size(0),temp_f_outgoing.size(1),temp_f_outgoing.size(2)),0,device=self._device,dtype=torch.float)
        else: return_F=Fmem
        with cp.cuda.Device(0):  
            #just paralellize over H for now, could parallelize time block transfers if wanted
            N,T,H = return_F.size()
            #h_per_block = 1024
            h_per_block = 256
            t_per_block = 4
            H_blocks=return_F.size(2)//h_per_block
            T_blocks = (return_F.size(1)//t_per_block)
            if return_F.size(1)%t_per_block!=0: T_blocks+=1
            trellis_incoming_indexes_gpu = trellis_incoming_indexes.to(self._device)
            trellis_incoming_value_gpu = trellis_incoming_value.to(self._device)
            max_vals = value_of_max_scores 
            HedgesBonitoBase.get_new_F_kernel(grid=(H_blocks,T_blocks,N),block=(h_per_block,t_per_block,1),shared_mem=0,args=(temp_f_outgoing.data_ptr(),
                                                                                                            trellis_incoming_indexes_gpu.data_ptr(),
                                                                                                            trellis_incoming_value_gpu.data_ptr(),
                                                                                                            max_vals.data_ptr(),
                                                                                                            return_F.data_ptr(),
                                                                                                            return_F.size(2),
                                                                                                            return_F.size(1),
                                                                                                            trellis_incoming_indexes.size(1),
                                                                                                            temp_f_outgoing.size(3)
                                                                                                         )
            )

        return return_F

    #@profile
    def decode(self,scores:torch.Tensor,reverse:torch.Tensor,time_range_end:torch.Tensor)->str:
        """
        @brief      Core algorithm for implementing hedges viterbi decoding

        @param      scores tensor that reflect probabilites of characters

        @return     string representing basecalled strand
        """

        N = scores.size(0)
        #setup backtracing matricies
        BT_index = np.zeros((N,self._H,self._L),dtype=int)#dtype=torch_get_index_dtype(self._H)) #index backtrace matrix
        BT_bases = np.zeros((N,self._H,self._L),dtype=int)#dtype=torch.uint8) #base value backtrace matrix
        C1 = ContextManager(N,self._H,self._global_hedge_state_init)
        C2 = ContextManager(N,self._H,self._global_hedge_state_init)
        current_C=C1
        other_C=C2
        scores_gpu= scores.to(self._device)

        #setup forward arrays
        F=self._scorer.init_initial_state_F(scores_gpu,reverse) #initialize the state corresponding to the initial valid state of the trellis
        temp_f_outgoing = torch.full((N,F.size(1),self._H,2**self._max_bits),Log.zero,device=self._device)
        
        
        current_scores = torch.full((N,self._H),Log.zero)
        """
        Perform core algorithm.
        1. iterate over length of strand we are guessing
        2. in each iteration we need to visit each trellis state in _H
        3. Calculate all valid edges into _H
        4. Calculate scores for edges into _H and take the max score, updating the state's C/BT/F matrices approriately
        """
        sub_length = self._full_message_length-self._L
        F=F.to(self._device)
        H_range=torch.arange(self._H)
        N_range=torch.arange(N)[:,None]
        pattern_counter=0
        accumulate_base_transition=torch.full((N,self._H,2**1,3*2),0,dtype=torch.int64)
        state_is_dead = torch.zeros((N,self._H),dtype=torch.uint8)
        for i in range(self._full_message_length-self._L,self._full_message_length):
            nbits = hedges_hooks.get_nbits(self._global_hedge_state_init,i)
            base_transition_outgoing=self.fill_base_transitions(N,self._H,2**nbits,current_C,nbits,reverse.numpy())
            pattern_range=pattern_counter*2
            accumulate_base_transition[:,:,:,pattern_range:pattern_range+2]=torch.stack([torch.zeros((N,self._H,2**1),dtype=torch.int64),torch.from_numpy(base_transition_outgoing).expand(-1,-1,2**self._max_bits)],dim=3)
            pattern_counter+=1            
            if nbits==0 and i<self._full_message_length-1 and not self._using_hedges_DNA_constraint:
                BT_index[:,:,i-sub_length] = np.arange(self._H)  #simply point to the same state
                BT_bases[:,:,i-sub_length] = base_transition_outgoing[:,:,-1] #set base back trace matrix
                other_C.const_update_context(current_C,BT_index,0,i-sub_length,nbits)
            else:
                trellis_incoming_indexes=self._trellis_connections[nbits] #Hx2^nbits matrix indicating incoming states from the previous time step
                trellis_incoming_value = self._trellis_transition_values[nbits]
                if i-sub_length==0:
                    forward_index_bases = torch.full((N,self._H),self._letter_to_index[self.fastforward_seq[-1]])[:,:,None].expand(-1,-1,2**nbits)
                    reverse_index_bases = torch.full((N,self._H),self._letter_to_index[complement(self.fastforward_seq)[-1]])[:,:,None].expand(-1,-1,2**nbits)
                    starting_bases=torch.where(reverse[:,None,None].expand(-1,self._H,2**nbits),reverse_index_bases,forward_index_bases)  
                else:
                    starting_bases = torch.from_numpy(BT_bases[:,:,i-sub_length-1-(pattern_counter-1)])[:,:,None].expand(-1,-1,2**nbits)

                state_transition_scores_outgoing = self._scorer.forward_step(scores_gpu,
                                                                                      accumulate_base_transition[:,:,:2**nbits,:pattern_range+2].to(self._device),
                                                                                      F,starting_bases.to(self._device),i,nbits,temp_f_outgoing,time_range_end.to(self._device))
                

                pattern_counter=0 #reset pattern counter
                #get incoming bases and scores coming in to each state so that the best one can be selected
                bases = base_transition_outgoing[:,trellis_incoming_indexes,trellis_incoming_value]#NxHx2^n matrix of bases to add
                mask = self.calculate_trellis_connections_mask(current_C,nbits,state_is_dead.numpy())
                state_scores = self.gather_trans_scores(state_transition_scores_outgoing,trellis_incoming_indexes,trellis_incoming_value)
                #masking allows us to effectively eliminate non-sensical scores for given contexts
                if not mask is None:
                    state_scores = torch.where(mask.to(self._device).bool()[None,:,:].expand(N,-1,-1),state_scores,state_scores.new_full(state_scores.size(),Log.zero))

                m,argmax_scores= torch.max(state_scores,dim=2) # NxH-length vectror indicating location of best score
                current_scores=m#state_scores.gather(2,argmax_scores[:,:,None])
                cpu_argmax_scores = argmax_scores.to("cpu")
                if not mask is None:    
                    state_is_dead=(m<=Log.zero).to(torch.uint8).to("cpu")
                #print(current_scores)
                #update back trace matrices
                BT_index[:,:,i-sub_length] = trellis_incoming_indexes[H_range,cpu_argmax_scores] #set the back trace index with best incoming state
                BT_bases[:,:,i-sub_length] = bases[N_range,H_range,cpu_argmax_scores] #set base back trace matrix
                #copy over F values
                F = self.get_new_F(temp_f_outgoing,self._device_trellis_connections[nbits],self._device_trellis_transition_values[nbits],argmax_scores,Fmem=F) 
                trellis_numpy=trellis_incoming_value.numpy()
                other_C.update_context(current_C,BT_index,trellis_numpy,i-sub_length,nbits)
            #swap contexts to make sure update happens properly
            t=current_C
            current_C=other_C
            other_C=t
        start_state = torch.argmax(current_scores,dim=1).to('cpu').numpy() 
        out_set=[]
        #print(current_scores)
        for i in range(N):
            seq =self.string_from_backtrace(BT_index[i,:,:],BT_bases[i,:,:],start_state[i])
            if reverse.numpy()[i]: out_set.append(self._fastforward_seq+complement(seq))  
            else: out_set.append(self._fastforward_seq+seq)   
        return out_set



class HedgesBonitoDelayStates(HedgesBonitoBase):
    _mask_cache={}
    def __init__(self, hedges_param_dict: dict, hedges_bytes: bytes, using_hedges_DNA_constraint: bool, alphabet: list, device: str, score: str,
                 window:int=0,mod_states:int=3) -> None:
        self._mod = mod_states #represents the number of states per history state
        super().__init__(hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint, alphabet, device, score,window=window)
        self._height = int(math.ceil(math.log2(self._mod)))
        self._mask = self._calculate_trellis_connections_mask(range(0,self._max_bits+1))
    
    def get_trellis_state_length(self,hedges_param_dict,using_hedges_DNA_constraint)->int:
        return 2**hedges_param_dict["prev_bits"]*self._mod
    
    def get_initial_trellis_index(self,global_hedge_state)->int:
        history_state = hedges_hooks.get_hedge_context_history(global_hedge_state)
        return history_state*self._mod

    def _calculate_trellis_connections_mask(self,bit_range:range)->list[torch.Tensor]:
        l = []
        cache_key = tuple([self._max_bits+1,self._H,self._mod]) 
        if cache_key in type(self)._mask_cache: return type(self)._mask_cache[cache_key]
        for nbits in bit_range:
            mask = torch.zeros((self._H,self._mod*2**nbits))
            for h in range(self._H):
                #break down the state into its mod and history component
                history = h//self._mod
                mod = h%self._mod
                if nbits==0: #if it is zero bits, state should only accept edge from itself
                    mask[h,mod]=1
                    continue
                #now, considering the history and the mod, determine which states the current state needs to source from, place 1's there 
                if mod==0:
                    value,incoming_states = hedges_hooks.get_incoming_states(self._global_hedge_state_init,nbits,history)
                    for s in range(len(incoming_states)):
                        for i in range(2**(self._height-1)-1,self._mod): mask[h,s*self._mod+i]=1
                else:
                    level = int(math.floor(math.log2(mod+1)))
                    #need to select the previous state and the mod from the previous state
                    base = mod - sum([2**i for i in range(0,level)])
                    mod_from_prev_state = base>>1
                    prev_state = base&0x1
                    mask[h,prev_state*self._mod+mod_from_prev_state]=1
            l.append(mask.bool())
        type(self)._mask_cache[cache_key]=l
        return l

    def calculate_trellis_connections_mask(self,context:ContextManager,nbits:int,dead_states:np.ndarray)->torch.Tensor|None:
        #should be able to calculate a static mask
        return self._mask[nbits]

    def calculate_trellis_connections(self, bit_range: range, trellis_states: int) -> tuple[list[torch.Tensor], ...]:
        cache_key = tuple([self._max_bits+1,self._H,self._mod]) 
        if cache_key in type(self)._trellis_connection_cache:
            return type(self)._trellis_connection_cache[cache_key]
        #trellis connections when considering additional mod states
        index_list=[]
        value_list=[]
        for nbits in bit_range:
            incoming_states_matrix=torch.full((trellis_states,(2**nbits)*self._mod),0,dtype=torch.int64)
            incoming_state_value_matrix=torch.full((trellis_states,(2**nbits)*self._mod),0,dtype=torch.int64)
            for h in range(trellis_states):
                history = h//self._mod
                value,incoming_states = hedges_hooks.get_incoming_states(self._global_hedge_state_init,nbits,history)
                for prev_index,s_in in incoming_states:
                    for m in range(self._mod):
                        prev_index_after_mod = prev_index*self._mod+m
                        incoming_states_matrix[h,prev_index_after_mod]=s_in*self._mod+m
                        incoming_state_value_matrix[h,prev_index_after_mod]=value
            index_list.append(incoming_states_matrix)
            value_list.append(incoming_state_value_matrix)
        type(self)._trellis_connection_cache
        return index_list,value_list
    


class HedgesBonitoBeam(HedgesBonitoBase):
    @property
    def is_beam(self):
        return True
    
    def __init__(self, hedges_param_dict: dict, hedges_bytes: bytes, using_hedges_DNA_constraint: bool, 
                 alphabet: list, device: str, score: str, beam: str = "beam_1") -> None:
        super().__init__(hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint, alphabet, device, score)
        self._omp_threads = 8
        self._list_size=1
        self._beam = beam
    def decode(self,scores:torch.Tensor,reverse:bool)->str:
        scores_cpu = scores.to("cpu")
        #launches beam viterbi decoding
        message_length = self._L
        #message_length=300
        gpu_scores = scores.to(self._device)
        out_seq = run_beam_1(int(math.log2(self._H)),self.get_initial_trellis_index(self._global_hedge_state_init),message_length,self._list_size,
                             scores_cpu.size(0),reverse,gpu_scores.data_ptr(),self._global_hedge_state_init,self._full_message_length-self._L,self._omp_threads) 
        out_seq = self.fastforward_seq+out_seq
        if reverse: out_seq=complement(out_seq)     
        return out_seq
