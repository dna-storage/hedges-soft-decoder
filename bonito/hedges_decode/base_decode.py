import os
import torch
import numpy as np
from collections import namedtuple
import dnastorage.codec.hedges as hedges
import dnastorage.codec.hedges_hooks as hedges_hooks
import math 
from bonito.hedges_decode.context_utils import ContextManager
import bonito.hedges_decode.context_utils as context_utils


#get env variables
PLOT = os.getenv("PLOT",False)

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
                                   


def torch_get_index_dtype(states)->torch.dtype:
    num_bits = int(math.ceil(math.log2(states)))
    if num_bits<=8:
        return torch.uint8
    elif num_bits<=15: #int16 is signed, so assume last bit is sign
        return torch.int16
    else:
        return torch.int32

class HedgesBonitoBase:
    """
    @brief      Base class for Hedges Bonito decoding

    @details    Base class for decoding CTC-type outputs of models used for basecalling nanopore signals, provides general functionality of implementing a CTC-based branch metric
    """
    def get_trellis_state_length(self,hedges_param_dict:dict,using_hedges_DNA_cosntraints:bool)->int:
        raise NotImplementedError()

    def get_initial_trellis_index(self,global_hedge_state:int)->int:
        raise NotImplementedError()

    def init_initial_state_F(self,scores:torch.Tensor)->torch.Tensor:
        raise NotImplementedError()

    def forward_step(self,scores:torch.Tensor,base_transitions:torch.Tensor,F:torch.Tensor,initial_bases:torch.Tensor,strand_index:int,
                     nbits:int)->tuple[torch.Tensor,torch.Tensor]:
        raise NotImplementedError()
    
    def calculate_trellis_connections(self, bit_range:range, trellis_states:int) -> tuple[list[torch.Tensor],...]:
        raise NotImplementedError()

    @property 
    def fastforward_seq(self):
        return self._fastforward_seq

    @fastforward_seq.setter
    def fastforward_seq(self,s):
        self._fastforward_seq=s

    def fill_base_transitions(self,H:int,transitions:int,C:ContextManager,nbits:int,reverse:bool)->np.ndarray:
        """
        @brief      Fills in base_transitions with indexes representing the characters at this point in the message
        @param      base_transitions tensor holding indexes of bases
        @param      C list of contexts
        @param      nbits number of bits on this transition
        @return     None
        """
        return context_utils.fill_base_transitions(H,transitions,C,nbits,reverse,self._letter_to_index)

    def __init__(self,hedges_param_dict:dict,hedges_bytes:bytes,using_hedges_DNA_constraint:bool,alphabet:list,device) -> None:
        self._global_hedge_state_init = hedges_hooks.make_hedge( hedges.hedges_state(**hedges_param_dict)) #stores pointer to a hedges state object
        self._fastforward_seq=hedges_hooks.fastforward_context(bytes(hedges_bytes),self._global_hedge_state_init) #modifies the global hedge state to reflect state at end of hedges_bytes
        self._H = self.get_trellis_state_length(hedges_param_dict,using_hedges_DNA_constraint) #length of history side of matrices
        self._full_message_length = hedges_hooks.get_max_index(self._global_hedge_state_init) #total message length
        self._L = self._full_message_length - len(self._fastforward_seq)#length of message-length side of matrices
        self._alphabet = alphabet #alphabet we are using
        self._letter_to_index = {_:i for i,_ in enumerate(self._alphabet)} #reverse map for the alphabet
        self._trellis_connections=[]
        self._trellis_transition_values=[]
        self._max_bits=1 #max number of bits per base
        self._device=device
        
    def string_from_backtrace(self,BT_index:np.ndarray,BT_bases:np.ndarray,start_state:int)->str:
        H,L = BT_index.shape
        current_state=torch.tensor(start_state)
        return_sequence=""
        for i in torch.flip(torch.arange(L,dtype=torch.int64),dims=(0,)):
            return_sequence+=self._alphabet[int(BT_bases[current_state,i])]
            if i==0: break
            current_state=BT_index[current_state,i]
        return return_sequence[::-1]

    @profile
    def decode(self,scores:torch.Tensor,reverse:bool)->str:
        """
        @brief      Core algorithm for implementing hedges viterbi decoding

        @param      scores tensor that reflect probabilites of characters

        @return     string representing basecalled strand
        """
        self._T = scores.size(0) #time dimension 

        #setup backtracing matricies
        BT_index = np.zeros((self._H,self._L),dtype=int)#dtype=torch_get_index_dtype(self._H)) #index backtrace matrix
        BT_bases = np.zeros((self._H,self._L),dtype=int)#dtype=torch.uint8) #base value backtrace matrix
        C1 = ContextManager(self._H)
        C2 = ContextManager(self._H)
        current_C=C1
        other_C=C2

        #setup forward arrays
        F=self.init_initial_state_F(scores) #initialize the state corresponding to the initial valid state of the trellis
        current_scores = torch.full((self._H,),Log.zero)
        """
        Perform core algorithm.
        1. iterate over length of strand we are guessing
        2. in each iteration we need to visit each trellis state in _H
        3. Calculate all valid edges into _H
        4. Calculate scores for edges into _H and take the max score, updating the state's C/BT/F matrices approriately
        """
        sub_length = self._full_message_length-self._L
        scores_gpu= scores.to(self._device)
        F=F.to(self._device)
        H_range=torch.arange(self._H)
        pattern_counter=0
        accumulate_base_transition=torch.full((self._H,2**1,3*2),0,dtype=torch.int64)
        #print("Real Scores size {}".format(self._T))
        for i in range(self._full_message_length-self._L,self._full_message_length):
            #print(i)
            nbits = hedges_hooks.get_nbits(self._global_hedge_state_init,i)
            base_transition_outgoing=self.fill_base_transitions(self._H,2**nbits,current_C,nbits,reverse)
            pattern_range=pattern_counter*2
            accumulate_base_transition[:,:,pattern_range:pattern_range+2]=torch.stack([torch.zeros((self._H,2**1),dtype=torch.int64),                                                                             torch.from_numpy(base_transition_outgoing).expand(-1,2**self._max_bits)],dim=2)
            pattern_counter+=1            
            if nbits==0 and i<self._full_message_length-1:
                BT_index[:,i-sub_length] = np.arange(self._H)  #simply point to the same state
                BT_bases[:,i-sub_length] = base_transition_outgoing[:,-1] #set base back trace matrix
                other_C.update_context(current_C,BT_index,0,i-sub_length,nbits)
            else:
                
                trellis_incoming_indexes=self._trellis_connections[nbits] #Hx2^nbits matrix indicating incoming states from the previous time step
                trellis_incoming_value = self._trellis_transition_values[nbits]
                if i-sub_length==0:
                    starting_bases = torch.full((self._H,),self._letter_to_index[self.fastforward_seq[-1]])[:,None].expand(-1,2**nbits)
                else:
                    starting_bases = torch.from_numpy(BT_bases[:,i-sub_length-1-(pattern_counter-1)])[:,None].expand(-1,2**nbits)

                state_transition_scores_outgoing, temp_f_outgoing = self.forward_step(scores_gpu,
                                                                                      accumulate_base_transition[:,:2**nbits,:pattern_range+2].to(self._device),
                                                                                      F,starting_bases.to(self._device),i,nbits)
                pattern_counter=0 #reset pattern counter
                #get incoming bases and scores coming in to each state so that the best one can be selected
                state_scores = state_transition_scores_outgoing[trellis_incoming_indexes,trellis_incoming_value] #should produce Hx2^n matrix of scores that need to be compared
                bases = base_transition_outgoing[trellis_incoming_indexes,trellis_incoming_value]#Hx2^n matrix of bases to add
                value_of_max_scores= torch.argmax(state_scores,dim=1) # H-length vectror indicating location of best score
                current_scores=state_scores.gather(1,value_of_max_scores[:,None])

                cpu_value_of_max_scores = value_of_max_scores.to("cpu")
                #update back trace matrices
                BT_index[:,i-sub_length] = trellis_incoming_indexes[H_range,cpu_value_of_max_scores] #set the back trace index with best incoming state
                BT_bases[:,i-sub_length] = bases[H_range,cpu_value_of_max_scores] #set base back trace matrix

                #update forward arrays
                incoming_F = temp_f_outgoing[:,trellis_incoming_indexes,trellis_incoming_value]
                F = incoming_F[:,H_range,value_of_max_scores]
                trellis_numpy=trellis_incoming_value.numpy()
                other_C.update_context(current_C,BT_index,trellis_numpy,i-sub_length,nbits)
            #swap contexts to make sure update happens properly
            t=current_C
            current_C=other_C
            other_C=t
        start_state = int(torch.argmax(current_scores))
        out_seq = self.fastforward_seq+self.string_from_backtrace(BT_index,BT_bases,start_state)
        if reverse: out_seq=complement(out_seq)     
        return out_seq
