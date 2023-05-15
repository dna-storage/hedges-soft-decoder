from os import listdir
import torch
import math
import numpy as np
import json
import inspect
import bonito.cuda_utils as cu
from collections import namedtuple
import dnastorage.codec.hedges as hedges
import dnastorage.codec.hedges_hooks as hedges_hooks
import cupy as cp



import time


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

    def init_initial_state_F(self,scores:torch.Tensor,F:torch.Tensor)->None:
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

    def fill_base_transitions(self,base_transitions:torch.Tensor,C:list,nbits:int,reverse:bool)->None:
        """
        @brief      Fills in base_transitions with indexes representing the characters at this point in the message
        @param      base_transitions tensor holding indexes of bases
        @param      C list of contexts
        @param      nbits number of bits on this transition
        @return     None
        """
        H,transitions = base_transitions.size()
        for i in torch.arange(H,dtype=torch.int64):
            for j in torch.arange(transitions,dtype=torch.int64):
                if reverse:
                    next_base=self._letter_to_index[reverse_map[hedges_hooks.peek_context(C[i],nbits,j)]]
                else:
                    next_base=self._letter_to_index[hedges_hooks.peek_context(C[i],nbits,j)]
                base_transitions[i,j] = next_base

    def __init__(self,hedges_param_dict:dict,hedges_bytes:bytes,using_hedges_DNA_constraint:bool,alphabet:list) -> None:
        self._global_hedge_state_init = hedges_hooks.make_hedge( hedges.hedges_state(**hedges_param_dict)) #stores pointer to a hedges state object
        print("Bytes {}".format(hedges_bytes))
        self._fastforward_seq=hedges_hooks.fastforward_context(bytes(hedges_bytes),self._global_hedge_state_init) #modifies the global hedge state to reflect state at end of hedges_bytes
        print("Fastforward sequence {}".format(self._fastforward_seq))
        self._H = self.get_trellis_state_length(hedges_param_dict,using_hedges_DNA_constraint) #length of history side of matrices
        self._full_message_length = hedges_hooks.get_max_index(self._global_hedge_state_init) #total message length
        self._L = self._full_message_length - len(self._fastforward_seq)#length of message-length side of matrices
        self._alphabet = alphabet #alphabet we are using
        self._letter_to_index = {_:i for i,_ in enumerate(self._alphabet)} #reverse map for the alphabet
        self._trellis_connections=[]
        self._trellis_transition_values=[]

    def string_from_backtrace(self,BT_index:torch.Tensor,BT_bases:torch.Tensor,start_state:int)->str:
        H,L = BT_index.size()
        current_state=torch.tensor(start_state)
        return_sequence=""
        for i in torch.flip(torch.arange(L,dtype=torch.int64),dims=(0,)):
            #print(current_state)
            return_sequence+=self._alphabet[int(BT_bases[current_state,i])]
            if i==0: break
            current_state=BT_index[current_state,i]
        return return_sequence[::-1]
    
    def decode(self,scores:torch.Tensor,reverse:bool)->str:
        """
        @brief      Core algorithm for implementing hedges viterbi decoding

        @param      scores tensor that reflect probabilites of characters

        @return     string representing basecalled strand
        """
        self._T = scores.size(0) #time dimension 
        #setup backtracing matricies
        BT_index = torch.full((self._H,self._L),0,dtype=torch.int64)#dtype=torch_get_index_dtype(self._H)) #index backtrace matrix
        BT_bases = torch.full((self._H,self._L),0,dtype=torch.int64)#dtype=torch.uint8) #base value backtrace matrix
        C1 = [hedges_hooks.make_context(self._global_hedge_state_init) for _ in range(0,self._H)]
        C2 = [hedges_hooks.make_context(self._global_hedge_state_init) for _ in range(0,self._H)]
        current_C=C1
        other_C=C2


        #setup forward arrays
        F = torch.full((self._T,self._H),Log.zero,dtype=torch.float32)
        self.init_initial_state_F(scores,F) #initialize the state corresponding to the initial valid state of the trellis
        current_scores = torch.full((self._H,),Log.zero)
        """
        Perform core algorithm.
        1. iterate over length of strand we are guessing
        2. in each iteration we need to visit each trellis state in _H
        3. Calculate all valid edges into _H
        4. Calculate scores for edges into _H and take the max score, updating the state's C/BT/F matrices approriately
        """
        sub_length = self._full_message_length-self._L
        print(F.size())
        for i in range(self._full_message_length-self._L,self._full_message_length):
            print(i)
            loop_time=time.time()
            nbits = hedges_hooks.get_nbits(self._global_hedge_state_init,i)
            trellis_incoming_indexes=self._trellis_connections[nbits] #Hx2^nbits matrix indicating incoming states from the previous time step
            trellis_incoming_value = self._trellis_transition_values[nbits]
            base_transition_outgoing = torch.full((self._H,2**nbits),0,dtype=torch.int64)
            self.fill_base_transitions(base_transition_outgoing,current_C,nbits,reverse)
            if i-sub_length==0:
                starting_bases = torch.full((self._H,),self._letter_to_index[self.fastforward_seq[-1]])[:,None].expand(-1,2**nbits)
            else:
                starting_bases = BT_bases[:,i-sub_length-1][:,None].expand(-1,2**nbits)

            init_time=time.time()
            print("Top loop init time {}".format(init_time-loop_time))
            state_transition_scores_outgoing, temp_f_outgoing = self.forward_step(scores,base_transition_outgoing,F,starting_bases,i,nbits)
            forward_step_time = time.time()
            print("Step time {}".format(forward_step_time-init_time))
            #get incoming bases and scores coming in to each state so that the best one can be selected
            state_scores = state_transition_scores_outgoing[trellis_incoming_indexes,trellis_incoming_value] #should produce Hx2^n matrix of scores that need to be compared
            bases = base_transition_outgoing[trellis_incoming_indexes,trellis_incoming_value]#Hx2^n matrix of bases to add
            value_of_max_scores= torch.argmax(state_scores,dim=1) # H-length vectror indicating location of best score
            current_scores=state_scores.gather(1,value_of_max_scores[:,None])

            #update back trace matrices
            BT_index[:,i-sub_length] = trellis_incoming_indexes[torch.arange(self._H),value_of_max_scores] #set the back trace index with best incoming state
            BT_bases[:,i-sub_length] = bases[torch.arange(self._H),value_of_max_scores] #set base back trace matrix

            back_trace_time = time.time()
            print("BT time {}".format(back_trace_time-forward_step_time))
            #update forward arrays
            incoming_F = temp_f_outgoing[:,trellis_incoming_indexes,trellis_incoming_value]
            F = incoming_F[:,torch.arange(self._H),value_of_max_scores]
            f_time = time.time()
            print("F update time {}".format(f_time-back_trace_time)) 
            
            for r in torch.arange(self._H):
                state = BT_index[r,i-sub_length]
                val = trellis_incoming_value[r,0]
                hedges_hooks.update_context(other_C[r],current_C[state],nbits,val)
            #swap contexts to make sure update happens properly
            t=current_C
            current_C=other_C
            other_C=t
            end_loop_time=time.time()
            c_time = time.time()
            print("C time {}".format(c_time-f_time))
            print("End loop time {}".format(end_loop_time-forward_step_time))
            
        print(current_scores)
        for x in range(current_scores.size(0)):
            print("{}:{}".format(x,self.string_from_backtrace(BT_index,BT_bases,x)))
            print("{}:{}".format(x,current_scores[x,0]))
        start_state = int(torch.argmax(current_scores))
        print("start state {}".format(start_state))
        out_seq = self.string_from_backtrace(BT_index,BT_bases,start_state)
        if reverse: out_seq=complement(out_seq)        
        #return self._fastforward_seq+out_seq
        return out_seq
            

class HedgesBonitoCTC(HedgesBonitoBase):
    """
    @brief      Hedges decoding on Bonito CTC output

    @details    Implements the necessary methods to extract information out of Bonito scores when using the Bonito CTC model
    """
    def get_trellis_state_length(self,hedges_param_dict,using_hedges_DNA_constraint)->int:
        return 2**hedges_param_dict["prev_bits"]
    
    def get_initial_trellis_index(self,global_hedge_state)->int:
        return 0

    @classmethod
    def _fwd_algorithm(cls,target_scores:torch.Tensor,mask:torch.Tensor,
                       F:torch.Tensor,lower_t_range:int,upper_t_range:int)->torch.Tensor:
        running_alpha_index=2
        T,H,E,L=target_scores.size()
        alpha_t = torch.full((T,H,E),Log.zero)
        running_alpha =torch.full((H,E,L+2),Log.zero)
        log_zeros = torch.full((H,E,L),Log.zero)
        print("zeros {}".format(log_zeros.size(0)*log_zeros.size(1)))
        results_stack=torch.full((3,running_alpha.size(0),running_alpha.size(1),running_alpha.size(2)-running_alpha_index),Log.zero)
        for t in torch.arange(lower_t_range,upper_t_range):
            loop_start_time=time.time()
            running_alpha[:,:,running_alpha_index-1] = F[t-1,:][:,None]
            slice_time=time.time()
            #print("slice data time {}".format(slice_time-loop_start_time))
            results_stack[0,:,:,:] = running_alpha[:,:,running_alpha_index:]
            results_stack[1,:,:,:] = running_alpha[:,:,running_alpha_index-1:-1]
            stack_time=time.time()
            #print("stack time {}".format(loop_start_time-stack_time))
            results_stack[2,:,:,:] = torch.where(mask,log_zeros,running_alpha[:,:,running_alpha_index-2:-2])
            running_alpha[:,:,running_alpha_index:]= Log.mul(target_scores[t,:,:,:],Log.sum(results_stack,dim=0))
            alpha_time=time.time()
            #print("alpha_time {}".format(alpha_time-loop_start_time))
            alpha_t[t,:,:]=running_alpha[:,:,-1]
        return alpha_t


    @classmethod
    def string_to_indexes(cls,seq:str,letter_to_index:dict)->torch.Tensor:
        ret_tensor = torch.zeros((len(seq),))
        for i,s in enumerate(seq):
            ret_tensor[i]=letter_to_index[s]
        return ret_tensor
    
    @classmethod
    def insert_blanks(cls,seq:torch.Tensor)->torch.Tensor:
        #blanks should be index 0 in the alphabet
        L = seq.size(0)
        ret_tensor = torch.zeros((L*2+1,),dtype=torch.int64)
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
        start=time.time()
        T,A = scores.size()
        H,E = base_transitions.size()
        #need to create a Hx2^nbitsxL tensor to represent all strings we are calculating alphas for
        targets = torch.stack([initial_bases,scores.new_zeros((H,E),dtype=torch.int64),base_transitions],dim=2)
        _,_2,L = targets.size()
        targets=targets[None,:,:,:].expand(T,-1,-1,-1) #expand the targets along the time dimension 
        target_scores = torch.gather(scores[:,None,None,:].expand(-1,H,E,-1),3,targets) #gather in the scores for the targets
        mask = torch.nn.functional.pad(targets[0,:,:,2:]==targets[0,:,:,:-2],(1,0),value=1)
        target_scores=target_scores[:,:,:,1:]
        #calculate valid ranges of t to avoid unnecessary iterations
        lower_t_range=strand_index
        upper_t_range=T-self._full_message_length+strand_index+1
        init_time=time.time()
        print("Time to init fwd {}".format(init_time-start))
        alpha_t = self._fwd_algorithm(target_scores,mask,F,lower_t_range,upper_t_range)        
        fwd_time=time.time()
        log_prob_no_transition=torch.nn.functional.pad(
            Log.sum(torch.stack([torch.zeros((T,H,E)),target_scores[:,:,:,-1]]),dim=0), 
            (0,0,0,0,0,1),value=Log.one)[1:,:,:] #TxHx2^nbits tensor to help with final score calculation
        out_scores = dot(log_prob_no_transition,alpha_t[:,:,:],dim=0) #Hx2^nbits output
        score_time=time.time()
        print("Time to calculate dot product {}".format(score_time-fwd_time))
        return out_scores,alpha_t[:,:,:]
        
    def __init__(self, hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint,alphabet) -> None:
            super().__init__(hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint,alphabet)
            self._trellis_connections,self._trellis_transition_values = self.calculate_trellis_connections(range(0,3),self._H) #list of matrices with trellis connections for different points in the codeword
         
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
    def __init__(self, hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint, alphabet) -> None:
        super().__init__(hedges_param_dict, hedges_bytes, using_hedges_DNA_constraint, alphabet)
        assert torch.cuda.is_available() #make sure we have cuda for this class
      

    @classmethod
    def _fwd_algorithm(cls, target_scores: torch.Tensor,mask: torch.Tensor,
                       F: torch.Tensor, lower_t_range: int, upper_t_range: int)->torch.Tensor:
        kernel=cu.load_cupy_func("cuda/ctc_fwd.cu","fwd_logspace",FLOAT='float',SUM='logsumexp3',MUL='add',ZERO='{:E}'.format(Log.zero),ONE='{:E}'.format(Log.one))
        T,H,E,L=target_scores.size()
        alpha_t = torch.full((T,H,E),Log.zero)
        #convert mask from bools to floats to avoid control flow in GPU kernel
        mask = torch.where(mask,torch.full(mask.size(),Log.zero),torch.full(mask.size(),Log.one))
        kernel_start_time=time.time()
        x = target_scores.to("cuda:0")
        y=alpha_t.to("cuda:0")
        z=mask.to("cuda:0")
        w=F.to("cuda:0")
        with cp.cuda.Device(0):
            kernel(grid=(1,1,1),block=(L,H,E),shared_mem=2*4*(L+2)*H*E,args=(x.data_ptr(),y.data_ptr(),
                                                                               z.data_ptr(),w.data_ptr(),lower_t_range,
                                                                               upper_t_range,H,E,L,T,2))                   
        print("Kernel run time {}".format(time.time()-kernel_start_time))
        
        return y.to("cpu")
        
class Align:
    def __init__(self,alphabet:list) -> None:
        self._alphabet=alphabet
        self._letter_to_index = {_:i for i,_ in enumerate(self._alphabet)} #reverse map for the alphabet
        pass

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
        target_indexes = HedgesBonitoCTC.string_to_indexes(seq,self._letter_to_index)
        target_indexes = HedgesBonitoCTC.insert_blanks(target_indexes)
        BT = torch.full((T,target_indexes.size(0)),0,dtype=torch.int64) #backtrace to know where alignment ranges over T
        F = torch.full((T,target_indexes.size(0)+1),Log.zero) #forward calculating Trellis
        target_indexes = target_indexes[None,:].expand(T,-1)
        emission_scores = scores.gather(1,target_indexes)
        emission_scores=torch.nn.functional.pad(emission_scores,(1,0),value=Log.one)
        running_alpha = torch.full((emission_scores.size(1)+2,),Log.zero)
        running_alpha[2]=Log.one
        mask = target_indexes[0,:-2]==target_indexes[0,2:]
        mask=torch.nn.functional.pad(mask,(3,0),value=1)
        r = torch.arange(1,emission_scores.size(1))
        zeros= torch.full((emission_scores.size(1),),Log.zero)
        for t in torch.arange(T):
            stay = Log.mul(running_alpha[2:],emission_scores[t,:])
            previous = Log.mul(running_alpha[1:-1],emission_scores[t,:])
            previous_previous = Log.mul(torch.where(mask,zeros,running_alpha[:-2]),emission_scores[t,:])
            F[t,:],arg_max = Max.sum(torch.stack([stay,previous,previous_previous]))
            running_alpha[2:]=F[t,:]
            BT[t,:]=r-arg_max[1:]

        lower,upper = self.get_index_range(BT,F)
        return lower,upper,torch.max(F[1:,-1])
    
    
class AlignCTC(Align):
    def __init__(self,alphabet:list) -> None:
        super().__init__(alphabet)
    def align(self, scores: torch.Tensor, seq: str) -> tuple[int, int, torch.Tensor]:
        return super().align(scores, seq)



   
def check_hedges_params(hedges_params_dict)->None:
    """
    @brief      Checks parameters in dictionary to make sure they line up with class

    @param      hedges_params_dict  dictionary of parameters

    @return     No return value, raise exception if error
    """
    args = inspect.signature(hedges.hedges_state.__init__)
    for param in hedges_params_dict:
        if param not in args.parameters: raise KeyError("{} not legal".format(param))
        



def hedges_decode(scores:torch.Tensor,hedges_params:str,hedges_bytes:bytes,
                  using_hedges_DNA_constraint:bool,alphabet:list,stride=1,
                  endpoint_seq:str="")->dict:
    """
    @brief      Top level function for decoding CTC-style outputes to hedges strands

    @details    Generates a base-call that should be a strand that satisfies the given hedges code

    @param      scores  Log-probabilities for bases at a particular point in the signal
    @param      hedges_params file that contains parameters for the hedges code
    @param      hedges_bytes optional string of bytes to fast forward the decode process to
    @param      using_hedges_DNA_cosntraint Boolean when True uses DNA constraint information of the hedges code in the trellis
    @param      stride parameter used to satisfy interface, has no purpose at the moment

    @return     Dictionary with entries related to the output seqeunce
    """

    print("IN HEDGES DECODE")
    
    assert(hedges_params!=None and hedges_bytes!=None)
    
    try:
        hedges_params_dict = json.load(open(hedges_params,'r'))
        check_hedges_params(hedges_params_dict)
    except Exception as e:
        print(e)
        exit(1)

    decoder = HedgesBonitoCTCGPU(hedges_params_dict,hedges_bytes,using_hedges_DNA_constraint,alphabet)
    #create aligner
    aligner = AlignCTC(alphabet)
    
    f_endpoint_upper_index=0
    r_endpoint_lower_index=len(scores)
    f_endpoint_score=Log.one
    r_endpoint_score=Log.one
    if len(endpoint_seq)>0:
        f_endpoint_lower_index,f_endpoint_upper_index,f_endpoint_score  = aligner.align(scores,endpoint_seq)
        r_endpoint_lower_index,r_endpoint_upper_index,r_endpoint_score  = aligner.align(scores,reverse_complement(endpoint_seq))

    f_hedges_bytes_lower_index,f_hedges_bytes_upper_index,f_hedges_score = aligner.align(scores,decoder.fastforward_seq[::-1])
    r_hedges_bytes_lower_index,r_hedges_bytes_upper_index,r_hedges_score = aligner.align(scores,complement(decoder.fastforward_seq))
    
    """
    We need to rearrange the scores based on alignments so that index is always at the beginning of the strand.
    Because we
    """
    seq=""
    print("hedges f score {}".format(f_hedges_score))
    print("endpoint f score {}".format(f_endpoint_score))
    print("hedges r score {}".format(r_hedges_score))
    print("endpoint r score {}".format(r_endpoint_score))
    if Log.mul(f_hedges_score,f_endpoint_score)>Log.mul(r_endpoint_score,r_hedges_score):
        print("IS FORWARD")     
        s=scores[f_endpoint_upper_index:f_hedges_bytes_upper_index]
        print(" {} {}".format(f_endpoint_upper_index,f_hedges_bytes_upper_index))
        s=s.flip([0])
        complement_trellis=False
        decoder.fastforward_seq = decoder.fastforward_seq[::-1] 
        seq = decoder.decode(s,complement_trellis)
    else:
        print("IS REVERSE")
        s=scores[r_hedges_bytes_lower_index:r_endpoint_lower_index]
        print("heges lower upper {} {}".format(r_hedges_bytes_lower_index,r_hedges_bytes_upper_index))
        print("{} {}".format(r_hedges_bytes_lower_index,r_endpoint_lower_index))
        complement_trellis=True
        decoder.fastforward_seq = complement(decoder.fastforward_seq)
        seq = decoder.decode(s,complement_trellis)

    return {'sequence':seq,'qstring':"*"*len(seq),'stride':stride,'moves':seq}

  
