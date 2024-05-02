import torch
import numpy as np
import os


alphabet = ["N","A","C","G","T"] #TODO: should confirm this
alphabet_num_map = {"N":0,"A":1,"C":2,"G":3,"T":4}


def reverse_strand(s:str):
    return_string = ""
    d = {"A":"T","T":"A","C":"G","G":"C"}
    for i in s:
        return_string = return_string+d[i]
    return return_string

def CTC_to_numpy(s:str): #converts alignment string to a numpy array #TODO: maybe won't need this 
    return_array = np.zeros((len(s),),dtype=np.uint8)
    for index,i in enumerate(s):
        return_array[index]=alphabet_num_map[i]
    return return_array

def num_to_string(c:np.ndarray):
    ctc_string=""
    for i in c:
        ctc_string+=alphabet[i]
    return ctc_string


def CTC_decode(s:str):
    #decode input ctc string s
    previous = None
    decoded_string=""
    for index,i in enumerate(s):
        if previous is None and i!="N":
            decoded_string+=i
            previous=i
        elif i!=previous and i!="N":
            decoded_string+=i
            previous=i
        elif i!=previous and i=="N":
            previous=i
    return decoded_string


def string_to_num(s:str):
    nums=[]
    for i in s:
        nums.append(alphabet_num_map[i])
    return np.array(nums)

    
