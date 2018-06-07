from abc import ABC,abstractmethod


class LabelsConverter(ABC):
    @abstractmethod
    def convert_labels(self,labels):
        ...

    @abstractmethod
    def get_num_outputs(self):
        ...

class BinLabelsConverter(LabelsConverter):

    def get_num_outputs(self):
        return self.num_bits

    def __init__(self,num_bits=8):
        self.num_bits=num_bits

    @staticmethod
    def append_lead_zeros(str_num:str,num_bits:int):
        l=len(str_num)
        lead_str='0'*(num_bits-l)
        return lead_str+str_num

    def dec_to_binary(self,dec_labels,num_bits=8):
        bin_labels=[]
        for step_label in dec_labels:
            bin_str=bin(step_label)[2:]
            bin_str=BinLabelsConverter.append_lead_zeros(bin_str,num_bits)
            arr=[int(x) for x in bin_str]
            bin_labels.append(arr)
        return bin_labels

    def convert_labels(self,labels):
        return self.dec_to_binary(labels,self.num_bits)

if __name__=='__main__':
    bc=BinLabelsConverter()
    b=bc.dec_to_binary([10,9])
    print(b)