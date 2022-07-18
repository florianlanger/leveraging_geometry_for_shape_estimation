from shutil import register_unpack_format
import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data.sampler import Sampler
import numpy as np

from scipy.spatial.transform import Rotation as scipy_rot



def get_index_infos(outputs,extra_infos,config,n_refinement):
    outputs = outputs.cpu().detach().numpy()

    list_of_dicts = []
    for i in range(len(outputs)):
        single_dict = {}
        for key in extra_infos:
            if key == 'T':
                single_dict[key] = extra_infos[key][i] + outputs[i][1:4]
            elif key == 'offset_t':
                single_dict[key] = extra_infos[key][i] - outputs[i][1:4]
            elif key == 'S':
                # single_dict[key] = extra_infos[key][i] * (outputs[i][4:7] + 1)
                single_dict[key] = extra_infos[key][i] + outputs[i][4:7]
            elif key == 'offset_s':
            
                single_dict[key] =  extra_infos['S_gt'][i] - (extra_infos['S'][i] + outputs[i][4:7])

            elif key == 'R':
                # offset_r = outputs[i][7:16].reshape(3,3)
                if 'R' in config["data"]["sample_what"]:
                    offset_r = scipy_rot.from_quat(outputs[i][7:11]).as_matrix()
                    single_dict[key] = np.matmul(extra_infos[key][i],offset_r)
                
                # if dont sample and learn R just keep gt R, so that dont add on existing augmentations 
                else:
                    single_dict[key] = np.array(extra_infos['R_gt'][i])


            elif key == 'offset_r':
                if 'R' in config["data"]["sample_what"]:
                    # offset_r_new = outputs[i][7:16].reshape(3,3)
                    # inv_offset_r_new = np.linalg.inv(offset_r_new)
                    # previous_offset = extra_infos['offset_r'][i].reshape(3,3)
                    # single_dict[key] = np.matmul(inv_offset_r_new,previous_offset).reshape(9)
                    offset_r_new = scipy_rot.from_quat(outputs[i][7:11])
                    previous_offset = scipy_rot.from_quat(extra_infos['offset_r'][i])
                    out = (offset_r_new.inv() * previous_offset).as_quat()
                    if out[3] < 0:
                        out = out * -1
                    single_dict[key] = out
                else:
                    single_dict[key] = np.array([0,0,0,1.])

            # elif key == 'history':
            #     start_index = (n_refinement + 1) * 3
            #     history = extra_infos['history'][i]
            #     R = scipy_rot.from_matrix(single_dict['R']).as_quat()
            #     history[start_index + 0,3:7] = R
            #     history[start_index + 1,3:6] = single_dict['T']
            #     history[start_index + 2,3:6] = single_dict['S']
            #     single_dict['history'] = history

                # print('history',history)

            else:
                single_dict[key] = extra_infos[key][i]
        list_of_dicts.append(single_dict)

    return list_of_dicts


def get_batch_from_dataset(dataset,indices,sample_just_classifier):
    batch = get_simple_batch_from_dataset(dataset,indices,sample_just_classifier)

    padded_info_all = torch.stack([item[0] for item in batch])
    target_all = torch.stack([item[1] for item in batch])

    extra_infos_all = {}
    for key in batch[0][2]:
        extra_infos_all[key] = np.stack([item[2][key] for item in batch])
    
    return padded_info_all,target_all,extra_infos_all

def get_simple_batch_from_dataset(dataset,indices,sample_just_classifier):
    batch = []

    for counter,i in enumerate(indices):
        # debug here replace output with 0 s 
        tuple_index_just_classifier = (i,sample_just_classifier)
        batch.append(dataset[tuple_index_just_classifier])
    return batch

class SimpleSampler(Sampler[int]):

    def __init__(self, indices) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return 1000000

    def invert_order(self):
        self.indices = self.indices[::-1]

class SequentialSampler_custom(Sampler[int]):

    def __init__(self, repeats,n_examples) -> None:
        self.indices = np.arange(n_examples)
        self.repeats = repeats
        self.n_examples = n_examples
        self.shuffle_indices()
        print('repeats',repeats)
        print('n_examples',n_examples)

    def __iter__(self) -> Iterator[int]:

        index_dicts = [{'index': i} for i in self.indices_repeated]
        # return iter(index_dicts)
        return iter(self.indices_repeated)

    def __len__(self) -> int:
        return self.n_examples

    def shuffle_indices(self):
        np.random.shuffle(self.indices)
        self.indices_repeated = np.repeat(self.indices, repeats=self.repeats)


class BatchSampler_repeat(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
    
        for idx in self.sampler:
            batch = [idx] * self.batch_size
            yield batch
        #         batch = []
        # if len(batch) > 0 and not self.drop_last:
        #     yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

if __name__ == '__main__':
    sampler = SequentialSampler_custom(repeats=10, n_examples=4)
    for i in sampler:
        print(i)
    print(len(sampler))