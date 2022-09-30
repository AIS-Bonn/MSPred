import torch


class SequenceDataset(torch.utils.data.Dataset):
    """ Abstract base class for all used Datasets """

    NUM_HMAP_CHANNELS = NotImplemented
    STRUCT_TYPE = NotImplemented

    def get_num_hmap_channels(self):
        return self.NUM_HMAP_CHANNELS

    def get_struct_type(self):
        return self.STRUCT_TYPE

    def get_heatmap_weights(self, w_easy_kpts=1.0, w_hard_kpts=1.0):
        return []
