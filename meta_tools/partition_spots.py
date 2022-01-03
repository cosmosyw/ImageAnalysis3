# required packages
import os
import time
import numpy as np
import pandas as pd
# required internal functions
from ..classes.preprocess import Spots3D
from ..io_tools.crop import generate_neighboring_crop


default_search_radius = 10
default_pixel_sizes = [250,108,108]

######################################################################
# Notes:
# 1. gene_readout_file must have columns: 'Bit number' and 'Gene' (or any given query label)
#
######################################################################

class Spots_Partition():
    """"""
    def __init__(self,
                 segmentation_masks:np.ndarray, 
                 readout_filename:str,
                 fov_id=None,
                 search_radius=default_search_radius,
                 pixel_sizes=default_pixel_sizes,
                 save_filename=None,
                 ):
        print("- Partition spots")
        # localize segmentation_masks
        self.segmentation_masks = np.array(segmentation_masks, dtype=np.int16)
        self.image_size = np.shape(segmentation_masks)
        self.fov_id=fov_id
        self.radius = int(search_radius)
        self.pixel_sizes = pixel_sizes
        # filenames
        self.readout_filename = readout_filename
        self.save_filename = save_filename
    
    def run(self, spots_list, bits=None,
            query_label='Gene', 
            save=True,
            overwrite=False, verbose=True):
        if os.path.exists(self.save_filename) and not overwrite:
            print(f"-- directly load from file: {self.save_filename}")
            _count_df = pd.read_csv(self.save_filename, header=0)
            setattr(self, f"{query_label.lower()}_count_df", _count_df)
        else:
            if bits is None:
                _bits = np.arange(1, len(spots_list)+1)
            elif len(bits) != len(spots_list):
                raise IndexError(f"length of spots_list and bits don't match")
            else:
                _bits = np.array(bits, dtype=np.int32)
            # read gene df
            if verbose:
                print(f"-- read gene_list")
            self.readout_df = self.read_gene_list(self.readout_filename)
            # initialize 
            _cells = np.unique(self.segmentation_masks)[1:]
            _cell_spots_list = {_c:{_bit:[] for _bit in _bits} 
                                for _c in _cells}
            _labels_list = []
            # loop through each bit
            for spots, bit in zip(spots_list, _bits):
                _bit = int(bit)
                _spots = Spots3D(spots, _bit, self.pixel_sizes)
                _labels = self.spots_to_labels(self.segmentation_masks,
                    _spots, self.image_size, self.radius, 
                    verbose=verbose)
                _labels_list.append(_labels)
                for _l in np.unique(_labels):
                    if _l > 0:
                        _cell_spots_list[_l][_bit] = _spots[_labels==_l]
            # use information in _cell_spots_list, update the gene_count_df
            _count_df = pd.DataFrame()
            for _cell, _cell_spots in _cell_spots_list.items():
                _info_dict = {'cell_id': _cell}
                if hasattr(self, 'fov_id') or self.fov_id is not None:
                    _info_dict['fov_id'] = self.fov_id
                for _bit, _spots in _cell_spots.items():
                    if _bit in self.readout_df['Bit number'].values:
                        _gene = self.readout_df.loc[self.readout_df['Bit number']==_bit, query_label].values[0]
                        _info_dict[_gene] = len(_spots)
                # append
                _count_df = _count_df.append(_info_dict, ignore_index=True, )
            # add to attribute
            setattr(self, f"{query_label.lower()}_count_df", _count_df)
            setattr(self, 'cell_spots_list', _cell_spots_list)
            setattr(self, 'labels_list', _labels_list)
            # save
            if save:
                if verbose:
                    print(f"-- save {query_label.lower()}_count_df into file: {self.save_filename}")
                _count_df.to_csv(self.save_filename, index=False, header=True)
        # return
        return _count_df

    @staticmethod
    def spots_to_labels(segmentation_masks:np.ndarray, 
                        spots:Spots3D, 
                        single_im_size:(list or np.ndarray),
                        search_radius:int=10,
                        verbose:bool=True,
                        ):
        if verbose:
            print(f"- partition barcodes for {len(spots)} spots")
        # initialize
        _spot_labels = []
        # loop through each spot
        for _coord in spots.to_coords():
            # generate crop
            _crop = generate_neighboring_crop(_coord, crop_size=search_radius, 
                                            single_im_size=single_im_size)
            # crop locally
            _mks, _counts = np.unique(segmentation_masks[_crop.to_slices()], 
                                    return_counts=True)
            # filter counts and mks
            _counts = _counts[_mks>0]
            _mks = _mks[_mks>0]
            # append the label
            if len(_mks) == 0:
                _spot_labels.append(-1)
            else:
                _spot_labels.append(_mks[np.argmax(_counts)])

        return np.array(_spot_labels, dtype=np.int16)

    @staticmethod
    def read_gene_list(readout_filename):
        return pd.read_csv(readout_filename, header=0, )


def Merge_GeneCounts(gene_counts_list,
                     fov_ids,
                     save=True, save_filename=None,
                     overwrite=False, verbose=True,
                     ):
    """Merge cell-locations from multiple field-of-views"""
    _start_time = time.time()
    
    if save_filename is None or not os.path.exists(save_filename) or overwrite:
        if verbose:
            print(f"- Start merging {len(gene_counts_list)} cell locations")
        # initialize
        merged_gene_counts_df = pd.DataFrame()
        # loop through each cell-location file
        for _fov_id, _gene_counts in zip(fov_ids, gene_counts_list):
            if isinstance(_gene_counts, str):
                _gene_counts =  pd.read_csv(_gene_counts, header=0)
            elif isinstance(_gene_counts, pd.DataFrame):
                _gene_counts = _gene_counts.copy()
            else:
                raise TypeError(f"Wrong input type for _gene_counts")
            # add fov 
            if 'fov_id' not in _gene_counts.columns or np.isnan(_gene_counts['fov_id']).all():
                
                _gene_counts['fov_id'] = _fov_id * np.ones(len(_gene_counts), dtype=np.int32)
            # merge
            merged_gene_counts_df = pd.concat([merged_gene_counts_df, _gene_counts],
                                                ignore_index=True)

        if verbose:
            print(f"-- {len(merged_gene_counts_df)} cells converted into MetaData")

        if save and (save_filename is not None or overwrite):
            if verbose:
                print(f"-- save {len(merged_gene_counts_df)} cells into file:{save_filename}")
            merged_gene_counts_df.to_csv(save_filename, index=False, header=True)
    else:
        merged_gene_counts_df = pd.read_csv(save_filename, header=0)
        if verbose:
            print(f"- directly load {len(merged_gene_counts_df)} cells file: {save_filename}")

    if verbose:
        _execute_time = time.time() - _start_time
        if verbose:
            print(f"-- merge cell-locations in {_execute_time:.3f}s")
            
    return merged_gene_counts_df


def batch_partition_spots(        
                        segmentation_masks:np.ndarray, 
                        readout_filename:str,
                        fov_id,
                        spots_list,
                        bits=None,
                        query_label='Gene', 
                        search_radius=default_search_radius,
                        pixel_sizes=default_pixel_sizes,
                        save_filename=None,
                        ):
    """
    """
    _partition_cls = Spots_Partition(segmentation_masks,
    readout_filename, fov_id=fov_id, search_radius=search_radius,
    pixel_sizes=pixel_sizes, save_filename=save_filename)
    # run
    _df = _partition_cls.run(spots_list, bits, query_label=query_label)
    # return
    return _df
