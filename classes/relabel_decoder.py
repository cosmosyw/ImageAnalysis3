# packages
import os
import h5py
import numpy as np
import pandas as pd
# functions/classes
from tqdm import tqdm
from scipy.spatial import KDTree
from itertools import combinations
from copy import copy
# local functions/classes
from .preprocess import SpotTuple,Spots3D
from ..io_tools.spots import CellSpotsDf_2_CandSpots, Dataframe_2_SpotGroups, spotTupleList_2_DataFrame, CandSpotDf_add_positions
# variables
default_radius = 250
default_eps = 0.25
default_weights = np.array([1,1,1,1,1,])
from .new_decoder import generate_score_metrics, generate_scores, summarize_score

## Starts here:
class relabel_SpotDecoder():
    """Class to decode one-codebook"""
    def __init__(self, 
                 codebook_name:str, # name of this codebook to be decoded
                 candSpotDf:pd.DataFrame=None, # candidate spots
                 codebook:pd.DataFrame=None, # Codebook, ReadoutName+binaryTable
                 readoutDf:pd.DataFrame=None, # Bit+ReadoutName
                 saveFile:str=None, # save filename
                 searchRadiusPair:float=default_radius,
                 searchEps:float=default_eps,
                 autoRun:bool=True,
                 preLoad:bool=True,
                 overwrite:bool=False,
                 verbose:bool=True,
                 ):
        # input name
        self.codebook_name = codebook_name
        # input dataframes
        self.candSpotDf = candSpotDf
        if isinstance(self.candSpotDf, pd.DataFrame):
            self.candSpots = CellSpotsDf_2_CandSpots(self.candSpotDf) # convert into cand_spots
        self.codebook = codebook
        self.readoutDf = readoutDf
        self.saveFile = saveFile
        # parameters
        self.search_radius_pair = searchRadiusPair
        self.search_eps = searchEps
        self.overwrite = overwrite
        self.verbose = verbose
        # Load from exist
        if not self._check_full_inputs() or preLoad:
            self._load()
        # if still not having full input, break
        if not self._check_full_inputs():
            raise AssertionError(f"Not enough inputs given to create decoder class, exit!")
        if autoRun:
            # step1: summarize bit_codebook
            self._match_bit_2_codebook()
            # step2: find valid bit pairs
            self._process_codebook_2_pairs()
            # step3: search pairs
            self._search_candidate_pairs()
            # step4: generate spot groups 
            self._generate_spotGroups()
            # step5: save
            self._save()
            # step6: plot stats
            self._plot_statistics()
    # check if all inputs are given
    def _check_full_inputs(self):
        return isinstance(self.candSpotDf, pd.DataFrame) \
            and isinstance(self.codebook, pd.DataFrame) \
            and isinstance(self.readoutDf, pd.DataFrame) 
    # bit_2_channel dict for references
    def _create_bit_2_channel(self):
        """Create bit_2_channel dict"""
        # try create bit_2_channel if possible
        try:
            _bit_2_channel = {}
            for _bit in self.bits:
                _chs = np.unique(self.candSpotDf.loc[self.candSpotDf['bit']==_bit, 'channel'].values)
                if len(_chs) == 1:
                    _bit_2_channel[_bit] = str(_chs[0])
        except:
            _bit_2_channel = {}
        self.bit_2_channel = _bit_2_channel
    # combine codebook to readout into ID
    def _match_bit_2_codebook(self):
        if self.verbose:
            print(f"- Matching {len(self.readoutDf)} bits to {self.codebook.shape} codebook")
        self.default_cols = ['name', 'id', 'chr','chr_order']
        self.bit_codebook = pd.DataFrame(self.codebook[self.default_cols])
        self.bits = []
        for _col in self.codebook.columns:
            if _col in self.default_cols:
                continue
            else:
                _matched_bit = self.readoutDf.loc[self.readoutDf['ReadoutName']==_col, 'Bit'].values
                if len(_matched_bit) > 0:
                    self.bit_codebook[_matched_bit[0]] = self.codebook[_col].copy()
                    self.bits.append(_matched_bit[0])
        #[_b for _b in bit_codebook.columns if isinstance(_b, np.int64)]
        # summarize
        self.bits = np.array(self.bits, dtype=np.int32)
        if self.verbose:
            print(f"-- {len(self.bits)} bits matched")
        return
    # Process codebook to find valid bit pairs/tuples
    def _process_codebook_2_pairs(self):
        if self.verbose:
            print(f"- Process {self.bit_codebook.shape} codebook into valid pairs")
        if not hasattr(self, 'bit_codebook') or not hasattr(self, 'bits'):
            self._match_bit_2_codebook()
        codebook_matrix = self.bit_codebook[self.bits].values
        self.ValidBitPair_2_RegionId = {}
        for _icode, _code in enumerate(codebook_matrix):
            # pairs
            for _p in combinations(np.where(_code > 0)[0], 2):
                _bs = tuple(np.sort(self.bits[np.array(_p)]))
                if _bs not in self.ValidBitPair_2_RegionId:
                    self.ValidBitPair_2_RegionId[_bs] = self.bit_codebook.loc[_icode, 'id']
        if self.verbose:
            print(f"-- {len(self.ValidBitPair_2_RegionId)} valid pairs detected.")
        # return
        return self.ValidBitPair_2_RegionId
    # search spot_pairs by KDtree
    def _search_candidate_pairs(self,):
        if self.verbose:
            print(f"- Searching for spot-pairs within {self.search_radius_pair}nm.")
        if hasattr(self, 'candSpotPairInds_list') and hasattr(self, 'candSpotPair_list') and not self.overwrite:
            return self.candSpotPair_list
        else:
            self.candSpotPairInds_list = []
            self.candSpotPair_list = []
        # extract all coordinates
        _cand_positions = self.candSpots.to_positions()
        # build kd-tree
        if not hasattr(self, 'kdtree'):
            if self.verbose:
                print(f"-- find candidate pairs by KDTree")
            self.kdtree = KDTree(_cand_positions)
        _candSpotPairInds_list = list(self.kdtree.query_pairs(self.search_radius_pair, eps=self.search_eps))
        # loop through all pairs
        if self.verbose:
            print(f"-- filter candidate pairs by codebook")
        for _inds in _candSpotPairInds_list:
            # only keep the valid pairs
            _pair_bits = tuple( np.sort(self.candSpots.bits[np.array(_inds)]) )
            if _pair_bits in self.ValidBitPair_2_RegionId:
                self.candSpotPairInds_list.append(_inds)
        # Convert into spotPairs
        self.candSpotPair_list = [SpotTuple(self.candSpots[np.array(_inds)], 
                                            spots_inds=np.array(_inds), 
                                            tuple_id=self.ValidBitPair_2_RegionId\
                                                [tuple(self.candSpots[np.array(_inds)].bits)])
                                  for _inds in self.candSpotPairInds_list]
        if self.verbose:
            print(f"-- {len(self.candSpotPair_list)} pairs selected.")
    
    # Load from savefile:
    def _load(self,):
        if self.verbose:
            print(f"- Load decoder from file: {self.saveFile}")
        if self.saveFile is None:
            if self.verbose:
                print(f"saveFile not given, skip loading!")
            return
        if not os.path.exists(self.saveFile):
            print(f"-- savefile:{self.saveFile} not exist, skip")
            return
        # load important results
        with h5py.File(self.saveFile, 'r') as _f:
            if self.codebook_name not in _f.keys():
                print(f"-- savefile:{self.saveFile} doesn't have information for {self.codebook_name}, skip")
                return
            _g = _f.require_group(self.codebook_name)
            # try loading datasets
            if 'bits' in _g.keys():
                self.bits = _g['bits'][:]
            # try loading attrs
            if 'search_radius_pair' in _g.attrs.keys():
                self.search_radius_pair = _g.attrs['search_radius_pair']
            if 'search_eps' in _g.attrs.keys():
                self.search_eps = _g.attrs['search_eps']
        # try loading dataframes
        try: 
            self.readoutDf = pd.read_hdf(self.saveFile, f'{self.codebook_name}/readoutDf')
            self.codebook = pd.read_hdf(self.saveFile, f'{self.codebook_name}/codebook')
            self.candSpotDf = pd.read_hdf(self.saveFile, f'{self.codebook_name}/candSpots')
            _spotGroupDf = pd.read_hdf(self.saveFile, f'{self.codebook_name}/spotGroups')
            # convert candSpots and spotGroups
            self.candSpots = CellSpotsDf_2_CandSpots(self.candSpotDf) # convert into cand_spots
            self.spotGroups = Dataframe_2_SpotGroups(_spotGroupDf)
        except:
            print(f"failed to load dataframes, skip.")
        return

    # According to spotPair scoring, select candidate pairs
    def _generate_spotGroups(self, _maxSpotUsage=1, 
                            _weights=default_weights):
        """Function to select spot tuples given self.candSpotPair_list found previously"""
        # initialize _spot_usage and tuples
        _spotUsage = np.zeros(len(self.candSpots))
        if hasattr(self, 'spotGroups') and not self.overwrite:
            for _g in self.spotGroups:
                _spotUsage[_g.spots_inds] += 1
            # check if this is valid
            if np.max(_spotUsage) <= _maxSpotUsage:
                # save spot_usage
                setattr(self, 'spotUsage', _spotUsage)
                if self.verbose:
                    print(f"-- directly return {len(self.spotGroups)} spot_groups.")
                return
                #return self.spotGroups, self.spotUsage
            # otherwise clear spot_usage
            else:
                _spotUsage = np.zeros(len(self.candSpots))
        # otherwise continue
        self.spotGroups = []
        # 1. if no candidate pairs detected, do the previous step
        if not hasattr(self, 'candSpotPair_list') or len(self.candSpotPair_list) == 0:
            self._search_candidate_pairs()
        # skip this step if still no cand_spots
        if len(self.candSpotPair_list) == 0:
            return
        # 2. scoring all pairs
        if self.verbose:
            print(f"-- calculate scores for candSpotPairs")
        _pair_ref_metrics = generate_score_metrics(self.candSpotPair_list,)
        _pair_ref_metrics = np.concatenate(_pair_ref_metrics, axis=0)
        _, _ = generate_scores(self.candSpotPair_list, _pair_ref_metrics,)
        _ = summarize_score(self.candSpotPair_list, weights=_weights) # directly added to Tuple attribute
        # 4. select pairs if allowing error_correction
        _num_pairs = 0
        for _pair in tqdm(sorted(self.candSpotPair_list, key=lambda _p:-_p.final_score)):
            # skip if spots are used
            if (_spotUsage[_pair.spots_inds] >= _maxSpotUsage).any():
                #print("--- spot used, skip")
                continue
            # append the pair
            self.spotGroups.append(copy(_pair))
            _spotUsage[_pair.spots_inds] += 1 
            _num_pairs += 1
        if self.verbose:
            print(f"-- {_num_pairs} pairs selected")
        if self.verbose:
            print(f"-- in total {len(self.spotGroups)} spot_groups detected")
        # add select orders as attribute
        for _i, _g in enumerate(self.spotGroups):
            _g.sel_ind = _i
        # save spot_usage
        setattr(self, 'spotUsage', _spotUsage)
        #return self.spotGroups, self.spotUsage
        return
    # save into savefile:
    def _save(self, _strict=True, _complevel=1, _complib='blosc:zstd',):
        if self.verbose:
            print(f"- Save decoder of {self.codebook_name} into file: {self.saveFile}")
        if self.saveFile is None:
            print(f"saveFile not given, skip saving!")
            if _strict:
                raise ValueError(f"saveFile not given, cannot save anything!")
        if not hasattr(self, 'bit_2_channel'):
            self._create_bit_2_channel()
        # list of saving:
        #self.bit_codebook, self.bits, self.candSpotDf, self.search_radius, self.search_eps, self.spotGroups,
        with h5py.File(self.saveFile, 'a') as _f:
            _g = _f.require_group(self.codebook_name)
            print(f"-- existing info: {list(_g.keys())} and {list(_g.attrs.keys())}")
            ## Datasets
            # save_bits
            if self.overwrite and 'bits' in _g.keys():
                if self.verbose:
                    print(f"-- remove existing bits")
                del(_g['bits'])
            if 'bits' not in _g.keys():
                if self.verbose:
                    print(f"-- save bits")
                _g.create_dataset('bits', data=self.bits)
            else:
                if self.verbose:
                    print(f"-- skip saving bits")
            ## Attrs:
            # search_radius
            if 'search_radius_pair' not in _g.attrs.keys() or self.overwrite:
                if self.verbose:            
                    print(f"-- save search_radius_pair to attrs")
                _g.attrs['search_radius_pair'] = self.search_radius_pair
            else:
                if self.verbose:
                    print(f"-- skip saving search_radius_pair")
            
            # search_eps
            if 'search_eps' not in _g.attrs.keys() or self.overwrite:
                if self.verbose:            
                    print(f"-- save search_eps to attrs")
                _g.attrs['search_eps'] = self.search_eps
            else:
                if self.verbose:
                    print(f"-- skip saving search_eps")
            # also judge whether save dataframes
            _save_readout = ('readoutDf' not in _g.keys()) or self.overwrite
            _save_codebook = ('codebook' not in _g.keys()) or self.overwrite
            _save_candspots = ('candSpots' not in _g.keys()) or self.overwrite
            _save_spotgroups = ('spotGroups' not in _g.keys()) or self.overwrite
        # Dataframes
        if _save_readout:
            if self.verbose:
                print(f"-- save readoutDf")
            self.readoutDf.to_hdf(self.saveFile, f'{self.codebook_name}/readoutDf', complevel=_complevel, complib=_complib)
        if _save_codebook:
            if self.verbose:
                print(f"-- save codebook")
            self.codebook.to_hdf(self.saveFile, f'{self.codebook_name}/codebook', complevel=_complevel, complib=_complib)
        if _save_candspots:
            if self.verbose:
                print(f"-- save candSpots")
            self.candSpotDf.to_hdf(self.saveFile, f'{self.codebook_name}/candSpots', complevel=_complevel, complib=_complib)
        if _save_spotgroups:
            if self.verbose:
                print(f"-- save spotGroups")
            # get parameters
            _fov_id = np.unique(self.candSpotDf['fov_id'])[0]
            _cell_id = np.unique(self.candSpotDf['cell_id'])[0]
            _cell_uid = np.unique(self.candSpotDf['uid'])[0]
            _spotGroupDf = spotTupleList_2_DataFrame(self.spotGroups, 
                                                     fov_id=_fov_id, cell_id=_cell_id, cell_uid=_cell_uid,
                                                     bit_2_channel=self.bit_2_channel, codebook=self.bit_codebook,
                                                     include_position=True,)
            _spotGroupDf.to_hdf(self.saveFile, f'{self.codebook_name}/spotGroups', complevel=_complevel, complib=_complib)
        return
    # generate_stats_plot
    def _plot_statistics(self, _max_usage=5, _save=True, _show_image=False):
        # generate plot
        from ..figure_tools.plot_decode import plot_relabel_spot_stats
        try:
            _stat_figure_file = self.saveFile.replace('.hdf5', f'_{self.codebook_name}_stats.png')
        except:
            _save=False
            _stat_figure_file = None
        if len(self.spotGroups) > 0:
            _ax = plot_relabel_spot_stats(self.spotGroups, self.spotUsage, _max_usage=_max_usage,
                codebook=self.codebook, save=_save, save_filename=_stat_figure_file,
                show_image=_show_image, verbose=self.verbose)
            if _show_image:
                return _ax

## Batch functions
def batch_process_relabel_SpotDecoder(
    codebook_name,
    candSpotDf,
    codebook,
    readoutDf,
    saveFile,
    searchRadiusPair=default_radius,
    searchEps=default_eps,
    overwrite=False,
    verbose=True,
    return_obj=False,
    ):
    # Create class
    _decoder = relabel_SpotDecoder(
        codebook_name=codebook_name, candSpotDf=candSpotDf,
        codebook=codebook, readoutDf=readoutDf,
        saveFile=saveFile, 
        searchRadiusPair=searchRadiusPair, searchEps=searchEps,
        autoRun=True, preLoad=(overwrite==False),
        overwrite=overwrite, verbose=verbose,
    )
    if return_obj:
        return _decoder

## Function for scoring
def generate_score_metrics(spot_groups, chr_tree=None, homolog_centers=None,
                          n_neighbors=10, 
                          update_attr=True, 
                          overwrite=False):
    """Five metrics:
    [mean_intensity, COV_intensity, median_internal_distance, distance_to_neighbor, distance_to_chr_center]"""
    # check inputs
    if chr_tree is None or isinstance(chr_tree, list) or isinstance(chr_tree, KDTree):
        pass
    else:
        raise TypeError(f"Wrong input type for chr_tree")
    if homolog_centers is None or isinstance(homolog_centers, np.ndarray):
        pass
    else:
        raise TypeError(f"Wrong input type for homolog_centers")
    
    # collect basic metrics
    _basic_metrics_list = []
    for _g in spot_groups:
        #
        if hasattr(_g, 'basic_score_metrics') and not overwrite:
            _metrics = list(getattr(_g, 'basic_score_metrics'))
        else:
            _metrics = [np.mean(_g.spots.to_intensities()), # mean intensity
                    np.std(_g.intensities())/np.mean(_g.intensities()), # Coefficient of variation of intensity
                    np.median(_g.dist_internal()), # median of internal distance
                    ]
            if update_attr:
                _g.basic_score_metrics = np.array(_metrics)
        # append
        _basic_metrics_list.append(_metrics)
        
    def neighboring_dists(_spot_groups, _chr_tree, _n_neighbors=10):
        if _chr_tree is None or _chr_tree.n < _n_neighbors:
            return np.nan * np.ones(len(_spot_groups))
        return np.mean(_chr_tree.query([_g.centroid_spot().to_positions()[0] for _g in _spot_groups], 
                                       _n_neighbors)[0], axis=1)
    def homolog_center_dists(_spot_groups, _homolog_center):
        if _homolog_center is None:
            return np.nan * np.ones(len(_spot_groups))
        #print(np.array([_g.centroid_spot().to_positions()[0] for _g in _spot_groups]).shape)
        #print(_homolog_center.shape)
        return np.linalg.norm([_g.centroid_spot().to_positions()[0] - _homolog_center 
                               for _g in _spot_groups], axis=1)
                         
    # no homologs:
    group_metrics_list = []
    if homolog_centers is None:
        if isinstance(chr_tree, list):
            _tree = chr_tree[0]
        else:
            _tree = chr_tree
        
        _nb_dists = neighboring_dists(spot_groups, _tree, _n_neighbors=n_neighbors)
        _ct_dists = homolog_center_dists(spot_groups, None)
        for _g, _bm, _nb_dist, _ct_dist in zip(spot_groups, _basic_metrics_list, _nb_dists, _ct_dists):
            _metric_list = np.array([_bm+[_nb_dist, _ct_dist]])
            group_metrics_list.append(_metric_list)

    # with homologs
    ## with different chr_tree:
    elif isinstance(chr_tree, list) and isinstance(homolog_centers, np.ndarray):
        _metrics_by_homologs = []
        _nb_dists, _ct_dists = [], []
        
        for _tree, _ct in zip(chr_tree, homolog_centers):
            _nb_dists.append(neighboring_dists(spot_groups, _tree, _n_neighbors=n_neighbors))
            _ct_dists.append( homolog_center_dists(spot_groups, _ct) )
        # merge homologs
        _nb_dists = np.array(_nb_dists).transpose()
        _ct_dists = np.array(_ct_dists).transpose()
        for _g, _bm, _nb_dist, _ct_dist in zip(spot_groups, _basic_metrics_list, _nb_dists, _ct_dists):
            _metric_list = np.array([_bm+[_nd, _cd] for _nd,_cd in zip(_nb_dist,_ct_dist)])
            group_metrics_list.append(_metric_list)

    ## with the same chr_tree:
    elif isinstance(homolog_centers, np.ndarray):
        _metrics_by_homologs = []
        _nb_dists = neighboring_dists(spot_groups, chr_tree, _n_neighbors=n_neighbors)
        _ct_dists = []
        for _ct in homolog_centers:
            _ct_dists.append( homolog_center_dists(spot_groups, _ct) )
        # merge homologs
        _nb_dists = np.array(_nb_dists).transpose()
        _ct_dists = np.array(_ct_dists).transpose()
        for _g, _bm, _nb_dist, _ct_dist in zip(spot_groups, _basic_metrics_list, _nb_dists, _ct_dists):
            _metric_list = np.array([_bm+[_nb_dist, _cd] for _cd in _ct_dist])
            group_metrics_list.append(_metric_list)

    group_metrics_list = np.array(group_metrics_list)
    if update_attr:
        for _g, _metrics_list in zip(spot_groups, group_metrics_list):
            _g.score_metrics = _metrics_list

    return group_metrics_list