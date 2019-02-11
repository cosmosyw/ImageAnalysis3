from . import get_img_info, visual_tools, alignment_tools, analysis, classes
from . import _correction_folder,_temp_folder,_distance_zxy,_sigma_zxy,_image_size,_allowed_colors
from .External import Fitting_v3
import numpy as np
import scipy
import pickle
import matplotlib.pylab as plt
import os, glob, sys, time
from scipy.stats import scoreatpercentile
import multiprocessing as mp
import ctypes
from scipy.ndimage.interpolation import map_coordinates
def __init__():
    pass



## Fit bead centers
def get_STD_centers(im, th_seed=150, dynamic=False, th_seed_percentile=95, 
                    close_threshold=0.01, fit_radius=5,
                    save=False, save_folder='', save_name='', 
                    plt_val=False, force=False, verbose=False):
    '''Fit beads for one image:
    Inputs:
        im: image, ndarray
        th_seeds: threshold for seeding, float (default: 150)
        close_threshold: threshold for removing duplicates within a distance, float (default: 0.01)
        plt_val: whether making plot, bool (default: False)
        save: whether save fitting result, bool (default: False)
        save_folder: full path of save folder, str (default: None)
        save_name: full name of save file, str (default: None)
        force: whether force fitting despite of saved file, bool (default: False)
        verbose: say something!, bool (default: False)
    Outputs:
        beads: fitted spots with information, n by 4 array'''
    import os
    import pickle as pickle
    if not force and os.path.exists(save_folder+os.sep+save_name) and save_name != '':
        if verbose:
            print("- loading file:,", save_folder+os.sep+save_name)
        beads = pickle.load(open(save_folder+os.sep+save_name, 'rb'))
        if verbose:
            print("--", len(beads), " of beads loaded.")
        return beads
    else:
        # seeding
        seeds = visual_tools.get_seed_in_distance(im, center=None, dynamic=dynamic, 
                                    th_seed_percentile=th_seed_percentile,
                                    gfilt_size=0.75,filt_size=3,th_seed=th_seed,hot_pix_th=4)
        # fitting
        fitter = Fitting_v3.iter_fit_seed_points(im, seeds.T, radius_fit=5)
        fitter.firstfit()
        pfits = fitter.ps
        #pfits = visual_tools.fit_seed_points_base_fast(im,seeds.T,width_z=1.8*1.5/2,width_xy=1.,radius_fit=5,n_max_iter=3,max_dist_th=0.25,quiet=not verbose)
        # get coordinates for fitted beads
        remove = 0
        if len(pfits) > 0:
            beads = np.array(pfits)[:,1:4]
            # remove very close spots
            for i,bead in enumerate(beads):
                if np.sum(np.sum((beads-bead)**2, axis=1)<close_threshold) > 1:
                    beads = np.delete(beads, i-remove, 0)
                    remove += 1
        else:
            beads = None
        if verbose:
            print(f"- fitting {len(pfits)} points")
            print(f"-- {remove} points removed given smallest distance {close_threshold}")
        # make plot if required
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im,0),interpolation='nearest')
            plt.plot(beads[:,2],beads[:,1],'or')
            plt.show()
        # save to pickle if specified
        if save:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if verbose:
                print("-- saving fitted spots to", save_folder+os.sep+save_name)
            pickle.dump(beads, open(save_folder+os.sep+save_name, 'wb'))

        return beads

def _align_single_image_from_file(_filename, _selected_crops, _ref_centers=None, _ref_filename=None, _bead_channel_id=None,
                                  _bead_channel='488', _channels=_allowed_colors, _im_size=_image_size, _buffer_frame=10,
                                  _illumination_corr=True, _correction_folder=_correction_folder,
                                  _align_cutoff=3, _align_res=1, _dynamic_seeding=True, _dynamic_th_percent=90, _seed_th=300,
                                  _drift_cutoff=3, _make_plot=False, _verbose=True):
    """align single image used by multiprocessing-drift-correction
    Inputs:
        _filename: filename of a dax file
        _selected_crops: crop limits for selected windows, ndarray, 3x3x2
        _ref_centers: fitted center coordinates for  
        
    """
    ## check inputs
    if _ref_centers is None and _ref_filename is None:
        raise ValueError(f"Either ref_center or ref_filename should be given!")
    if '.dax' not in _filename:
        raise IOError(
            f"Wrong input file type, {_filename} should be .dax file")
    if _ref_filename is not None and '.dax' not in _ref_filename:
        raise IOError(
            f"Wrong input reference file type, {_ref_filename} should be .dax file")
    # check bead_channel_id
    if _bead_channel_id is None:
        _bead_channel_id = _channels.index(_bead_channel)
    ## slice the image for given filename
    _full_im_size, _num_colors = get_img_info.get_num_frame(
        _filename, buffer_frame=_buffer_frame, frame_per_color=_im_size[0])
    # correct full image
    _full_target_im = correct_single_image(_filename, _bead_channel, 
                        single_im_size=_im_size, all_channels=_channels, channel_id=_bead_channel_id,
                        num_buffer_frames=_buffer_frame, correction_folder=_correction_folder,
                        z_shift_corr=True, hot_pixel_remove=True, 
                        illumination_corr=_illumination_corr, chromatic_corr=False)
    # crop images
    _target_ims = [_full_target_im[_crop[0, 0]:_crop[0, 1],
                                   _crop[1, 0]:_crop[1, 1],
                                   _crop[2, 0]:_crop[2, 1]] for _crop in _selected_crops[:2]]
    ## case 1: ref_centers are given:
    # initialize
    _failed_count = 0
    # initialize seeding threshold
    if _dynamic_seeding:
        _target_seed_ths = [scoreatpercentile(
            rim, _dynamic_th_percent)*0.45 for rim in _target_ims]
    else:
        _target_seed_ths = [_seed_th for rim in _target_ims]
    # fitting
    _target_centers = [get_STD_centers(rim, th_seed=rseed, verbose=_verbose)
                       for rim, rseed in zip(_target_ims, _target_seed_ths)]

    # printing info
    _print_name = os.path.join(_filename.split(os.sep)[-2], _filename.split(os.sep)[-1])


    if _ref_centers is not None:
        if _verbose:
            print(f"- Aligning {_print_name}")
        _aligns = [visual_tools.translation_aling_pts(_ref_cent, _tar_cent, cutoff=_align_cutoff,
                                                      xyz_res=_align_res, plt_val=_make_plot, verbose=_verbose) for _ref_cent, _tar_cent in zip(_ref_centers[:2], _target_centers[:2])]
        _drift = np.mean(_aligns, 0)
        if np.linalg.norm(_aligns[0]-_aligns[1], 1) > _drift_cutoff:
            if _verbose:
                print(f"Suspecting failure for {_print_name}")
            _failed_count += 1
            _target_ims.append(visual_tools.slice_image(_filename, _full_im_size, [_buffer_frame+_num_colors*_selected_crops[2][0, 0], _buffer_frame+_num_colors*_selected_crops[2][0, 1]],
                                                        _selected_crops[2][1], _selected_crops[2][2], zstep=_num_colors, zstart=_bead_channel_id))
            if _dynamic_seeding:
                _target_seed_ths.append(scoreatpercentile(
                    _target_ims[-1], _dynamic_th_percent)*0.45)
            else:
                _target_seed_ths.append(_seed_th)
            _target_centers.append(get_STD_centers(
                _target_ims[-1], th_seed=_target_seed_ths[-1], verbose=_verbose))
            _aligns.append(visual_tools.translation_aling_pts(_ref_centers[2], _target_centers[2],
                                                              cutoff=_align_cutoff, xyz_res=_align_res, plt_val=True, verbose=_verbose))
            if np.linalg.norm(_aligns[0]-_aligns[2], 1) > _drift_cutoff and np.linalg.norm(_aligns[1]-_aligns[2], 1) > _drift_cutoff:
                if _verbose:
                    print(f"-- keep the previous drift.")
            elif np.linalg.norm(_aligns[0]-_aligns[2], 1) <= np.linalg.norm(_aligns[1]-_aligns[2], 1):
                _drift = (_aligns[0]+_aligns[2])/2
                if _verbose:
                    print(f"-- selected drifts:{_aligns[0]}, {_aligns[2]}")
            else:
                _drift = (_aligns[1]+_aligns[2])/2
                if _verbose:
                    print(f"-- selected drifts:{_aligns[1]}, {_aligns[2]}")
    ## do alignment if ref-filename is given
    elif _ref_filename is not None:
        _ref_print_name = os.path.join(_ref_filename.split(os.sep)[-2], _ref_filename.split(os.sep)[-1])
        if _verbose:
            print(f"- Aligning {_print_name} to {_ref_print_name}")
        _full_ref_im_size, _ref_num_colors = get_img_info.get_num_frame(
            _ref_filename, buffer_frame=_buffer_frame, frame_per_color=_im_size[0])

        _full_ref_im = correct_single_image(_ref_filename, _bead_channel, 
                    single_im_size=_im_size, all_channels=_channels, channel_id=_bead_channel_id,
                    num_buffer_frames=_buffer_frame, correction_folder=_correction_folder,
                    z_shift_corr=True, hot_pixel_remove=True, 
                    illumination_corr=_illumination_corr, chromatic_corr=False)
        # crop images
        _ref_ims = [_full_ref_im[_crop[0, 0]:_crop[0, 1],
                                 _crop[1, 0]:_crop[1, 1],
                                 _crop[2, 0]:_crop[2, 1]] for _crop in _selected_crops[:2]]
        # seeding
        if _dynamic_seeding:
            _ref_seed_ths = [scoreatpercentile(
                rim, _dynamic_th_percent)*0.5 for rim in _ref_ims]
        else:
            _ref_seed_ths = [_seed_th for rim in _ref_ims]
        _ref_centers = [get_STD_centers(_rim, th_seed=_rseed, verbose=_verbose)
                        for _rim, _rseed in zip(_ref_ims, _ref_seed_ths)]
        _aligns = [visual_tools.translation_aling_pts(_ref_cent, _tar_cent, cutoff=_align_cutoff,
                                                      xyz_res=_align_res, plt_val=_make_plot, verbose=_verbose) for _ref_cent, _tar_cent in zip(_ref_centers[:2], _target_centers[:2])]
        _drift = np.mean(_aligns, 0)
        if np.linalg.norm(_aligns[0]-_aligns[1], 1) > _drift_cutoff:
            if _verbose:
                print(f"Suspecting failure for {_print_name}")
            _failed_count += 1
            # append target image
            # crop new images
            _new_target_im = _full_target_im[_selected_crops[2][0, 0]:_selected_crops[2][0, 1],
                                            _selected_crops[2][1, 0]:_selected_crops[2][1, 1],
                                            _selected_crops[2][2, 0]:_selected_crops[2][2, 1]] 
            _target_ims.append(_new_target_im)
            if _dynamic_seeding:
                _target_seed_ths.append(scoreatpercentile(
                    _target_ims[-1], _dynamic_th_percent)*0.45)
            else:
                _target_seed_ths.append(_seed_th)
            _target_centers.append(get_STD_centers(
                _target_ims[-1], th_seed=_target_seed_ths[-1], verbose=_verbose))
            # append ref image
            _new_ref_im = _full_ref_im[_selected_crops[2][0, 0]:_selected_crops[2][0, 1],
                                       _selected_crops[2][1, 0]:_selected_crops[2][1, 1],
                                       _selected_crops[2][2, 0]:_selected_crops[2][2, 1]] 
            _ref_ims.append(_new_ref_im)
            if _dynamic_seeding:
                _ref_seed_ths.append(scoreatpercentile(
                    _ref_ims[-1], _dynamic_th_percent)*0.45)
            else:
                _ref_seed_ths.append(_seed_th)
            _ref_centers.append(get_STD_centers(
                _ref_ims[-1], th_seed=_ref_seed_ths[-1], verbose=_verbose))
            # align the new image
            _aligns.append(visual_tools.translation_aling_pts(_ref_centers[2], _target_centers[2],
                                                              cutoff=_align_cutoff, xyz_res=_align_res, plt_val=True, verbose=_verbose))

            if np.linalg.norm(_aligns[0]-_aligns[2], 1) > _drift_cutoff and np.linalg.norm(_aligns[1]-_aligns[2], 1) > _drift_cutoff:
                if _verbose:
                    print(f"-- keep the previous drift.")
            elif np.linalg.norm(_aligns[0]-_aligns[2], 1) <= np.linalg.norm(_aligns[1]-_aligns[2], 1):
                _drift = (_aligns[0]+_aligns[2])/2
                if _verbose:
                    print(f"-- selected drifts:{_aligns[0]}, {_aligns[2]}")
            else:
                _drift = (_aligns[1]+_aligns[2])/2
                if _verbose:
                    print(f"-- selected drifts:{_aligns[1]}, {_aligns[2]}")

    return _drift, _failed_count

# merged function to calculate bead drift directly from files
def Calculate_Bead_Drift(folders, fovs, fov_id, bead_channel_id=None, num_threads=12, 
                         bead_channel='488', channels=_allowed_colors, illumination_corr=True, correction_folder=_correction_folder,
                         sequential_mode=False, ref_id=0, drift_size=400, im_size=_image_size, buffer_frame=10,
                         coord_sel=None, align_cutoff=3, align_res=1,
                         dynamic_seeding=True, dynamic_th_percent=80, seed_th=300, drift_cutoff=3,
                         save=True, save_folder=None, save_postfix='_current_cor.pkl', make_plot=False, overwrite=False, verbose=True):
    """Function to generate drift profile given a list of corrected bead files
    Inputs:
    filenames: filenames of temp-files or raw images, list of string(path)s
    bead_names: reference names(H1R1\Conv_Zscan_03.dax), used as reference key for drift_dic, list of strings(default:None, generated by filenames)
    ref_id: reference frame index, int (defualt: 0)
    drift_size: window size of drift image which is used for cropping, int (default: 300)
    num_threads: number of threads used in fitting beads, int (default: 12)
    coord_sel: coordinate to pick windows, array-like of 2 elements (default: None, which means center)
    align_cutoff: cutoff for trans-point alignment, int (default: 3)
    align_res: alignment resolution, int (default: 1)
    dynamic_seeding: whether use dynamic seeding threshold, bool (defualt: True)
    dynamic_th_percent: dynamic seeding threshold percentile, int (defualt: 80)
    seed_th: seeding threshold as default, which is used when dynamic=False or dynamic seeding failed, float (default: 300)
    save: whether save drift result dictionary to pickle file, bool (default: True)
    save_folder: folder to save drift result, required if save is True, string of path (default: None)
    save_postfix: drift file postfix, string (default: '_sequential_current_cor.pkl')
    overwrite: whether overwrite existing drift_dic, bool (default: False)
    verbose: say something during the process! bool (default:True)
    """
    from scipy.stats import scoreatpercentile
    ## check inputs
    # check folders
    if not isinstance(folders, list):
        raise ValueError("Wrong input type of folders, should be a list")
    if len(folders) == 0: # check folders length
        raise ValueError("Kwd folders should have at least one element")
    if not isinstance(fovs, list):
        raise ValueError("Wrong input type of fovs, should be a list")
    # check if all images exists
    _fov_name = fovs[fov_id]
    for _fd in folders:
        _filename = os.path.join(_fd, _fov_name)
        if not os.path.isfile(_filename):
            raise IOError(f"one of input file:{_filename} doesn't exist, exit!")
    # check bead_channel_id
    if bead_channel_id is None:
        bead_channel_id = channels.index(str(bead_channel))
    # check ref-id
    if ref_id >= len(folders) or ref_id <= -len(folders):
        raise ValueError(f"Ref_id should be valid index of folders, however {ref_id} is given")
    # check save-folder
    if save:
        if save_folder is None:
            save_folder = os.path.join(os.path.dirname(folders[0]), 'Analysis', 'drift')
            print(f"No save_folder specified, use default save-folder:{save_folder}")
        elif not os.path.exists(save_folder):
            if verbose:
                print(f"Create drift_folder:{save_folder}")
            os.makedirs(save_folder)
    # check save_name 
    if sequential_mode: # if doing drift-correction in a sequential mode:
        save_postfix = '_sequential'+save_postfix
    # check coord_sel
    if coord_sel is None:
        coord_sel = np.array([int(im_size[-2]/2), int(im_size[-1]/2)], dtype=np.int)
    # collect crop coordinates (slices)
    crop0 = np.array([[0,im_size[0]],
                      [max(coord_sel[-2]-drift_size,0), coord_sel[-2]],
                      [max(coord_sel[-1]-drift_size,0), coord_sel[-1]]],dtype=np.int)
    crop1 = np.array([[0,im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2]+drift_size, im_size[-2])],
                      [coord_sel[-1], min(coord_sel[-1]+drift_size, im_size[-1])]], dtype=np.int)
    crop2 = np.array([[0, im_size[0]],
                      [coord_sel[-2], min(coord_sel[-2]+drift_size, im_size[-2])],
                      [max(coord_sel[-1]-drift_size, 0), coord_sel[-1]]],dtype=np.int)
    selected_crops = np.stack([crop0,crop1,crop2]) # merge into one array which is easier to feed into function

    ## start loading existing profile
    _save_name = _fov_name.replace('.dax', save_postfix)
    _save_filename = os.path.join(save_folder, _save_name)
    # try to load existing profiles
    if not overwrite and os.path.isfile(_save_filename):
        if verbose:
            print(f"-- loading existing drift info from file:{_save_filename}")
        old_drift_dic = pickle.load(open(_save_filename, 'rb'))
        _exist_keys = [os.path.join(os.path.basename(_fd), _fov_name) in old_drift_dic for _fd in folders]
        if sum(_exist_keys) == len(folders):
            if verbose:
                print("-- All frames exists in original drift file, exit.")
            return old_drift_dic, 0
    # no existing profile or force to do de novo correction:
    else:
        if verbose:
            print(f"-- starting a new drift correction for field-of-view:{_fov_name}")
        old_drift_dic = {}
    
    ## initialize drift correction
    _start_time = time.time() # record start time
    # whether drift for reference frame changes
    if len(old_drift_dic) > 0:
        old_ref_frames = [_hyb_name for _hyb_name,
                         _dft in old_drift_dic.items() if (np.array(_dft)==0).all()]
        if len(old_ref_frames) > 1 or len(old_ref_frames) == 0:
            print("Ref frame not unique, start over!")
            old_drift_dic = {}
        else:
            old_ref_frame = old_ref_frames[0]
            # if ref-frame not overlapping, remove the old one for now
            if old_ref_frame != os.path.join(os.path.basename(folders[ref_id]),_fov_name):
                if verbose:
                    print(f"old-ref:{old_ref_frame}, delete old refs because ref doesn't match")
                del old_drift_dic[old_ref_frame]
    else:
        old_ref_frame = None

    if not sequential_mode:
        if verbose:
            print(f"-- Start drift-correction, image mapped to image:{ref_id}")
        # ref filename
        _ref_filename = os.path.join(folders[ref_id], _fov_name) # get ref-frame filename
        _ref_keyname = os.path.join(os.path.basename(folders[ref_id]), _fov_name)
        # get number of colors etc.
        _ref_im_size, _ref_num_colors = get_img_info.get_num_frame(_ref_filename,buffer_frame=buffer_frame,frame_per_color=im_size[0]) 
        # collect ref image
        ref_ims = [visual_tools.slice_image(_ref_filename, _ref_im_size, 
                                            [buffer_frame+_ref_num_colors*_crop[0,0], buffer_frame+_ref_num_colors*_crop[0,1]],_crop[1],_crop[2],
                                            zstep=_ref_num_colors, zstart=bead_channel_id) for _crop in selected_crops]
        # seeding
        if dynamic_seeding:
            ref_seed_ths = [scoreatpercentile(rim, dynamic_th_percent)*0.45 for rim in ref_ims]
        else:
            ref_seed_ths = [seed_th for rim in ref_ims]
        ref_centers = [get_STD_centers(rim, th_seed=rseed, verbose=verbose) for rim, rseed in zip(ref_ims, ref_seed_ths)]
    
        # retrieve files requiring drift correction
        args = []
        new_keynames = []
        for _i, _fd in enumerate(folders):
            # extract filename
            _filename = os.path.join(_fd, _fov_name)
            _keyname = os.path.join(os.path.basename(_fd), _fov_name)
            # append all new frames except ref-frame. ref-frame will be assigned to 0
            if _keyname not in old_drift_dic and _keyname != _ref_keyname:
                new_keynames.append(_keyname)
                args.append((_filename, selected_crops, ref_centers, None, bead_channel_id,
                             bead_channel, channels, im_size, buffer_frame,
                             illumination_corr, correction_folder,
                             align_cutoff, align_res,dynamic_seeding, dynamic_th_percent*0.99**_i, seed_th*0.99**_i,
                             drift_cutoff,make_plot, verbose))
    else: # sequential mode
        # force ref-id to be 0
        ref_id = 0
        # retrieve files requiring drift correction
        args = []
        new_keynames = []
        for _ref_fd, _fd in zip(folders[:-1], folders[1:]):
            # extract filename
            _filename = os.path.join(_fd, _fov_name) # full filename for image
            _keyname = os.path.join(os.path.basename(_fd), _fov_name) # key name used in dic
            _ref_filename = os.path.join(_ref_fd, _fov_name) # full filename for ref image
            # append
            new_keynames.append(_keyname)
            args.append((_filename, selected_crops, None, _ref_filename, bead_channel_id,
                         bead_channel, channels, im_size, buffer_frame,
                         illumination_corr, correction_folder,
                         align_cutoff, align_res,dynamic_seeding, dynamic_th_percent, seed_th,
                         drift_cutoff,make_plot, verbose))
            
    ## multiprocessing 
    if verbose:
        print(f"- Start multi-processing drift correction with {num_threads} threads")
    with mp.Pool(num_threads) as drift_pool:
        align_results = drift_pool.starmap(_align_single_image_from_file, args)
        drift_pool.close()
        drift_pool.terminate()
        drift_pool.join()
    # clear 
    del(args)
    classes.killchild()
    if not sequential_mode:
        new_drift_dic = {_ref_name:_ar[0] for _ref_name, _ar in zip(new_keynames, align_results)}
    else:
        _total_dfts = [_ar[0][0] for _ar in zip(align_results)]
        _total_dfts = [sum(_total_dfts[:i+1]) for i, _d in enumerate(_total_dfts)]
        new_drift_dic = {_ref_name:_dft for _ref_name, _dft in zip(new_keynames, _total_dfts)}
    fail_count = sum([_ar[1] for _ar in align_results])
    # append ref frame drift
    _ref_keyname = os.path.join(os.path.basename(folders[ref_id]), _fov_name)
    new_drift_dic[_ref_keyname] = np.zeros(3) # for ref frame, assign to zero
    if old_ref_frame is not None and old_ref_frame in new_drift_dic:
        ref_drift = new_drift_dic[old_ref_frame]
        if verbose:
            print(f"-- drift of reference: {ref_drift}")
        # update drift in old ones
        for _old_name, _old_dft in old_drift_dic.items():
            if _old_name not in new_drift_dic:
                new_drift_dic[_old_name] = _old_dft + ref_drift
    ## save
    if save:
        if verbose:
            print(f"- Saving drift to file:{_save_filename}")
        if not os.path.exists(os.path.dirname(_save_filename)):
            if verbose:
                print(f"-- creating save-folder: {os.path.dirname(_save_filename)}")
            os.makedirs(os.path.dirname(_save_filename))
        pickle.dump(new_drift_dic, open(_save_filename, 'wb'))
    if verbose:
        print(f"-- Total time cost in drift correction: {time.time()-_start_time}")
        if fail_count > 0:
            print(f"-- number of failed drifts: {fail_count}")
    return new_drift_dic, fail_count


def STD_beaddrift_sequential(bead_ims, bead_names, drift_folder, fovs, fov_id,
                             drift_size=200, coord_sel=None, cutoff_=3, xyz_res_=1,
                             dynamic=False, dynamic_th_percent=80, th_seed=300,
                             illumination_correction=False, ic_channel='488', correction_folder=_correction_folder,
                             overwrite=False, plt_val=False, save=True, save_postfix='_sequential_current_cor.pkl',
                             verbose=True):
    """Given a list of bead images This handles the fine bead drift correction.
    If save is true this requires global paramaters drift_folder,fovs,fov_id
    Inputs:
        bead_ims: list of images, list of ndarray
        bead_names: names for bead files, list of strings
        drift_folder: full directory to store analysis files, string
        fovs: names for all field of views, list of strings
        fov_id: the id for field of view to be analysed, int
        drift_size: size of window used in drift correction, int (defualt: 200)
        coord_sel: whether select a coordinate as center, None or len-3 vector (default:None, means middle of image)
        cutoff_: point alignment cutoff, int (default: 3)
        xyz_res_: resolution in xyz during alignment, int (default: 1)
        dynamic: whether use dynamic seeding threashold, bool (default: False)
        dynamic_th_percent: if dynamic, the threshold is this percentile of total image intensity, float in [0,100] (default:80)
        th_seed: fixed seed threshold if not dynamic, float (default:150)
        Illumination_correction: whether do illumination correction, bool
        ic_channel: illumination_correction channel, str or int (default: 488)
        correction_folder: folder for correction files, string (path)
        overwrite: whether overwrite existing drifts, bool (default: False)
        plt_val: whether plot alignment result, bool (default: False)
        save: whether save drift to file, bool (default: True)
        save_postfix: postfix for save filename after field-of-view name, str(default:'_sequential_current_cor.pkl')
        verbose: Say something during the process! bool (default: True)
    Outputs:
        total_drift: dictionary from bead_names to len=3 vector representing drifts in z,x,y, dict
        _failed_count: number of suspicious failures during drifts, int
        """
    if verbose:
        print("- Drift correction starts!")
    # Initialize failed counts
    fail_count = 0
    # whether drift for this frame changes
    change_markers = {}
    # record start time
    _start_time = time.time()
    # check existing pickle file, if exist, don't repeat
    save_filename = os.path.join(drift_folder, fovs[fov_id].replace('.dax', save_postfix))
    # if savefile already exists, load
    if os.path.exists(save_filename):
        total_drift = pickle.load(open(save_filename,'rb'))
    else: # else create an empty dic
        total_drift = {}
    # if do illumination_correction:
    if illumination_correction:
        _bead_ims = [Illumination_correction(_im,ic_channel, correction_folder=correction_folder, verbose=False) for _im in bead_ims]
    else:
        _bead_ims = bead_ims

    # repeat if no fitted data exist
    # initialize drifts
    if len(_bead_ims) == 0:
        raise ValueError("Wrong dimension of _bead_ims, at least 1 image required")
    else:
        # initialize old_ref_frame
        old_ref_frame = ''
        txyzs = [np.array([0.,0.,0.])]; # initialize with zeros, representing image0 itself
        if len(total_drift) > 0 and bead_names[0] in total_drift:
            change_markers[bead_names[0]] = False # marker for whether made any changes
        else: # ref frame is not in total_drift dic: create a zero array
            if verbose:
                print("Ref frame changed")
            # old ref-frame
            for _hyb_name, _dft in list(total_drift.items()):
                if np.sum(_dft == 0 ) == len(_dft):
                    print('old-ref:',_hyb_name)
                    old_ref_frame = _hyb_name
                    del total_drift[_hyb_name]
            # new ref-frame
            change_markers[bead_names[0]] = True
            total_drift[bead_names[0]] = np.array([0,0,0])
            if verbose:
                print(f"-- Modifying existing refrence frame to {bead_names[0]}")

    # define selected coordinates in each field of view
    if coord_sel is None:
        coord_sel = np.array(_bead_ims[0].shape)/2
    coord_sel1 = np.array([0,-drift_size,-drift_size]) + coord_sel
    coord_sel2 = np.array([0,drift_size,drift_size]) + coord_sel

    # initialize ref image
    ref = 0; # initialize reference id
    im_ref = _bead_ims[ref]; # initialize reference image
    # dynamic seeding threhold
    if dynamic:
        if np.max(im_ref) < 30000:
            th_seed = scoreatpercentile(im_ref, dynamic_th_percent) * 0.5;
    # start fitting image 0
    im_ref_sm = visual_tools.grab_block(im_ref,coord_sel1,[drift_size]*3)
    cents_ref1 = get_STD_centers(im_ref_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the ref cube 1
    im_ref_sm = visual_tools.grab_block(im_ref,coord_sel2,[drift_size]*3)
    cents_ref2 = get_STD_centers(im_ref_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the ref cube 2

    for iim,(im, _name) in enumerate(zip(_bead_ims[1:], bead_names[1:])):
        # check if key exists
        if _name in total_drift and not overwrite:
            change_markers[_name] = False
            continue
        else:
            change_markers[_name] = True
            # dynamic seeding
            if dynamic:
                th_seed = scoreatpercentile(im, dynamic_th_percent) * 0.45;
            # fit target image
            im_sm = visual_tools.grab_block(im,coord_sel1,[drift_size]*3)
            cents1 = get_STD_centers(im_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the cube 1
            im_sm = visual_tools.grab_block(im,coord_sel2,[drift_size]*3)
            cents2 = get_STD_centers(im_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the cube 2
            if verbose:
                print("Aligning "+str(iim+1), _name)
            # calculate drift
            txyz1 = visual_tools.translation_aling_pts(cents_ref1,cents1,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=plt_val)
            txyz2 = visual_tools.translation_aling_pts(cents_ref2,cents2,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=plt_val)
            txyz = (txyz1+txyz2)/2.
            # if two drifts are really different:
            if np.sum(np.abs(txyz1-txyz2))>3:
                fail_count += 1; # count times of suspected failure
                print(f"Suspecting failure for {_name}")
                #drift_size+=10
                coord_sel3 = np.array([0,drift_size,-drift_size])+coord_sel
                im_ref_sm = visual_tools.grab_block(im_ref,coord_sel3,[drift_size]*3)
                cents_ref3 = get_STD_centers(im_ref_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the ref cube 3
                im_sm = visual_tools.grab_block(im,coord_sel3,[drift_size]*3)
                cents3 = get_STD_centers(im_sm, th_seed=th_seed, verbose=verbose)#list of fits of beads in the cube 3
                txyz3 = visual_tools.translation_aling_pts(cents_ref3,cents3,cutoff=cutoff_,xyz_res=xyz_res_,plt_val=plt_val)
                #print txyz1,txyz2,txyz3
                if np.sum(np.abs(txyz3-txyz1))<np.sum(np.abs(txyz3-txyz2)):
                    txyz = (txyz1+txyz3)/2.
                    print(txyz1,txyz3)
                else:
                    txyz = (txyz2+txyz3)/2.
                    print(txyz2,txyz3)

            txyzs.append(txyz)
            # store in total_drift if overwrite / not exist
            total_drift[_name] = sum(txyzs)
            # inherit centers and ref image
            cents_ref1 = cents1
            cents_ref2 = cents2
            ref = iim
            im_ref = im
    # if ref-frame changed, modify old drift files:
    if old_ref_frame != '':
        ref_drift = total_drift[old_ref_frame]
        for _hyb_name, _mk in change_markers.items():
            if not _mk: # if not changed
                total_drift[_hyb_name] += ref_drift
                if verbose:
                    print(f"Update drift for {_hyb_name}")

    # if any_changes exist and save, do save
    if True in list(change_markers.values()) and save:
        if not os.path.exists(drift_folder):
            os.makedirs(drift_folder)
        pickle.dump(total_drift,open(save_filename,'wb'))
    if verbose:
        print("-- total time spent in drift correction:", time.time()-_start_time)

    return total_drift, fail_count


# function to generate illumination profiles
def generate_illumination_correction(ims, threshold_percentile=98, gaussian_sigma=40,
                                     save=True, save_name='', save_dir=r'.', make_plot=False):
    '''Take into a list of beads images, report a mean illumination strength image'''
    from scipy import ndimage
    import os
    import pickle as pickle
    from scipy.stats import scoreatpercentile
    # initialize total image
    _ims = ims
    total_ims = []
    # calculate total(average) image
    for _i,_im in enumerate(_ims):
        im_stack = _im.mean(0)
        threshold = scoreatpercentile(_im, threshold_percentile);
        im_stack[im_stack > threshold] = np.nan;
        total_ims.append(im_stack);
    # gaussian fliter total image to denoise
    total_im = np.nanmean(np.array(total_ims),0)
    total_im[np.isnan(total_im)] = np.nanmean(total_im)
    if gaussian_sigma:
        fit_im = ndimage.gaussian_filter(total_im, gaussian_sigma);
    else:
        fit_im = total_im;
    fit_im = fit_im / np.max(fit_im);

    # make plot
    if make_plot:
        f = plt.figure()
        plt.imshow(fit_im)
        plt.title('Illumination profile for '+ save_name)
        plt.colorbar()
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_filename = save_dir + os.sep + 'illumination_correction_'+save_name+'.pkl';
        pickle.dump(fit_im, open(save_filename, 'wb'));
        if make_plot:
            plt.savefig(save_filename.replace('.pkl','.png'));
    return fit_im

# illumination correction for one image

def Illumination_correction(im, correction_channel, crop_limits=None, all_channels=_allowed_colors,
                            single_im_size=_image_size, correction_folder=_correction_folder,
                            profile_dtype=np.float, image_dtype=np.uint16,
                            ic_profile_name='illumination_correction', correction_power=1, verbose=True):
    """Function to do fast illumination correction in a RAM-efficient manner
    Inputs:
        im: 3d image, np.ndarray or np.memmap
        channel: the color channel for given image, int or string (should be in single_im_size list)
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        correction_power: power for correction factor, float (default: 1)
        verbose: say something!, bool (default: True)
    Outputs:
        corr_im: corrected image
    """
    ## check inputs
    # im
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            f"Wrong input type for im: {type(im)}, np.ndarray or np.memmap expected")
    if verbose:
        print(f"-- correcting illumination for image size:{im.shape} for channel:{correction_channel}")
    # channel
    channel = str(correction_channel)
    if channel not in all_channels:
        raise ValueError(
            f"Input channel:{channel} is not in allowed channels:{allowed_channels}")
    # check correction profile exists:
    ic_filename = os.path.join(correction_folder,
                               ic_profile_name+'_'+str(channel)+'_'
                               + str(single_im_size[-2])+'x'+str(single_im_size[-1])+'.npy')
    if not os.path.isfile(ic_filename):
        raise IOError(
            f"Illumination correction profile file:{ic_filename} doesn't exist, exit!")
    # image shape and ref-shape
    im_shape = np.array(im.shape, dtype=np.int)
    single_im_size = np.array(single_im_size, dtype=np.int)
    # check crop_limits
    if crop_limits is None:
        if (im_shape[-2:]-single_im_size[-2:]).any():
            raise ValueError(
                f"Test image is not of full image size:{single_im_size}, while crop_limits are not given, exit!")
        else:
            # change crop_limits into full image size
            crop_limits = np.stack([np.zeros(3), im_shape]).T
    elif len(crop_limits) <= 1 or len(crop_limits) > 3:
        raise ValueError("crop_limits should have 2 or 3 elements")
    elif len(crop_limits) == 2:
        crop_limits = np.stack(
            [np.array([0, im_shape[0]])] + list(crop_limits)).astype(np.int)
    else:
        crop_limits = np.array(crop_limits, dtype=np.int)
    # convert potential negative values to positive for further calculation
    for _s, _lims in zip(im_shape, crop_limits):
        if _lims[1] < 0:
            _lims[1] += _s
    crop_shape = crop_limits[:, 1] - crop_limits[:, 0]
    # crop image if necessary
    if not (im_shape[-2:]-single_im_size[-2:]).any():
        cim = im[crop_limits[0, 0]:crop_limits[0, 1],
                 crop_limits[1, 0]:crop_limits[1, 1],
                 crop_limits[2, 0]:crop_limits[2, 1], ]
    elif not (im_shape-crop_shape).any():
        cim = im
    else:
        raise IndexError(
            f"Wrong input size for im:{im_shape} compared to crop_limits:{crop_limits}")

    ## do correction
    # get cropped correction profile
    cropped_profile = visual_tools.slice_2d_image(
        ic_filename, single_im_size[-2:], crop_limits[1], crop_limits[2], image_dtype=profile_dtype)
    # correction
    corr_im = (cim / cropped_profile**correction_power).astype(image_dtype)

    return corr_im

# Function to generate chromatic abbrevation profile
def generate_chromatic_abbrevation_correction(ims, names, master_folder, channels, corr_channel, ref_channel,
                                              fitting_save_subdir=r'Analysis\Beads', seed_th_per=99.92,
                                              correction_folder=_correction_folder,
                                              make_plot=False,
                                              save=True, save_folder=_correction_folder,
                                              force=False, verbose=True):
    '''Generate chromatic abbrevation profile from list of images
    Inputs:
        ims: images, list of 3D image for beads, in multi color
        names: list of corresponding names
        master_folder: master directory of beads data, string
        channels: list of channels used in this data, list of string or ints
        corr_channel: channel to be corrected, str or int
        ref_channel: reference channel, str or int
        fitting_save_subdir: sub_directory under master_folder to save fitting results, str (default: None)
        seed_th_per: intensity percentile for seeds during beads-fitting, float (default: 99.92)
        correction_folder: full directory for illumination correction profiles, string (default: ...)
        save: whether directly save profile, bool (default: True)
        save_folder: full path to save correction file, str (default: correction_folder)
        force: do fitting and saving despite of existing files, bool (default: False)
        verbose: say something!, bool (default: True)
    Output:
        _cc_profiles: list of correction profiles (in order of z,x,y, based on x,y coordinates)
    '''
    # Check inputs
    if len(ims) != len(names): # check length
        raise ValueError('Input images and names doesnt match!');
    _default_channels = ['750','647','561','488','405'];
    _channels = [str(_ch) for _ch in channels];
    for _ch in _channels: # check channels
        if _ch not in _default_channels:
            raise ValueError('Channel '+_ch+" not exist in default channels");
    if str(corr_channel) not in _channels:
        raise ValueError("correction channel "+str(corr_channel)+" is not given in channels");
    if str(ref_channel) not in _channels:
        raise ValueError("reference channel "+str(ref_channel)+" is not given in channels");

    # imports
    from scipy.stats import scoreatpercentile
    import os
    import matplotlib.pyplot as plt
    import scipy.linalg

    # split images into channels
    _splitted_ims = get_img_info.split_channels(ims, names, num_channel=len(_channels), buffer_frames=10, DAPI=False)
    _cims = _splitted_ims[_channels.index(str(corr_channel))]
    _rims = _splitted_ims[_channels.index(str(ref_channel))]
    # initialize list of beads centers:
    _ccts, _rcts, _shifts = [], [], []

    # loop through all images and calculate profile
    for _cim,_rim,_name in zip(_cims,_rims,names):
        try:
            # fit correction channel
            _cim = Illumination_correction(_cim, corr_channel, correction_folder=correction_folder,
                                                        verbose=verbose)[0]
            _cct = get_STD_centers(_cim, scoreatpercentile(_cim, seed_th_per), verbose=verbose,
                                        save=save, force=force, save_folder=master_folder+os.sep+fitting_save_subdir,
                                        save_name=_name.split(os.sep)[-1].replace('.dax', '_'+str(corr_channel)+'_fitting.pkl'))
            # fit reference channel
            _rim = Illumination_correction(_rim, ref_channel, correction_folder=correction_folder,
                                                        verbose=verbose)[0]
            _rct = get_STD_centers(_rim, scoreatpercentile(_rim, seed_th_per), verbose=verbose,
                                        save=save, force=force, save_folder=master_folder+os.sep+fitting_save_subdir,
                                        save_name=_name.split(os.sep)[-1].replace('.dax', '_'+str(ref_channel)+'_fitting.pkl'))
            # Align points
            _aligned_cct, _aligned_rct, _shift = visual_tools.beads_alignment_fast(_cct ,_rct, outlier_sigma=1, unique_cutoff=2)
            # append
            _ccts.append(_aligned_cct)
            _rcts.append(_aligned_rct)
            _shifts.append(_shift)
            # make plot
            if make_plot:
                fig = plt.figure()
                plt.plot(_aligned_rct[:,2], _aligned_rct[:,1],'r.', alpha=0.3)
                plt.quiver(_aligned_rct[:,2], _aligned_rct[:,1], _shift[:,2], _shift[:,1])
                plt.imshow(_rim.sum(0))
                plt.show()
        except:
            pass
    ## do fitting to explain chromatic abbrevation
    # merge
    _corr_beads = np.concatenate(_ccts)
    _ref_beads = np.concatenate(_rcts)
    _shift_beads = np.concatenate(_shifts)
    # initialize
    dz, dx, dy = _cims[0].shape
    _cc_profiles = []
    # loop through
    for _j in range(3): # 3d correction
        if _j == 0: # for z, do order-2 polynomial fitting
            _data = np.concatenate((_ref_beads[:,1:]**2, (_ref_beads[:,1] * _ref_beads[:,2])[:,np.newaxis], _ref_beads[:,1:], np.ones([_ref_beads.shape[0],1])),1);
            _C,_r,_,_ = scipy.linalg.lstsq(_data, _shift_beads[:,_j])    # coefficients
            if verbose:
                print('Z axis fitting R^2=', 1 - _r / sum((_shift_beads[:,_j] - _shift_beads[:,_j].mean())**2))
            _cc_profile = np.zeros([dx,dy])
            for _m in range(dx):
                for _n in range(dy):
                    _cc_profile[_m, _n] = np.dot(_C, [_m**2, _n**2, _m*_n, _m, _n, 1]);
        else: # for x and y, do linear decomposition
            _data = np.concatenate((_ref_beads[:,1:], np.ones([_ref_beads.shape[0],1])),1);
            _C,_r,_,_ = scipy.linalg.lstsq(_data, _shift_beads[:,_j])    # coefficients
            if verbose:
                print('axis'+str(_j)+' fitting R^2=', 1 - _r / sum((_shift_beads[:,_j] - _shift_beads[:,_j].mean())**2))
            _cc_profile = np.zeros([dx,dy])
            for _m in range(dx):
                for _n in range(dy):
                    _cc_profile[_m, _n] = np.dot(_C, [_m,_n,1])
        _cc_profiles.append(_cc_profile)
    if save:
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        _save_file = save_folder + os.sep + "Chromatic_correction_"+str(corr_channel)+'_'+str(ref_channel)+'.pkl';
        if verbose:
            print("-- save chromatic abbrevation correction to file", _save_file)
        pickle.dump(_cc_profiles, open(_save_file, 'wb'))
        if make_plot:
            for _d, _cc in enumerate(_cc_profiles):
                plt.figure()
                plt.imshow(_cc)
                plt.colorbar()
                plt.savefig(_save_file.replace('.pkl', '_'+str(_d)+'.png'))

    return _cc_profiles

def Chromatic_abbrevation_correction(im, correction_channel, target_channel='647', crop_limits=None, 
                                     all_channels=_allowed_colors, single_im_size=_image_size,
                                     correction_folder=_correction_folder, 
                                     profile_dtype=np.float, image_dtype=np.uint16,
                                     cc_profile_name='chromatic_correction', verbose=True):
    """Chromatic abbrevation correction for given image and crop
        im: 3d image, np.ndarray or np.memmap
        correction_channel: the color channel for given image, int or string (should be in single_im_size list)
        target_channel: the target channel that image should be correct to, int or string (default: '647')
        crop_limits: 2d or 3d crop limits given for this image,
            required if im is already sliced, 2x2 or 3x2 np.ndarray (default: None, no cropping at all)
        all_channels: allowed channels to be corrected, list 
        single_im_size: full image size before any slicing, list of 3 (default:[30,2048,2048])
        correction_folder: correction folder to find correction profile, string of path (default: Z://Corrections/)
        correction_power: power for correction factor, float (default: 1)
        verbose: say something!, bool (default: True)"""
    
    ## check inputs
    # im
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            f"Wrong input type for im: {type(im)}, np.ndarray or np.memmap expected")
    if verbose:
        print(f"-- correcting chromatic abbrevation for iamge with size:{im.shape}")
    # correction channel
    correction_channel = str(correction_channel)
    if correction_channel not in all_channels:
        raise ValueError(f"Input channel:{correction_channel} is not in allowed channels:{allowed_channels}")
    # target channel
    target_channel = str(target_channel)
    if target_channel not in all_channels:
        raise ValueError(f"Input channel:{target_channel} is not in allowed channels:{allowed_channels}")
    # if no correction required, directly return
    if correction_channel == target_channel:
        if verbose:
            print(f"-- no chromatic abbrevation required for channel:{_color}")
        return im
    
    # check correction profile exists:
    cc_filename = os.path.join(correction_folder,
                               cc_profile_name+'_'+str(correction_channel)+'_'+str(target_channel)+'_'
                               + str(single_im_size[-2])+'x'+str(single_im_size[-1])+'.npy')
    if not os.path.isfile(cc_filename):
        raise IOError(
            f"Chromatic correction profile file:{cc_filename} doesn't exist, exit!")
    
    # image shape and ref-shape
    im_shape = np.array(im.shape, dtype=np.int)
    single_im_size = np.array(single_im_size, dtype=np.int)
    
    # check crop_limits
    if crop_limits is None:
        if (im_shape[-2:]-single_im_size[-2:]).any():
            raise ValueError(
                f"Test image is not of full image size:{single_im_size}, while crop_limits are not given, exit!")
        else:
            # change crop_limits into full image size
            crop_limits = np.stack([np.zeros(3), im_shape]).T
    elif len(crop_limits) <= 1 or len(crop_limits) > 3:
        raise ValueError("crop_limits should have 2 or 3 elements")
    elif len(crop_limits) == 2:
        crop_limits = np.stack(
            [np.array([0, im_shape[0]])] + list(crop_limits)).astype(np.int)
    else:
        crop_limits = np.array(crop_limits, dtype=np.int)
    
    # convert potential negative values to positive for further calculation
    for _s, _lims in zip(im_shape, crop_limits):
        if _lims[1] < 0:
            _lims[1] += _s
    crop_shape = crop_limits[:, 1] - crop_limits[:, 0]
    
    # crop image if necessary
    if not (im_shape[-2:]-single_im_size[-2:]).any():
        cim = im[crop_limits[0, 0]:crop_limits[0, 1],
                 crop_limits[1, 0]:crop_limits[1, 1],
                 crop_limits[2, 0]:crop_limits[2, 1] ]
    elif not (im_shape-crop_shape).any():
        cim = im
    else:
        raise IndexError(f"Wrong input size for im:{im_shape} compared to crop_limits:{crop_limits}")
    
    ## do correction
    # 1. get coordiates to be mapped
    _coords = np.meshgrid( np.arange(crop_limits[0][1]-crop_limits[0][0]), 
                           np.arange(crop_limits[1][1]-crop_limits[1][0]), 
                           np.arange(crop_limits[2][1]-crop_limits[2][0]))
    _coords = np.stack(_coords).transpose((0, 2, 1, 3)) # transpose is necessary
    # 2. load chromatic profile
    _cropped_cc_profile = visual_tools.slice_image(cc_filename, [3, single_im_size[1], single_im_size[2]],
                                                   [0,3], [crop_limits[1][0],crop_limits[1][1]],
                                                   [crop_limits[2][0],crop_limits[2][1]], image_dtype=profile_dtype)
    # 3. calculate corrected coordinates as a reference
    _corr_coords = _coords + _cropped_cc_profile[:,np.newaxis]
    # 4. map coordinates
    _corr_im = map_coordinates(cim, _corr_coords.reshape(_corr_coords.shape[0], -1), mode='nearest')
    _corr_im = _corr_im.reshape(np.shape(cim))

    return _corr_im

## FFT alignment, used for fast alignment
def fftalign(im1,im2,dm=100,plt_val=False):
	"""
	Inputs: 2 images im1, im2 and a maximum displacement max_disp.
	This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
	"""
	from scipy.signal import fftconvolve
	sh = np.array(im2.shape)
	dim1,dim2 = np.max([sh-dm,sh*0],0),sh+dm
	im2_=np.array(im2[dim1[0]:dim2[0],dim1[1]:dim2[1],dim1[2]:dim2[2]][::-1,::-1,::-1],dtype=float)
	im2_-=np.mean(im2_)
	im2_/=np.std(im2_)
	sh = np.array(im1.shape)
	dim1,dim2 = np.max([sh-dm,sh*0],0),sh+dm
	im1_=np.array(im1[dim1[0]:dim2[0],dim1[1]:dim2[1],dim1[2]:dim2[2]],dtype=float)
	im1_-=np.mean(im1_)
	im1_/=np.std(im1_)
	im_cor = fftconvolve(im1_,im2_, mode='full')

	xyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)
	if np.sum(im_cor>0)>0:
		im_cor[im_cor==0]=np.min(im_cor[im_cor>0])
	else:
		im_cor[im_cor==0]=0
	if plt_val:
		plt.figure()
		x,y=xyz[-2:]
		im_cor_2d = np.array(im_cor)
		while len(im_cor_2d.shape)>2:
			im_cor_2d = np.max(im_cor_2d,0)
		plt.plot([y],[x],'ko')
		plt.imshow(im_cor_2d,interpolation='nearest')
		plt.show()
	xyz=np.round(-1 * np.array(im_cor.shape)/2.+xyz).astype(int)
	return xyz

## Translate images given drift
def fast_translate(im,trans):
	shape_ = im.shape
	zmax=shape_[0]
	xmax=shape_[1]
	ymax=shape_[2]
	zmin,xmin,ymin=0,0,0
	trans_=np.array(np.round(trans),dtype=int)
	zmin-=trans_[0]
	zmax-=trans_[0]
	xmin-=trans_[1]
	xmax-=trans_[1]
	ymin-=trans_[2]
	ymax-=trans_[2]
	im_base_0 = np.zeros([zmax-zmin,xmax-xmin,ymax-ymin])
	im_zmin = min(max(zmin,0),shape_[0])
	im_zmax = min(max(zmax,0),shape_[0])
	im_xmin = min(max(xmin,0),shape_[1])
	im_xmax = min(max(xmax,0),shape_[1])
	im_ymin = min(max(ymin,0),shape_[2])
	im_ymax = min(max(ymax,0),shape_[2])
	im_base_0[(im_zmin-zmin):(im_zmax-zmin),(im_xmin-xmin):(im_xmax-xmin),(im_ymin-ymin):(im_ymax-ymin)]=im[im_zmin:im_zmax,im_xmin:im_xmax,im_ymin:im_ymax]
	return im_base_0

# correct for illumination _shifts across z layers
def Z_Shift_Correction(im, dtype=np.uint16, verbose=False):
    '''Function to correct for each layer in z, to make sure they match in term of intensity'''
    if verbose:
        print("-- correcting Z axis illumination shifts.")
    _nim = im / np.mean(im, axis=(1, 2))[:,np.newaxis,np.newaxis] * np.mean(im)
    return _nim.astype(dtype)

# remove hot pixels
def Remove_Hot_Pixels(im, dtype=np.uint16, hot_pix_th=0.50, hot_th=4, 
                      interpolation_style='nearest', verbose=False):
    '''Function to remove hot pixels by interpolation in each single layer'''
    if verbose:
        print("-- removing hot pixels")
    # create convolution matrix, ignore boundaries for now
    _conv = (np.roll(im,1,1)+np.roll(im,-1,1)+np.roll(im,1,2)+np.roll(im,1,2))/4
    # hot pixels must be have signals higher than average of neighboring pixels by hot_th in more than hot_pix_th*total z-stacks
    _hotmat = im > hot_th * _conv
    _hotmat2D = np.sum(_hotmat,0)
    _hotpix_cand = np.where(_hotmat2D > hot_pix_th*np.shape(im)[0])
    # if no hot pixel detected, directly exit
    if len(_hotpix_cand[0]) == 0:
        return im
    # create new image to interpolate the hot pixels with average of neighboring pixels
    _nim = im.copy()
    if interpolation_style == 'nearest':
        for _x, _y in zip(_hotpix_cand[0],_hotpix_cand[1]):
            if _x > 0 and  _y > 0 and _x < im.shape[1]-1 and  _y < im.shape[2]-1:
                _nim[:,_x,_y] = (_nim[:,_x+1,_y]+_nim[:,_x-1,_y]+_nim[:,_x,_y+1]+_nim[:,_x,_y-1])/4
    return _nim.astype(dtype)


# fast function to generate illumination profiles
def fast_generate_illumination_correction(color, data_folder, correction_folder, image_type='H', num_of_images=50,
                                          folder_id=0, buffer_frame=10, frame_per_color=30, target_color_ind=-1,
                                          gaussian_sigma=40, seeding_th_per=99.5, seeding_th_base=300, seeding_crop_size=9,
                                          remove_cap=False, cap_th_per=99.5,
                                          force=False, save=True, save_name='illumination_correction_', make_plot=False, verbose=True):
    """Function to generate illumination correction profile from hybridization type of image or bead type of image
    Inputs:
        color: 
        data_folder: master folder for images, string of path
        correction_folder: folder to save correction_folder, string of path
        image_type: type of images to generate this profile, int (default: 50)
        gaussian_sigma: sigma of gaussian filter used to smooth image, int (default: 40)
    Outputs:
        ic_profile: 2d illumination correction profile 
        """
    from scipy.stats import scoreatpercentile
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve
    ## check inputs
    # color
    _color = str(color)
    _allowed_colors = ['750','647','561','488','405']
    if _color not in _allowed_colors:
        raise ValueError(f"Wrong color input, {color} is given, color among {_allowed_colors} is expected")
    # extract index of this color
    if target_color_ind == -1:
        target_color_ind = _allowed_colors.index(_color)
    # image type
    image_type = image_type[0].upper()
    if image_type not in ['H', 'B']:
        raise ValueError(f"Wrong input image_type, {image_type} is given, 'H' or 'B' expected!")
    ## get images
    _folders, _fovs = get_img_info.get_folders(data_folder, feature=image_type,verbose=verbose)
    if len(_folders)==0 or len(_fovs)==0:
        raise IOError(f"No folders or fovs detected with given data_folder:{data_folder} and image_type:{image_type}!")
    ## load image info
    _im_filename = os.path.join(_folders[folder_id], _fovs[0])
    _info_filename = _im_filename.replace('.dax', '.inf')
    with open(_info_filename, 'r') as _info_hd:
        _infos = _info_hd.readlines()
    # get frame number and color information
    _num_frame, _num_color = 0, 0
    _dx, _dy = 0, 0
    for _line in _infos:
        _line = _line.rstrip()
        if "number of frames" in _line:
            _num_frame = int(_line.split('=')[1])
            _num_color = (_num_frame - 2*buffer_frame) / frame_per_color
            if _num_color != int(_num_color):
                raise ValueError("Wrong num_color, should be integer!")
            _num_color = int(_num_color)
        if "frame dimensions" in _line:
            _dx = int(_line.split('=')[1].split('x')[0])
            _dy = int(_line.split('=')[1].split('x')[1])
    if not _num_frame or not _num_color:
        raise ValueError(f"No frame number info exists in {_info_filename}")
    if not _dx or not _dy:
        raise ValueError(f"No frame dimension info exists in {_info_filename}")    
    # save filename    
    save_filename = os.path.join(correction_folder, save_name+str(_color)+'_'+str(_dx)+'x'+str(_dy))
    if os.path.isfile(save_filename+'.npy') and not force:
        if verbose:
            print(f"- directly loading illumination correction profile from file:{save_filename}.npy")
        _mean_profile = np.load(save_filename+'.npy')
    else:
        # initialize 
        _picked_profiles = []
        _fd = _folders[folder_id]
        for _fov in _fovs:
            # exit if enough images loaded
            if len(_picked_profiles) >= num_of_images:
                break
            # start slicing image
            if os.path.isfile(os.path.join(_fd, _fov)):
                ## get info
                _im_filename = os.path.join(_fd, _fov)
                if verbose:
                    print(f"-- loading {_color} from image file {_im_filename}, color_ind:{target_color_ind}")
                _im = visual_tools.slice_image(_im_filename, [_num_frame, _dx, _dy], [buffer_frame, _num_frame-buffer_frame], [0, _dx],
                                                [0, _dy], _num_color, target_color_ind)
                # do corrections
                _im = Remove_Hot_Pixels(_im, verbose=False)
                _im = Z_Shift_Correction(_im, normalization=False, verbose=False)
                _im = np.array(_im, dtype=np.float)
                # remove top values if necessary
                if remove_cap:
                    _cap_thres = scoreatpercentile(_im, cap_th_per)
                    _im[_im > _cap_thres] = np.nan
                # seed and exclude these blocks
                _seed_thres = scoreatpercentile(_im, seeding_th_per) - np.median(_im)
                _seeds = visual_tools.get_seed_points_base(_im, th_seed=_seed_thres)
                for _sd in np.transpose(_seeds):
                    _l_lims = np.array(_sd - int(seeding_crop_size/2), dtype=np.int)
                    _l_lims[_l_lims<0] = 0
                    _r_lims = [_sd+int(seeding_crop_size/2), np.array(_im.shape)]
                    _r_lims = np.min(np.array(_r_lims, dtype=np.int), axis=0)
                    _im[_l_lims[0]:_r_lims[0], _l_lims[1]:_r_lims[1], _l_lims[2]:_r_lims[2]] = np.nan
                # append
                _picked_profiles.append(_im)
        # generate averaged profile
        if verbose:
            print("- generating averaged profile")
        _mean_profile = np.nanmean(np.concatenate(_picked_profiles), axis=0)
        _mean_profile[_mean_profile == 0] = np.nan
        ## gaussian filter
        if verbose:
            print("- applying gaussian filter to averaged profile")
        # set gaussian kernel
        _kernel = Gaussian2DKernel(x_stddev=gaussian_sigma)
        # convolution, which will interpolate any NaN numbers
        _mean_profile = convolve(_mean_profile, _kernel, boundary='extend')
        _mean_profile = _mean_profile / np.max(_mean_profile)
        if save:
            if verbose:
                print(f"-- saving correction profile to file:{save_filename}.npy")
            if not os.path.exists(os.path.dirname(save_filename)):
                os.makedirs(os.path.dirname(save_filename))
            np.save(save_filename, _mean_profile)
    if make_plot:
        plt.figure()
        plt.imshow(_mean_profile)
        plt.colorbar()
        plt.show()

    return _mean_profile


def fast_generate_chromatic_abbrevation_from_spots(corr_spots, ref_spots, corr_channel, ref_channel, 
                                                   image_size=_image_size, fitting_order=2,
                                                   correction_folder=_correction_folder, make_plot=False, 
                                                   save=True, save_name='chromatic_correction_',force=False, verbose=True):
    """Code to generate chromatic abbrevation from fitted and matched spots"""
    ## check inputs
    if len(corr_spots) != len(ref_spots):
        raise ValueError("corr_spots and ref_spots are of different length, so not matched")
    # color 
    _allowed_colors = ['750', '647', '561', '488', '405']
    if str(corr_channel) not in _allowed_colors:
        raise ValueError(f"corr_channel given:{corr_channel} is not valid, {_allowed_colors} are expected")
    if str(ref_channel) not in _allowed_colors:
        raise ValueError(f"corr_channel given:{ref_channel} is not valid, {_allowed_colors} are expected")
        
    ## savefile
    filename_base = save_name + str(corr_channel)+'_'+str(ref_channel)+'_'+str(image_size[1])+'x'+str(image_size[2])
    saved_profile_filename = os.path.join(correction_folder, filename_base+'.npy')
    saved_const_filename = os.path.join(correction_folder, filename_base+'_const.npy')
    # whether have existing profile
    if os.path.isfile(saved_profile_filename) and os.path.isfile(saved_const_filename) and not force:
        _cac_profiles = np.load(saved_profile_filename)
        _cac_consts = np.load(saved_const_filename)
        if make_plot:
            for _i,_grid_shifts in enumerate(_cac_profiles):
                plt.figure()
                plt.imshow(_grid_shifts)
                plt.colorbar()
                plt.title(f"chromatic-abbrevation {corr_channel} to {ref_channel}, axis-{_i}")
                plt.show()
    else:
        ## start correction
        _cac_profiles = []
        _cac_consts = []
        # variables used in polyfit
        ref_spots, corr_spots = np.array(ref_spots), np.array(corr_spots) # convert to array
        _x = ref_spots[:,1]
        _y = ref_spots[:,2]
        _data = [] # variables in polyfit
        for _order in range(fitting_order+1): # loop through different orders
            for _p in range(_order+1):
                _data.append(_x**_p * _y**(_order-_p))
        _data = np.array(_data).transpose()

        for _i in range(3): # 3D
            if verbose:
                print(f"-- fitting chromatic-abbrevation in axis {_i} with order:{fitting_order}")
            _value =  corr_spots[:,_i] - ref_spots[:,_i] # target-value for polyfit
            _C,_r,_r2,_r3 = scipy.linalg.lstsq(_data, _value)    # coefficients and residues
            _cac_consts.append(_C) # store correction constants
            _rsquare =  1 - np.var(_data.dot(_C) - _value)/np.var(_value)
            if verbose:
                print(f"--- fitted rsquare:{_rsquare}")

            ## generate correction function
            def _get_shift(coords):
                # traslate into 2d
                if len(coords.shape) == 1:
                    coords = coords[np.newaxis,:]
                _cx = coords[:,1]
                _cy = coords[:,2]
                _corr_data = []
                for _order in range(fitting_order+1):
                    for _p in range(_order+1):
                        _corr_data.append(_cx**_p * _cy**(_order-_p))
                _shift = np.dot(np.array(_corr_data).transpose(), _C)
                return _shift

            ## generate correction_profile
            _xc_t, _yc_t = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[2])) # initialize meshgrid
            _xc, _yc = _xc_t.transpose(), _yc_t.transpose()
            # transpose to the shape compatible with function
            _grid_coords = np.array([np.zeros(np.size(_xc)), _xc.reshape(-1), _yc.reshape(-1)]).transpose() 
            # calculate shift and trasform back
            _grid_shifts = _get_shift(_grid_coords)
            _grid_shifts = _grid_shifts.reshape(np.shape(_xc))
            _cac_profiles.append(_grid_shifts) # store correction profile across 2d

            ## make plot
            if make_plot:
                plt.figure()
                plt.imshow(_grid_shifts)
                plt.colorbar()
                plt.title(f"chromatic-abbrevation {corr_channel} to {ref_channel}, axis-{_i}")
                plt.show()
        _cac_profiles = np.array(_cac_profiles)
        _cac_consts = np.array(_cac_consts)
        # save 
        if save:
            if verbose:
                print("-- save profiles to file:{saved_profile_filename}")
            np.save(saved_profile_filename.split('.npy')[0], _cac_profiles)
            if verbose:
                print("-- save shift functions to file:{saved_const_filename}")
            np.save(saved_const_filename.split('.npy')[0], _cac_consts)
    
    def _cac_func(coords, consts=_cac_consts, max_order=fitting_order):
        # traslate into 2d
        if len(coords.shape) == 1:
            coords = coords[np.newaxis,:]
        _cx = coords[:,1]
        _cy = coords[:,2]
        _corr_data = []
        for _order in range(max_order+1):
            for _p in range(_order+1):
                _corr_data.append(_cx**_p * _cy**(_order-_p))
        _shifts = []
        for _c in consts:
            _shifts.append(np.dot(np.array(_corr_data).transpose(), _c))
        _shifts = np.array(_shifts).transpose()
        _corr_coords = coords - _shifts
        return _corr_coords    
    
    return _cac_profiles, _cac_func

def fast_chromatic_abbrevation_correction(im, correction_channel, target_channel='647', single_im_size=_image_size,
                                          crop_limits=None, all_channels=_allowed_colors,
                                          buffer_frame=10, frame_per_color=30, target_color_ind=-1,
                                          correction_folder=_correction_folder, cac_profile=None,
                                          verbose=True):
    """Chromatic abbrevation correction"""
    from scipy.ndimage.interpolation import map_coordinates
    ## check inputs
    # color
    _color = str(correction_channel)
    all_channels = ['750', '647', '561', '488', '405']
    if _color not in all_channels:
        raise ValueError(
            f"Wrong color input, {_color} is given, color among {all_channels} is expected")
    if target_color_ind == -1:
        target_color_ind = all_channels.index(_color)
    # crop_limits
    if crop_limits is not None:
        if len(crop_limits) <= 1 or len(crop_limits) > 3:
            raise ValueError("crop_limits should have 2 elements")
        else:
            for _lims in crop_limits:
                if len(_lims) < 2:
                    raise ValueError(
                        f"given limit {_lims} has less than 2 elements, exit")
    # if no correction required, directly return
    if _color == target_channel:
        if verbose:
            print(f"-- no chromatic abbrevation required for channel:{_color}")
        return im
    else:
        if verbose:
            print(f"-- chromatic abbrevation for image with size:{im.shape}")
    if cac_profile is not None and (isinstance(cac_profile, list) or isinstance(cac_profile, np.ndarray)):
        # directly adopt from input profile
        _cac_profile = cac_profile
        print("direct chromatic")
    elif cac_profile is not None and "multiprocessing" in str(type(cac_profile)):
        _cac_shape = np.array([3, single_im_size[1], single_im_size[2]], dtype=int)
        _cac_profile = np.frombuffer(cac_profile).reshape(_cac_shape)
    else:
        # load profile from correction file
        _cac_profile_file = os.path.join(correction_folder,
                                        'chromatic_correction_'+str(_color)+'_'+str(target_channel)+'_'+str(single_im_size[1])+'x'+str(single_im_size[2])+'.npy')
        if not os.path.isfile(_cac_profile_file):
            raise IOError(
                f"chromatic abbreviation correction file:{_cac_profile_file} doesn't exist, exit! ")
        else:
            _cac_profile = np.load(_cac_profile_file)
    # check image
    if not isinstance(im, np.ndarray) and not isinstance(im, np.memmap):
        raise ValueError(
            "Wrong input type for im, should be np.ndarray or memmap")

    ## process the whole image
    if crop_limits is None and (np.array(np.shape(im))[:3] == np.array(single_im_size)).all():
        if verbose:
            print(f"-- chromatic abbrevation for the whole image")
        # initialize grid-points
        _coords = np.meshgrid(np.arange(im.shape[0]), np.arange(
            single_im_size[1]), np.arange(single_im_size[2]))
        _coords = np.stack(_coords).transpose((0, 2, 1, 3))
        _corr_coords = _coords + _cac_profile[:, np.newaxis, :, :]
        # map coordinates
        _corr_im = map_coordinates(im, _corr_coords.reshape(
            _corr_coords.shape[0], -1), mode='nearest')
        _corr_im = _corr_im.reshape(single_im_size)
    ## process cropped image
    elif crop_limits is not None:
        if len(crop_limits) == 2:
            if (np.array(np.shape(im))[:3] == np.array(single_im_size)).all():
                _cropped_im = im[:, crop_limits[0][0]:crop_limits[0]
                                [1], crop_limits[1][0]:crop_limits[1][1]]
            elif (np.array(np.shape(im))[:3] == np.array([single_im_size[0], crop_limits[0][1]-crop_limits[0][0], crop_limits[1][1]-crop_limits[1][0]])).all():
                _cropped_im = im.copy()
            else:
                raise ValueError(
                    "Input image should be of original-image-size or cropped image size given by crop_limits")
            if verbose:
                print(f"-- chromatic abbrevation for cropped image")
            # initialize grid-points for cropped region
            _coords = np.meshgrid(np.arange(_cropped_im.shape[0]), np.arange(
                crop_limits[0][1]-crop_limits[0][0]), np.arange(crop_limits[1][1]-crop_limits[1][0]))
            _coords = np.stack(_coords).transpose((0, 2, 1, 3))
            _corr_coords= _coords + _cac_profile[:, np.newaxis,
                                                 crop_limits[0][0]:crop_limits[0][1],
                                                 crop_limits[1][0]:crop_limits[1][1]]
        else: # 3 limits are given:
            if (np.array(np.shape(im))[:3] == np.array(single_im_size)).all():
                _cropped_im = im[crop_limits[0][0]:crop_limits[0][1], 
                                 crop_limits[1][0]:crop_limits[1][1],
                                 crop_limits[2][0]:crop_limits[2][1]
                                 ]
            elif (np.array(np.shape(im))[:3] == np.array([crop_limits[0][1]-crop_limits[0][0], crop_limits[1][1]-crop_limits[1][0], crop_limits[2][1]-crop_limits[2][0]])).all():
                _cropped_im = im.copy()
            else:
                raise ValueError(
                    "Input image should be of original-image-size or cropped image size given by crop_limits")
            if verbose:
                print(f"-- chromatic abbrevation for cropped image")
            # initialize grid-points for cropped region
            _coords = np.meshgrid( np.arange(crop_limits[0][1]-crop_limits[0][0]), 
                                   np.arange(crop_limits[1][1]-crop_limits[1][0]), 
                                   np.arange(crop_limits[2][1]-crop_limits[2][0]))

            _coords = np.stack(_coords).transpose((0, 2, 1, 3))
            _corr_coords = _coords + _cac_profile[:, np.newaxis, 
                                                  crop_limits[1][0]:crop_limits[1][1], 
                                                  crop_limits[2][0]:crop_limits[2][1]]
        # map coordinates
        _corr_im = map_coordinates(_cropped_im, _corr_coords.reshape(
            _corr_coords.shape[0], -1), mode='nearest')
        _corr_im = _corr_im.reshape(np.shape(_cropped_im))
 
    return _corr_im

def load_correction_profile(channel, corr_type, correction_folder=_correction_folder, ref_channel='647', 
                            im_size=_image_size, verbose=False):
    """Function to load chromatic/illumination correction profile"""
    ## check inputs
    # type
    _allowed_types = ['chromatic', 'illumination']
    _type = str(corr_type).lower()
    if _type not in _allowed_types:
        raise ValueError(f"Wrong input corr_type, should be one of {_allowed_types}")
    # channel
    _allowed_channels = ['750', '647', '561', '488', '405']
    _channel = str(channel).lower()
    _ref_channel = str(ref_channel).lower()
    if _channel not in _allowed_channels:
        raise ValueError(f"Wrong input channel, should be one of {_allowed_channels}")
    if _ref_channel not in _allowed_channels:
        raise ValueError(f"Wrong input ref_channel, should be one of {_allowed_channels}")
    ## start loading file
    _basename = _type+'_correction_'+_channel+'_'
    if _type == 'chromatic':
        _basename += _ref_channel+'_'
    _basename += str(im_size[1])+'x'+str(im_size[2])+'.npy'
    # filename 
    _corr_filename = os.path.join(correction_folder, _basename)
    if os.path.isfile(_corr_filename):
        _corr_profile = np.load(_corr_filename)
    elif _type == 'chromatic' and _channel == _ref_channel:
        return None
    else:
        raise IOError(f"File {_corr_filename} doesn't exist, exit!")

    return np.array(_corr_profile)


# merged function to crop and correct single image
def correct_single_image(filename, channel, seg_label=None, drift=np.array([0, 0, 0]),
                         single_im_size=_image_size, all_channels=_allowed_colors, channel_id=None,
                         num_buffer_frames=10, extend_dim=20, correction_folder=_correction_folder,
                         z_shift_corr=True, hot_pixel_remove=True, illumination_corr=True, chromatic_corr=True,
                         return_limits=False, verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        filename: full filename of a dax_file or npy_file for one image, string
        channel: channel to be extracted, str (for example. '647')
        seg_label: segmentation label, 2D array
        drift: 3d drift vector for this image, 1d-array
        single_im_size: z-x-y size of the image, list of 3
        all_channels: all_channels used in this image, list of str(for example, ['750','647','561'])
        raw_im: directly give image rather than loading, this will overwrite filename etc.
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        extend_dim: extension pixel number if doing cropping, int (default: 20)
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        return_limits: whether return cropping limits
        verbose: whether say something!, bool (default:True)
        """
    ## check inputs
    # color channel
    channel = str(channel)
    all_channels = [str(ch) for ch in all_channels]
    if channel not in all_channels:
        raise ValueError(
            f"Target channel {channel} doesn't exist in all_channels:{all_channels}")
    # only 3 all_channels requires chromatic correction
    if channel not in ['750', '647', '561']:
        chromatic_corr = False
    # channel id
    if channel_id is None:
        channel_id = all_channels.index(channel)
    # check filename
    if not isinstance(filename, str):
        raise ValueError(
            f"Wrong input of filename {filename}, should be a string!")
    if not os.path.isfile(filename):
        raise IOError("Input filename:{filename} doesn't exist, exit!")
    elif '.dax' not in filename:
        raise IOError("Input filename should be .dax format!")

    # check drift
    if len(drift) != 3:
        raise ValueError(f"Wrong input drift:{drift}, should be an array of 3")
    
    ## load image
    _ref_name = os.path.join(filename.split(os.sep)[-2], filename.split(os.sep)[-1])
    if verbose:
        print(f"- Start correcting {_ref_name} for channel:{channel}")
    _full_im_shape, _num_color = get_img_info.get_num_frame(filename,
                                                            frame_per_color=single_im_size[0],
                                                            buffer_frame=num_buffer_frames)
    ## crop image
    _cropped_im, _limits = visual_tools.crop_single_image(filename, channel, all_channels=all_channels,
                                             channel_id=channel_id, seg_label=seg_label,
                                             drift=drift, single_im_size=single_im_size,
                                             num_buffer_frames=num_buffer_frames,
                                             extend_dim=extend_dim, return_limits=True)

    ## corrections
    _corr_im = _cropped_im.copy()
    start = time.time()
    if z_shift_corr:
        # correct for z axis shift
        _corr_im = Z_Shift_Correction(_corr_im, verbose=verbose)
        mid = time.time()
        print(mid-start)
        start = mid
    if hot_pixel_remove:
        # correct for hot pixels
        _corr_im = Remove_Hot_Pixels(_corr_im, verbose=verbose)
        mid = time.time()
        print(mid-start)
        start = mid
    if illumination_corr:
        # illumination correction
        _corr_im = Illumination_correction(_corr_im, channel,
                                           crop_limits=_limits,
                                           correction_folder=correction_folder,
                                           single_im_size=single_im_size,
                                           verbose=verbose)
        mid = time.time()
        print(mid-start)
        start = mid
    if chromatic_corr:
        # chromatic correction
        _corr_im = Chromatic_abbrevation_correction(_corr_im, channel,
                                                    single_im_size=single_im_size,
                                                    crop_limits=_limits[1:],
                                                    correction_folder=correction_folder,
                                                    verbose=verbose)
        mid = time.time()
        print(mid-start)
        start = mid
    ## return
    if return_limits:
        return _corr_im, _limits
    else:
        return _corr_im

# merged function to crop DNA combo-group
def correct_combo_group(filenames, channel, seg_label=None, drift_dic=None,
                        single_im_size=_image_size, all_channels=_allowed_colors, channel_id=None,
                        num_buffer_frames=10, extend_dim=20, correction_folder=_correction_folder,
                        z_shift_corr=True, hot_pixel_remove=True, illumination_corr=True, chromatic_corr=True,
                        return_limits=False, verbose=False):
    """wrapper for all correction steps to one image, used for multi-processing
    Inputs:
        filenames: list of full filenames of a dax_file or npy_file for one image, list of string
        channel: channel to be extracted, str (for example. '647')
        seg_label: segmentation label, 2D array (default: None, which means the whole image)
        drift_dic: dictionary for drift, from hyb_folder/fov_name to drift array, dict (default: all zeros)
        single_im_size: z-x-y size of the image, list of 3 (default: 30, 2048, 2048, depends on camera)
        all_channels: all_channels used in this image, list of str(for example, ['750','647','561'])
        raw_im: directly give image rather than loading, this will overwrite filename etc.
        num_buffer_frames: number of buffer frame in front and back of image, int (default:10)
        extend_dim: extension pixel number if doing cropping, int (default: 20)
        correction_folder: path to find correction files
        z_shift_corr: whether do z-shift correction, bool (default: True)
        hot_pixel_remove: whether remove hot-pixels, bool (default: True)
        illumination_corr: whether do illumination correction, bool (default: True)
        chromatic_corr: whether do chromatic abbrevation correction, bool (default: True)
        return_limits: whether return cropping limits
        verbose: whether say something!, bool (default:True)
        """

    if ims is None and temp_filenames is None:
        raise ValueError(
            "Keywords ims and temp_filenames cannot be both None!")
    if np.max(seg_label) > 1:
        raise ValueError(
            "seg_label must be binary label, either 0-1 label or bool")
    if temp_filenames is None:
        if group_names is None or fov_name is None:
            raise ValueError(
                "When image are given, group_names and fov_name should be given as well")
        if len(group_names) != len(ims):
            raise ValueError(
                "number of images and number of group_names doesn't match")
    else:
        matched_filenames = []
        for _gname in group_names:
            _matches = [_fl for _fl in temp_filenames if _gname in _fl and str(
                group_channel) in _fl]
            if len(_matches) == 1:
                matched_filenames.append(_matches[0])
            else:
                raise ValueError(
                    "there are no matched temp files matched with group_names")
    # binarilize segmentation label
    seg_label = np.array(seg_label > 0, dtype=np.int)
    # extract reference name, used to extract drift
    if group_names is not None and fov_name is not None:
        ref_names = [os.path.join(_gn, fov_name) for _gn in group_names]
    else:
        ref_names = [os.path.basename(_fl).split(
            '_'+group_channel)[0].replace('-', os.sep)+'.dax' for _fl in matched_filenames]
    # match filename
    for _rn in ref_names:
        if _rn not in drift_dic:
            raise KeyError(f"Drift_dic doesn't have key:{_rn} for ref_name")
    # initialzie
    cropped_ims = []

    if temp_filenames is not None:
        for filename, ref_name in zip(matched_filenames, ref_names):
            cropped_ims.append(crop_single_image(
                None, filename, seg_label, drift=drift_dic[ref_name], im_size=im_size, extend_dim=extend_dim))
    else:
        for im, ref_name in zip(ims, ref_names):
            cropped_ims.append(
                crop_cell(im, seg_label, drift=drift_dic[ref_name], extend_dim=extend_dim))
    # return
    return cropped_ims
# merged function to crop merfish-group

