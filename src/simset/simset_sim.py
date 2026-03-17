# -*- coding: utf-8 -*-
import os
import re
import shutil
import time
import numpy as np
import nibabel as nib
import warnings
from multiprocessing import Process
from pathlib import Path
from os import PathLike
from os.path import join, dirname, abspath, exists
import src.simset.simset_tools as simset_tools
from utils import tools


def read_ws_from_simset_log(simset_log: PathLike) -> float:
    """
    Read W and W^2 from SimSet log file.

    Args:
        simset_log: path to simset log file.

    Returns:
        W^2/W as ``float``.
    """
    simset_log = Path(simset_log)

    scintific_number_regex = "(\d+\.\d+e\+\d+)"
    w_regex = re.compile(
        f"Sum of accepted coincidence weights in this simulation = {scintific_number_regex}")
    w2_regex = re.compile(
        f"Sum of accepted coincidence squared weights in this simulation = {scintific_number_regex}")

    weights = {"W": None, "W2": None}
    with open(simset_log, 'r') as slog:
        for line in slog:
            w_match = w_regex.match(line)
            w2_match = w2_regex.match(line)
            if w_match:
                weights["W"] = float(w_match.groups()[0])
            elif w2_match:
                weights["W2"] = float(w2_match.groups()[0])
            if weights["W"] is not None and weights["W2"] is not None:
                quotient = weights["W2"] / weights["W"]
                if quotient < 1:
                    warnings.warn("W2/W is less than 1!")
                return quotient


class SimSET_Simulation(object):
    """This class provides functions to run a SimSET simulation."""

    def __init__(
        self, params, config, act_map, att_map, scanner, projections_dir, debug=False
    ):  ##Debug True must be implemented....
        # Initialization
        self.simpet_dir = dirname(abspath(__file__))

        self.params = params
        self.config = config
        self.scanner = scanner

        self.simset_dir = self.config.get("dir_simset")

        self.act_map = act_map
        self.att_map = att_map
        self.output_dir = projections_dir
        self.center_slice = params.get("center_slice")

        if self.center_slice == 0:  # we calculate the half of the total z voxels
            img = nib.load(act_map)
            self.center_slice = img.shape[2] / 2

        self.sim_dose = params.get("total_dose")
        self.s_photons = params.get("sampling_photons")
        self.photons = params.get("photons")
        self.sim_time = params.get("simulation_time")
        self.divisions = params.get("divisions")

        self.detlistmode = params.get("detlistmode")
        self.phglistmode = params.get("phglistmode")
        self.add_randoms = params.get("add_randoms")

    def run(self):
        processes = []

        print("\nLaunching simulation with %d divisions" % self.divisions)
        print('------------------------------------------------------------')
        print('Scanner: %s' % self.scanner.get("scanner_name"))
        print('Activity map: %s' % self.act_map)
        print('Attenuation map: %s' % self.att_map)
        print('Dose: %s mCi' % self.sim_dose)
        print('Acquisition time: %s seconds' % self.sim_time)
        print('------------------------------------------------------------')

        for division in range(self.divisions):
            division_dir = join(self.output_dir, "division_" + str(division))
            os.makedirs(division_dir)
            p = Process(target=self.run_simset_simulation, args=(division_dir,))
            processes.append(p)
            p.start()
            time.sleep(5)


        for process in processes:
            process.join()

        print(" ")

        self.simulation_postprocessing()

        print('------------------------------------------------------------')
        print("Final report")
        div_0_dir = join(self.output_dir, "division_0")

        for j in ['trues', 'scatter', 'randoms']:
            #hdr_ = join(div_0_dir, "%s.hdr" % j)
            hdr_ = join(div_0_dir, "%s.nii" % j)
            #hdr_ = join(div_0_dir, "%s.nii.gz" % j)
            if exists(hdr_):
                counts_ = tools.ncounts(hdr_)
                print("Number of %s in simulation: %s" % (j, counts_))

        print('------------------------------------------------------------')


    def run_simset_simulation(self, sim_dir):

        log_file = join(sim_dir, "logging.log")

        act, act_data = tools.nib_load(self.act_map)

        # It generates a new act_table for the simulation
        if self.sim_dose != 0:
            phantom_counts = np.sum(act_data)
            voxel_size = act.affine[0, 0] * act.affine[1, 1] * act.affine[2, 2] / 1000
            phantom_dose = abs(phantom_counts * voxel_size)  # uCi
            act_table_factor = self.sim_dose * 1000 / phantom_dose
        else:
            act_table_factor = 1

        # Creates the data files from the simulation maps
        act_img = self.act_map[0:-3] + "img"
        att_img = self.att_map[0:-3] + "img"
        shutil.copy(act_img, join(sim_dir, "act.dat"))
        shutil.copy(att_img, join(sim_dir, "att.dat"))

        # If Sampling Photons is 0, importance sampling is deactivated, so we use photons as the input.
        if self.add_randoms == 1:
            sim_photons = 0
            needed_sims = 1

        elif self.s_photons == 0 and self.photons != 0:
            sim_photons = self.photons / self.divisions
            needed_sims = 1

        elif self.s_photons == 0 and self.photons == 0:
            sim_photons = self.photons
            needed_sims = 1

        elif self.s_photons != 0 and self.photons !=0:
            sim_photons = self.s_photons
            needed_sims = 2

        else:
            sim_photons = self.s_photons
            needed_sims = 3

        sim_time = float(self.sim_time) / self.divisions

        my_phg = self.prepare_simset_files(
            sim_dir, act_table_factor, act, sim_photons, sim_time, 0
        )
        # Copying phg file for posterior analysis
        sim_phg = Path(sim_dir).joinpath("phg.rec")
        shutil.copy(
            sim_phg,
            sim_phg.with_name("phg_first_sampling_sim.rec")
        )
        my_log = join(sim_dir, "simset_s0_init.log")

        print("Running first simulation...(Of %s simulations needed for %s)" %
              (needed_sims, os.path.basename(sim_dir)))

        if self.add_randoms == 1:
            print("WARNING: add_randoms=1, so simulation is forced to realistic noise")
            print("Importance sampling is also being deactivated")
            print("All these means the simulation can take very long...")

        command = "%s/bin/phg %s > %s" % (self.simset_dir, my_phg, my_log)
        tools.osrun(command, log_file)

        rec_weight = join(sim_dir, "rec.weight")
        det_hf = join(sim_dir, "det_hf.hist")
        phg_hf = join(sim_dir, "phg_hf.hist")

        if self.s_photons != 0 and self.params.get("add_randoms") != 1:

            # If the user did not state photons it will be calculated from sampling
            if self.photons == 0:

                # Removes counts for preparing for the next simulation
                os.remove(rec_weight)
                if exists(det_hf):
                    os.remove(det_hf)
                if exists(phg_hf):
                    os.remove(phg_hf)

                my_phg = self.prepare_simset_files(
                    sim_dir, act_table_factor, act, sim_photons, sim_time, 1
                    )

                # Copying phg file for posterior analysis
                sim_phg = Path(sim_dir).joinpath("phg.rec")
                shutil.copy(
                    sim_phg,
                    sim_phg.with_name("phg_second_sampling_sim.rec")
                )
                my_log = join(sim_dir, "simset_s0.log")

                print("Running second simulation...(Of %s simulations needed for %s)" %
                      (needed_sims, os.path.basename(sim_dir)))

                command = "%s/bin/phg %s > %s" % (self.simset_dir, my_phg, my_log)
                tools.osrun(command, log_file)
                w_quotient = read_ws_from_simset_log(my_log)

                sim_photons = int(self.s_photons * w_quotient)

            else:
                # If the user stated photons, the provided value will be used
                sim_photons = self.photons

            # Removes counts for preparing for the next simulation
            os.remove(rec_weight)
            if exists(det_hf):
                os.remove(det_hf)
            if exists(phg_hf):
                os.remove(phg_hf)

            my_phg = self.prepare_simset_files(
                sim_dir, act_table_factor, act, sim_photons, sim_time, 1
            )
            my_log = join(sim_dir, "simset_s1.log")

            print("Running final simulation...(Of %s simulations needed for %s)" %
                  (needed_sims, os.path.basename(sim_dir)))

            command = "%s/bin/phg %s > %s" % (self.simset_dir, my_phg, my_log)
            tools.osrun(command, log_file)

        if self.add_randoms == 1:
            coincidence_window = self.scanner.get("coincidence_window")

            simset_tools.add_randoms(
                sim_dir,
                self.simset_dir,
                coincidence_window,
                rebin=True,
                log_file=log_file,
            )

        simset_tools.process_weights(
            rec_weight, sim_dir, self.scanner, self.add_randoms
        )

        print("Finished simulation for %s" % os.path.basename(sim_dir))

    def prepare_simset_files(
        self, sim_dir, act_table_factor, act, sim_photons, sim_time, sampling
    ):
        log_file = join(sim_dir, "logging.log")
        # Establishing necessary parameters
        model_type = self.params.get("model_type")
        scanner_radius = self.scanner.get("scanner_radius")
        scanner_axial_fov = self.scanner.get("axial_fov")

        # We activate det_listmode if demanded by user or if add_randoms is on
        if self.add_randoms == 1:
            det_listmode = 1
            add_randoms = True
        elif self.detlistmode == 1:
            det_listmode = 1
            add_randoms = False
        else:
            det_listmode = 0
            add_randoms = False

        # Creating the act table for the simulation....
        my_act_table = join(sim_dir, "phg_act_table")
        simset_tools.make_simset_act_table(
            act_table_factor, my_act_table, log_file=log_file
        )

        # Creating the phg for the simulation...
        my_phg_file = join(sim_dir, "phg.rec")
        simset_tools.make_simset_phg(
            self.config,
            my_phg_file,
            sim_dir,
            act,
            scanner_radius,
            scanner_axial_fov,
            self.center_slice,
            sim_photons,
            sim_time,
            add_randoms,
            self.phglistmode,
            sampling,
            log_file=log_file,
        )

        my_det_file = join(sim_dir, "det.rec")
        if model_type == "simple_pet":
            simset_tools.make_simset_simp_det(
                self.scanner, my_det_file, sim_dir, det_listmode, log_file=log_file
            )
        elif model_type == "cylindrical":
            simset_tools.make_simset_cyl_det(
                self.scanner, my_det_file, sim_dir, det_listmode, log_file=log_file
            )

        my_bin_file = join(sim_dir, "bin.rec")
        simset_tools.make_simset_bin(
            self.config,
            my_bin_file,
            sim_dir,
            self.scanner,
            add_randoms,
            log_file=log_file,
        )

        simset_tools.make_index_file(sim_dir, self.simset_dir, log_file=log_file)

        return my_phg_file

    def simulation_postprocessing(self):
        print("Postprocessing simulation...")
        print(" ")
        # All the parallel simulations are combined in division 0

        log_file = join(self.output_dir, "postprocessing.log")

        division_zero = join(self.output_dir, "division_0")

        for image in ["trues", "scatter", "randoms"]:
            #zero_image = join(division_zero, image + ".hdr")
            zero_image = join(division_zero, image + ".nii")
            #zero_image = join(division_zero, image + ".nii.gz")

            if exists(zero_image):
                print("Adding sinograms for %s" % image)
                
                #This is to create the postlog file in case there is only one simulation, to avoid double attenuation computing!
                message = "Adding sinograms for %s" % image
                tools.log_message(log_file, message)
                
                for division in range(1, self.divisions):
                    division_dir = join(self.output_dir, "division_" + str(division))
                    #division_image = join(division_dir, image + ".hdr")
                    division_image = join(division_dir, image + ".nii")
                    #division_image = join(division_dir, image + ".nii.gz")
                    message = "Adding %s from simulation %s" % (image, division)
                    tools.log_message(log_file, message)
                    
                    #tools.operate_images_nii( #RESTABLISH IF FAILS [NOT COMPUTING NANS!]
                    tools.operate_sinograms_nii(
                        zero_image, division_image, zero_image, "sum"
                    )
                    #os.remove(division_image)
                    #os.remove(division_image[0:-3] + "img")
                    os.remove(division_image[0:-3] + "nii")
                    #os.remove(division_image[0:-6] + "nii.gz")

        for hist in ["phg_hf.hist", "det_hf.hist"]:
            zero_hist = join(division_zero, hist)

            if exists(zero_hist):
                print(" ")
                print("Adding History Files for %s" % hist)
                
                #PUT IN FUNCTION OF DETLISTMODE (MISSING NOW?)
                output = join(division_zero, "tmp_" + hist)

                for division in range(1, self.divisions):
                    division_dir = join(self.output_dir, "division_" + str(division))
                    division_hist = join(division_dir, hist)
                    file_list = zero_hist + " " + division_hist
                    #simset_tools.combine_history_files(
                    #    self.simset_dir, file_list, output, log_file
                    #) #IMPORTANT! RESTABLISH IF FAILS!
                    # shutil.move(output, zero_hist)
                    # Once everything is combined in division_0, remove the other division
                    shutil.rmtree(division_dir)
                
                os.remove(join(division_zero, "det_hf.hist")) #IMPORTANT! REMOVE IF FAILS.
                    

        if self.add_randoms == 1:
            # To have randoms in the final history file, we need to add randoms to the final det_hf.hist
            print("Adding randoms to the history file...")

            coincidence_window = self.scanner.get("coincidence_window")

            simset_tools.add_randoms(
                division_zero,
                self.simset_dir,
                coincidence_window,
                rebin=False,
                log_file=log_file,
            )

            # os.remove(join(division_zero, "sorted_det_hf.hist"))

            det_hist = join(division_zero, "det_hf.hist")
            randoms_hist = join(division_zero, "randoms.hist")
            output = join(division_zero, "full_det_hf.hist")

            file_list = det_hist + " " + randoms_hist

            simset_tools.combine_history_files(
                self.simset_dir, file_list, output, log_file
            )
        
        self.stir_norm_from_att_map = self.scanner.get("stir_norm_from_att_map")
        
        if self.stir_norm_from_att_map != 1:
        
            print("Calculating attenuation map...")
            print(" ")

            output_atten = "attenuationsino"
            #RESTABLISH IF FAILS:
            #hdr_to_copy = join("trues.hdr")
            hdr_to_copy = join("trues.nii")
            #hdr_to_copy = join("trues.nii.gz")

            simset_tools.simset_calcattenuation(
                self.simset_dir, division_zero, output_atten, hdr_to_copy, nrays=1, timeout=None
            )
        
        #To remove unnecessary files once endeded the sim. Add condition: If exists.
        #Also: Change directory name to "simuulation". IMPORTANT! REMOVE IF FAILS:
        #os.remove(join(division_zero, "rec.weight"))
        #os.remove(join(division_zero, "rec.act_indexes"))
        #os.remove(join(division_zero, "rec.activity_image"))
        #os.remove(join(division_zero, "rec.att_indexes"))
        #os.remove(join(division_zero, "rec.attenuation_image"))
        #os.remove(join(division_zero, "sampling_rec"))
        #os.remove(join(division_zero, "attenuationsino"))
        

class SimSET_Reconstruction(object):
    """This class provides functions to reconstruct a SimSET simulation."""

    def __init__(
        self, params, config, projections_dir, scanner, reconstructions_dir, recons_type
    ):
        # Initialization
        self.simpet_dir = dirname(abspath(__file__))
        self.dir_stir = config.get("dir_stir")

        self.input_dir = join(projections_dir, "division_0") #CHANGE TO "SIMULATION"!!!
        self.output_dir = reconstructions_dir

        self.params = params
        self.config = config
        self.scanner = scanner
        self.add_randoms = params.get("add_randoms")

        self.scatt_corr_factor = scanner.get("analytic_scatt_corr_factor")
        self.random_corr_factor = scanner.get("analytic_randoms_corr_factor")

        self.do_pre_att_correction = scanner.get("analytical_att_correction")
        self.do_recons_att_correction = scanner.get("stir_recons_att_corr")

        if self.do_pre_att_correction == 1 and self.do_recons_att_correction == 1:
            self.do_pre_att_correction == 0
            raise Warning(
                "WARNING: both pre and recons att corrections are both active...Ignoring pre-correction"
            )

        self.log_file = join(self.output_dir, "recons.log")

    def run(self):
        if not exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        #if (not exists(join(self.output_dir, "stir_sinogram.nii.gz")) or (not exists(join(self.output_dir, "stir_sinogram.nii.gz"))
        
        self.prepare_recons()
        self.run_recons()

    def prepare_recons(self):
        from src.stir import stir_tools

        print("Preparing files for reconstruction")
        
        self.stir_norm_from_att_map = self.scanner.get("stir_norm_from_att_map")
        
        
        #RESTABLISH IF FAILS:
        #trues_sino = join(self.input_dir, "trues.hdr")
        #scatter_sino = join(self.input_dir, "scatter.hdr")
        #randoms_sino = join(self.input_dir, "randoms.hdr")

        #corr_scatter_sino = join(self.input_dir, "corr_scatter.hdr")
        #corr_randoms_sino = join(self.input_dir, "corr_randoms.hdr")
        #my_simset_sino = join(self.input_dir, "my_sinogram.hdr")
        #additive_sinogram = join(self.input_dir, "additive_sinogram.hdr")
        #att_sino = join(self.input_dir, "attenuationsino.hdr")
        
        
        #trues_sino = join(self.input_dir, "trues.nii")
        #scatter_sino = join(self.input_dir, "scatter.nii")
        #randoms_sino = join(self.input_dir, "randoms.nii")

        #corr_scatter_sino = join(self.input_dir, "corr_scatter.nii")
        #corr_randoms_sino = join(self.input_dir, "corr_randoms.nii")
        #my_simset_sino = join(self.input_dir, "my_sinogram.nii")
        #additive_sinogram = join(self.input_dir, "additive_sinogram.nii")
        #att_sino = join(self.input_dir, "attenuationsino.nii")
        
        """
        trues_sino = join(self.input_dir, "trues.nii.gz")
        scatter_sino = join(self.input_dir, "scatter.nii.gz")
        randoms_sino = join(self.input_dir, "randoms.nii.gz")

        corr_scatter_sino = join(self.input_dir, "corr_scatter.nii.gz")
        corr_randoms_sino = join(self.input_dir, "corr_randoms.nii.gz")
        my_simset_sino = join(self.input_dir, "my_sinogram.nii.gz")
        additive_sinogram = join(self.input_dir, "additive_sinogram.nii.gz")
        att_sino = join(self.input_dir, "attenuationsino.nii.gz")
        """
        
        #Delete:
        trues_sino = join(self.input_dir, "trues.nii")
        scatter_sino = join(self.input_dir, "scatter.nii")
        randoms_sino = join(self.input_dir, "randoms.nii")

        corr_scatter_sino = join(self.input_dir, "corr_scatter.nii")
        corr_randoms_sino = join(self.input_dir, "corr_randoms.nii")
        my_simset_sino = join(self.input_dir, "my_sinogram.nii")
        additive_sinogram = join(self.input_dir, "additive_sinogram.nii")
        att_sino = join(self.input_dir, "attenuationsino.nii")
        
        #RESTABLISH IF FAILS:
        #tools.operate_single_image(
        
        print("MULTIPLYING CORR_SCATTER_SINO")        
        
        if not exists(corr_scatter_sino):
            tools.operate_single_image_nii(
                scatter_sino,
                "mult",
                self.scatt_corr_factor,
                corr_scatter_sino,
                self.log_file,
                check_nans = False,
            )
        
        print("SUMMING TRUES AND CORR SCATTER")   
        
        #RESTABLISH IF FAILS:
        #tools.operate_images_analyze(
        if not exists(my_simset_sino):
            tools.operate_images_nii(
                trues_sino, corr_scatter_sino, my_simset_sino, operation="sum", check_nans = False,
            )

        if self.add_randoms == 1:
            tools.operate_single_image(
                randoms_sino,
                "mult",
                self.random_corr_factor,
                corr_randoms_sino,
                self.log_file,
            )
            tools.operate_images_analyze(
                my_simset_sino, corr_randoms_sino, my_simset_sino, operation="sum"
            )

            if self.scanner.get("stir_randoms_corr_smoothing") == 1:
                tools.operate_images_analyze(
                    scatter_sino, randoms_sino, additive_sinogram, operation="sum"
                )
            else:
                #RESTABLISH IF FAILS
                #tools.copy_analyze(scatter_sino, additive_sinogram)
                tools.copy_nifti(scatter_sino, additive_sinogram)

        else:
            #RESTABLISH IF FAILS
            #tools.copy_analyze(scatter_sino, additive_sinogram)
            
            print("COPYING ADDITIVE SINO")  
            if not exists(additive_sinogram):
                tools.copy_nifti(scatter_sino, additive_sinogram)
                print("SMOOTHING ADDITIVE SINO")
                tools.smooth_nifti(additive_sinogram, 10, additive_sinogram)
        
        #RESTABLISH IF FAILS
        #tools.smooth_analyze(additive_sinogram, 10, additive_sinogram)
        
        #if not exists(additive_sinogram):
        #    tools.smooth_nifti(additive_sinogram, 10, additive_sinogram)
        
        
        #RESTABLISH IF FAILS:
        """
        sinogram_stir = join(self.output_dir, "stir_sinogram.hdr")
        tools.convert_simset_sino_to_stir(my_simset_sino, sinogram_stir)
        shutil.copy(sinogram_stir[0:-3] + "bin", sinogram_stir[0:-3] + "s") #PREVIOUS: img
        stir_tools.create_stir_hs_from_detparams(
            self.scanner, sinogram_stir[0:-3] + "hs"
        )

        additive_sino_stir = join(self.output_dir, "stir_additivesino.hdr")
        tools.convert_simset_sino_to_stir(additive_sinogram, additive_sino_stir)
        shutil.copy(additive_sino_stir[0:-3] + "img", additive_sino_stir[0:-3] + "s") #PREVIOUS: img
        stir_tools.create_stir_hs_from_detparams(
            self.scanner, additive_sino_stir[0:-3] + "hs"
        )

        #att_sino = join(self.input_dir, "attenuationsino.hdr")
        #att_sino = join(self.input_dir, "attenuationsino.nii")
        att_stir = join(self.output_dir, "stir_att.hdr")
        tools.convert_simset_sino_to_stir(att_sino, att_stir)
        shutil.copy(att_stir[0:-3] + "bin", att_stir[0:-3] + "s") #PREVIOUS: img
        stir_tools.create_stir_hs_from_detparams(self.scanner, att_stir[0:-3] + "hs")
        """
        
        ##### THIS IS FOR .nii:
        #sinogram_stir = join(self.output_dir, "stir_sinogram.nii")
        #tools.convert_simset_sino_to_stir_nii(my_simset_sino, sinogram_stir)
        #shutil.copy(sinogram_stir[0:-3] + "img", sinogram_stir[0:-3] + "s")
        #stir_tools.create_stir_hs_from_detparams(
        #    self.scanner, sinogram_stir[0:-3] + "hs"
        #)
        #os.remove(sinogram_stir[0:-3] + "img")
        #os.remove(sinogram_stir[0:-3] + "hdr")
        
        #additive_sino_stir = join(self.output_dir, "stir_additivesino.nii")
        #tools.convert_simset_sino_to_stir_nii(additive_sinogram, additive_sino_stir)
        #shutil.copy(additive_sino_stir[0:-3] + "img", additive_sino_stir[0:-3] + "s")
        #stir_tools.create_stir_hs_from_detparams(
        #    self.scanner, additive_sino_stir[0:-3] + "hs"
        #)
        #os.remove(additive_sino_stir[0:-3] + "img")
        #os.remove(additive_sino_stir[0:-3] + "hdr")

        #att_sino = join(self.input_dir, "attenuationsino.nii")
        #att_stir = join(self.output_dir, "stir_att.nii")
        #tools.convert_simset_sino_to_stir_nii(att_sino, att_stir)
        #shutil.copy(att_stir[0:-3] + "img", att_stir[0:-3] + "s")
        #stir_tools.create_stir_hs_from_detparams(
        #    self.scanner, att_stir[0:-3] + "hs"
        #)
        #os.remove(att_stir[0:-3] + "img")
        #os.remove(att_stir[0:-3] + "hdr")
        
        
        ##### THIS IS FOR .nii.gz:
        #sinogram_stir = join(self.output_dir, "stir_sinogram.nii.gz")
        #tools.convert_simset_sino_to_stir_nii(my_simset_sino, sinogram_stir)
        #shutil.copy(sinogram_stir[0:-6] + "img", sinogram_stir[0:-6] + "s")
        #stir_tools.create_stir_hs_from_detparams(
        #    self.scanner, sinogram_stir[0:-6] + "hs"
        #)
        #os.remove(sinogram_stir[0:-6] + "img")
        #os.remove(sinogram_stir[0:-6] + "hdr")
        
        #additive_sino_stir = join(self.output_dir, "stir_additivesino.nii.gz")
        #tools.convert_simset_sino_to_stir_nii(additive_sinogram, additive_sino_stir)
        #shutil.copy(additive_sino_stir[0:-6] + "img", additive_sino_stir[0:-6] + "s")
        #stir_tools.create_stir_hs_from_detparams(
        #    self.scanner, additive_sino_stir[0:-6] + "hs"
        #)
        #os.remove(additive_sino_stir[0:-6] + "img")
        #os.remove(additive_sino_stir[0:-6] + "hdr")

        #att_stir = join(self.output_dir, "stir_att.nii.gz")
        #tools.convert_simset_sino_to_stir_nii(att_sino, att_stir)
        #shutil.copy(att_stir[0:-6] + "img", att_stir[0:-6] + "s")
        #stir_tools.create_stir_hs_from_detparams(
        #    self.scanner, att_stir[0:-6] + "hs"
        #)
        #os.remove(att_stir[0:-6] + "img")
        #os.remove(att_stir[0:-6] + "hdr")
        
        
        ##### THIS IS FOR .nii.gz [NEW VERSION (FILE REOIRGANIZATION):
        
        num_rings = self.scanner.get("num_rings")
        max_segment = self.scanner.get("max_segment")
        
        print("GENERATING STIR SINOGRAM")
        
        #sinogram_stir_nii = join(self.input_dir, "stir_sinogram.nii.gz")
        sinogram_stir_nii = join(self.input_dir, "stir_sinogram.nii")
        sinogram_stir_s = join(self.output_dir, "stir_sinogram.s")
        sinogram_stir_hs = join(self.output_dir, "stir_sinogram.hs")
        
        if not exists(sinogram_stir_nii):
            tools.convert_simset_sino_to_stir_nii(my_simset_sino, sinogram_stir_nii)
            
        tools.copy_sinogram_stir_to_output(sinogram_stir_nii, sinogram_stir_s)
        #tools.copy_reduced_sinogram_stir_to_output(sinogram_stir_nii, sinogram_stir_s, num_rings, max_segment)
        
        stir_tools.create_stir_hs_from_detparams(
        #stir_tools.create_reduced_stir_hs_from_detparams(
            self.scanner, sinogram_stir_hs
        )
        
        print("GENERATING STIR ADDITIVE")
        
        #additivesino_stir_nii = join(self.input_dir, "stir_additivesino.nii.gz")
        additivesino_stir_nii = join(self.input_dir, "stir_additivesino.nii")
        additivesino_stir_s = join(self.output_dir, "stir_additivesino.s")
        additivesino_stir_hs = join(self.output_dir, "stir_additivesino.hs")
        
        if not exists(additivesino_stir_nii):
            tools.convert_simset_sino_to_stir_nii(additive_sinogram, additivesino_stir_nii)
            
        tools.copy_sinogram_stir_to_output(additivesino_stir_nii, additivesino_stir_s)
        #tools.copy_reduced_sinogram_stir_to_output(additivesino_stir_nii, additivesino_stir_s, num_rings, max_segment)
        
        stir_tools.create_stir_hs_from_detparams(
        #stir_tools.create_reduced_stir_hs_from_detparams(
            self.scanner, additivesino_stir_hs
        )
    
        print("GENERATING STIR ATT")
        
        if self.stir_norm_from_att_map != 1:
            
            #att_stir_nii = join(self.input_dir, "stir_att.nii.gz")
            att_stir_nii = join(self.input_dir, "stir_att.nii")
            att_stir_s = join(self.output_dir, "stir_att.s")
            att_stir_hs = join(self.output_dir, "stir_att.hs")
            
            if not exists(att_stir_nii):
                tools.convert_simset_sino_to_stir_nii(att_sino, att_stir_nii)
                
            tools.copy_sinogram_stir_to_output(att_stir_nii, att_stir_s)
            #tools.copy_reduced_sinogram_stir_to_output(att_stir_nii, att_stir_s, num_rings, max_segment)
            
            stir_tools.create_stir_hs_from_detparams(
            #stir_tools.create_reduced_stir_hs_from_detparams(
                self.scanner, att_stir_hs
            )
        
        """ #RESTABLISH IF FAILS:
        if self.stir_norm_from_att_map == 1:
        
            attmap = nib.load(join('Data', self.params.get("patient_dirname"), self.params.get("act_map")))

            attmap_dim = np.shape(attmap)
            attmap_pixsize = attmap.header['pixdim'][1:4]
            
            
            filedims = [attmap_dim[2], attmap_dim[0], attmap_dim[1]]  #[618, 440, 440]
            dtype = np.uint32 #np.float32   # or np.uint8, np.float32, etc.

            with open(join(self.input_dir, "rec.att_indexes"), "rb") as f:
                data = np.frombuffer(f.read(), dtype=dtype)

            att_indexes = data.reshape(filedims)
            
            mu_map = np.zeros(np.shape(att_indexes))

            for attidx in np.unique(att_indexes):
                mu_map[att_indexes == attidx] = tools.mu_coef_511keV(attidx)
                
            
            mu_map_img = nib.Nifti2Image(mu_map, np.eye(4))
            nib.save(mu_map_img, join(self.input_dir, 'mu_map.hdr'))
            
            shutil.copy(join(self.input_dir, 'mu_map.img'), join(self.output_dir, 'mu_map.v'))
            tools.write_interfile_header(join(self.output_dir, 'mu_map.hv'), attmap_dim[2], attmap_pixsize[2],
                      attmap_dim[0], attmap_pixsize[0],
                      attmap_dim[1], attmap_pixsize[1])
        
        """
        
        
        if self.stir_norm_from_att_map == 1:
        
            attmap = nib.load(join('Data', self.params.get("patient_dirname"), self.params.get("att_map")))
            
            attmap_data = (attmap.get_fdata()).astype(np.uint8)

            attmap_dim = np.shape(attmap)
            attmap_pixsize = attmap.header['pixdim'][1:4]
            
            mu_map = np.zeros(attmap_dim)
            
            for attidx in np.unique(attmap_data):
                mu_map[attmap_data == attidx] = tools.mu_coef_511keV(attidx)
            
            mu_map_img = nib.Nifti1Image(mu_map, np.eye(4))
            nib.save(mu_map_img, join(self.input_dir, 'mu_map.hdr'))
            
            shutil.copy(join(self.input_dir, 'mu_map.img'), join(self.output_dir, 'mu_map.v'))
            tools.write_interfile_header(join(self.output_dir, 'mu_map.hv'), attmap_dim[0], attmap_pixsize[0],
                      attmap_dim[1], attmap_pixsize[1],
                      attmap_dim[2], attmap_pixsize[2])
        
        
        
        
        ######THIS WAS FOR .nii.gz [NOT WORKING AT THE MOMENT]:
        """
        sinogram_stir = join(self.output_dir, "stir_sinogram.nii.gz")
        tools.convert_simset_sino_to_stir_nii(my_simset_sino, sinogram_stir)
        nib.save(nib.load(sinogram_stir), join(self.output_dir, "stir_sinogram.nii"))
        shutil.copy(sinogram_stir[0:-6] + "nii", sinogram_stir[0:-6] + "s")
        stir_tools.create_stir_hs_from_detparams(
            self.scanner, sinogram_stir[0:-6] + "hs"
        )
        os.remove(join(self.output_dir, "stir_sinogram.nii"))

        additive_sino_stir = join(self.output_dir, "stir_additivesino.nii.gz")
        tools.convert_simset_sino_to_stir_nii(additive_sinogram, additive_sino_stir)
        nib.save(nib.load(additive_sino_stir), join(self.output_dir, "stir_additivesino.nii"))
        shutil.copy(additive_sino_stir[0:-6] + "nii", additive_sino_stir[0:-6] + "s")
        stir_tools.create_stir_hs_from_detparams(
            self.scanner, additive_sino_stir[0:-6] + "hs"
        )
        os.remove(join(self.output_dir, "stir_additivesino.nii"))

        att_sino = join(self.input_dir, "attenuationsino.nii.gz")
        att_stir = join(self.output_dir, "stir_att.nii.gz")
        tools.convert_simset_sino_to_stir_nii(att_sino, att_stir)
        nib.save(nib.load(att_stir), join(self.output_dir, "stir_att.nii"))
        shutil.copy(att_stir[0:-6] + "nii", att_stir[0:-6] + "s")
        stir_tools.create_stir_hs_from_detparams(self.scanner, att_stir[0:-6] + "hs")
        os.remove(join(self.output_dir, "stir_att.nii"))
        """
        
        
        #EDIT IN FUTURE:
        if self.scanner.get("analytical_att_correction") == 1:
            catt_sino = join(self.output_dir, "catt_sinogram.hdr")
            tools.operate_images_analyze(
                sinogram_stir, att_stir, catt_sino, operation="mult"
            )
            shutil.copy(catt_sino[0:-3] + "img", sinogram_stir[0:-3] + "s")

            catt_add_sino = join(self.output_dir, "my_catt_additivesino.hdr")
            tools.operate_images_analyze(
                additive_sino_stir, att_stir, catt_add_sino, operation="mult"
            )
            shutil.copy(catt_add_sino[0:-3] + "img", additive_sino_stir[0:-3] + "s")

        if self.scanner.get("psf_value") != 0:
            stir_tools.apply_psf(self.scanner, sinogram_stir, self.log_file)

        if self.scanner.get("add_noise") != 0:
            stir_tools.add_noise(
                self.config, self.scanner, sinogram_stir, self.log_file
            )

    def run_recons(self):
        from src.stir import stir_tools
        
        start_recons = False
        
        print("Starting STIR reconstruction")

        recons_algorithm = self.scanner.get("recons_type")
        sinogram_stir = join(self.output_dir, "stir_sinogram.hs")
        additive_sino_stir = join(self.output_dir, "stir_additivesino.hs")
        att_stir = join(self.output_dir, "stir_att.hs")
        
        if self.scanner.get("stir_norm_from_att_map") != 1:
        
            if any(
                exists(i) == False for i in [sinogram_stir, additive_sino_stir, att_stir]
            ):
                print("Something is not ready for the reconstruction")
            else:
                start_recons = True
                print("Starting STIR reconstruction")
        
        else:
            
            if any(
                exists(i) == False for i in [sinogram_stir, additive_sino_stir]
            ):
                print("Something is not ready for the reconstruction")
            else:
                start_recons = True
                print("Starting STIR reconstruction")
         
        if start_recons:

            if recons_algorithm == "FBP2D":
                reconsFile_hdr = stir_tools.FBP2D_recons(
                    self.config,
                    self.scanner,
                    sinogram_stir,
                    self.output_dir,
                    self.log_file,
                )

            elif recons_algorithm == "FBP3D":
                reconsFile_hdr = stir_tools.FBP3D_recons(
                    self.config,
                    self.scanner,
                    sinogram_stir,
                    self.output_dir,
                    self.log_file,
                )

            elif recons_algorithm == "OSEM2D":
                reconsFile_hdr = stir_tools.OSEM2D_recons(
                    self.config,
                    self.scanner,
                    sinogram_stir,
                    additive_sino_stir,
                    att_stir,
                    self.output_dir,
                    self.log_file,
                )

            elif recons_algorithm == "OSEM3D":
                reconsFile_hdr = stir_tools.OSEM3D_recons(
                    self.config,
                    self.scanner,
                    sinogram_stir,
                    additive_sino_stir,
                    att_stir,
                    self.output_dir,
                    self.log_file,
                )

            if exists(reconsFile_hdr):
                print("Reconstruction finished")
            else:
                print(
                    "Reconstruction output file does not exists. Something went wrong"
                )
