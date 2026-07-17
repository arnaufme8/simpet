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
        
        self.list_mode = params.get("listmode")

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
                if exists(rec_weight):
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
            if exists(rec_weight):
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
            
            #WARNING: TESTING IF THIS IS NOT NEEDED. DE-COMMENT IF FAILS:
            simset_tools.add_randoms(
                sim_dir,
                self.simset_dir,
                coincidence_window,
                rebin=True,
                log_file=log_file,
            )
        
        if not self.list_mode:
            
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
        list_mode = self.params.get("listmode")

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
            list_mode=list_mode
        )

        simset_tools.make_index_file(sim_dir, self.simset_dir, log_file=log_file)

        return my_phg_file

    def simulation_postprocessing(self):
        print("Postprocessing simulation...")
        print(" ")
        
        
        # All the parallel simulations are combined in division 0. NOTE: CHANGE TO simulation folder at some point.

        log_file = join(self.output_dir, "postprocessing.log")

        division_zero = join(self.output_dir, "division_0")
        
        #ATTENTION: We need to think a better place to do this. At the moment, we keep it here.
        #This is generating rec.weight files when listmoding. NOTE: WE WILL USE phg_hf.hist at the moment. Ignoring det.hist.
        
        if self.list_mode:
            
            zero_hist = join(division_zero, "phg_hf.hist")
            full_hist = join(division_zero, "full_phg_hf.hist")
            rec_weight = join(division_zero, "rec.weight")
            my_phg = join(division_zero, "phg.rec")
            file_list = zero_hist
            
            print("Adding History Files for PHG")
            
            message = "Adding History Files for PHG"
            tools.log_message(log_file, message)
            
            for division in range(1, self.divisions):
                
                division_dir = join(self.output_dir, "division_" + str(division))
                division_hist = join(division_dir, "phg_hf.hist")
                #file_list = zero_hist + " " + division_hist
                file_list = file_list + " " + division_hist
            
            if self.divisions > 1:
                simset_tools.combine_history_files(
                    self.simset_dir, file_list, full_hist, log_file
                )
                #shutil.move(output, zero_hist)
                
                os.remove(zero_hist)
                os.rename(full_hist, zero_hist)
            
            #ATTENTION: THIS IS ADDED ONLY FOR LIST MODE. MAY BECOME UNUSED OR PROBLEMATIC.
            
            with open(join(division_zero, "bin.rec"), "a") as f:
                f.write('STR      weight_image_path = "' + rec_weight + '"\n')
            f.close()
                
            command = "%s/bin/bin -p %s" % (self.simset_dir, my_phg)
            tools.osrun(command, log_file)
            
            
            
            simset_tools.process_weights(
                rec_weight, division_zero, self.scanner, self.add_randoms
            )
        
        
    
        else:

            for image in ["trues", "scatter", "randoms"]: #TODO: randoms needed here?
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
                        division_image = join(division_dir, image + ".nii")
                        message = "Adding %s from simulation %s" % (image, division)
                        tools.log_message(log_file, message)
                        
                        tools.operate_sinograms_nii(
                        #tools.operate_images_nii(
                            zero_image, division_image, zero_image, "sum"
                        )
                        
                        #os.remove(division_image[0:-3] + "nii") #NOTE: Not needed as full division folder will be removed
        
        #TODO: ADDED TO OPTIMIZE:
        #histtypes_list = []
        # if detlistmode == 1... if phglistmode == 1... and remove divisions apart.
        """ #This has become obsolete.
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
                    
                    #WARNING: AT THE MOMENT THIS IS NOT NEEDED. RECOVER WHEN DETLISTMODE IS USED.
                    #simset_tools.combine_history_files(
                    #    self.simset_dir, file_list, output, log_file
                    #)
                    #shutil.move(output, zero_hist)
                    
                    #shutil.rmtree(division_dir) # Once everything is combined in division_0, remove the other division
                
                #os.remove(join(division_zero, "det_hf.hist")) #IMPORTANT! REMOVE IF FAILS.
        """
        
        #Remove unused divisions...
        for division in range(1, self.divisions):
            division_dir = join(self.output_dir, "division_" + str(division))
            shutil.rmtree(division_dir) # Once everything is combined in division_0, remove the other division
                

        if self.add_randoms == 1: #TODO: THIS HAS TO BE TESTED IN NEW SETUP.
            # To have randoms in the final history file, we need to add randoms to the final det_hf.hist
            print("Adding randoms to the history file...")

            coincidence_window = self.scanner.get("coincidence_window")
            
            #NOTE: I THINK THIS PROCESS SHOULD BE DONE BEFORE REMOVING EACH DIVISION. THAT IS: EXTRACTING AND COMBINING FILES FOR EACH DIVISION. CHECK IF WORKS!
            #WARNING: AT THE MOMENT THIS IS NOT NEEDED. RECOVER WHEN DETLISTMODE IS USED.
            #simset_tools.add_randoms(
            #    division_zero,
            #    self.simset_dir,
            #    coincidence_window,
            #    rebin=False,
            #    log_file=log_file,
            #)

            # os.remove(join(division_zero, "sorted_det_hf.hist"))
            
            #WARNING: AT THE MOMENT THIS IS NOT NEEDED. RECOVER WHEN DETLISTMODE IS USED.
            #det_hist = join(division_zero, "det_hf.hist")
            #randoms_hist = join(division_zero, "randoms.hist")
            #output = join(division_zero, "full_det_hf.hist")

            #file_list = det_hist + " " + randoms_hist

            #simset_tools.combine_history_files(
            #    self.simset_dir, file_list, output, log_file
            #) 
            
            
        #self.stir_norm_from_att_map = self.scanner.get("stir_norm_from_att_map")
        self.attenuation_mode = self.scanner.get("attenuation_mode")
        
        if self.attenuation_mode == 1: #self.stir_norm_from_att_map != 1:
        
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
        
        #TODO: To remove unnecessary files once endeded the sim. Add condition: If exists.
        #TODO: Change directory name to "simuulation". IMPORTANT! REMOVE IF FAILS:
        
        """
        if exists(join(division_zero, "rec.weight")):
            os.remove(join(division_zero, "rec.weight"))
        
        if exists(join(division_zero, "rec.act_indexes")):
            os.remove(join(division_zero, "rec.act_indexes"))
            
        if exists(join(division_zero, "rec.att_indexes")):
            os.remove(join(division_zero, "rec.att_indexes"))
            
        if exists(join(division_zero, "rec.activity_image")):
            os.remove(join(division_zero, "rec.activity_image"))
            
        if exists(join(division_zero, "rec.attenuation_image")):
            os.remove(join(division_zero, "rec.attenuation_image"))
        """
        
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
        
        self.simset_dir = self.config.get("dir_simset")
        self.stir_dir = self.config.get("dir_stir")
        
        num_rings = self.scanner.get("num_rings")
        max_segment = self.scanner.get("max_segment")
        
        
        #self.stir_norm_from_att_map = self.scanner.get("stir_norm_from_att_map")
        self.attenuation_mode = self.scanner.get("attenuation_mode")
        
        #if ((self.stir_norm_from_att_map != 1) and (not exists(join(self.input_dir, "attenuationsino.nii")))):
        if ((self.attenuation_mode == 1) and (not exists(join(self.input_dir, "attenuationsino.nii")))):
        
            print("Attenuation map was not computed: Calculating attenuation map...")
            print(" ")

            output_atten = "attenuationsino"
            hdr_to_copy = join("trues.nii")

            simset_tools.simset_calcattenuation(
                self.simset_dir, self.input_dir, output_atten, hdr_to_copy, nrays=1, timeout=None
            )
        
        trues_sino = join(self.input_dir, "trues.nii")
        scatter_sino = join(self.input_dir, "scatter.nii")
        randoms_sino = join(self.input_dir, "randoms.nii")

        corr_scatter_sino = join(self.input_dir, "corr_scatter.nii")
        corr_randoms_sino = join(self.input_dir, "corr_randoms.nii")
        my_simset_sino = join(self.input_dir, "my_sinogram.nii")
        additive_sinogram = join(self.input_dir, "additive_sinogram.nii")
        att_sino = join(self.input_dir, "attenuationsino.nii")
        
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
        
        if not exists(my_simset_sino):
            tools.operate_images_nii(
                trues_sino, corr_scatter_sino, my_simset_sino, operation="sum", check_nans = False,
            )

        if self.add_randoms == 1:
            #tools.operate_single_image(
            tools.operate_single_image_nii(
                randoms_sino,
                "mult",
                self.random_corr_factor,
                corr_randoms_sino,
                self.log_file,
                #check_nans = False
            )
            #tools.operate_images_analyze(
            tools.operate_sinograms_nii(
                my_simset_sino, corr_randoms_sino, my_simset_sino, operation="sum", #check_nans = False,
            )

            if self.scanner.get("stir_randoms_corr_smoothing") == 1:
                #tools.operate_images_analyze(
                tools.operate_sinograms_nii(
                    scatter_sino, randoms_sino, additive_sinogram, operation="sum", #check_nans = False,
                )
            else:
                tools.copy_nifti(scatter_sino, additive_sinogram)

        else:
            if self.scanner.get("stir_scatt_corr_smoothing") == 1:
                print("COPYING ADDITIVE SINO")  #TODO: CHECK IF NECESSARY IN FUNCTION OF stir_scatt_corr_smoothing:
                if not exists(additive_sinogram):
                    tools.copy_nifti(scatter_sino, additive_sinogram)
                    print("SMOOTHING ADDITIVE SINO")
                    tools.smooth_nifti(additive_sinogram, 10, additive_sinogram)
                
        
        print("GENERATING STIR SINOGRAM")
        
        sinogram_stir_nii = join(self.input_dir, "stir_sinogram.nii")
        sinogram_stir_s = join(self.output_dir, "stir_sinogram.s")
        sinogram_stir_hs = join(self.output_dir, "stir_sinogram.hs")
        
        if not exists(sinogram_stir_nii):
            tools.convert_simset_sino_to_stir_nii(my_simset_sino, sinogram_stir_nii)
            
        tools.copy_sinogram_stir_to_output(sinogram_stir_nii, sinogram_stir_s) #WARNING! GET BACK IF NEEDED.
        #tools.copy_reduced_sinogram_stir_to_output(sinogram_stir_nii, sinogram_stir_s, num_rings, max_segment)
        
        stir_tools.create_stir_hs_from_detparams(
            self.scanner, sinogram_stir_hs
        )
        
        
        if self.scanner.get("stir_scatt_corr_smoothing") == 1:
            
            print("GENERATING STIR ADDITIVE") #TODO: CHECK IF NECESSARY IN FUNCTION OF stir_scatt_corr_smoothing:
        
            additivesino_stir_nii = join(self.input_dir, "stir_additivesino.nii")
            additivesino_stir_s = join(self.output_dir, "stir_additivesino.s")
            additivesino_stir_hs = join(self.output_dir, "stir_additivesino.hs")
            
            if not exists(additivesino_stir_nii):
                tools.convert_simset_sino_to_stir_nii(additive_sinogram, additivesino_stir_nii)
            
            tools.copy_sinogram_stir_to_output(additivesino_stir_nii, additivesino_stir_s) #WARNING! GET BACK IF NEEDED.
            #tools.copy_reduced_sinogram_stir_to_output(additivesino_stir_nii, additivesino_stir_s, num_rings, max_segment)
        
            stir_tools.create_stir_hs_from_detparams(
                self.scanner, additivesino_stir_hs
            )
    
        print("GENERATING STIR ATT")
        att_stir_nii = join(self.input_dir, "stir_att.nii")
        att_stir_s = join(self.output_dir, "stir_att.s")
        att_stir_hs = join(self.output_dir, "stir_att.hs")
        
        #if self.stir_norm_from_att_map != 1: #TODO: REFINE THIS.
        
        if self.attenuation_mode == 1:
            if not exists(att_stir_nii):
                tools.convert_simset_sino_to_stir_nii(att_sino, att_stir_nii)
                
            tools.copy_sinogram_stir_to_output(att_stir_nii, att_stir_s) #WARNING! GET BACK IF NEEDED.
            #tools.copy_reduced_sinogram_stir_to_output(att_stir_nii, att_stir_s, num_rings, max_segment)
            
            stir_tools.create_stir_hs_from_detparams(
                self.scanner, att_stir_hs
            )
        
        
        #OBTAIN MUMAP DIRECTLY FROM ATTMAP:
        #if self.stir_norm_from_att_map == 1:
        if self.attenuation_mode == 2:
            
            #if exists(att_stir_nii): #inactive at the moment but this is very time consuming...
            
            if not exists(att_stir_nii): #Added to save time... Remove if does not work.
            
                nib.save(nib.load(join(self.input_dir, 'stir_sinogram.nii')), join(self.input_dir, 'stir_sinogram.hdr')) #DELETE, this is only for testing...
                
                attmap = nib.load(join('Data', self.params.get("patient_dirname"), self.params.get("att_map")))
                att_indexes = np.uint8(attmap.get_fdata())
                
                attmap_dim = np.shape(attmap)
                attmap_pixsize = attmap.header['pixdim'][1:4]
                
                mu_map = np.zeros(np.shape(att_indexes), dtype=np.float32)

                for attidx in np.unique(att_indexes):
                    mu_map[att_indexes == attidx] = tools.mu_coef_511keV(attidx)
                
                mu_map_img = nib.Nifti1Image(mu_map, attmap.affine)
                nib.save(mu_map_img, join(self.input_dir, 'mu_map.hdr'))
                nib.save(mu_map_img, join(self.input_dir, 'mu_map.nii'))
                
                shutil.copy(join(self.input_dir, 'mu_map.img'), join(self.output_dir, 'mu_map.v'))
                tools.write_interfile_header_mu(join(self.output_dir, 'mu_map.hv'), attmap_dim[0], attmap_pixsize[0],
                        attmap_dim[1], attmap_pixsize[1],
                        attmap_dim[2], attmap_pixsize[2]) #offset_z = 0) #offset_z = None) #offset_z = -0.936025*128)
                
                
                tools.write_fwdproj_parfile(join(self.output_dir, "fwdproj_par.par"))
                
                #This works but removed for logging.
                #ACF_command = "%s --ACF %s %s %s %s" % (join(self.stir_dir, "bin", "calculate_attenuation_coefficients"), join(self.output_dir, "stir_att.hs"), join(self.output_dir, 'mu_map.hv'), sinogram_stir_hs, join(self.output_dir, "fwdproj_par.par"))
                #TODO: ADD LOGGING FOR ERROR MESSAGES AS WELL AS PRINTING TO TERMINAL, JUST LIKE HERE:
                ACF_command = "%s --ACF %s %s %s %s 2>&1 | tee %s" % (join(self.stir_dir, "bin", "calculate_attenuation_coefficients"), join(self.output_dir, "stir_att.hs"), join(self.output_dir, 'mu_map.hv'), sinogram_stir_hs, join(self.output_dir, "fwdproj_par.par"), join(self.output_dir, "att_logging.log"))
                
                #TODO: Still have to think about this... Way to save the nifti to the simulation folder.
                os.system(ACF_command)
                
                #Added to save time and keep the stir_att... Remove if does not work.
                shutil.copyfile(join(self.output_dir, "stir_att.s"), join(self.input_dir, "stir_att.img"))
                
                os.rename(join(self.input_dir, 'stir_sinogram.hdr'), join(self.input_dir, 'stir_att.hdr'))
                
                stiratt_img = nib.load(join(self.input_dir, "stir_att.img"))
                stiratt_data = stiratt_img.dataobj
                
                stiratt_nifti2 = nib.Nifti2Image(stiratt_data, stiratt_img.affine, stiratt_img.header)
                nib.save(stiratt_nifti2, join(self.input_dir, "stir_att.nii")) 
                
                #Remove not needed files anymore...
                os.remove(join(self.input_dir, 'mu_map.hdr'))
                os.remove(join(self.input_dir, 'mu_map.img'))
                os.remove(join(self.input_dir, 'stir_att.hdr'))
                os.remove(join(self.input_dir, 'stir_att.img'))
                os.remove(join(self.input_dir, 'stir_sinogram.img'))
                
            else:
                
                tools.copy_sinogram_stir_to_output(att_stir_nii, att_stir_s) #WARNING! GET BACK IF NEEDED.

            stir_tools.create_stir_hs_from_detparams(
                self.scanner, att_stir_hs, output_format = "STIR"
            )
                
                
        
        
        
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
            #stir_tools.apply_psf(self.scanner, sinogram_stir, self.log_file) #RECOVER IF NOT WORKING.
            stir_tools.apply_psf(self.scanner, join(self.output_dir, "stir_sinogram.nii"), self.log_file)
            

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
        
        #if self.scanner.get("stir_norm_from_att_map") != 1:
        if self.scanner.get("attenuation_mode") != 0:
        
            if any(
                exists(i) == False for i in [sinogram_stir, att_stir] #[sinogram_stir, additive_sino_stir, att_stir] #NOTE: Integrate in the future...
            ):
                print("Something is not ready for the reconstruction")
            else:
                start_recons = True
                print("Starting STIR reconstruction")
        
        else:
            
            if any(
                exists(i) == False for i in [sinogram_stir] #[sinogram_stir, additive_sino_stir] #NOTE: Integrate in the future...
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
