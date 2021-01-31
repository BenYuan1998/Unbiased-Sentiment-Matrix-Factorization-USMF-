#!/bin/bash
#


MODELCODEPY_SWEEP="wg_wmf_obs wg_wmf_cau_ips wg_wmf_cau_const_add wg_wmf_cau_user_add wg_pmf_obs wg_pmf_cau_ips wg_pmf_cau_const_add wg_pmf_cau_user_add wg_pf_obs wg_pf_cau_ips wg_pf_cau_const_add wg_pf_cau_user_add"
DIR_PREFIX="dat/proc/"
DATADIR_SWEEP="simulation"
ODIR_PREFIX="out/"
LOCALFITDIR_SWEEP="_Afit"
OUTDIR_SWEEP="_Yfit"
OUTDIM_SWEEP="5 10 20 50"
CAUDIM_SWEEP="5 10 20 50"
THOLD_SWEEP="2 3"
BATCHSIZE_SWEEP="100"
NITER_SWEEP="20000"
PRIORU_SWEEP="1"
PRIORV_SWEEP="1"
ALPHA_SWEEP="40"
BINARY_SWEEP="0"
SIMULATION_SWEEP="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"

DATADIRSUFFIX="_wg"

SUFFIX=".py"

export DATA="R3"



for MODELCODEPYi in ${MODELCODEPY_SWEEP}; do
    export MODELCODEPY=${MODELCODEPYi}${SUFFIX}
    for DATADIRi in ${DATADIR_SWEEP}; do
    	for SIMULATIONi in ${SIMULATION_SWEEP}; do
	    	export DATADIR=${DIR_PREFIX}${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}
		    for LOCALFITDIRi in ${LOCALFITDIR_SWEEP}; do
		    	export LOCALFITDIR=${ODIR_PREFIX}${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}${LOCALFITDIRi}
			    for OUTDIRi in ${OUTDIR_SWEEP}; do
			    	export OUTDIR=${ODIR_PREFIX}${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}${OUTDIRi}
				    for OUTDIMi in ${OUTDIM_SWEEP}; do
				    	export OUTDIM=${OUTDIMi}
					    for CAUDIMi in ${CAUDIM_SWEEP}; do
					    	export CAUDIM=${CAUDIMi}
						    for THOLDi in ${THOLD_SWEEP}; do
						    	export THOLD=${THOLDi}
							    for BATCHSIZEi in ${BATCHSIZE_SWEEP}; do
							    	export BATCHSIZE=${BATCHSIZEi}
								    for NITERi in ${NITER_SWEEP}; do
								    	export NITER=${NITERi}
									    for PRIORUi in ${PRIORU_SWEEP}; do
									    	export PRIORU=${PRIORUi}
										    for PRIORVi in ${PRIORV_SWEEP}; do
										    	export PRIORV=${PRIORVi}
											    for ALPHAi in ${ALPHA_SWEEP}; do
											    	export ALPHA=${ALPHAi}
											    	for BINARYi in ${BINARY_SWEEP}; do
											    		export BINARY=${BINARYi}
											            NAME=data_${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}_model_${MODELCODEPYi}_odim_${OUTDIM}_cdim_${CAUDIM}_th_${THOLD}_M_${BATCHSIZE}_nitr_${NITER}_pU_${PRIORU}_pV_${PRIORV}_alpha_${ALPHA}_binary_${BINARY}
											            echo ${NAME}
											            sbatch --job-name=${NAME} \
											            --output=${NAME}.out \
											            run_scripts.sh
											        done
										        done
	       									done
	    								done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done



MODELCODEPY_SWEEP="sg_wmf_obs sg_wmf_cau_ips sg_wmf_cau_const_add sg_wmf_cau_user_add sg_pmf_obs sg_pmf_cau_ips sg_pmf_cau_const_add sg_pmf_cau_user_add sg_pf_obs sg_pf_cau_ips sg_pf_cau_const_add sg_pf_cau_user_add"
DIR_PREFIX="dat/proc/"
DATADIR_SWEEP="simulation"
ODIR_PREFIX="out/"
LOCALFITDIR_SWEEP="_Afit"
OUTDIR_SWEEP="_Yfit"
OUTDIM_SWEEP="5 10 20 50"
CAUDIM_SWEEP="5 10 20 50"
THOLD_SWEEP="2 3"
BATCHSIZE_SWEEP="100"
NITER_SWEEP="20000"
PRIORU_SWEEP="1"
PRIORV_SWEEP="1"
ALPHA_SWEEP="40"
BINARY_SWEEP="0"
SIMULATION_SWEEP="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21"


SUFFIX=".py"

export DATA="R3"


DATADIRSUFFIX="_sg"


for MODELCODEPYi in ${MODELCODEPY_SWEEP}; do
    export MODELCODEPY=${MODELCODEPYi}${SUFFIX}
    for DATADIRi in ${DATADIR_SWEEP}; do
    	for SIMULATIONi in ${SIMULATION_SWEEP}; do
	    	export DATADIR=${DIR_PREFIX}${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}
		    for LOCALFITDIRi in ${LOCALFITDIR_SWEEP}; do
		    	export LOCALFITDIR=${ODIR_PREFIX}${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}${LOCALFITDIRi}
			    for OUTDIRi in ${OUTDIR_SWEEP}; do
			    	export OUTDIR=${ODIR_PREFIX}${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}${OUTDIRi}
				    for OUTDIMi in ${OUTDIM_SWEEP}; do
				    	export OUTDIM=${OUTDIMi}
					    for CAUDIMi in ${CAUDIM_SWEEP}; do
					    	export CAUDIM=${CAUDIMi}
						    for THOLDi in ${THOLD_SWEEP}; do
						    	export THOLD=${THOLDi}
							    for BATCHSIZEi in ${BATCHSIZE_SWEEP}; do
							    	export BATCHSIZE=${BATCHSIZEi}
								    for NITERi in ${NITER_SWEEP}; do
								    	export NITER=${NITERi}
									    for PRIORUi in ${PRIORU_SWEEP}; do
									    	export PRIORU=${PRIORUi}
										    for PRIORVi in ${PRIORV_SWEEP}; do
										    	export PRIORV=${PRIORVi}
											    for ALPHAi in ${ALPHA_SWEEP}; do
											    	export ALPHA=${ALPHAi}
											    	for BINARYi in ${BINARY_SWEEP}; do
											    		export BINARY=${BINARYi}
											            NAME=data_${DATADIRi}${SIMULATIONi}${DATADIRSUFFIX}_model_${MODELCODEPYi}_odim_${OUTDIM}_cdim_${CAUDIM}_th_${THOLD}_M_${BATCHSIZE}_nitr_${NITER}_pU_${PRIORU}_pV_${PRIORV}_alpha_${ALPHA}_binary_${BINARY}
											            echo ${NAME}
											            sbatch --job-name=${NAME} \
											            --output=${NAME}.out \
											            run_scripts.sh
											        done
										        done
	       									done
	    								done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

