#!/bin/bash
# Full CICIoT2023 experiment queue. Usage (Colab):  !bash run_matrix.sh /content/ciciot /content/drive/MyDrive/abacfl
CSV=${1:-/content/ciciot}; OUT=${2:-/content/drive/MyDrive/abacfl}; mkdir -p "$OUT"
COMMON="--csv $CSV --cache $OUT/ciciot.npz --seeds 5 --rounds 30 --attack_scale 1.0 --cluster_strength 0.9 --alpha_client 1.0 --z0 2.0"
# E1 main matrix (defaults)
python run_ciciot.py $COMMON --attacks none,sign_flip,label_flip --rhos 0.0,0.5,1.0 --variants fedavg,median,krum,A0,A2,A3 | tee $OUT/E1_main.txt
# E2 stricter revocation evidence (compare hCli / hRev against E1)
python run_ciciot.py $COMMON --attacks sign_flip,label_flip --rhos 0.0,1.0 --variants A0,A2,A3 --tau_rep 0.7 --M 4 --N 6 | tee $OUT/E2_strict_revocation.txt
# E3 concentrated attackers (all inside one device type)
python run_ciciot.py $COMMON --attacks sign_flip,label_flip --rhos 1.0 --variants fedavg,A0,A2,A3 --placement cohort | tee $OUT/E3_cohort_capture.txt
# E4 adaptive on-off attacker
python run_ciciot.py $COMMON --attacks sign_flip --rhos 0.0,1.0 --variants fedavg,A0,A2,A3 --on_off 6,2,5 --rounds 40 | tee $OUT/E4_on_off.txt
STRICT="--tau_rep 0.7 --M 4 --N 6"
# E5 capture-guard variants under the two scenarios that stress it (strict revocation on)

for PM in cos_or_D D; do
  python run_ciciot.py $COMMON $STRICT --attacks sign_flip,label_flip --rhos 1.0 --variants A2,A3 --placement cohort --phi_mode $PM | tee $OUT/E5_capture_${PM}.txt
  python run_ciciot.py $COMMON $STRICT --attacks sign_flip --rhos 1.0 --variants A2,A3 --on_off 6,2,5 --rounds 40 --phi_mode $PM | tee $OUT/E5_onoff_${PM}.txt
  python run_ciciot.py $COMMON $STRICT --attacks none --rhos 1.0 --variants A0,A2,A3 --phi_mode $PM | tee $OUT/E5_clean_${PM}.txt
done
# E6 breakdown point of cohort scoring: attackers concentrated in ONE device type (14 clients), growing share.
# frac 0.08/0.13/0.16/0.20 of 38 clients = 3/5/6/8 attackers = 21%/36%/43%/57% of that cohort.
for F in 0.08 0.13 0.16 0.20; do
  python run_ciciot.py $COMMON $STRICT --attacks sign_flip,label_flip --rhos 1.0 --variants A0,A2,A3 --placement cohort --frac $F | tee $OUT/E6_capture_frac_${F}.txt
done
