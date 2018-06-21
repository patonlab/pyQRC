%chk=acetaldehyde.chk
%nproc=4
%mem=8GB
#opt freq B3LYP/def2tzvp empiricaldispersion=GD3BJ

 acetaldehyde = [H]C(C)[D=O

0  1
H           1.02598         0.12223         0.06942
C           2.12749         0.06671         0.05614
C           2.85038         1.34175         0.36112
O           2.70357        -0.98671        -0.19583
H           2.11900         2.13190         0.55012
H           3.46686         1.62591        -0.49461
H           3.46687         1.20794         1.25280

