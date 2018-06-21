pyQRC


## Example 1
'''
pyQRC.py acetaldehyde.log --nproc 4 --mem 8GB
'''
The initial optimization inadvertently produced a transition structure. Displacement along the normal mode and another optimization fixed this problem.


## Example 2
'''
pyQRC.py claisen_ts.log --nproc 4 --mem 8GB --amp 0.3 --name QRCF
pyQRC.py claisen_ts.log --nproc 4 --mem 8GB --amp -0.3 --name QRCR
'''
The initial optimization located a transition structure. The quick reaction coordinate (QRC) is obtained from two optmizations, started from points displaced along the reaction coordinate in either direction. 
