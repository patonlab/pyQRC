#!/usr/bin/python
from __future__ import print_function, absolute_import

#######################################################################
#                             pyQRC.py                                #
#  A Python implementation of Silva and Goodman's QRC approach        #
#  From a Gaussian frequency calculation it is possible to create     #
#  new input files which are displaced along any normal modes which   #
#  have an imaginary frequency.                                       #
#######################################################################
__version__ = '2.0' # last modified May 5 2018
__author__ = 'Robert Paton'
__email__= 'robert.paton@colostate.edu'
#######################################################################

#Python Libraries
import sys, os, re
from glob import glob
from optparse import OptionParser
import cclib
import numpy as np
import shutil
import subprocess

#Some useful arrays
periodictable = ["","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl",
    "Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Uub","Uut","Uuq","Uup","Uuh","Uus","Uuo"]

atomic_masses = [0.0, 1.007825, 4.00, 7.00, 9.00,\
                 11.00,  12.01, 14.0067, 15.9994,\
                 19.00, 20.180, 22.990,  24.305,\
                  26.982, 28.086, 30.973762, 31.972071,\
                  35.453, 39.948, 39.098, 40.078,\
                  44.956, 47.867, 50.942, 51.996,\
                  54.938, 55.845,58.933,  58.693,\
                  63.546, 65.38, 69.723,  72.631,\
                  74.922, 78.971, 79.904, 84.798,\
                  84.468, 87.62, 88.906, 91.224,\
                  92.906, 95.95, 98.907, 101.07,\
                  102.906, 106.42, 107.868, 112.414,\
                  114.818, 118.711, 121.760, 126.7,\
                  126.904, 131.294, 132.905, 137.328,\
                  138.905, 140.116, 140.908, 144.243,\
                  144.913, 150.36, 151.964, 157.25,\
                  158.925, 162.500, 164.930, 167.259,\
                  168.934, 173.055, 174.967, 178.49,\
                  180.948, 183.84, 186.207, 190.23,\
                  192.217, 195.085, 196.967, 200.592,\
                  204.383, 207.2, 208.980, 208.982,\
                  209.987, 222.081, 223.020, 226.025,\
                  227.028, 232.038, 231.036, 238.029,\
                  237, 244, 243, 247, 247,\
                  251, 252, 257, 258, 259,\
                  262, 261, 262, 266, 264,\
                  269, 268, 271, 272, 285,\
                  284, 289, 288, 292, 294,\
                  294]

rcov = {"H": 0.32,"He": 0.46,
	"Li":1.33,"Be":1.02,"B":0.85,"C":0.75,"N":0.71,"O":0.63,"F":0.64,"Ne":0.67,
	"Na":1.55,"Mg":1.39,"Al":1.26, "Si":1.16,"P":1.11,"S":1.03,"Cl":0.99, "Ar":0.96,
	"K":1.96,"Ca":1.71,"Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn":1.19, "Fe":1.16, "Co":1.11, "Ni":1.10,"Zn":1.18, "Ga":1.24, "Ge":1.21, "As":1.21, "Se":1.16, "Br":1.14, "Kr":1.17,
	"Rb":2.10, "Sr":1.85,"Y":1.63, "Zr":1.54, "Nb":1.47, "Mo":1.38, "Tc":1.28, "Ru":1.25,"Rh":1.25,"Pd":1.20,"Ag":1.28,"Cd":1.36, "In":1.42, "Sn":1.40,"Sb":1.40,"Te":1.36,"I":1.33,"Xe":1.31}

def elementID(massno):
        if massno < len(periodictable): return periodictable[massno]
        else: return "XX"

# Enables output to terminal and to text file
class Logger:
   # Designated initializer
   def __init__(self,filein,suffix,append):
      # Create the log file at the input path
      self.log = open(filein+"_"+append+"."+suffix,'w')
   # Write a message only to the log and not to the terminal
   def Writeonlyfile(self, message):
       self.log.write(message+"\n")

#Read data from an output file - data not currently provided by cclib
class getoutData:
    def __init__(self, file):
        if not os.path.exists(file):
            print(("\nFATAL ERROR: Output file [ %s ] does not exist"%file))

        def getFORMAT(self, outlines):
            for i in range(0,len(outlines)):
                if outlines[i].find('Gaussian, Inc.') > -1: self.format = "Gaussian"
                if outlines[i].find('* O   R   C   A *') > -1: self.format = "ORCA"
                if outlines[i].find('Q-Chem, Inc.') > -1: self.format = "QChem"

        def level_of_theory(file):
            """Read output for the level of theory and basis set used."""
            repeated_theory = 0
            with open(file) as f:
                data = f.readlines()
            level, bs = 'none', 'none'

            for line in data:
                if line.strip().find('External calculation') > -1:
                    level, bs = 'ext', 'ext'
                    break
                if '\\Freq\\' in line.strip() and repeated_theory == 0:
                    try:
                        level, bs = (line.strip().split("\\")[4:6])
                        repeated_theory = 1
                    except IndexError:
                        pass
                elif '|Freq|' in line.strip() and repeated_theory == 0:
                    try:
                        level, bs = (line.strip().split("|")[4:6])
                        repeated_theory = 1
                    except IndexError:
                        pass
                if '\\SP\\' in line.strip() and repeated_theory == 0:
                    try:
                        level, bs = (line.strip().split("\\")[4:6])
                        repeated_theory = 1
                    except IndexError:
                        pass
                elif '|SP|' in line.strip() and repeated_theory == 0:
                    try:
                        level, bs = (line.strip().split("|")[4:6])
                        repeated_theory = 1
                    except IndexError:
                        pass
                if 'DLPNO BASED TRIPLES CORRECTION' in line.strip():
                    level = 'DLPNO-CCSD(T)'
                if 'Estimated CBS total energy' in line.strip():
                    try:
                        bs = ("Extrapol." + line.strip().split()[4])
                    except IndexError:
                        pass
                # Remove the restricted R or unrestricted U label
                if level[0] in ('R', 'U'):
                    level = level[1:]
            level_of_theory = '/'.join([level, bs])
            return level_of_theory

        def getJOBTYPE(self, outlines):
            if self.format == "Gaussian":
                level = "none"; bs = "none"
                for i in range(0,len(outlines)):
                    if outlines[i].strip().find('----------') > -1:
                        if outlines[i+1].strip().find('#') > -1:
                            self.JOBTYPE = ''
                            for j in range(i+1,len(outlines)):
                                if outlines[j].strip().find('----------') > -1:
                                    break
                                else:
                                    self.JOBTYPE = self.JOBTYPE+re.sub('#','',outlines[j].strip())
                            self.JOBTYPE = re.sub(r' geom=\S+','',self.JOBTYPE)
                            self.LEVELOFTHEORY = level_of_theory(file)
                            break
            if self.format == "ORCA":
                level = "none"; bs = "none"
                for i in range(0,len(outlines)):
                    if outlines[i].strip().find('> !') > -1:
                        self.JOBTYPE = outlines[i].strip().split('> !')[1].lstrip()
                        self.LEVELOFTHEORY = level_of_theory(file)
                        break

        def getTERMINATION(self, outlines):
                    if self.format == "Gaussian":
                      for i in range(0,len(outlines)):
                                    if outlines[i].find("Normal termination") > -1:
                                            self.TERMINATION = "normal"

        outfile = open(file,"r")
        outlines = outfile.readlines()
        getFORMAT(self, outlines)
        getJOBTYPE(self, outlines)
        getTERMINATION(self, outlines)


# compute mass-weighted Cartesian displacment between two structures (bohr amu^1/2)
def mwdist(coords1, coords2, elements):
    dist = 0

    for n, atom in enumerate(elements):

        dist += atomic_masses[atom] * (np.linalg.norm(coords1[n] - coords2[n])) ** 2
        #print(coords1[n], coords2[n], atomic_masses[atom] ** 0.5, (np.linalg.norm(coords1[n] - coords2[n])))
        #print(atomic_masses[atom] ** 0.5 * (np.linalg.norm(coords1[n] - coords2[n])))
        #print((np.linalg.norm(coords1[n] - coords2[n])), dist)

    dist = 1.88972612456506 * dist ** 0.5

    return dist

class gen_qrc:
   def __init__(self, file, amplitude, nproc, mem, route, verbose, suffix, val, num):

        # parse compchem output with cclib
        parser = cclib.io.ccopen(file)
        data = parser.parse()

        #try:
        nat, charge, atomnos = data.natom, data.charge, data.atomnos
        try:
            mult = data.mult 
        except:
            AttributeError 
            mult = '1'    # surface level workaround to set default value of multiplicity to 1 if not parsed properly by cclib
            print('Warning - multiplicity not parsed from input: defaulted to 1 in input files')
        elements = [periodictable[z] for z in atomnos]
        cartesians = data.atomcoords[-1]
        freq, disps = data.vibfreqs, data.vibdisps
        nmodes = len(freq)
        if hasattr(data, 'vibrmasses'): rmass = data.vibrmasses
        else: rmass = [0.0] * nmodes
        if hasattr(data, 'vibfconsts'): fconst = data.vibfconsts
        else: fconst = [0.0] * nmodes

        self.CARTESIAN = []
        for atom in range(0,nat):
            self.CARTESIAN.append([cartesians[atom][0], cartesians[atom][1], cartesians[atom][2]])

        # Write an output file
        if verbose: log = Logger(file.split(".")[0],"qrc", suffix)

        # The molecular data as read in from the frequency calculation, including atomic masses
        if verbose:
            log.Writeonlyfile(' pyQRC - a quick alternative to IRC calculations')
            log.Writeonlyfile(' version: '+__version__+' / author: '+__author__+' / email: '+__email__)
            log.Writeonlyfile(' Based on: Goodman, J. M.; Silva, M. A. Tet. Lett. 2003, 44, 8233-8236; Tet. Lett. 2005, 46, 2067-2069.\n')
            log.Writeonlyfile('                -----ORIGINAL GEOMETRY------')
            log.Writeonlyfile('{0:>4} {1:>9} {2:>9} {3:>9} {4:>9}'.format('', '', 'X', 'Y', 'Z'))
            for atom in range(0,nat):
                log.Writeonlyfile('{0:>4} {1:>9} {2:9.6f} {3:9.6f} {4:9.6f}'.format(elements[atom], '', cartesians[atom][0], cartesians[atom][1], cartesians[atom][2]))
            log.Writeonlyfile('\n                ----HARMONIC FREQUENCIES----')
            log.Writeonlyfile('{0:>24} {1:>9} {2:>9}'.format('Freq', 'Red mass', 'F const'))
            for mode in range(0,nmodes):
                log.Writeonlyfile('{0:24.4f} {1:9.4f} {2:9.4f}'.format(freq[mode], rmass[mode], fconst[mode]))

        shift = []

        # Save the original Cartesian coordinates before they are altered
        orig_carts = []
        for atom in range(0,nat):
            orig_carts.append([cartesians[atom][0], cartesians[atom][1], cartesians[atom][2]])

        # Based on user input select the appropriate displacements
        for mode, wn in enumerate(freq):

            # Either moves along any and all imaginary freqs, or a specific mode requested by the user
            if wn < 0.0 and val == None and num == None:
                shift.append(amplitude)
                if verbose:
                    log.Writeonlyfile('\n                -SHIFTING ALONG NORMAL MODE-')
                    log.Writeonlyfile('                -AMPLIFIER = '+str(shift[mode]))

                    log.Writeonlyfile('{0:>4} {1:>9} {2:>9} {3:>9} {4:>9}'.format('', '', 'X', 'Y', 'Z'))
                    for atom in range(0,nat):
                        log.Writeonlyfile('{0:>4} {1:>9} {2:9.6f} {3:9.6f} {4:9.6f}'.format(elements[atom], '', disps[mode][atom][0], disps[mode][atom][1], disps[mode][atom][2]))

            elif wn == val or mode+1 == num:
                # print(wn, num)
                shift.append(amplitude)
                if verbose:
                    log.Writeonlyfile('\n                -SHIFTING ALONG NORMAL MODE-')
                    log.Writeonlyfile('                -AMPLIFIER = '+str(shift[mode]))

                    log.Writeonlyfile('{0:>4} {1:>9} {2:>9} {3:>9} {4:>9}'.format('', '', 'X', 'Y', 'Z'))
                    for atom in range(0,nat):
                        log.Writeonlyfile('{0:>4} {1:>9} {2:9.6f} {3:9.6f} {4:9.6f}'.format(elements[atom], '', disps[mode][atom][0], disps[mode][atom][1], disps[mode][atom][2]))
            else: shift.append(0.0)

            # This is where a perturbed structure is generated
            # The starting geometry is displaced along the each normal mode multipled by a user-specified amplitude
            for atom in range(0,nat):
                for coord in range(0,3):
                    cartesians[atom][coord] = cartesians[atom][coord] + disps[mode][atom][coord] * shift[mode]

        # useful information
        self.NEW_CARTESIAN = cartesians
        self.ATOMTYPES = elements

        # Record by how much the structure has been altered
        mw_distance = mwdist(self.NEW_CARTESIAN, self.CARTESIAN, atomnos)
        log.Writeonlyfile('\n   STRUCTURE MOVED BY {:.3f} Bohr amu^1/2 \n'.format(mw_distance))

        # Create a new compchem input file
        if hasattr(data, 'metadata'):
            try: format = data.metadata['package']
            except: format = None
            try: func = data.metadata['functional']
            except: func = None
            try: basis = data.metadata['basis_set']
            except: basis = None

        # if not specified, the job specification will be cloned from the previous calculation
        # In practice this works better than cclib metadata so is the default for now
        gdata = getoutData(file)

        if format == None: format = gdata.format
        if route == None: route = gdata.JOBTYPE

        if format == "Gaussian": input = "com"
        elif format == "ORCA" or format == "QChem": input = "inp"

        new_input = Logger(file.split(".")[0],input, suffix)

        if format == "Gaussian":
            new_input.Writeonlyfile('%chk='+file.split(".")[0]+"_"+suffix+".chk")
            new_input.Writeonlyfile('%nproc='+str(nproc)+'\n%mem='+mem+'\n#'+route+'\n\n'+file.split(".")[0]+'_'+suffix+'\n\n'+str(charge)+" "+str(mult))
        elif format == "ORCA":

            ## split maxcore string for ORCA
            memory_number = re.findall(r'\d+',mem)
            unit = re.findall(r'GB',mem)
            if len(unit) > 0:
                mem = int(memory_number[0])*1024

            else:
                ## asuuming memory is given in MB
                mem = memory_number[0]
            
            new_input.Writeonlyfile('! '+route+'\n %pal nprocs '+str(nproc) + ' end\n %maxcore '+ str(mem)+ '\n\n# '+file.split(".")[0]+'_'+suffix+'\n\n* xyz '+str(charge)+" "+str(mult))
        elif format == "QChem":
            new_input.Writeonlyfile('$molecule\n'+str(charge)+" "+str(mult))
        # Save the new Cartesian coordinates
        for atom in range(0,nat):
            new_input.Writeonlyfile('{0:>2} {1:12.8f} {2:12.8f} {3:12.8f}'.format(elements[atom], cartesians[atom][0], cartesians[atom][1], cartesians[atom][2]))
        if format == "Gaussian": new_input.Writeonlyfile("")
        elif format == "ORCA": new_input.Writeonlyfile("*")
        elif format == "QChem":
            new_input.Writeonlyfile("$end\n\n$rem")
            new_input.Writeonlyfile("   JOBTYPE opt\n   METHOD "+func+"\n   BASIS "+basis)
            new_input.Writeonlyfile("$end\n\n@@@\n\n$molecule\n   read\n$end\n\n$rem")
            new_input.Writeonlyfile("   JOBTYPE freq\n   METHOD "+func+"\n   BASIS "+basis)
            new_input.Writeonlyfile("$end\n")



        def gen_overlap(mol_atoms, coords, covfrac):
            ## Use VDW radii to infer a connectivity matrix
            over_mat = np.zeros((len(mol_atoms), len(mol_atoms)))
            for i, atom_no_i in enumerate(mol_atoms):
                for j, atom_no_j in enumerate(mol_atoms):
                    if j > i:
                        rcov_ij = rcov[atom_no_i] + rcov[atom_no_j]
                        dist_ij = np.linalg.norm(np.array(coords[i])-np.array(coords[j]))
                        if dist_ij / rcov_ij < covfrac:
                            #print((i+1), (j+1), dist_ij, vdw_ij, rcov_ij)
                            over_mat[i][j] = 1
                        else: pass
            return over_mat

        def check_overlap(self, covfrac=0.8):
            overlapped = None
            over_mat = gen_overlap(self.ATOMTYPES, self.NEW_CARTESIAN, covfrac)
            overlapped = np.any(over_mat)
            return overlapped

        self.OVERLAPPED = check_overlap(self)

        #except:
        #    print('o   Unable to parse information from {} with cclib ...'.format(file))


def g16_opt( comfile):
    ''' run g16 using shell script and args '''
    # check whether job has already been run
    logfile = os.path.splitext(comfile)[0] + '.log'
    command = [os.path.abspath(os.path.dirname(__file__))+'/run_g16.sh', str(comfile)]
    g16_result = subprocess.run(command)


def run_irc(file,options,num,amp,lot_bs,suffix,charge,mult,log_output):
    #checking amplitutes energy for a given node and creating the single point file
    qrc = gen_qrc(file, amp, options.nproc, options.mem, lot_bs, options.verbose, suffix, None, num)
    #do check of GEOMETRY if its valid and no atoms overlappins
    if not qrc.OVERLAPPED:
        g16_opt(file.split('.')[0]+'_'+suffix+'.com')
    else:
        log_output.Writeonlyfile('x  Skipping {} due to overlap in atoms'.format(file.split('.')[0]+'_'+suffix+'.com'))

def main():
    # get command line inputs. Use -h to list all possible arguments and default values
    parser = OptionParser(usage="Usage: %prog [options] <input1>.log <input2>.log ...")
    parser.add_option("--amp", dest="amplitude", action="store", help="amplitude (default 0.2)", default="0.2", type="float", metavar="AMPLITUDE")
    parser.add_option("--nproc", dest="nproc", action="store", help="number of processors (default 1)", default="1", type="int", metavar="NPROC")
    parser.add_option("--mem", dest="mem", action="store", help="memory (default 4GB)", default="4GB", type="string", metavar="MEM")
    parser.add_option("--route", dest="route", action="store", help="calculation route (defaults to same as original file)", default=None, type="string", metavar="ROUTE")
    parser.add_option("-v", dest="verbose", action="store_true", help="verbose output", default=True, metavar="VERBOSE")
    parser.add_option("--auto", dest="auto", action="store_true", help="turn on automatic batch processing", default=False, metavar="AUTO")
    parser.add_option("--name", dest="suffix", action="store", help="append to file name (defaults to QRC)", default="QRC", type="string", metavar="SUFFIX")
    parser.add_option("-f", "--freq", dest="freq", action="store", help="request motion along a particular frequency (cm-1)", default=None, type="float", metavar="FREQ")
    parser.add_option("--freqnum", dest="freqnum", action="store", help="request motion along a particular frequency (number)", default=None, type="int", metavar="FREQNUM")

    #arguments for running calcs
    parser.add_option("--qcoord", dest="qcoord", action="store_true", help="request automatic single point calculation along a particular normal mode (number)", default=False, metavar="QCOORD")
    parser.add_option("--nummodes", dest="nummodes", action="store", help="number of modes for automatic single point calculation", default='all', type='string',metavar="NUMMODES")

    (options, args) = parser.parse_args()

    files = []
    if len(sys.argv) > 1:
      for elem in sys.argv[1:]:
         try:
            if os.path.splitext(elem)[1] in [".out", ".log"]:
               for file in glob(elem): files.append(file)
         except IndexError: pass


    for file in files:
        # parse compchem output with cclib & count imaginary frequencies
        parser = cclib.io.ccopen(file)
        data = parser.parse()

        #for i in range(1,len(data.atomcoords)):
        #    mw_distance = mwdist(data.atomcoords[i], data.atomcoords[1], data.atomnos)
        #    print(mw_distance)

        if hasattr(data, 'vibfreqs'):
            im_freq = len([val for val in data.vibfreqs if val < 0])
        else: print('o   {} has no frequency information: exiting'.format(file)); sys.exit()

        if not options.qcoord:
            if im_freq == 0 and options.auto != False:
               print('x   {} has no imaginary frequencies: skipping'.format(file))
            else:
                if options.freq == None and options.freqnum == None:
                    print('o   {} has {} imaginary frequencies: processing'.format(file, im_freq))
                elif options.freq != None:
                    print('o   {} will be distorted along the frequency of {} cm-1: processing'.format(file, options.freq))
                elif options.freqnum != None:
                    print('o   {} will be distorted along the frequency number {}: processing'.format(file, options.freqnum))
                qrc = gen_qrc(file, options.amplitude, options.nproc, options.mem, options.route, options.verbose, options.suffix, options.freq, options.freqnum)

        #doing automatic calcualtions (single points for stability check)
        else:
            log_output = Logger("RUNIRC",'dat',options.nummodes)
            if im_freq == 0:
               log_output.Writeonlyfile('o   {} has no imaginary frequencies: check for stability'.format(file))
            amp_base =  [round(elem, 2) for elem in np.arange(0,1,0.1) ]
            # amp_base_backward =  [round(elem, 2) for elem in np.arange(-1,0,0.1) ]
            energy_base = []
            root_dir = os.getcwd()
            parent_dir = os.getcwd()+'/'+file.split('.')[0]
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            log_output.Writeonlyfile('o  Entering directory {}'.format(parent_dir))

            # getting energetics of the current molecule
            energy_base.append(data.freeenergy)
            #creating folders for number of normal modes
            if options.nummodes=='all':
                freq_range = range(1,len(data.vibfreqs)+1)
            else:
                freq_range = range(1,len(data.vibfreqs)+1)
                freq_range = freq_range[:int(options.nummodes)]
            for num in freq_range:
                num_dir = parent_dir +'/'+ 'num_'+str(num)
                if not os.path.exists(num_dir):
                    os.makedirs(num_dir)
                log_output.Writeonlyfile('o  Entering directory {}'.format(num_dir))
                shutil.copyfile(root_dir+'/'+file, num_dir+'/'+file)
                os.chdir(num_dir)
                for amp in amp_base:
                    suffix = 'num_'+str(num)+'_amp_'+str(amp).split('.')[0]+str(amp).split('.')[1]
                    run_irc(file,options,num,amp,freq.LEVELOFTHEORY,suffix,freq.CHARGE,freq.MULT,log_output)
                    log_output.Writeonlyfile('o  Writing to file {}'.format(file.split('.')[0]+'_'+suffix))
                os.chdir(parent_dir)

if __name__ == "__main__":
    main()
