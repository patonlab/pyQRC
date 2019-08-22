#!/usr/bin/python
from __future__ import print_function, absolute_import

#######################################################################
#                             pyQRC.py                                #
#  A Python implementation of Silva and Goodman's QRC approach        #
#  From a Gaussian frequency calculation it is possible to create     #
#  new input files which are displaced along any normal modes which   #
#  have an imaginary frequency.                                       #
#######################################################################
__version__ = '1.0' # last modified May 5 2018
__author__ = 'Rob Paton'
__email__= 'robert.paton@colostate.edu'
#######################################################################

#Python Libraries
import sys, os
from glob import glob
from optparse import OptionParser

#Some useful arrays
periodictable = ["","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl",
    "Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Uub","Uut","Uuq","Uup","Uuh","Uus","Uuo"]

def elementID(massno):
        if massno < len(periodictable): return periodictable[massno]
        else: return "XX"

# Enables output to terminal and to text file
class Logger:
   # Designated initializer
   def __init__(self,filein,suffix,append):
      # Create the log file at the input path
      self.log = open(filein+"_"+append+"."+suffix, 'w' )

   # Write a message only to the log and not to the terminal
   def Writeonlyfile(self, message):
      self.log.write("\n"+message)

#Read data from an output file
class getoutData:
    def __init__(self, file):
        if not os.path.exists(file):
            print(("\nFATAL ERROR: Output file [ %s ] does not exist"%file))

        def getJOBTYPE(self, outlines, format):
            if format == "Gaussian":
                level = "none"; bs = "none"
                for i in range(0,len(outlines)):
                    if outlines[i].strip().find('----------') > -1:
                        if outlines[i+1].strip().find('#') > -1:
                            self.JOBTYPE = outlines[i+1].strip().split('#')[1]
                            break;

        def getTERMINATION(self, outlines,format):
                    if format == "Gaussian":
                      for i in range(0,len(outlines)):
                                    if outlines[i].find("Normal termination") > -1:
                                            self.TERMINATION = "normal"

        def getCHARGE(self, outlines, format):
            if format == "Gaussian":
                for i in range(0,len(outlines)):
                    if outlines[i].find("Charge = ") > -1:
                        self.CHARGE = int(outlines[i].split()[2])
                        self.MULT = int(outlines[i].split()[5].rstrip("\n")); break

        def getMASSES(self, outlines, format):
            self.MASSES = []
            if format == "Gaussian":
                for i in range(0,len(outlines)):
                    if outlines[i].find("has atomic number") > -1:
                        self.MASSES.append(float(outlines[i].split()[8]))

        def getATOMTYPES(self, outlines, format):
            self.ATOMTYPES = []; self.CARTESIANS = []
            if format == "Gaussian":
                for i in range(0,len(outlines)):
                    if outlines[i].find("Input orientation") > -1: standor = i
                    if outlines[i].find("Standard orientation") > -1: standor = i
                    if outlines[i].find("Distance matrix") > -1 or outlines[i].find("Rotational constants") >-1:
                        if outlines[i-1].find("-------") > -1:
                            self.NATOMS = i-standor-6
                try: standor
                except NameError: pass
                else:
                    for i in range (standor+5,standor+5+self.NATOMS):
                        self.ATOMTYPES.append(elementID(int(outlines[i].split()[1])))
                        self.CARTESIANS.append([float(outlines[i].split()[3]), float(outlines[i].split()[4]), float(outlines[i].split()[5])])

        def getFREQS(self, outlines, natoms, format):
            self.FREQS = []; self.REDMASS = []; self.FORCECONST = []; self.NORMALMODE = []; self.IM_FREQS = 0
            freqs_so_far = 0
            if format == "Gaussian":

                for i in range(0,len(outlines)):
                    if outlines[i].find(" Frequencies -- ") > -1:

                        nfreqs = len(outlines[i].split())
                        for j in range(2, nfreqs):
                            self.FREQS.append(float(outlines[i].split()[j]))
                            self.NORMALMODE.append([])
                            if float(outlines[i].split()[j]) < 0.0: self.IM_FREQS += 1
                        for j in range(3, nfreqs+1): self.REDMASS.append(float(outlines[i+1].split()[j]))
                        for j in range(3, nfreqs+1): self.FORCECONST.append(float(outlines[i+2].split()[j]))

                        for j in range(0,natoms):
                            for k in range(0, nfreqs-2):
                                self.NORMALMODE[(freqs_so_far + k)].append([float(outlines[i+5+j].split()[3*k+2]), float(outlines[i+5+j].split()[3*k+3]), float(outlines[i+5+j].split()[3*k+4])])
                        freqs_so_far = freqs_so_far + nfreqs - 2

        outfile = open(file,"r")
        outlines = outfile.readlines()
        getJOBTYPE(self, outlines, "Gaussian")
        getTERMINATION(self, outlines,"Gaussian")
        getCHARGE(self, outlines, "Gaussian")
        getATOMTYPES(self, outlines, "Gaussian")
        getFREQS(self, outlines, self.NATOMS, "Gaussian")
        getMASSES(self, outlines, "Gaussian")

class gen_qrc:
   def __init__(self, file, amplitude, nproc, mem, route, verbose, suffix, val, num):

        freq = getoutData(file)
        # Write an output file
        if verbose: log = Logger(file.split(".")[0],"qrc", suffix)

        # The molecular data as read in from the frequency calculation, including atomic masses
        if verbose:
            log.Writeonlyfile(' pyQRC - a quick alternative to IRC calculations')
            log.Writeonlyfile(' version: '+__version__+' / author: '+__author__+' / email: '+__email__)
            log.Writeonlyfile(' Based on: Goodman, J. M.; Silva, M. A. Tet. Lett. 2003, 44, 8233-8236; Tet. Lett. 2005, 46, 2067-2069.\n')
            log.Writeonlyfile('                -----ORIGINAL GEOMETRY------')
            log.Writeonlyfile('{0:>4} {1:>9} {2:>9} {3:>9} {4:>9}'.format('', '', 'X', 'Y', 'Z'))
            for atom in range(0,freq.NATOMS):
                log.Writeonlyfile('{0:>4} {1:>9} {2:9.6f} {3:9.6f} {4:9.6f}'.format(freq.ATOMTYPES[atom], '', freq.CARTESIANS[atom][0], freq.CARTESIANS[atom][1], freq.CARTESIANS[atom][2]))
            log.Writeonlyfile('\n                ----HARMONIC FREQUENCIES----')
            log.Writeonlyfile('{0:>24} {1:>9} {2:>9}'.format('Freq', 'Red mass', 'F const'))
            for mode in range(0,3*freq.NATOMS-6):
                log.Writeonlyfile('{0:24.4f} {1:9.4f} {2:9.4f}'.format(freq.FREQS[mode], freq.REDMASS[mode], freq.FORCECONST[mode]))

        shift = []

        # Save the original Cartesian coordinates before they are altered
        orig_carts = []
        for atom in range(0,freq.NATOMS):
            orig_carts.append([freq.CARTESIANS[atom][0], freq.CARTESIANS[atom][1], freq.CARTESIANS[atom][2]])

        # could get rid of atomic units here, if zpe_rat definition is changed
        for mode, wn in enumerate(freq.FREQS):
            # Either moves along any and all imaginary freqs, or a specific mode requested by the user
            if freq.FREQS[mode] < 0.0 and val == None and num == None:
                shift.append(amplitude)
                if verbose:
                    log.Writeonlyfile('\n                -SHIFTING ALONG NORMAL MODE-')
                    log.Writeonlyfile('                -AMPLIFIER = '+str(shift[mode]))

                    log.Writeonlyfile('{0:>4} {1:>9} {2:>9} {3:>9} {4:>9}'.format('', '', 'X', 'Y', 'Z'))
                    for atom in range(0,freq.NATOMS):
                        log.Writeonlyfile('{0:>4} {1:>9} {2:9.6f} {3:9.6f} {4:9.6f}'.format(freq.ATOMTYPES[atom], '', freq.NORMALMODE[mode][atom][0], freq.NORMALMODE[mode][atom][1], freq.NORMALMODE[mode][atom][2]))
            elif freq.FREQS[mode] == val or mode+1 == num:
                print(wn, num)
                shift.append(amplitude)
                if verbose:
                    log.Writeonlyfile('\n                -SHIFTING ALONG NORMAL MODE-')
                    log.Writeonlyfile('                -AMPLIFIER = '+str(shift[mode]))

                    log.Writeonlyfile('{0:>4} {1:>9} {2:>9} {3:>9} {4:>9}'.format('', '', 'X', 'Y', 'Z'))
                    for atom in range(0,freq.NATOMS):
                        log.Writeonlyfile('{0:>4} {1:>9} {2:9.6f} {3:9.6f} {4:9.6f}'.format(freq.ATOMTYPES[atom], '', freq.NORMALMODE[mode][atom][0], freq.NORMALMODE[mode][atom][1], freq.NORMALMODE[mode][atom][2]))
            else: shift.append(0.0)
            
            # The starting geometry is displaced along the each normal mode according to the random shift
            for atom in range(0,freq.NATOMS):
                for coord in range(0,3):
                    freq.CARTESIANS[atom][coord] = freq.CARTESIANS[atom][coord] + freq.NORMALMODE[mode][atom][coord] * shift[mode]

        new_input = Logger(file.split(".")[0],"com", suffix)
        if route == None:
            route = freq.JOBTYPE
        new_input.Writeonlyfile('%chk='+file.split(".")[0]+"_"+suffix+".chk")
        new_input.Writeonlyfile('%nproc='+str(nproc)+'\n%mem='+mem+'\n#'+route+'\n\n'+file.split(".")[0]+'_'+suffix+'\n\n'+str(freq.CHARGE)+" "+str(freq.MULT))
        for atom in range(0,freq.NATOMS):
            new_input.Writeonlyfile('{0:>2} {1:12.8f} {2:12.8f} {3:12.8f}'.format(freq.ATOMTYPES[atom], freq.CARTESIANS[atom][0], freq.CARTESIANS[atom][1], freq.CARTESIANS[atom][2]))
        new_input.Writeonlyfile("\n")

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
    (options, args) = parser.parse_args()

    files = []
    if len(sys.argv) > 1:
      for elem in sys.argv[1:]:
         try:
            if os.path.splitext(elem)[1] in [".out", ".log"]:
               for file in glob(elem): files.append(file)
         except IndexError: pass

    for file in files:
        freq = getoutData(file)
        if freq.IM_FREQS == 0 and options.auto != False:
           print('x   {} has no imaginary frequencies: skipping'.format(file))
        else:
            if options.freq == None and options.freqnum == None:
                print('o   {} has {} imaginary frequencies: processing'.format(file, freq.IM_FREQS))
            elif options.freq != None:
                print('o   {} will be distorted along the frequency of {} cm-1: processing'.format(file, options.freq))
            elif options.freqnum != None:
                print('o   {} will be distorted along the frequency number {}: processing'.format(file, options.freqnum))
            qrc = gen_qrc(file, options.amplitude, options.nproc, options.mem, options.route, options.verbose, options.suffix, options.freq, options.freqnum)

if __name__ == "__main__":
    main()
