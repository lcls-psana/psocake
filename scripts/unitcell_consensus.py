import sys
sys.path.append("/reg/neh/home/yoon82/Software/autosfx/scripts")
import os
import numpy as np
import subprocess
import evaluateIndexing as ei 
import sys
import argparse
from cctbx import uctbx
from cctbx import crystal
import collections
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
import subprocess
import stat

expName = 'cxih0115'# cxic0415 cxih0115 cxid9114 cxi78513 mfxp17318
runNum = 21 # (47, 95, 101) (21) (101) (14) (82)
runStr = str(runNum).zfill(4)
myPath = '/reg/data/ana03/scratch/yoon82/psocake/'+expName+'/'+expName+'/yoon82/psocake'
streamfile = os.path.join(myPath,'r'+runStr+'/'+expName+'_'+runStr+'.stream')
atol = 0.9 # unitcell values considered close enough (Angstrom and deg)
binSize = 0.1
snr_max = 10
print "Analysing streamfile: ", streamfile

rankName = expName+"_rank.txt"
if os.path.exists(rankName):
    os.remove(rankName)

spacegroup_guess = collections.defaultdict(list)
spacegroup_guess["triclinic_P"]="P1"
spacegroup_guess["monoclinic_P"]="P2"
spacegroup_guess["monoclinic_C"]="C2"
spacegroup_guess["orthorhombic_P"]="P222"
spacegroup_guess["orthorhombic_C"]="C222"
spacegroup_guess["orthorhombic_F"]="F222"
spacegroup_guess["orthorhombic_I"]="I222"
spacegroup_guess["tetragonal_P"]="P4"
spacegroup_guess["tetragonal_I"]="I4"
spacegroup_guess["rhombohedral_R"]="R3"
spacegroup_guess["hexagonal_P"]="P3"
spacegroup_guess["cubic_P"]="P23"
spacegroup_guess["cubic_F"]="F23"
spacegroup_guess["cubic_I"]="I23"

def write_cell(fname, bravais, unitcell):
    """
    fname:   name of CrystFEL unitcell filename to be written
    bravais: List containing [lattice type, centering, unique_axis]
    unitcell: List containing [a, b, c, alpha, beta, gamma]
    """
    if not fname.endswith(".cell"):
        fname += ".cell" 
    str1 = "CrystFEL unit cell file version 1.0\n"
    str2 = "lattice_type = "+ bravais[0]+ "\n"
    str3 = "centering = "+ bravais[1]+ "\n"
    str4 = "unique_axis = "+ bravais[2]+ "\n"
    str5 = "a = "+ str(float("{:.3f}".format(unitcell[0])))+ " A\n"
    str6 = "b = "+ str(float("{:.3f}".format(unitcell[1])))+ " A\n"
    str7 = "c = "+ str(float("{:.3f}".format(unitcell[2])))+ " A\n"
    str8 = "al = "+ str(float("{:.3f}".format(unitcell[3])))+ " deg\n"
    str9 = "be = "+ str(float("{:.3f}".format(unitcell[4])))+ " deg\n"
    str10 = "ga = "+ str(float("{:.3f}".format(unitcell[5])))+ " deg\n"
    with open(fname,"w") as f:
        L = [str1, str2, str3, str4, str5, str6, str7, str8, str9, str10]
        f.writelines(L)

def write_rank(fname, bravais, unitcell, population):
    if not fname.endswith(".txt"):
        fname += ".txt" 
    with open(fname,"a+") as f:
        # lattice_type centering unique_axis a b c al be ga population
        for i in range(3):
           f.write(bravais[i]+" ")
        for i in range(6):
           f.write(str(float("{:.2f}".format(unitcell[i])))+" ")
        f.write(str(population)+"\n")

# Enter CELL and SYMM in create-mtz
def write_createMtz(expName, unique_unitcell, spacegroup, fname):
    cmd="""#!/bin/sh
OUTFILE=`echo $1 | sed -e 's/\.hkl$/.mtz/'`

echo " Input: $1"
echo "Output: $OUTFILE"
sed -n '/End\ of\ reflections/q;p' $1 > create-mtz.temp.hkl

echo "Running 'f2mtz'..."
f2mtz HKLIN create-mtz.temp.hkl HKLOUT $OUTFILE > out.html << EOF
TITLE Reflections from CrystFEL
NAME PROJECT wibble CRYSTAL wibble DATASET wibble
CELL %.2f %.2f %.2f %.2f %.2f %.2f
SYMM %s
SKIP 3
LABOUT H K L IMEAN SIGIMEAN
CTYPE  H H H J     Q
FORMAT '(3(F4.0,1X),F10.2,10X,F10.2)'
EOF

if [ $? -ne 0 ]; then echo "Failed."; exit; fi

rm -f create-mtz.temp.hkl
echo "Done."
""" % (unique_unitcell[0], unique_unitcell[1], unique_unitcell[2], \
unique_unitcell[3], unique_unitcell[4], unique_unitcell[5], spacegroup)
    with open(fname,"w") as f:
        f.write(cmd)
    os.chmod(fname, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    print "Prepared: ", fname

#process = subprocess.Popen("find "+stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#out, err = process.communicate()
#out = out.split('\n')
#streamfile = out[0]

numIndex, numHits, indexHistogram, bravais = ei.getIndexHistogram(streamfile)
lattice = indexHistogram[:,9:15].copy() 

#spacegroup = "P1"
#for i in range(numIndex):
#    lattice[i,:] = nig(stringify(lattice[i,:]),spacegroup)

#plt.hist(lattice[:,0],100); plt.title("a"), plt.show()

## Look at distribution over all centering types
myUC=["a","b","c","al","be","ga"]
cen_types = list(set(bravais[1]));
cen_ind = collections.defaultdict(list) # Dictionary of 'centering':index
for i in cen_types:
    cen_ind[i] = []
for i, val in enumerate(bravais[1]):
    cen_ind[val].append(i)

for cen in cen_types:
    print "***************************"
    print "*** Trying centering: ", cen
    print "***************************"
    lattice_cen = lattice[cen_ind[cen]]
    lattice_type = [bravais[0][k] for k in cen_ind[cen]]
    unique_axis = [bravais[2][k] for k in cen_ind[cen]]

    # Create uc: dict of key" [a,b,c,al,be,ga] and value: array of values corresponding to this centering type
    uc=collections.defaultdict(list)
    for i in range(len(myUC)):
        print "max: ", np.max(lattice_cen[:,i])
        hist,bin_edges=np.histogram(lattice_cen[:,i],bins=np.arange(0,np.max(lattice_cen[:,i]),binSize))
        uc[myUC[i]].append(hist)

    ## 1D histogram peak finding
    foundPeaks = True
    possible = []
    hannPts = 15
    win=signal.hann(hannPts) # Hann window
    for j in range(3):
        conv=signal.convolve(uc[myUC[j]][0],win,mode='same')/sum(win)
        peakind = signal.find_peaks(uc[myUC[j]][0], prominence=10, distance=15)
        plt.subplot(211); plt.plot(uc[myUC[j]][0],'b-'); plt.title(myUC[j])
        plt.subplot(212); plt.plot(conv,'g-'); plt.plot(peakind[0],conv[peakind[0]],'ro'); plt.title("blurred and peaks found (red)"); plt.show()
        if len(peakind[0]) == 0:
            print "Unable to find peak in unitcell "+myUC[j]+" for "+cen+". Try again after indexing more data."
            foundPeaks = False
        possible.append(np.array([p*binSize for p in peakind[0]]))
    # possible: list of possible candidate lengths for a, b, and c
    print "possible uc values: ", possible
    if not foundPeaks:
        continue

    # Select lattices that appear at the possible peaks 
    ind = []
    for i, latt in enumerate(lattice_cen):
        if np.min(np.abs(possible[0]-latt[0])) < atol and \
           np.min(np.abs(possible[1]-latt[1])) < atol and \
           np.min(np.abs(possible[2]-latt[2])) < atol:
            ind.append(i)  

    sublatt = lattice_cen[ind,:]
    sublattice_type = [lattice_type[x] for x in ind]
    subunique_axis = [unique_axis[x] for x in ind]

    # K-means clustering
    topK = np.max([len(x) for x in possible])
    kmeans = KMeans(n_clusters=topK, random_state=0).fit(sublatt)
    print "kmeans: ", kmeans.cluster_centers_

    # Plot softmax based on cluster size
    score = np.zeros(topK,)
    for i in range(topK):
        score[i] = len(np.where(kmeans.labels_==i)[0])   

    score_norm = score / np.max(score) # normalize
    soft=np.exp(score_norm)/np.sum(np.exp(score_norm)) # softmax
    candidates = np.where(soft>=1./topK)[0] # choose ones with higher than 1/topK probability
    idx = np.argsort(soft[candidates])[::-1] # highest -> lowest prob
    plt.plot(soft, 'bx'); plt.plot(candidates,soft[candidates],'ro'); plt.title("softmax unitcell probability"); plt.show()
    for i in range(topK):
        print i, ": unitcell=", kmeans.cluster_centers_[i], ", Prob=", soft[i]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sublatt[:,0], sublatt[:,1], sublatt[:,2], c='k', marker='.', s=3)
    ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], c='r', marker='o', s=1000*soft, alpha=0.2)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    plt.title('Distribution of unitcell axis lengths: '+cen)
    plt.show()

    # Write unitcell candidates for each centering
    unique_unitcell = []
    with open(expName+"_"+cen+"_unitcell_candidates.txt","w") as f:
        f.write("# a b c al be ga probability population *:candidate\n")
        for i in range(topK):
            for x in kmeans.cluster_centers_[i]: # unitcell_candidates.txt file
                _u = '%.2f' % x
                f.write(_u+" ")
            _u = '%.2f' % soft[i]
            f.write(_u+" ")
            _u = '%d' % score[i]
            f.write(_u)
            if i in candidates: 
                f.write("*\n")
            else:
                f.write("\n")
            unique_unitcell.append(kmeans.cluster_centers_[i])
    print "unique uc: ", unique_unitcell

    probable_centers = np.zeros((len(idx),6))
    for i, val in enumerate(idx):
        c = candidates[val]
        print i, ": unitcell=", kmeans.cluster_centers_[c], ", Prob=", soft[c]
        probable_centers[i,:] = kmeans.cluster_centers_[c]

    # remove similar unitcells
    rm_list = []
    for i in range(len(idx)):
        for j in range(len(idx)):
            remove = True
            if i < j:
                print "comparing: ",i,j, probable_centers[i,:], probable_centers[j,:]
                for k in np.arange(3):
                    print "val: ", probable_centers[i,k],probable_centers[j,k]
                    if not np.isclose(probable_centers[i,k],probable_centers[j,k],atol=atol):
                        print "Different unitcell axis. Move on to next one"
                        remove = False
                for k in np.arange(3,6):
                    print "val: ", probable_centers[i,k],probable_centers[j,k]
                    if not np.isclose(probable_centers[i,k],probable_centers[j,k],atol=atol):# and \
#                       not np.isclose(probable_centers[i,k],np.abs(180-probable_centers[j,k]),atol=atol):
                        print "*Different unitcell angle. Move on to next one"
                        remove = False
                if remove:
                    print "Similar unitcell. Remove this one"
                    rm_list.append(j)

    # Write unitcell_candidates and .cell files and create-mtz files
    num_candidates = 0
    createMtzList = []
    for i, val in enumerate(idx):
        c = candidates[val]
        if not i in rm_list:
            # figure out what Bravais lattice to use
            _ind = np.where(kmeans.labels_==c)[0]
            _latt_type = [sublattice_type[x] for x in _ind]
            _uniq_axis = [subunique_axis[x] for x in _ind]
            _set_latt = list(set(_latt_type))
            _set_axis = list(set(_uniq_axis))
            _latt_count = [_latt_type.count(x) for x in _set_latt]
            _axis_count = [_uniq_axis.count(x) for x in _set_axis]
            probable_latt_type = _set_latt[np.argmax(_latt_count)]
            probable_uniq_axis = _set_axis[np.argmax(_axis_count)]
            _cname = expName+"-"+probable_latt_type[:4]+cen+str(num_candidates)
            write_cell(_cname, (probable_latt_type,cen,probable_uniq_axis), probable_centers[i,:])
 # write .cell file
            write_rank(rankName, (probable_latt_type,cen,probable_uniq_axis), probable_centers[i,:], np.max(_latt_count))
            sg = probable_latt_type + "_" + cen
            _fname = "create-mtz"+"-"+_cname
            write_createMtz(expName, probable_centers[i,:], spacegroup_guess[sg], _fname) # write create-mtz file
            createMtzList.append(_fname)
            num_candidates += 1

# Rank the unitcells of possible Bravais lattices
population = []
with open(expName+"_unitcell_rank.txt","w") as g:
    for cen in cen_types:
        _fname = expName+"_"+cen+"_unitcell_candidates.txt"
        if os.path.exists(_fname):
            with open(_fname,"r") as f:
                lines = f.readlines()
            for i,val in enumerate(lines):
                if val.endswith("*\n"): # candidate
                    g.write(cen+str(i)+": "+val.split("*")[0]+"\n")
                    population.append(int(val.split()[-1].split("*")[0]))

# rank higher symmetries as more favorable
population = []
recommend = []
with open(expName+"_rank.txt") as f:
    lines = f.readlines()
    for i,val in enumerate(lines):
        recommend.append(val.split()[0])
        population.append(int(val.split()[-1]))

population = np.array(population)
population_norm = population / np.max(population) # normalize
soft_pop=np.exp(population_norm)/np.sum(np.exp(population_norm)) # softmax
idx_pop = np.argsort(soft_pop)[::-1] # highest -> lowest prob
print "Most likely unitcell is: %s %.2f%%" %(recommend[idx_pop[0]], soft_pop[idx]*100)

exit()
##############################################################
## At this point, launch indexamajig with all .cell files
## Merge: process_hkl -i cxih0115_0021_orthC0.stream -o cxih0115_0021_orthC0.hkl -y 1
##############################################################
## Wait for cxih0115_0021_orthC0.stream
## Wait for cxih0115_0021_orthC0.hkl

# ./create-mtz-cxih0115-orthC0 cxih0115_0021_orthC0.hkl --> cxih0115_0021_orthC0.mtz
for i in range(num_candidates):


    #_fname = "create-mtz"+"-"+expName+"-"+






    p = subprocess.Popen([_fname, myHKL], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate() #now wait plus that you can send commands to process
    print "Done create-mtz", i

# Run Pointless on mtz, find most likely spacegroups + unitcell param + twinning
# pointless cxic0415_0095_triclinic0.mtz > pointless.out
for i in range(num_candidates):
    myMTZ = myHKL.split(".")[0] + str(i) + ".mtz"
    myOut = myHKL.split(".")[0] + str(i) + ".out"
    f = open(myOut, "w")
    p = subprocess.Popen(["pointless", myMTZ], stderr=subprocess.PIPE, stdout=f)
    out, err = p.communicate()

# Parse pointless output: cxic0415_0095_triclinic00.out
search0 = "The L-test suggests"
search1 = "   Spacegroup         TotProb"
search2 = "The crystal system chosen for output is"
def get_sg_prob(lines, lineNum):
    endLine = "-"*5
    sg = []
    while 1:
        if "(" in lines[lineNum]:
            spacegroup = lines[lineNum].split("(")[0].strip()
            totprob = lines[lineNum].split(")")[1].split("[")[0].strip().split(" ")[0]
            sysabsprob = lines[lineNum].split(")")[1].split("[")[0].strip().split(" ")[-1]
            sg.append((spacegroup,totprob,sysabsprob))
        elif endLine in lines[lineNum]:
            break
        lineNum += 1
    return sg

spacegroups = None
twinned = True
proposed_uc = None
for i in range(num_candidates):
    myOut = myHKL.split(".")[0] + str(i) + ".out"
    with open(myOut, "r") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if line.startswith(search0):
                if "not" in line:
                    twinned = False
            elif line.startswith(search1):
                spacegroups = get_sg_prob(lines, i)
            elif line.startswith(search2):
                proposed_uc = lines[i+1].split(":")[-1].split(')')[0].split()
print spacegroups
print twinned
print proposed_uc                


exit()

def get_bravais_cluster(i):
    ind = np.where(kmeans.labels_==i)[0]
    lt = []
    ce = []
    ua = []
    for j in ind:
        lt.append(bravais[0][j])
        ce.append(bravais[1][j])
        ua.append(bravais[2][j])
    return zip(lt, ce, ua)

fcounter = 0
for cluster_num in idx:
    print "cluster_num: ", cluster_num
    # tabulate occurence of all Bravais lattices
    myBr = get_bravais_cluster(cluster_num)
    set_br = set(myBr)
    br = collections.defaultdict(list)
    for i in set_br:
        br[i] = 0
    for i in myBr:
        br[i] += 1
    # merge all triclinic (can only be Primitive)
    # hexagonal (can only be Primitive unique_axis=c)
    num_tri = 0
    del_list = []
    for val in br:
        latt_type = val[0]
        if latt_type == 'triclinic':
            num_tri += br[val]
            del_list.append(val)
    for val in del_list:
        del br[val]
    if num_tri: br[('triclinic','P','*')] = num_tri
    score_br = br.values() / np.max(br.values()) # normalize
    soft_br=np.exp(score_br)/np.sum(np.exp(score_br)) # softmax
    candidates_br = np.where(soft_br>=1./len(br.values()))[0] # choose ones with higher than 1/br probability
    plt.plot(soft_br, 'ro'); plt.title("softmax bravais probability"); plt.show()
    idx_br = np.argsort(soft_br[candidates_br])[::-1] # highest -> lowest prob
    for i, val in enumerate(idx_br):
        k = candidates_br[val]
        print i, ": Bravais=", br.keys()[k], ", Prob=", soft_br[k]
        cand = candidates[cluster_num]
        print "unitcell=", kmeans.cluster_centers_[cand], ", Prob=", soft[cand]
        # check
        a = kmeans.cluster_centers_[cand][0]
        b = kmeans.cluster_centers_[cand][1]
        c = kmeans.cluster_centers_[cand][2]
        al = kmeans.cluster_centers_[cand][3]
        be = kmeans.cluster_centers_[cand][4]
        ga = kmeans.cluster_centers_[cand][5]
        #if fcounter == 0:
        #    write_cell("triclinic", br.keys()[k], kmeans.cluster_centers_[cand])
        # triclinic, a<b<c
        if br.keys()[k][0] == 'triclinic':
            if a < b < c:
                print "consistent. create .cell file"
                write_cell(str(fcounter), br.keys()[k], kmeans.cluster_centers_[cand])
                fcounter += 1
            else:
                print "inconsistent. skip this Bravais"
        # monoclinic, a!=c, al=ga=90, be>=90
        elif br.keys()[k][0] == 'monoclinic':
            if a != c and \
               np.isclose(al,90,atol=atol) and \
               np.isclose(ga,90,atol=atol) and \
               be >= (90-atol):
                print "consistent. create .cell file"
                write_cell(str(fcounter), br.keys()[k], kmeans.cluster_centers_[cand])
                fcounter += 1
            else:
                print "inconsistent. skip this Bravais"                
        # orthorhombic, a!=b!=c, al=be=ga=90
        elif br.keys()[k][0] == 'orthorhombic':
            if a!=b and a!=c and b!=c and \
               np.isclose(al,90,atol=atol) and \
               np.isclose(be,90,atol=atol) and \
               np.isclose(ga,90,atol=atol):
                print "consistent. create .cell file"
                write_cell(str(fcounter), br.keys()[k], kmeans.cluster_centers_[cand])
                fcounter += 1
            else:
                print "inconsistent. skip this Bravais" 
        # tetragonal, a=b!=c, al=be=ga=90
        elif br.keys()[k][0] == 'tetragonal':
            if np.isclose(a,b,atol=atol) and \
               not np.isclose(a,c,atol=atol) and \
               np.isclose(al,90,atol=atol) and \
               np.isclose(al,90,atol=atol) and \
               np.isclose(be,90,atol=atol) and \
               np.isclose(ga,90,atol=atol):
                print "consistent. create .cell file"
                write_cell(str(fcounter), br.keys()[k], kmeans.cluster_centers_[cand])
                fcounter += 1
            else:
                print "inconsistent. skip this Bravais"
        # rhombohedral, a=b=c, a=b=ga!=90
        elif br.keys()[k][0] == 'rhombohedral':
            if np.isclose(a,b,atol=atol) and \
               np.isclose(a,c,atol=atol) and \
               np.isclose(b,c,atol=atol) and \
               not np.isclose(al,90,atol=atol) and \
               not np.isclose(be,90,atol=atol) and \
               not np.isclose(ga,90,atol=atol):
                print "consistent. create .cell file"
                write_cell(str(fcounter), br.keys()[k], kmeans.cluster_centers_[cand])
                fcounter += 1
            else:
                print "inconsistent. skip this Bravais"
        # hexagonal, a=b, al=be=90, ga=120
        elif br.keys()[k][0] == 'hexagonal':
            if np.isclose(a,b,atol=atol) and \
               np.isclose(al,90,atol=atol) and \
               np.isclose(be,90,atol=atol) and \
               np.isclose(ga,120,atol=atol):
                print "consistent. create .cell file"
                write_cell(str(fcounter), br.keys()[k], kmeans.cluster_centers_[cand])
                fcounter += 1
            else:
                print "inconsistent. skip this Bravais"
        # cubic, a=b=c, al=be=ga=90
        elif br.keys()[k][0] == 'cubic':
            if np.isclose(a,b,atol=atol) and \
               np.isclose(a,c,atol=atol) and \
               np.isclose(b,c,atol=atol) and \
               np.isclose(al,90,atol=atol) and \
               np.isclose(be,90,atol=atol) and \
               np.isclose(ga,90,atol=atol):
                print "consistent. create .cell file"
                write_cell(str(fcounter), br.keys()[k], kmeans.cluster_centers_[cand])
                fcounter += 1
            else:
                print "inconsistent. skip this Bravais"

# Peak finder
# Index with Bravais= ('triclinic', 'P', '*')
# Merge: process_hkl -i cxic0415_0095.stream -o cxic0415_0095.hkl -y 1
# Get most common cell param from stream
# Enter CELL in create-mtz and ./create-mtz cxic0415_0095.hkl
# Run Pointless on mtz, find most likely spacegroups and unitcell param
# Convert spacegroup to Bravais and create .cell
# Index with .cell

#0 : unitcell= [  51.56454568  100.1889997    53.89832284   90.03537849  112.64066315
#   90.00985842] , Prob= 0.615862593229
#1 : unitcell= [  51.6550899    54.04134754  100.16388925   89.57792645   89.53863601
#   67.30936805] , Prob= 0.233219713201

#0 : Bravais= ('triclinic', 'P', '*') , Prob= 0.232295061052
#1 : Bravais= ('monoclinic', 'P', 'b') , Prob= 0.179211499672
#2 : Bravais= ('triclinic', 'P', '?') , Prob= 0.12798963258

## merge using process_hkl
#process_hkl -i cxic0415_0095_0.stream -o cxic0415_0095_0.hkl -y 2_uab
#process_hkl -i cxic0415_0095_1.stream -o cxic0415_0095_1.hkl -y 1

## Run Pointless

#Change CELL and SYMM in create-mtz:
#CELL 78 149.9 200.8  90  90 90
#SYMM P1

#./create-mtz mfxlt3017_ponan-mbs-day2-120.hkl
#pointless mfxlt3017_ponan-mbs-day2-120.mtz > pointless.out


## Need to know spacegroup to call niggli



