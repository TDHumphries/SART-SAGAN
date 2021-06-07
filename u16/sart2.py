
# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
import argparse
from glob import glob
import astra
import numpy as np
import os
import pdb
import pylab
from matplotlib import pyplot as plt # for png conversion

def create_projector(geom, numbin, angles, dso, dod, fan_angle):
    if geom == 'parallel':
        proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
    elif geom == 'fanflat':
        dso *=10; dod *=10;                         #convert to mm for astra
        ft = np.tan( np.deg2rad(fan_angle / 2) )    #compute tan of 1/2 the fan angle
        det_width = 2 * (dso + dod) * ft / numbin  #width of one detector pixel, calculated based on fan angle

        proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)

    p = astra.create_projector('cuda',proj_geom,vol_geom);
    return p

def grad_TV(img,numpix):
    #pdb.set_trace()
    epsilon = 1e-6
    ind_m1 = np.arange(numpix)
    ind_m2 = [(i + 1) % numpix for i in ind_m1]
    ind_m0 = [(i - 1) % numpix for i in ind_m1]

    m2m1 = np.ix_(ind_m2,ind_m1)
    m1m2 = np.ix_(ind_m1,ind_m2)
    m0m1 = np.ix_(ind_m0,ind_m1)
    m1m0 = np.ix_(ind_m1,ind_m0)
    
    diff1 = ( img[m2m1] - img) ** 2
    diff2 = ( img[m1m2] - img) ** 2
    diffttl = np.sqrt( diff1 + diff2 + epsilon**2)
    TV = np.sum(diffttl)

    dTV = -1/diffttl * (img[m2m1]-2*img + img[m1m2]) + \
          1/diffttl[m0m1] * (img-img[m0m1]) + \
          1/diffttl[m1m0] * (img-img[m1m0])
    return TV, dTV

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sino', dest='infile', default='.', help='input file -- sinogram in .flt format')
parser.add_argument('--out', dest='outfile', default='.', help='output directory')
parser.add_argument('--numpix',dest='numpix',type=int,default=512,help='size of volume (n x n )')
parser.add_argument('--dx',dest='dx',type=float,default=1.,help='pixel size (cm)');
parser.add_argument('--numbin',dest='numbin',type=int,default=729,help='number of detector pixels')
parser.add_argument('--ntheta',dest='numtheta',type=int,default=900,help='number of angles')
parser.add_argument('--nsubs',dest='ns',type=int,default=1,help='number of subsets. must divide evenly into number of angles')
parser.add_argument('--range', dest='theta_range',type=float,nargs=2,default=[0, 180],help='starting and ending angles (deg)')
parser.add_argument('--geom', dest='geom',default='fanflat',help='geometry (parallel or fanflat)')
parser.add_argument('--dso',dest='dso',type=float,default=100,help='source-object distance (cm) (fanbeam only)')
parser.add_argument('--dod',dest='dod',type=float,default=100,help='detector-object distance (cm) (fanbeam only)')
parser.add_argument('--fan_angle',dest='fan_angle',default=35,type=float,help='fan angle (deg) (fanbeam only)')
parser.add_argument('--numits',dest='num_its',default=32,type=int,help='maximum number of iterations')
parser.add_argument('--beta',dest='beta',default=1.,type=float,help='relaxation parameter beta')
parser.add_argument('--x0',dest='x0_file',default='',help='initial image (default: zeros)')
parser.add_argument('--xtrue',dest='xtrue_file',default='',help='true image (if available)')
parser.add_argument('--sup_params',dest='sup_params', type=float,nargs=3,help='superiorization parameters gamma, N, alpha_init')

#get arguments from command line
args = parser.parse_args()
infile, outfile, x0file, xtruefile = args.infile, args.outfile, args.x0_file, args.xtrue_file
numpix, dx, numbin, numtheta, ns, numits, beta = args.numpix, args.dx, args.numbin, args.numtheta, args.ns, args.num_its, args.beta
theta_range, geom, dso, dod, fan_angle = args.theta_range, args.geom, args.dso, args.dod, args.fan_angle

if args.sup_params is None:
    use_sup = False
else:
     use_sup = True
     gamma = args.sup_params[0]
     N = np.uint8(args.sup_params[1])
     alpha = args.sup_params[2]
          

eps = np.finfo(float).eps

#read in sinogram
sino = np.fromfile(infile,dtype='f')
sino = sino.reshape(numtheta,numbin)

head, tail = os.path.split(infile)      #get name of file for output
head, tail = tail.split("_",1)    #extract numerical part of filename only. Assumes we have ######_sino.flt
outname = outfile + "/" + head + "_img.flt"

# create projection geometry
vol_geom = astra.create_vol_geom(numpix, numpix)

#generate array of angular positions
theta_range = np.deg2rad(theta_range) #convert to radians
angles = theta_range[0] + np.linspace(0,numtheta-1,numtheta,False)*(theta_range[1]-theta_range[0])/numtheta #

if x0file == '':        
    f = np.zeros((numpix,numpix))
else:
    f = np.fromfile(x0file,dtype='f')
    f = f.reshape(numpix,numpix)

if xtruefile == '':
    calc_error = False
    xtrue = np.zeros((numpix,numpix))
else:
    xtrue = np.fromfile(xtruefile,dtype='f')
    xtrue = xtrue.reshape(numpix,numpix)
    calc_error = True
    
#create projectors and normalization terms (corresponding to diagonal matrices M and D) for each subset of projection data
P, Dinv, D_id, Minv, M_id = [None]*ns,[None]*ns,[None]*ns,[None]*ns,[None]*ns
for j in range(ns):
    ind1 = range(j,numtheta,ns);
    p = create_projector(geom,numbin,angles[ind1],dso,dod,fan_angle)
    
    D_id[j], Dinv[j] = astra.create_backprojection(np.ones((numtheta//ns,numbin)),p)
    M_id[j], Minv[j] = astra.create_sino(np.ones((numpix,numpix)),p)
    #avoid division by zero, also scale M to pixel size
    Dinv[j] = np.maximum(Dinv[j],eps)
    Minv[j] = np.maximum(Minv[j]*dx,eps)
    P[j] = p

for k in range(numits):
    #Superiorization loop
    if use_sup:
        #pdb.set_trace()
        g,dg = grad_TV(f,numpix)
        g_old = g;
        dg = -dg / (np.linalg.norm(dg,'fro') + eps)
        for j in range(N):
            while True:
                f_tmp = f + alpha * dg
                g_new = grad_TV(f_tmp,numpix)[0]
                alpha = alpha*gamma
                if g_new <= g_old:
                    f = f_tmp
                    break

            dg = grad_TV(f,numpix)[1]
            dg = -dg / (np.linalg.norm(dg,'fro') + eps)

    #SART loop
    for j in range(ns):
        ind1 = range(j,numtheta,ns);
        p = P[j]
        fp_id, fp = astra.create_sino(f,p)      #forward projection step
        diffs = (sino[ind1,:] - fp*dx)/Minv[j]                  #should do elementwise division
        bp_id, bp = astra.create_backprojection(diffs,p)
        ind2 = np.abs(bp) > 1e3
        bp[ind2] = 0             #get rid of spurious large values
        f = f + beta * bp/Dinv[j];                   #update f
        astra.data2d.delete(fp_id)
        astra.data2d.delete(bp_id)
        
    f = np.maximum(f,eps);                  #set any negative values to small positive value
    
    #compute residual and error (if xtrue is provided)
    fp = np.zeros((numtheta,numbin))
    for j in range(ns):
        ind = range(j,numtheta,ns)
        p = P[j]
        fp_tempid, fp_temp = astra.create_sino(f,p)
        fp[ind,:] = fp_temp*dx
        astra.data2d.delete(fp_tempid)

    res = np.linalg.norm(fp-sino,'fro') / np.linalg.norm(sino,'fro')
    #pdb.set_trace()
    if calc_error:
        err = np.linalg.norm(f-xtrue,'fro')/np.linalg.norm(xtrue,'fro')
        print('Iteration #{0:d}: Residual = {1:1.4f}\tError = {2:1.4f}\n'.format(k+1,res,err))
    else:
        print('Iteration #{0:d}: Residual = {1:1.4f}\n'.format(k+1,res))
   
#save image
f = np.float32(f)
f.tofile(outname)

#**********save image as png**********
max_pixel = np.amax(f)
img = (f/max_pixel) * 255
img = np.round(img)

plt.figure(num=None, figsize=(30, 40), facecolor='w', edgecolor='k')
plt.style.use('grayscale')
plt.imshow(img, interpolation = 'nearest')
png_outname = (outname + '.png')
plt.savefig(png_outname)
#**************************************


for j in range(ns):
    astra.data2d.delete(D_id[j])
    astra.data2d.delete(M_id[j])
    astra.projector.delete(P[j])