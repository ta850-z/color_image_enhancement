# -*- coding: utf-8 -*-
# CIELAB Chroma Enhancement

import numpy as np
import cv2
ep=1e-06
def inv_gamma_srgb(rgb):
    rgb2=np.zeros((rgb.shape[0],rgb.shape[1]),dtype=np.float64)
    rgb2[rgb[:,0]<=0.03928,0] = rgb[rgb[:,0]<=0.03928,0]/12.92
    rgb2[rgb[:,1]<=0.03928,1] = rgb[rgb[:,1]<=0.03928,1]/12.92
    rgb2[rgb[:,2]<=0.03928,2] = rgb[rgb[:,2]<=0.03928,2]/12.92

    rgb2[rgb[:,0]>0.03928,0] = ((rgb[rgb[:,0]>0.03928,0]+0.055)/1.055)**2.4
    rgb2[rgb[:,1]>0.03928,1] = ((rgb[rgb[:,1]>0.03928,1]+0.055)/1.055)**2.4
    rgb2[rgb[:,2]>0.03928,2] = ((rgb[rgb[:,2]>0.03928,2]+0.055)/1.055)**2.4
 
    return rgb2

def gamma_srgb(rgb):
    rgb2=np.zeros((rgb.shape[0],rgb.shape[1]),dtype=np.float64)
    rgb2[rgb[:,0]<=0.00304,0] = 12.92*rgb[rgb[:,0]<=0.00304,0]
    rgb2[rgb[:,1]<=0.00304,1] = 12.92*rgb[rgb[:,1]<=0.00304,1]
    rgb2[rgb[:,2]<=0.00304,2] = 12.92*rgb[rgb[:,2]<=0.00304,2]

    rgb2[rgb[:,0]>0.00304,0] = 1.055*rgb[rgb[:,0]>0.00304,0]**(1/2.4)-0.055
    rgb2[rgb[:,1]>0.00304,1] = 1.055*rgb[rgb[:,1]>0.00304,1]**(1/2.4)-0.055
    rgb2[rgb[:,2]>0.00304,2] = 1.055*rgb[rgb[:,2]>0.00304,2]**(1/2.4)-0.055

    return rgb2

def rgb2xyz_2(rgb):
    A=np.array([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])
    return rgb@A.T

def xyz2rgb_2(xyz):
    B=np.array([[3.2410,-1.5374,-0.4986],[-0.9692,1.8760,0.0416],[0.0556,-0.2040,1.0570]])
    return xyz@B.T

def lsasbs_f(x,xn):
    f=np.zeros((x.shape[0]),dtype=np.float64)
    x=x/xn
    a=0.008856
    f[x>a]=x[x>a]**(1/3)
    f[x<=a]=7.787*x[x<=a]+16/116
    return f

def lsasbs_invf(f):
    x=np.zeros((f.shape[0]),dtype=np.float64)
    b=0.20689
    x[f>b]=f[f>b]**(3)
    x[f<=b]=(f[f<=b]-16/116)/7.787
    return x

def xyz2lsasbs_2(xyz):
    lsasbs=np.zeros((xyz.shape[0],xyz.shape[1]),dtype=np.float64)
    xn=0.9505;yn=1.000;zn=1.089
    fx=lsasbs_f(xyz[:,0],xn)
    fy=lsasbs_f(xyz[:,1],yn)
    fz=lsasbs_f(xyz[:,2],zn)
    lsasbs[:,0]=116*fy-16
    lsasbs[:,1]=500*(fx-fy)
    lsasbs[:,2]=200*(fy-fz)
    return lsasbs

def lsasbs2xyz_2(lsasbs):
    xyz_out=np.zeros((lsasbs.shape[0],lsasbs.shape[1]),dtype=np.float64)
    xn=0.9505;yn=1.000;zn=1.089
    fy=(lsasbs[:,0]+16)/116
    fx=lsasbs[:,1]/500+fy
    fz=-lsasbs[:,2]/200+fy
    xyz_out[:,0]=xn*lsasbs_invf(fx)
    xyz_out[:,1]=yn*lsasbs_invf(fy)
    xyz_out[:,2]=zn*lsasbs_invf(fz)
    return xyz_out

def gamut_descript(data):
    if data[0] <-0.001 or data[0] > 1.001:
        r=0
    else:
        r=1

    if data[1] <-0.001 or data[1] > 1.001:
        g=0
    else:
        g=1

    if data[2] <-0.001 or data[2] > 1.001:
        b=0
    else:
        b=1
        
    return r*g*b
    
file_inp='cat.jpg'
file_out='cat_out.jpg'
file_out_correct='cat_out_correct.jpg'

rgb_in=cv2.imread(file_inp,1)
cstar_gmax = np.loadtxt(fname="cmax.csv",dtype="float",delimiter=",")
rgb=cv2.cvtColor(rgb_in,cv2.COLOR_BGR2RGB)
rgb=rgb/255
cx, cy, cc=rgb.shape[:3]
rgb=rgb.reshape((cx*cy,3),order="F")
rgb=inv_gamma_srgb(rgb)
xyz=rgb2xyz_2(rgb)
lsasbs=xyz2lsasbs_2(xyz)
lsasbs=lsasbs.reshape((cx,cy,cc),order="F")

#Chroma enhancement
k1=5
########################################
las_out=k1*lsasbs[:,:,1]
lbs_out=k1*lsasbs[:,:,2]
########################################

#Lightness enhancement
k2=1.5
########################################
ls_out=100*(lsasbs[:,:,0]/100)**(1/k2)
########################################

cs_out=np.sqrt(las_out**2+lbs_out**2)
las_out[np.abs(las_out)<ep]=ep
h_out=lbs_out/las_out
h_out[cs_out<0.1]=ep;
hangle_out=np.arctan2(lbs_out,las_out)*180/np.pi +360*(lbs_out<0)

fY=(ls_out+16)/116;
fX=np.sign(las_out)*cs_out/(500*np.sqrt(1+h_out**2))+fY;
fZ=-np.sign(lbs_out)*cs_out/(200*np.sqrt(1+(1./h_out**2)))+fY

X=np.zeros((fX.shape[0],fX.shape[1]),dtype=np.float64)
X[fX>0.20689]=0.9505*fX[fX>0.20689]**3
X[fX<=0.20689]=(fX[fX<=0.20689]-16/116)*(0.9505/7.78)

Z=np.zeros((fZ.shape[0],fZ.shape[1]),dtype=np.float64)
Z[fZ>0.20689]=1.089*fZ[fZ>0.20689]**3
Z[fZ<=0.20689]=(fZ[fZ<=0.20689]-16/116)*(1.089/7.78)

Y=np.zeros((fY.shape[0],fY.shape[1]),dtype=np.float64)
Y[fY>0.20689]=1*fY[fY>0.20689]**3
Y[fY<=0.20689]=(fY[fY<=0.20689]-16/116)*(1/7.78)

indexY=np.round(100*Y).astype(int)-1
indexh=np.round(hangle_out).astype(int)-1

X=X.reshape((cx*cy,1),order="F")
Y=Y.reshape((cx*cy,1),order="F")
Z=Z.reshape((cx*cy,1),order="F")

xyz_out=np.hstack([X,Y,Z])

rgb_out=xyz2rgb_2(xyz_out)
rgb_out=gamma_srgb(rgb_out)
rgb_out=255*rgb_out
rgb_out[rgb_out>255]=255
rgb_out[rgb_out<0]=0
rgb_out=rgb_out.astype(np.uint8)
rgb_out=rgb_out.reshape((cx,cy,cc),order="F")
rgb_out=cv2.cvtColor(rgb_out,cv2.COLOR_RGB2BGR)

X=X.reshape((cx,cy),order="F")
Y=Y.reshape((cx,cy),order="F")
Z=Z.reshape((cx,cy),order="F")

Xn=0.9505
Zn=1.089
cs_out_max=np.zeros((cs_out.shape[0],cs_out.shape[1]),dtype=np.float64)
for i in range(cx):
    for j in range(cy):   
        if gamut_descript(xyz2rgb_2([Xn*fX[i,j]**3,1*fY[i,j]**3,Zn*fZ[i,j]**3])) == 1:
            cs_out_max[i,j]=cs_out[i,j]
        else:
            if indexY[i,j]==-1:
                if indexh[i,j]==-1 or indexh[i,j]==359:
                    cs_out_max[i,j]=np.min([cstar_gmax[indexY[i,j]+1,0],cstar_gmax[indexY[i,j]+1,359]])
                else:
                    cs_out_max[i,j]=np.min([cstar_gmax[indexY[i,j]+1,indexh[i,j]],cstar_gmax[indexY[i,j]+1,indexh[i,j]+1]])
            elif indexY[i,j]==99:
                if indexh[i,j]==-1 or indexh[i,j]==359:
                    cs_out_max[i,j]=np.min([cstar_gmax[indexY[i,j],0],cstar_gmax[indexY[i,j],359]])
                else:
                    cs_out_max[i,j]=np.min([cstar_gmax[indexY[i,j],indexh[i,j]],cstar_gmax[indexY[i,j],indexh[i,j]+1]])
            elif indexh[i,j]==-1 or indexh[i,j]==359:
                cs_out_max[i,j]=np.min([cstar_gmax[indexY[i,j],0],cstar_gmax[indexY[i,j]+1,0],cstar_gmax[indexY[i,j],359],cstar_gmax[indexY[i,j]+1,359]])
            else:
                cs_out_max[i,j]=np.min([cstar_gmax[indexY[i,j],indexh[i,j]],cstar_gmax[indexY[i,j]+1,indexh[i,j]],cstar_gmax[indexY[i,j],indexh[i,j]+1],cstar_gmax[indexY[i,j]+1,indexh[i,j]+1]])

fX=np.sign(las_out)*cs_out_max/(500*np.sqrt(1+h_out**2))+fY
fZ=-np.sign(lbs_out)*cs_out_max/(200*np.sqrt(1+(1/h_out**2)))+fY

X[fX>0.20689]=0.9505*fX[fX>0.20689]**3
X[fX<=0.20689]=(fX[fX<=0.20689]-16/116)*(0.9505/7.78)

Z[fZ>0.20689]=1.089*fZ[fZ>0.20689]**3
Z[fZ<=0.20689]=(fZ[fZ<=0.20689]-16/116)*(1.089/7.78)

Y[fY>0.20689]=1*fY[fY>0.20689]**3
Y[fY<=0.20689]=(fY[fY<=0.20689]-16/116)*(1/7.78)

X=X.reshape((cx*cy,1),order="F")
Y=Y.reshape((cx*cy,1),order="F")
Z=Z.reshape((cx*cy,1),order="F")

XYZ_out=np.hstack([X,Y,Z])
RGB_correct=xyz2rgb_2(XYZ_out)
RGB_correct=gamma_srgb(RGB_correct)
RGB_correct=255*RGB_correct
RGB_correct[RGB_correct>255]=255
RGB_correct[RGB_correct<0]=0
RGB_correct=RGB_correct.astype(np.uint8)
RGB_correct=RGB_correct.reshape((cx,cy,cc),order="F")
RGB_correct=cv2.cvtColor(RGB_correct,cv2.COLOR_RGB2BGR)


cv2.imwrite(file_out,rgb_out)
cv2.imwrite(file_out_correct,RGB_correct)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img',rgb_in)
cv2.namedWindow('out', cv2.WINDOW_NORMAL)
cv2.imshow('out',rgb_out)
cv2.namedWindow('out_correct', cv2.WINDOW_NORMAL)
cv2.imshow('out_correct',RGB_correct)

cv2.waitKey(0)
cv2.destroyAllWindows()