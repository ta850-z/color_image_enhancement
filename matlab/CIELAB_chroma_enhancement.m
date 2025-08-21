% CIELAB Chroma Enhancement
clear all; 
close all;

%Read image file
RGB=imread('cat','jpg');

%Lookup table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load cmax;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RGB = double(RGB);
figure; imshow(uint8(RGB),'Border','tight');

%% 
[s1,s2,s3]=size(RGB);
RGB = RGB./255;
Out = RGB;
m = size(RGB,1);
n = size(RGB,2);
F_out = reshape(Out(:),m*n,3);

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F_out = inv_gamma_srgb(F_out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F_out_hold=reshape(F_out,m,n,3);
xyz_out=rgb2xyz_2(F_out);
lsasbs_out=xyz2lsasbs_2(xyz_out);
lsasbs_out=reshape(lsasbs_out,m,n,3);
ls_out=lsasbs_out(:,:,1);
ls_out_hold=ls_out;

%Chroma enhancement
k1=5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
las_out=k1*lsasbs_out(:,:,2);
lbs_out=k1*lsasbs_out(:,:,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Lightness enhancement
k2=1.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ls_out=100*(ls_out./100).^(1/k2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cs_out=sqrt(las_out.^2+lbs_out.^2);
h_out=lbs_out./las_out;
h_out(cs_out<0.1)=0;
hangle_out=atan2d(lbs_out,las_out)+360*(lbs_out<0);

%%
fY=(ls_out+16)/116;
fX=sign(las_out).*cs_out./(500*sqrt(1+h_out.^2))+fY;
fZ=-sign(lbs_out).*cs_out./(200*sqrt(1+(1./h_out.^2)))+fY;

X(fX>0.20689)=0.9505*fX(fX>0.20689).^3;
X(fX<=0.20689)=(fX(fX<=0.20689)-16/116)*(0.9505/7.78);
 
Z(fZ>0.20689)=1.089*fZ(fZ>0.20689).^3;
Z(fZ<=0.20689)=(fZ(fZ<=0.20689)-16/116)*(1.089/7.78);

Y(fY>0.20689)=1*fY(fY>0.20689).^3;
Y(fY<=0.20689)=(fY(fY<=0.20689)-16/116)*(1/7.78);

XYZ=cat(3,X,Y,Z);
XYZ_out = reshape(XYZ(:),m*n,3);
RGB_out=xyz2rgb_2(XYZ_out);
RGB_out_hold=reshape(RGB_out(:),m,n,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RGB_out=gamma_srgb(RGB_out);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RGB_out=reshape(RGB_out(:),m,n,3);
figure; imshow(RGB_out,'Border','tight');

indexY=round(100*Y);
indexh=round(hangle_out);

%%
X=reshape(X,m,n);
Y=reshape(Y,m,n);
Z=reshape(Z,m,n);
indexY=reshape(indexY,m,n);

Xn=0.9505;
Zn=1.089;
cs_out_max=zeros(s1,s2);
for i=1:s1
    for j=1:s2   
        if gamut_descript(xyz2rgb_2([Xn*fX(i,j)^(3),1*fY(i,j)^(3),Zn*fZ(i,j)^(3)])) == 1
            cs_out_max(i,j)=cs_out(i,j);
        else   
            if indexY(i,j)==0
                if indexh(i,j)==0 || indexh(i,j)==360
                    cs_out_max(i,j)=min([cstar_gmax(indexY(i,j)+1,1),cstar_gmax(indexY(i,j)+1,360)]);
                else
                    cs_out_max(i,j)=min([cstar_gmax(indexY(i,j)+1,indexh(i,j)),cstar_gmax(indexY(i,j)+1,indexh(i,j)+1)]);
                end
            elseif indexY(i,j)==100
                if indexh(i,j)==0 || indexh(i,j)==360
                    cs_out_max(i,j)=min([cstar_gmax(indexY(i,j),1),cstar_gmax(indexY(i,j),360)]);
                else
                    cs_out_max(i,j)=min([cstar_gmax(indexY(i,j),indexh(i,j)),cstar_gmax(indexY(i,j),indexh(i,j)+1)]);
                end
            elseif indexh(i,j)==0 || indexh(i,j)==360
                cs_out_max(i,j)=min([cstar_gmax(indexY(i,j),1),cstar_gmax(indexY(i,j)+1,1),cstar_gmax(indexY(i,j),360),cstar_gmax(indexY(i,j)+1,360)]);
            else
                cs_out_max(i,j)=min([cstar_gmax(indexY(i,j),indexh(i,j)),cstar_gmax(indexY(i,j)+1,indexh(i,j)),cstar_gmax(indexY(i,j),indexh(i,j)+1),cstar_gmax(indexY(i,j)+1,indexh(i,j)+1)]);
            end
        end
    end
end
fX=sign(las_out).*cs_out_max./(500*sqrt(1+h_out.^2))+fY;
fZ=-sign(lbs_out).*cs_out_max./(200*sqrt(1+(1./h_out.^2)))+fY;

X(fX>0.20689)=0.9505*fX(fX>0.20689).^3;
X(fX<=0.20689)=(fX(fX<=0.20689)-16/116)*(0.9505/7.78);
 
Z(fZ>0.20689)=1.089*fZ(fZ>0.20689).^3;
Z(fZ<=0.20689)=(fZ(fZ<=0.20689)-16/116)*(1.089/7.78);

Y(fY>0.20689)=1*fY(fY>0.20689).^3;
Y(fY<=0.20689)=(fY(fY<=0.20689)-16/116)*(1/7.78);

%%
XYZ=cat(3,X,Y,Z);
XYZ_out = reshape(XYZ(:),m*n,3);
RGB_correct=xyz2rgb_2(XYZ_out);
RGB_correct_hold=reshape(RGB_correct(:),m,n,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RGB_correct=gamma_srgb(RGB_correct);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RGB_correct=reshape(RGB_correct(:),m,n,3);
figure; imshow(RGB_correct,'Border','tight');
