%XYZ to LAB conversion
function lsasbs = xyz2lsasbs_2(xyz)

xn=0.9505;yn=1.000;zn=1.089;

fx = lsasbs_f(xyz(:,1),xn);
fy = lsasbs_f(xyz(:,2),yn);
fz = lsasbs_f(xyz(:,3),zn);

lsasbs(:,1) = 116*fy-16.;   %Ls
lsasbs(:,2) = 500*(fx-fy);  %as
lsasbs(:,3) = 200*(fy-fz);  %bs