function t = gamut_descript( data )

%if data(1) <0 || data(1) > 1
if data(1) <-0.001 || data(1) > 1.001
    r=0;
else
    r=1;
end

%if data(2) <0 || data(2) > 1
if data(2) <-0.001 || data(2) > 1.001
    g=0;
else
    g=1;
end

%if data(3) <0 || data(3) > 1
if data(3) <-0.001 || data(3) > 1.001
   b=0;
else
   b=1;
end
    t=r*g*b;
end