%inverse gamma correction
function rgb2 = inv_gamma_srgb(rgb)

[m,n] = size(rgb);

srgb_dush = rgb;

flag_r = srgb_dush(:,1)<=0.03928;
flag_g = srgb_dush(:,2)<=0.03928;
flag_b = srgb_dush(:,3)<=0.03928;

rgb2 = zeros(m,n);

rgb2(flag_r,1) = srgb_dush(flag_r,1)/12.92;
rgb2(flag_g,2) = srgb_dush(flag_g,2)/12.92;
rgb2(flag_b,3) = srgb_dush(flag_b,3)/12.92;

rgb2(not(flag_r),1) = ((srgb_dush(not(flag_r),1)+0.055)/1.055).^2.4;
rgb2(not(flag_g),2) = ((srgb_dush(not(flag_g),2)+0.055)/1.055).^2.4;
rgb2(not(flag_b),3) = ((srgb_dush(not(flag_b),3)+0.055)/1.055).^2.4;