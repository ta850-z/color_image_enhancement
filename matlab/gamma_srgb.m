%gamma correction
function rgb = gamma_srgb(rgb2)

[m,n] = size(rgb2);

flag_r = rgb2(:,1)<=0.00304;
flag_g = rgb2(:,2)<=0.00304;
flag_b = rgb2(:,3)<=0.00304;

srgb_dush = zeros(m,n);

srgb_dush(flag_r,1) = 12.92*rgb2(flag_r,1);
srgb_dush(flag_g,2) = 12.92*rgb2(flag_g,2);
srgb_dush(flag_b,3) = 12.92*rgb2(flag_b,3);

srgb_dush(not(flag_r),1) = 1.055*rgb2(not(flag_r),1).^(1/2.4)-0.055;
srgb_dush(not(flag_g),2) = 1.055*rgb2(not(flag_g),2).^(1/2.4)-0.055;
srgb_dush(not(flag_b),3) = 1.055*rgb2(not(flag_b),3).^(1/2.4)-0.055;

rgb = srgb_dush;