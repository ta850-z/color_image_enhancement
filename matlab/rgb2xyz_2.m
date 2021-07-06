%RGB to XYZ conversion
function xyz = rgb2xyz_2(rgb)

rgb_to_xyz=[0.4124 0.3576 0.1805 ; 0.2126 0.7152 0.0722 ; 0.0193 0.1192 0.9505];

xyz = rgb*rgb_to_xyz';
