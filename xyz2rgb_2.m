%XYZ to RGB conversion
function rgb = xyz2rgb_2(xyz)

xyz_to_rgb = [ 3.2410 -1.5374 -0.4986 ; -0.9692 1.8760 0.0416 ; 0.0556 -0.2040 1.0570 ];

rgb = xyz*xyz_to_rgb';