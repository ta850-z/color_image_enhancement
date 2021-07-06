%function f
function f = lsasbs_f(x,xn)

m = size(x,1);

x2 = x/xn;

a = 0.008856*ones(m,1);

s = x2>a;
t = x2<=a;

f(s,1) = x2(s,1).^(1./3);
f(t,1) = 7.787*x2(t,1)+16./116;