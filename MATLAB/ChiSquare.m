function [chi2stat,p] = ChiSquare(f1,Ftotal,m1,Mtotal)
%chi square  test

p0 = (f1+m1) / (Ftotal+Mtotal);

%Expected count

f10 = Ftotal * p0;
m10 = Mtotal * p0;

observed = [f1 Ftotal-f1 m1 Mtotal-m1];
expected = [f10 Ftotal-f10 m10 Mtotal-m10];

chi2stat = sum((observed-expected).^2 ./ expected);

p = 1 - chi2cdf(chi2stat,1);



end