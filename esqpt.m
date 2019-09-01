%%%%%%%%%%%%% CALL METHOD %%%%%%%%%%%%%%

clc; clear all;
% plot energy spectra

N = 100;
xi_vec = linspace(0.0,1.0,500);
[E,V] = Diagonalize(xi_vec, N);

% Eigenenergies
figure;
plot(xi_vec, E(1:1:end,:)-E(1,:))


%%
function [EigenE, EigenV] = Diagonalize(xi_vec,N)
% input：vector of $\xi$, particle number $N$
% output: corresopondigng eigenenrgies and eigenvectors.
%  for M =0 only.
hdim = N/2 + 1; % dimension of hilbert space.
l_xi = length(xi_vec);
EigenE = zeros(hdim, l_xi); % 
EigenV = zeros(hdim, l_xi, hdim); % 


n1 = (0:N/2);
% h_single = (M+2k)
h_single = 2.0*diag(n1); 

temp = 2.0*sqrt((N-2.0*n1+1.0).*(N-2.0*n1+2.0)).*n1;
temp_up = temp(2:end);
temp_0 = (2.0*(N-2*n1)-1).*(2.0*n1);
h_int = diag(temp_0,0) + diag(temp_up,1) + diag(temp_up,-1);

 for id_xi=1:l_xi
   xi = xi_vec(id_xi);
   h = ((1.0-xi)/N)*h_single - (xi/N/N)*h_int;
   [vve, eve] = eig(h,'vector');
   EigenE(:,id_xi) = eve(:);
   EigenV(:,id_xi,:) = vve(:,:);
 end
   
end




