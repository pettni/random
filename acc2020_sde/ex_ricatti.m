noise = 0.4;

lim = 2;

A = [0 1; -1 0];
B = [1 0;0 1];
E = noise * [1;1];

Q = 0.2*eye(2);

x0 = [1; 1];

Tmax = 15;
periods = 3000;
N = 1000;
SEP = 6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tt = linspace(0, Tmax, periods+1);

P = icare(A, B, Q);
K = 0.5*B'*P;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% exact solution %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

V_fun = @(x,y) [x y] * P * [x;y];

ex_paths = simulate(sde(@(t,X) (A-B*K)*X, @(t,X) E, 'StartState', x0), ...
                    periods, ...
                    'DeltaTime', Tmax/periods, ... 
                    'nTrials', N, ...
                    'Antithetic', false);

V_ex_paths = arrayfun(V_fun, ex_paths(:,1,:), ex_paths(:,2,:));
V_sup_ex_paths = cummax(V_ex_paths, 1);

ex_prob_tresh = sum(V_sup_ex_paths > lim, 3)/size(V_sup_ex_paths,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% lyapunov dynam %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% minimal alpha s.t. -Q <= -alpha * P
alpha_var = sdpvar;
sol = optimize([Q >= P*alpha_var], -alpha_var, sdpsettings('solver','mosek','verbose',0));

alpha_val = value(alpha_var); 
c_val = trace(E'*P*E);  % Frobenius product <E, PE>
L = chol(P, 'lower');
sig_val = 2*norm(L'*E);

sc_lim = alpha_val/c_val*lim;

scly_paths = simulate(sde(@(t,V) (1-V), ...
                          @(t,V) sig_val/sqrt(c_val)*sqrt(abs(V)), ...
                         'StartState', (alpha_val/c_val) *  V_fun(x0(1), x0(2))), ...
                    periods, 'DeltaTime', ...
                    alpha_val*Tmax/periods, ... 
                    'nTrials', N, ...
                    'Antithetic', false);

V_sup_scly_paths = cummax(scly_paths, 1);

scly_prob_tresh = sum(V_sup_scly_paths > sc_lim, 3)/size(V_sup_scly_paths,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% analytical %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

b_kush = (V_fun(x0(1), x0(2)) + (exp(tt*min(alpha_val, c_val/lim)) - 1) * max(lim, c_val/alpha_val))./(lim * exp(tt*min(alpha_val, c_val/lim)));

b_tedr = (V_fun(x0(1), x0(2)) + c_val*tt)/lim;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% plotting %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clf; 
subplot(311)
hold on
for i=1:100
  plot(tt, V_ex_paths(:,1,i))
end
xlim([0, Tmax]);
ylim([0, 2*lim]);
plot([0, tt(end)], [lim lim], 'k', 'linewidth', 3);
title("Actual SDE $V(X_t)$", 'interpreter', 'latex')

subplot(312)
hold on
for i=1:100
  plot(tt, scly_paths(:,1,i))
end
xlim([0, Tmax]);
ylim([0, 2*sc_lim]);
plot([0, tt(end)], [sc_lim sc_lim], 'k', 'linewidth', 3);
title("Scaled Lyapunov", 'interpreter', 'latex')

subplot(313)
hold on
plot(tt, ex_prob_tresh(:,1,1))
plot(tt, scly_prob_tresh(:,1,1))
plot(tt, b_kush)
plot(tt, b_tedr)
legend('Actual', 'Scaled Lyapunov', 'Kushner', 'Prajna/Steinhardt')
xlim([0, Tmax]);
title("Exit probabilities", 'interpreter', 'latex')

disp(strcat("alpha=", num2str(alpha_val)))
disp(strcat("c=", num2str(c_val)))
disp(strcat("sigma=", num2str(sig_val)))

csvwrite('lyap_exact.txt', [tt(1:SEP:end); lim*ones(size(tt(1:SEP:end))); permute(V_ex_paths(1:SEP:end, :, 1:100), [3 1 2])]')
csvwrite('lyap_bound.txt', [alpha_val*tt(1:SEP:end); sc_lim*ones(size(tt(1:SEP:end))); real(permute(scly_paths(1:SEP:end, :, 1:100), [3 1 2]))]')

csvwrite('est_prob_04.txt', [tt(1:SEP:end)' ex_prob_tresh(1:SEP:end) scly_prob_tresh(1:SEP:end) b_kush(1:SEP:end)' b_tedr(1:SEP:end)'])

