% DYNAMICS
A = [0 1; 0 0];
B = [0;1];
E = [0;1];
x0 = [4; -3];
c_sigm = 0.2;   % noise magnitude

% CBF CONSTANTS
c_alph = 0.1;
c_beta = 1;
M = -1;

% SIMULATION CONSTANTS
Tmax = 14;
periods = 3000;
N = 10000;

SEP = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

V = @(x1, x2) - x1 - c_beta * x2;
lim = 0;

u = @(X) (c_alph * (V(X(1),X(2)) - M) - X(2))/c_beta;

% Exact system
drift = @(t, X) A*X + B*u(X);
ex_paths = simulate(sde(@(t,X) A*X + B*u(X), @(t,X) c_sigm * E, 'StartState', x0), ...
                    periods, ...
                    'DeltaTime', Tmax/periods, ... 
                    'nTrials', N, ...
                    'Antithetic', true);

V_ex_paths = arrayfun(V, ex_paths(:,1,:), ex_paths(:,2,:));
V_sup_ex_paths = cummax(V_ex_paths, 1);

% probability of exceeding V = M
ex_prob_tresh = sum(V_sup_ex_paths > lim, 3)/N;

% probability of having x1 <= 0
fail_ex = cummin(ex_paths(:, 1, :), 1);
ex_prob_fail = sum(fail_ex < 0, 3)/N;

figure(1)
clf; 
plot([-1 6], -1/c_beta*[-1 6])
hold on
for i=1:2
	plot(ex_paths(:, 1, i), ex_paths(:, 2, i))
end
xlim([-0.1 x0(1)])
ylim([x0(2) 1])

% Comparison system
if true  % scaled
	comp_x0 = sqrt(c_alph)/(c_beta * c_sigm) * (V(x0(1), x0(2)) - M);
	comp_Tmax = c_alph * Tmax;
	comp_lim = sqrt(c_alph)/(c_beta * c_sigm) * (lim - M);

	comp_paths = simulate(sde(@(t,X) -X, @(t,X) 1, 'StartState', comp_x0), ...
											  periods, ...
											  'DeltaTime', comp_Tmax/periods, ...
											  'nTrials', N, ...
											  'Antithetic', true);

	sup_scly_paths = cummax(comp_paths, 1);
	scly_prob_tresh = sum(sup_scly_paths > comp_lim, 3)/N;

else % non-scaled
	comp_x0 = V(x0(1), x0(2)) - M;
	comp_Tmax = Tmax;
	comp_lim = lim - M;

	comp_paths = simulate(sde(@(t,X) -c_alph*X, @(t,X) c_beta * c_sigm, 'StartState', comp_x0), ...
											  periods, ...
											  'DeltaTime', comp_Tmax/periods, ...
											  'nTrials', N, ...
											  'Antithetic', true);

	sup_scly_paths = cummax(comp_paths, 1);
	scly_prob_tresh = sum(sup_scly_paths > comp_lim, 3)/N;
end

figure(2)
subplot(211)
hold on
for i=1:20
	plot(V_ex_paths(:,1,i))
end
subplot(212)
hold on
for i=1:20
	plot(comp_paths(:,1,i))
end

figure(3)
clf; hold on
plot(ex_prob_fail)
plot(ex_prob_tresh)
plot(scly_prob_tresh)
legend('fail', 'tresh', 'bound')

tt = linspace(0, Tmax, periods+1);

csvwrite('cbf_prob.txt', [tt(1:SEP:end)' ex_prob_fail(1:SEP:end) ex_prob_tresh(1:SEP:end) scly_prob_tresh(1:SEP:end)])
