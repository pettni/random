function msfuntmpl_basic(block)
	setup(block);
end

function setup(block)

	% Register number of ports
	block.NumInputPorts  = 1;
	block.NumOutputPorts = 0;

	% Setup port properties to be inherited or dynamic
	block.SetPreCompInpPortInfoToDynamic;
	block.SetPreCompOutPortInfoToDynamic;

	block.InputPort(1).Dimensions        = 2;
	block.InputPort(1).DatatypeID  = 0;  % double
	block.InputPort(1).Complexity  = 'Real';
	block.InputPort(1).DirectFeedthrough = true;
	block.NumDialogPrms     = 0;
	block.SampleTimes = [0 0];

	block.SimStateCompliance = 'DefaultSimState';

	block.RegBlockMethod('PostPropagationSetup',    @DoPostPropSetup);
	block.RegBlockMethod('InitializeConditions', @InitializeConditions);
	block.RegBlockMethod('Start', @Start);
	block.RegBlockMethod('Update', @Update);
	block.RegBlockMethod('Terminate', @Terminate); % Required

	%%
	%% PostPropagationSetup:
	%%   Functionality    : Setup work areas and state variables. Can
	%%                      also register run-time methods here
	%%   Required         : No
	%%   C MEX counterpart: mdlSetWorkWidths
end

function DoPostPropSetup(block)
	block.NumDworks = 0;
end

%%
%% InitializeConditions:
%%   Functionality    : Called at the start of simulation and if it is 
%%                      present in an enabled subsystem configured to reset 
%%                      states, it will be called when the enabled subsystem
%%                      restarts execution to reset the states.
%%   Required         : No
%%   C MEX counterpart: mdlInitializeConditions
%%
function InitializeConditions(block)

end


%%
%% Start:
%%   Functionality    : Called once at start of model execution. If you
%%                      have states that should be initialized once, this 
%%                      is the place to do it.
%%   Required         : No
%%   C MEX counterpart: mdlStart
%%
function Start(block)

	figHandle = get_param(block.BlockHandle, 'UserData')

	if ~isvalid(figHandle)
		figHandle = figure('Units',          'pixel',...
		                   'Position',       [0 0 300 300],...
		                   'Name',           '2D Particle',...
		                   'NumberTitle',    'off',...
		                   'IntegerHandle',  'off',...
		                   'Toolbar',        'none',...
		                   'Menubar',        'none');

		ud.XYAxes = axes(figHandle);
		hold on;

		fill(ud.XYAxes, [0 0 5 5], [0 5 5 0], 'k');

		fill(ud.XYAxes, ...
				 [2 3 3 5 5 3 3 2 2 0 0 2 2],...
				 [0 0 2 2 3 3 5 5 3 3 2 2 0], 'w');
	
		h1 = fill(ud.XYAxes, [2.75 2.75 2.25 2.25], [4.00 4.75 4.75 4.00], 'r', 'EdgeColor','none');
		h2 = fill(ud.XYAxes, [2.75 2.75 2.25 2.25], [0.25 1.00 1.00 0.25], 'r', 'EdgeColor','none');

		h3 = fill(ud.XYAxes, [4.00 4.75 4.75 4.00], [2.75 2.75 2.25 2.25], 'b', 'EdgeColor','none');
		h4 = fill(ud.XYAxes, [0.25 1.00 1.00 0.25], [2.75 2.75 2.25 2.25], 'b', 'EdgeColor','none');

		h5 = fill(ud.XYAxes, [2 2 3 3], [2 3 3 2], 'y', 'EdgeColor','none');


		set(h1, 'facealpha', 0.2);
		set(h2, 'facealpha', 0.2);
		set(h3, 'facealpha', 0.2);
		set(h4, 'facealpha', 0.2);
		set(h5, 'facealpha', 0.2);

		ud.plot_hist = plot(ud.XYAxes, 2.5*ones(1,200), 2.5*ones(1,200), '--g');
	
		ud.plot_mark = plot(ud.XYAxes, [2.5], [2.5],'s', ...
											'Markersize', 8, ...
											'MarkerFaceColor','g', ...
											'erasemode','background');

		% store figure handle in block data
		set_param(block.BlockHandle, 'UserData', figHandle);

		% store userdata in figure handle	
		set(figHandle, 'HandleVisibility', 'callback' , 'UserData', ud);

	end

  ud = get(figHandle, 'UserData');

  ud.plot_hist.XData = 2.5*ones(1,200);
  ud.plot_hist.YData = 2.5*ones(1,200);

  ud.plot_mark.XData = [2.5];
  ud.plot_mark.YData = [2.5];

end


%%
%% Update:
%%   Functionality    : Called to update discrete states
%%                      during simulation step
%%   Required         : No
%%   C MEX counterpart: mdlUpdate
%%
function Update(block)
	% Retrieve UserData
  figHandle = get_param(block.BlockHandle, 'UserData');
	
	if ~isvalid(figHandle),
	   return
	end

  ud = get(figHandle, 'UserData');

  x_new = block.InputPort(1).Data(1);
  y_new = block.InputPort(1).Data(2);

  ud.plot_hist.XData = [ud.plot_hist.XData(2:end) x_new];
  ud.plot_hist.YData = [ud.plot_hist.YData(2:end) y_new];

  ud.plot_mark.XData = [x_new];
  ud.plot_mark.YData = [y_new];

	set(ud.XYAxes, 'XLim', [0, 5]);
	set(ud.XYAxes, 'YLim', [0, 5]);
end

function Terminate(block)

end


function SetSfunXYFigure(block,figHandle)

	if strcmp(get_param(bdroot,'BlockDiagramType'),'model')
	  if strcmp(get_param(block,'BlockType'),'S-Function')
	    block=get_param(block,'Parent');
	  end

	  set_param(block,'UserData',figHandle);
	end

end