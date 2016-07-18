require 'torch'
require 'nn'
require 'optim'
require 'loadcaffe'
require 'hdf5'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Input
cmd:option('-caffe_model', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-input_file', 'path/to/HDF5file', 
'Contains the targets for the activations of layers in the network that should be optimised for in order to synthesis an image')
cmd:option('-init_file', 'path/to/HDF5file', 'Initialisation of the gradient procedure for image synthesis')

-- Options
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-max_iter', 1000)
cmd:option('-backend', 'nn', 'nn|cudnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-layer_order', 'none', 'order of layers to be used in maybe_print function')

-- Output 
cmd:option('-output_file', 'path/to/HDF5file', 'Name of the torch output file')
cmd:option('-loss_file', 'path/to/HDF5file', 'Name of file in which the tracked loss is saved')

local function main(params)
    -- Load auxillary functions
    paths.dofile('LossLayers.lua')
    paths.dofile('Misc.lua')

    -- Set gpu mode
    if params.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(params.gpu + 1)
    else
        params.backend = 'nn'
    end
    if params.backend == 'cudnn' then
        require 'cudnn'
        if params.cudnn_autotune then
            cudnn.benchmark = true
        end
        cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
    end
    
    -- Load network from caffemodel
    local loadcaffe_backend = params.backend
    local cnn = loadcaffe.load('network', params.caffe_model, params.backend):float()
    cnn = set_datatype(cnn, params.gpu)

    -- Load optimisation targets 
    local f = hdf5.open(params.input_file, 'r')
    local opt_targets = f:all()
    f:close()

    -- Set up new network with appropriate loss layers
    local net = nn.Sequential()
    local loss_modules = {}
    local next_layer_ndx = 1
    -- Loss layers acting directly on the image
    if opt_targets['data'] then
        loss_modules['data'] = {}
        for loss_layer, args in pairs(opt_targets['data']) do
            local loss_module = get_loss_module(loss_layer, args)
            loss_module = set_datatype(loss_module, params.gpu)
            net:add(loss_module)
            loss_modules['data'][loss_layer] = loss_module
        end
        next_layer_ndx = next_layer_ndx + 1
    end
    -- Loss layers acting on CNN features
    for i = 1, #cnn do
        if next_layer_ndx <= length(opt_targets) then
            local layer = cnn:get(i)
            local layer_name = layer.name
            local layer_type = torch.type(layer)
            net:add(layer)
            if opt_targets[layer_name] then
                loss_modules[layer_name] = {}
                for loss_layer, args in pairs(opt_targets[layer_name]) do
                    local loss_module = get_loss_module(loss_layer, args)
                    loss_module = set_datatype(loss_module, params.gpu)
                    net:add(loss_module)
                    loss_modules[layer_name][loss_layer] = loss_module
                end
                next_layer_ndx = next_layer_ndx + 1
            end
        end
    end

    -- Get flat list of loss modules to call in feval
    local loss_modules_flat = {}
    for layer_name, layer_table in pairs(loss_modules) do
        for loss_layer, loss_module in pairs(layer_table) do
            loss_modules_flat[#loss_modules_flat + 1] = loss_module
        end
    end

    -- We don't need the base CNN anymore, so clean it up to save memory.
    cnn = nil
    for i=1,#net.modules do
        local module = net.modules[i]
        if torch.type(module) == 'nn.SpatialConvolutionMM' then
                module.gradWeight = nil
                module.gradBias = nil
        end
    end
    collectgarbage()
  
    -- Load initialisation 
    local f = hdf5.open(params.init_file, 'r')
    local img = f:all()['init']
    f:close()
    img = set_datatype(img, params.gpu)

    -- Run it through the network once to get the proper size for the gradient
    -- All the gradients will come from the extra loss modules, so we just pass
    -- zeros into the top of the net on the backward pass.
    local y = net:forward(img)
    local dy = img.new(#y):zero()

    -- Declare optimisation options
    local optim_state = {
      maxIter = params.max_iter,
      verbose = true,
      tolX = 0,
      tolFun = 0,
    }
    
    -- Get layer_order for use in maybe_print
    local layer_order = params.layer_order:split(",")

    -- Function to evaluate loss and gradient. We run the net forward and
    -- backward to get the gradient, and sum up losses from the loss modules.
    -- optim.lbfgs internally handles iteration and calls this fucntion many
    -- times, so we manually count the number of iterations to handle printing
    -- and saving intermediate results.
    local num_calls = 0
    local function feval(x)
        num_calls = num_calls + 1
        net:forward(x)
        local grad = net:updateGradInput(x, dy)
        local loss = 0
        for _, mod in ipairs(loss_modules_flat) do
            loss = loss + mod.loss
        end
        maybe_print(num_calls, params.print_iter, params.max_iter, layer_order, loss_modules, loss)
        maybe_save(num_calls, params.save_iter, params.max_iter, params.output_file, img)

        collectgarbage()
        -- optim.lbfgs expects a vector for gradients
        return loss, grad:view(grad:nElement())
    end

     -- Run optimization.
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)

    -- Also save result if optimisation stops before max iter is reached
    if num_calls < params.max_iter then
        maybe_save(params.max_iter, params.save_iter, params.max_iter, params.output_file, img)
    end

    -- Optionally save the loss as tracked over the optimisation
    if params.loss_file ~= 'path/to/HDF5file' then
        local f = hdf5.open(params.loss_file, 'w')
        f:write('losses', torch.Tensor(losses):double())
        f:close()
    end
end

local params = cmd:parse(arg)
main(params)
