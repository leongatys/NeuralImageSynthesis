require 'torch'
require 'nn'
require 'loadcaffe'
require 'hdf5'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Input
cmd:option('-caffe_model', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-images', 'path/to/HDF5file', 'images to compute network activations for')
cmd:option('-layers', 'none', 'layers for which to return the dilated activations')
cmd:option('-dilation', '1', 'comma separated list what dilations to return')

-- Options
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-backend', 'nn', 'nn|cudnn')

-- Output 
cmd:option('-output_file', 'path/to/HDF5file', 'Name of the torch output file containing the activations')

local function main(params)
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
        cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
    end
    
    -- Load network from caffemodel
    local loadcaffe_backend = params.backend
    local cnn = loadcaffe.load('network', params.caffe_model, params.backend):float()
    cnn = set_datatype(cnn, params.gpu)

    -- Load images
    local f = hdf5.open(params.images, 'r')
    local images = f:all()['images']
    f:close()
    images = set_datatype(images, params.gpu)
    
    -- Pass image through network
    cnn:forward(images)
    
    -- For each layer in layers generate dilated activations
    local layers = params.layers:split(",")
    local dilation = params.dilation:split(",")
    for k, d in pairs(dilation) do
        dilation[k] = tonumber(d)
    end
    local dilated_activations = {}
    local next_layer_ndx = 1
    for i = 1, #cnn do
        if next_layer_ndx <= length(layers) then
            local layer = cnn:get(i)
            local layer_name = layer.name
            if layer_name == layers[next_layer_ndx] then
                dilated_activations[layer_name] = {}
                local input_layer = nil
                if i < 3 then
                    input_layer = nn.Identity()
                    input_layer = set_datatype(input_layer, params.gpu)
                    input_layer:forward(images)
                else
                    input_layer = cnn:get(i-2) -- get input from previous relu layer
                end
                local conv_layer = cnn:get(i-1) -- get original conv layer
                local n_in = input_layer.output:size()[1]
                local n_out = layer.output:size()[1]
                local kH,kW,dH,dW = conv_layer.kH, conv_layer.kW, conv_layer.dH, conv_layer.dW -- kernel size and stride
                for _, d in pairs(dilation) do
                    local dilation_net = nn.Sequential()
                    local p = d -- padding, equal to dilation for constant feature map size
                    local dilation_layer = nn.SpatialDilatedConvolution(n_in, n_out, kW, kH, dW, dH, p, p, d, d)
                    dilation_layer.weight = conv_layer.weight:clone()
                    dilation_layer.bias = conv_layer.bias:clone()
                    dilation_net:add(dilation_layer):add(nn.ReLU())
                    dilation_net = set_datatype(dilation_net, params.gpu)
                    dilation_net:forward(input_layer.output)
                        -- local debugger = require('fb.debugger')
                        -- debugger.enter()
                    dilated_activations[layer_name][d] = dilation_net.output:clone()
                end
                next_layer_ndx = next_layer_ndx + 1
            end
        end
    end
    cnn = nil
    collectgarbage()

    -- Write dilated activations in output file
    local f = hdf5.open(params.output_file, 'w')
    for _, layer in pairs(layers) do 
        for d, act in pairs(dilated_activations[layer]) do
                f:write(string.format(layer .. '/' .. '%i', d), act:double())
        end
    end
    f:close()
end

local params = cmd:parse(arg)
main(params)


