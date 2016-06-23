require 'torch'
require 'nn'
require 'loadcaffe'
require 'hdf5'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Input
cmd:option('-caffe_model', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-images', 'path/to/HDF5file', 'images to compute network activations for')
cmd:option('-layers', 'all', 'layers for which to return the activations')

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
    local cnn = loadcaffe.load('none', params.caffe_model, params.backend):float()
    cnn = set_datatype(cnn, params.gpu)

    -- Load images
    local f = hdf5.open(params.images, 'r')
    local images = f:all()['images']
    f:close()
    images = set_datatype(images, params.gpu)

    -- Pass images through the network
    cnn:forward(images)

    local f = hdf5.open(params.output_file, 'w')
    if params.layers == 'all' then
        for i = 1, #cnn do
            local layer = cnn:get(i)
            local layer_name = layer.name
            local layer_type = torch.type(layer)
            f:write(layer_name, layer.output:double())
        end
    else
        local layers = params.layers :split(",")
        next_layer_ndx = 1
        for i = 1, #cnn do
            if next_layer_ndx <= length(layers) then
                local layer = cnn:get(i)
                local layer_name = layer.name
                local layer_type = torch.type(layer)
                if layer_name == layers[next_layer_ndx] then
                    f:write(layer_name, layer.output:double())
                    next_layer_ndx = next_layer_ndx + 1
                end
            end
        end
    end
    print(f:all()['relu1_1']:size())
    f:close()
    g = hdf5.open(params.output_file, 'r')
    print(g:all()['relu1_1']:size())
    g:close()
-- local debugger = require('fb.debugger')
-- debugger.enter()
end

local params = cmd:parse(arg)
main(params)


