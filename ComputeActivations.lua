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
cmd:option('-reflectance', false, 'if true, use reflectance padding')

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
    
    -- Set up new network only using necessary layers 
    local net = nn.Sequential()
    local layers = nil 
    if params.layers == 'all' then
        net = cnn:clone()
        if params.reflectance then
            print('Warning, no reflectance padding for layers "all"')
        end
    else
        layers = params.layers:split(",")
        local next_layer_ndx = 1
        for i = 1, #cnn do
            if next_layer_ndx <= length(layers) then
                local layer = cnn:get(i)
                local layer_name = layer.name
                local is_convolution = (layer_type == 'cudnn.SpatialConvolution' or layer_type == 'nn.SpatialConvolution')
                if is_convolution and params.reflectance then
                    local padW, padH = layer.padW, layer.padH
                    local pad_layer = nn.SpatialReflectionPadding(padW, padW, padH, padH)
                    pad_layer = set_datatype(pad_layer, params.gpu)
                    net:add(pad_layer)
                    layer.padW = 0
                    layer.padH = 0
                end
                net:add(layer)
                if layer_name == layers[next_layer_ndx] then
                    next_layer_ndx = next_layer_ndx + 1
                end
            end
        end
    end
    cnn = nil
    collectgarbage()

    -- Pass images through the network
    net:forward(images)

    local f = hdf5.open(params.output_file, 'w')
    if params.layers == 'all' then
        for i = 1, #net do
            local layer = net:get(i)
            local layer_name = layer.name
            local layer_type = torch.type(layer)
            f:write(layer_name, layer.output:double())
        end
    else
        next_layer_ndx = 1
        if 'data'  == layers[next_layer_ndx] then
            f:write('data', images:double())
            next_layer_ndx = next_layer_ndx + 1
        end
        for i = 1, #net do
            if next_layer_ndx <= length(layers) then
                local layer = net:get(i)
                local layer_name = layer.name
                local layer_type = torch.type(layer)
                if layer_name == layers[next_layer_ndx] then
                    f:write(layer_name, layer.output:double())
                    next_layer_ndx = next_layer_ndx + 1
                end
            end
        end
    end
    f:close()
end

local params = cmd:parse(arg)
main(params)


