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
    else
        layers = params.layers:split(",")
        local next_layer_ndx = 1
        for i = 1, #cnn do
            if next_layer_ndx <= length(layers) then
                local layer = cnn:get(i)
                local layer_name = layer.name
                net:add(layer)
                if layer_name == layers[next_layer_ndx] then
                    next_layer_ndx = next_layer_ndx + 1
                    net:add(nn.MeanMod())
                end
            end
        end
    end
    net = set_datatype(net, params.gpu)
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
            if layer_name == 'mean_fm' then
                f:write(layer_name..string.format('_%i',i), layer.mean_fm.output:double())
            end
        end
    else
        next_layer_ndx = 1
        for i = 1, #net do
            if next_layer_ndx <= length(layers) then
                local layer = net:get(i)
                local layer_name = layer.name
                local layer_type = torch.type(layer)
                if layer_name == 'mean_fm' then
                    f:write(layer_name..string.format('_%i',next_layer_ndx), layer.mean_fm.output:double())
                    next_layer_ndx = next_layer_ndx + 1
                end
            end
        end
    end
    f:close()
end

-- Computes mean feature maps
function MeanFM()
    local net = nn.Sequential()
    net:add(nn.View(-1):setNumInputDims(2))
    net:add(nn.Mean(2))
    return net
end

-- Define module that computes the mean feature maps in-place
local MeanMod, parent = torch.class('nn.MeanMod', 'nn.Module')

function MeanMod:__init()
    parent.__init(self)
    self.name = 'mean_fm'
    self.mean_fm = MeanFM()
end

function MeanMod:updateOutput(input)
    self.mean_fm:forward(input)
    self.output = input
    return self.output
end

function MeanMod:updateGradInput(input, gradOutput)
    self.gradInput:add(gradOutput)
    return self.gradInput
end

local params = cmd:parse(arg)
main(params)
