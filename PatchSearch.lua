require 'torch'
require 'nn'
require 'hdf5'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Input
cmd:option('-input_image', 'path/to/HDF5file', 'the image to get the nearest neighbour field for')
cmd:option('-input_patches', 'path/to/HDF5file', 'the patches to fiend the neighbours from')

-- Options
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-backend', 'nn', 'nn|cudnn')

-- Output 
cmd:option('-nnf', 'path/to/HDF5file', 'the file with the nearest neibour field')

local function main(params)
    -- Load auxillary functions
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

    -- Load image file 
    local f = hdf5.open(params.input_image, 'r')
    local img = f:all()['data']
    f:close()
    img = set_datatype(img, params.gpu)

    -- Load patches file 
    local f = hdf5.open(params.input_patches, 'r')
    local patches = f:all()['data']
    f:close()

    -- Build network for patch comparison
    local net = nn.Sequential()
    local n_in = img:size()[1]
    local n_out = patches:size()[1]
    local kh = patches:size()[3]
    local kw = patches:size()[4]
    local conv = nn.SpatialConvolution(n_in, n_out, kw, kh, 1, 1, 0, 0)
    for i =1, n_out do
        conv.weight[{{i},{},{},{}}] = patches[{{i},{},{},{}}]:clone()
    end
    net:add(conv)
    net = set_datatype(net, params.gpu)

    -- Get local energy of image for normalisation
    local norm_conv = nn.SpatialConvolution(n_in, 1, kw, kh, 1, 1, 0, 0)
    norm_conv.weight:fill(1)
    norm_conv = set_datatype(norm_conv, params.gpu)
    local pow_img = torch.pow(img,2)
    local img_norm = norm_conv:forward(pow_img):double()
    norm_conv = nil
    pow_img = nil
    collectgarbage()
    -- Find best matching patches by convolving with them
    out = net:forward(img):double()
    out = 2 * out -img_norm:expand(n_out, img_norm:size()[2], img_norm:size()[3])
    local _, nnf = torch.max(out:view(n_out,-1), 2)

    -- Save nearest neighbour field
    local f = hdf5.open(params.nnf, 'w')
    f:write('data', nnf:double())
    f:close()
end

local params = cmd:parse(arg)
main(params)




