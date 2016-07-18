require 'torch'
require 'cutorch'
require 'hdf5'

-- Function to return the number of key value pairs in a table
function length(table)
    l = 0 
    for k,v in pairs(table) do
        l = l+1
    end
    return l
end

-- Function to iterate over nested table and return all non-table entries in flat table
function deep_table_iter(data, flat_table)
    if type(data) == "table" then
        for k,v in pairs(data) do 
            deep_table_iter(v, flat_table)  
        end
    else 
        flat_table[#flat_table + 1] = data
    end
    return flat_table
end

-- Function to set gpu/cpu datatype
function set_datatype(data, gpu)
    if gpu >= 0 then
        data = data:cuda()
    else
        data = data:float()
    end
    return data
end

-- Function to get the appropriate loss module with arguments
function get_loss_module(loss_layer, args)
    if loss_layer == 'MSE' then
        return nn.MSE(args['targets'], args['weights']) 
    elseif loss_layer == 'GramMSE' then
        if args['guidance'] then
            return nn.GramMSE(args['targets'], args['weights'], args['guidance'])
        else
            return nn.GramMSE(args['targets'], args['weights'])
        end
    elseif loss_layer == 'TVLoss' then
        return nn.TVLoss(args['weight'])
    elseif loss_layer == 'L1Penalty' then
        return nn.L1Penalty(args['weight'][1], true)
    elseif loss_layer == 'L2Penalty' then
        return nn.L2Penalty(args['weight'][1], true)
    end
end

-- Function to print intermediate loss values
function maybe_print(t, print_iter, max_iter, layer_order, loss_modules, loss)
    local verbose = (print_iter > 0 and t % print_iter == 0)
    if verbose then
        print(string.format('Iteration %d / %d', t, max_iter))
        for _, layer_name in ipairs(layer_order) do
            local layer_table = loss_modules[layer_name]
            print(layer_name)
            for loss_layer, loss_module in pairs(layer_table) do
                print(string.format(loss_layer .. ' loss: %f', loss_module.loss))
            end
        end
        print(string.format('Total loss: %f', loss))
    end
end

-- Function to save intermediate results
function maybe_save(t, save_iter, max_iter, output_file, opt_result)
    local should_save = save_iter > 0 and t % save_iter == 0
    should_save = should_save or t == max_iter
    if should_save then
        -- local filename = build_filename(output_file, t)
        local filename = nil
        if t == max_iter then
            filename = output_file
        end
        local f = hdf5.open(filename, 'w')
        f:write('opt_result', opt_result:double())
        f:close()
    end
end

