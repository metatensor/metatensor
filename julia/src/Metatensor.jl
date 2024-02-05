module Metatensor
    module lib
        using Preferences
        using ..Metatensor

        if has_preference(Metatensor, "libmetatensor")
            # if the users told us to use a custom libmetatensor (e.g. built
            # from the local checkout), load it!
            const libmetatensor = load_preference(Metatensor, "libmetatensor")
            const custom_build = true
        else
            using Metatensor_jll
            const custom_build = false
        end

        include("generated/_c_api.jl")

        function version()
            unsafe_string(lib.mts_version())
        end

        function __init__()
            if custom_build
                @info "using custom libmetatensor v$(version()) from $(libmetatensor)"
            end
        end
    end

end # module Metatensor
