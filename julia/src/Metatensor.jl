module Metatensor

    module lib
        using Metatensor_jll
        include("generated/_c_api.jl")
    end

    function version()
        unsafe_string(lib.mts_version())
    end

end # module Metatensor
