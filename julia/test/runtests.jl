using Metatensor
using Test

TESTS = [
    # TODO: add tests files here
]

function main()
    @testset "Version" begin
        @test startswith(Metatensor.lib.version(), "0.")
    end

    for test in TESTS
        include(test)
    end
end

main()
