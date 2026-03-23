
using Test

using TM_Test

function test1()
    boost = transfer_matrix(Dist,22.025e9,ones(20)*0.00721)[2][1]

    if !isapprox(boost,154680.45,atol=10)
        println("Value: ",boost,"\nTarget: ",154680.45); return false
    end

    return true
end

Test.@testset "main" begin
    @test test1()
end