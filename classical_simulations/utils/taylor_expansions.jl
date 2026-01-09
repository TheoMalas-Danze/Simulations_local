# Define Taylor expansion of sin
function Taylor_sin(x, n)
    result = zero(x)
    for i in 0:div(n-1,2)
        result += (-1)^i * x^(2i + 1) / factorial(2i + 1)
    end
    return result
end

function Taylor_cos(x, n)
    result = zero(x)
    for i in 0:div(n,2)
        result += (-1)^i * x^(2i) / factorial(2i)
    end
    return result
end